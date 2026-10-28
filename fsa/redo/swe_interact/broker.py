"""Host-side broker for the SWE-Interact arms and pushback overlay (redo_v1 Stage 1).

The benchmark's ``user-server`` sidecar (``overlay/server.py``) forwards its two hooks as files in
the trial's ``user-sim/rpc/`` bind mount; one ``TrialBroker`` thread per trial, started by the
runner before ``harbor run`` and stopped after it, answers them:

``turn_end-N.req.json``  (agent turn N ended with ``ask_user``; snapshot ``ask-N`` taken)
    * A5.14 (``a5_<arm>``): the sidecar Claim Book gate (``a5.Gate``): the first call at step >= 02_implement is
      withheld in artifact arms (``a5_gate.action = withhold``, the demand is the tool result), the next payloads are
      validated, then ``release`` makes the simulator answer the withheld question (record ``user-sim/a5/gate.json``);
    * commit_before / claims_no_p: write the claim book now (``user-sim/claims/ask-N.json``);
      the sidecar consults the simulator only after this response, so the book provably exists
      before the reply (``written_at`` < the sidecar's ``sim_consult_started_ts``);
    * overlay trial: tag the simulator messages delivered so far (revealed intents) and start the
      snapshot judge on the revealed intents in the background (concurrently with the simulator).
``sim_reply-N.req.json``  (the simulator decided; its text is in the request)
    * overlay trial: wait for the judge (at most ``overlay_wait`` s after the simulator decided;
      unfinished -> no overlay at this turn, recorded); ``Overlay.decide`` -> the overlay message
      replaces the reply (it is also what the simulator's history holds);
    * claims_after: write the claim book now, with the (possibly replaced) reply visible;
    * tracker / mediator (controlled ports of IDSS / Mediator-Assistant, ``fsa/redo/ports.py``): one
      call of the agent's own model now, with the (possibly replaced) reply visible - the tracker
      updates the situation state (``user-sim/ports/ask-N.json``), the mediator rewrites the reply;
    * compose the delivered message (reply + claim book + rules / recap / minimal-edit note /
      tracker state; the mediator's rewrite instead of the reply).

All outputs stay in ``user-sim/``: ``claims/``, ``claim_calls.jsonl`` (the claim-book seat),
``overlay_judge/ask-N.json``, ``overlay/ask-N.json``, ``broker.jsonl`` (one line per request with
timings), ``rpc/`` (requests and responses, kept). A broker failure never blocks the trial: the
sidecar falls back to the plain reply and records the error (``asks.jsonl`` ``redo``).
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from . import a5 as A5
from . import tagger
from .. import ports as P
from .arms import (MAX_OVERLAYS, Overlay, RedoConfig, SwiArms, dialogue_messages, session_transcript,
                   step_streams)
from .config import TASKS

OVERLAY_WAIT_SEC = 240.0   # max wait for the in-trial judge after the simulator decided
POLL_SEC = 0.2


def _atomic_write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, p)


class TrialBroker:
    def __init__(self, job_dir: Path, task: str, agent_model: str, cfg: RedoConfig, *, judge=None,
                 overlay_wait: float = OVERLAY_WAIT_SEC, arms: SwiArms | None = None, tag_fn=None):
        self.job_dir = Path(job_dir)
        self.task = task
        self.task_dir = TASKS / task
        self.cfg = cfg
        self.agent_model = agent_model
        self.overlay_wait = overlay_wait
        self._judge = judge
        self._arms = arms
        self._port: P.PortArm | None = None
        self._tag_fn = tag_fn or tagger.tag_message
        self.trial_dir: Path | None = None
        self.overlay = Overlay(cfg, task) if cfg.overlay else None
        self.asks: list[dict] = []           # completed asks: question + reply the agent got (plain)
        self.delivered_sim: list[dict] = []  # simulator messages actually delivered (reveal intents)
        self._tags: dict[int, Future] = {}
        self._judge_futs: dict[int, Future] = {}
        self._pending_q: dict[int, str] = {}
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"broker-{task[:20]}")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._intents = None
        self._stats_lock = threading.Lock()
        self._tags_counted: set[int] = set()
        self.errors: list[str] = []
        self._gate: A5.Gate | None = None   # A5.14 sidecar gate (a5_ arms only)

    # ── lifecycle ──

    def start(self) -> "TrialBroker":
        self._thread = threading.Thread(target=self._loop, name=f"broker-{self.task[:30]}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)
        self._pool.shutdown(wait=False, cancel_futures=True)

    @property
    def arms(self) -> SwiArms:
        if self._arms is None:
            self._arms = SwiArms(self.cfg, self.user_sim, self.agent_model)
        return self._arms

    @property
    def port(self) -> P.PortArm:
        if self._port is None:
            self._port = P.PortArm(self.cfg.port, self.user_sim, self.agent_model, prefix="ask")
        return self._port

    @property
    def user_sim(self) -> Path:
        return self.trial_dir / "user-sim"

    @property
    def judge(self):
        if self._judge is None:
            from .judge import SnapshotJudge
            self._judge = SnapshotJudge()
        return self._judge

    @property
    def intents(self) -> dict:
        if self._intents is None:
            from .intents import load
            self._intents = load(self.task)
        return self._intents

    def _log(self, rec: dict) -> None:
        try:
            with open(self.user_sim / "broker.jsonl", "a") as f:
                f.write(json.dumps({"ts": time.time(), **rec}, default=str) + "\n")
        except OSError:
            pass

    def _loop(self) -> None:
        done: set[str] = set()
        last_alive = 0.0
        while not self._stop.is_set():
            try:
                if self.trial_dir is None:
                    cands = [d for d in self.job_dir.glob("*/user-sim") if d.is_dir()]
                    if cands:
                        self.trial_dir = cands[0].parent
                if self.trial_dir is not None:
                    rpc = self.user_sim / "rpc"
                    if time.time() - last_alive > 5:
                        try:
                            rpc.mkdir(parents=True, exist_ok=True)
                            (rpc / ".alive.tmp").write_text(str(time.time()))
                            os.replace(rpc / ".alive.tmp", rpc / "broker.alive")
                            last_alive = time.time()
                        except OSError:
                            pass
                    for req in sorted(rpc.glob("*.req.json"), key=lambda p: p.stat().st_mtime) if rpc.exists() else []:
                        if req.name in done:
                            continue
                        done.add(req.name)
                        self._handle(req)
            except Exception as e:  # noqa: BLE001 - keep serving
                self.errors.append(f"{type(e).__name__}: {e}")
            self._stop.wait(POLL_SEC)

    def _handle(self, req_path: Path) -> None:
        t0 = time.time()
        try:
            req = json.loads(req_path.read_text())
        except (json.JSONDecodeError, OSError):
            time.sleep(0.2)
            req = json.loads(req_path.read_text())
        kind, n = req["kind"], int(req["call_index"])
        try:
            resp = self.turn_end(n, req) if kind == "turn_end" else self.sim_reply(n, req)
            resp["ok"] = True
        except Exception as e:  # noqa: BLE001
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
            self.errors.append(resp["error"])
        resp.update(broker_received_ts=t0, broker_done_ts=time.time())
        _atomic_write(req_path.with_name(req_path.name.replace(".req.json", ".resp.json")), resp)
        self._log({"kind": kind, "call_index": n, "sec": round(time.time() - t0, 3), "ok": resp["ok"],
                   **({"error": resp["error"]} if not resp["ok"] else {})})

    # ── hooks ──

    def _transcript(self, pending_question: str | None) -> str:
        return session_transcript(self.trial_dir, self.task_dir, self.asks, pending_question,
                                  self.cfg.transcript_chars)

    def _snapshot_text(self, n: int) -> str:
        p = self.user_sim / "snapshots" / f"ask-{n}.patch"
        return p.read_text(errors="surrogateescape") if p.exists() else ""

    def turn_end(self, n: int, req: dict) -> dict:
        q = req.get("question") or ""
        self._pending_q[n] = q
        out: dict = {}
        if A5.base_arm(self.cfg.arm) is not None:
            out.update(self._a5_turn_end(n, q, req))
        if self.overlay is not None and not self.overlay.done:
            self._judge_futs[n] = self._pool.submit(self._overlay_verdict, n)
            out["overlay_judge_started_ts"] = time.time()
        if self.cfg.claim_timing == "before":
            doc = self.arms.write_claim_book(call_index=n, transcript=self._transcript(q),
                                             diff_text=self._snapshot_text(n), reply=None)
            out.update(claims_book=f"claims/ask-{n}.json", claims_ok=doc["ok"], claims_phase=doc["phase"],
                       claims_call_started_ts=doc["call_started_at"], claims_written_ts=doc["written_at"])
        return out

    def sim_reply(self, n: int, req: dict) -> dict:
        reply = req.get("reply") or ""
        out: dict = {"opened_by": "sim"}
        eff = reply
        if n in self._judge_futs:
            deadline = float(req.get("sim_decided_ts") or time.time()) + self.overlay_wait
            fut = self._judge_futs.pop(n)
            try:
                verdict = fut.result(timeout=max(0.0, deadline - time.time()))
            except Exception as e:  # noqa: BLE001 - timeout or judge error: no overlay this turn
                verdict = {"all_revealed_satisfied": False, "error": f"{type(e).__name__}: {e}"[:300],
                           "judge_unfinished": not fut.done()}
                _atomic_write(self.user_sim / "overlay_judge" / f"ask-{n}.wait.json",
                              {"call_index": n, **verdict, "waited_until": time.time()})
            out["overlay_judge"] = f"overlay_judge/ask-{n}.json"
            out["overlay_all_revealed_satisfied"] = verdict.get("all_revealed_satisfied")
            ov = self.overlay.decide(n, verdict)
            if ov is not None:
                eff = ov["message"]
                out.update(opened_by="overlay", overlay_kind=ov["kind"], overlay_index=ov["index"],
                           overlay_target_intent=ov["target_intent"], sim_message_replaced=reply)
                _atomic_write(self.user_sim / "overlay" / f"ask-{n}.json", {**ov, "sim_message_replaced": reply})
        # bookkeeping for revelation and the transcript (the simulator "said" eff)
        self.asks.append({"call_index": n, "question": self._pending_q.pop(n, ""), "reply": eff})
        if out["opened_by"] == "sim" and not req.get("last_resort"):
            self.delivered_sim.append({"call_index": n, "reply": reply})
            if self.overlay is not None:
                self._tags[n] = self._pool.submit(self._tag_fn, reply, self.intents)
        book = None
        if self.cfg.claim_timing == "after":
            book = self.arms.write_claim_book(call_index=n, transcript=self._transcript(None),
                                              diff_text=self._snapshot_text(n), reply=eff)
            out.update(claims_book=f"claims/ask-{n}.json", claims_ok=book["ok"], claims_phase=book["phase"],
                       claims_call_started_ts=book["call_started_at"], claims_written_ts=book["written_at"])
        elif self.cfg.claim_timing == "before":
            p = self.user_sim / "claims" / f"ask-{n}.json"
            book = json.loads(p.read_text()) if p.exists() else None
        out["history_reply"] = eff
        if self.cfg.port:  # tracker / mediator: after the simulator decided, reply visible
            meta = {"call_index": n, "sim_decided_ts": req.get("sim_decided_ts"), "opened_by": out["opened_by"]}
            if self.cfg.port == "tracker":
                doc = self.port.tracker_update(index=n, transcript=self._transcript(None),
                                               streams=step_streams(self.trial_dir),
                                               diff_text=self._snapshot_text(n), snapshot=f"ask-{n}", reply=eff,
                                               meta=meta)
            else:
                doc = self.port.mediate(index=n, dialogue=dialogue_messages(self.trial_dir, self.task_dir, self.asks),
                                        meta=meta)
            out.update(port_doc=f"ports/ask-{n}.json", port_ok=doc["ok"], port_phase=doc["phase"],
                       port_call_started_ts=doc["call_started_at"], port_written_ts=doc["written_at"])
            out["deliver"] = self.port.compose(eff, doc)
        else:
            out["deliver"] = self.arms.compose(eff, book, "")
        if A5.base_arm(self.cfg.arm) is not None:
            a5out = self._a5_note(n, eff)
            out.update(a5out)
            out["deliver"] = A5.compose(out["deliver"], a5out.get("a5_note") or "")
        return out

    # ── A5.11 / A5.14 sidecar gate and notes ──

    def _a5_chat(self, model: str, messages: list[dict]) -> dict:
        from ..swe_pushback import llm as dsllm
        return dsllm.chat(model, messages, effort="low", temperature=0, max_tokens=4096, json_mode=True)

    @property
    def gate(self) -> "A5.Gate":
        if self._gate is None:
            self._gate = A5.Gate(A5.base_arm(self.cfg.arm))
        return self._gate

    def _a5_save_gate(self) -> None:
        _atomic_write(self.user_sim / "a5" / "gate.json", {**self.gate.doc, "written_at": time.time()})

    def _a5_turn_end(self, n: int, q: str, req: dict) -> dict:
        """A5.14 (1): the gate decision for ask_user call ``n`` (before the simulator is consulted). Returns
        ``a5_gate`` = {phase, action?, message? | question?} for the sidecar; a broker error never withholds."""
        try:
            step = A5.current_step(self.trial_dir)
            dec = self.gate.on_ask(n, q, step)
            if dec.get("action") == "release":
                self._pending_q[n] = dec["question"]   # the simulator answers the withheld question
            rec = {"phase": dec["phase"], "step": step, "gate_status": self.gate.doc["status"],
                   "gate_call": self.gate.doc["call_index"], "snapshot_patch_sha256": req.get("snapshot_patch_sha256")}
            self._a5_save_gate()
            return {"a5_gate": {**dec, **rec}}
        except Exception as e:  # noqa: BLE001 - recorded; the sidecar proceeds (and releases a held question)
            self.gate.fail(n, f"{type(e).__name__}: {e}")
            self.errors.append(f"a5 gate: {type(e).__name__}: {e}")
            try:
                self._a5_save_gate()
            except Exception:  # noqa: BLE001
                pass
            return {"a5_gate": {"phase": "broker_error", "error": f"{type(e).__name__}: {e}"[:300]}}

    def _a5_note(self, n: int, reply: str) -> dict:
        """The note for simulator reply ``n``: none before the gate has resolved (01_plan replies; arms byte-identical
        until the gate); from the gate on, the arm's dspush note (``a5.note``), routed through the claim book for
        p-arms with a valid artifact. Written to ``user-sim/a5/ask-N.json``."""
        arm = A5.base_arm(self.cfg.arm)
        if not self.gate.resolved:
            return {"a5_phase": "pre_gate"}
        doc = self.gate.doc
        ok = doc["status"] == "ok" and isinstance(doc.get("artifact"), dict)
        rt = None
        if arm in A5.T.P_ARMS and ok:
            rt = A5.route(doc["artifact"], reply, chat=self._a5_chat)
        text, enforced = A5.note(arm, rt, ok)
        rec = {"call_index": n, "arm": arm, "phase": "post_gate", "artifact_status": doc["status"],
               "gate_call": doc["call_index"], "resolved_call": doc["resolved_call"],
               "route": rt, "note": text, "enforced": enforced, "written_at": time.time()}
        _atomic_write(self.user_sim / "a5" / f"ask-{n}.json", rec)
        return {"a5_phase": "post_gate", "a5_note": text, "a5_enforced": enforced, "a5_route": rt,
                "a5_artifact_status": doc["status"], "a5_gate_call": doc["call_index"]}

    # ── overlay judge ──

    def _overlay_verdict(self, n: int) -> dict:
        t0 = time.time()
        doc = self.intents
        tags = []
        for m in self.delivered_sim:
            fut = self._tags.get(m["call_index"])
            if fut is None:
                fut = self._tags[m["call_index"]] = self._pool.submit(self._tag_fn, m["reply"], doc)
            tags.append(fut.result())
        rev = tagger.revealed_sets(tags, doc)["tagger"]
        # trigger scope: revealed intents except rf's behaviour_preserved (its executable check is the
        # full test suite, minutes long; the overlay fires on the rubric items there)
        its = [i for i in doc["intents"] if i["key"] in rev and i["judge"] != "exec_rf"]
        mode = (doc.get("validation") or {}).get("trigger_judge", "exec")
        if mode == "llm":  # slow hidden tests (validation exec_wall_sec_max > 150 s): LLM judge in-trial
            its = [{**i, "judge": "llm"} if i["judge"] == "exec" else i for i in its]
        patch = self._snapshot_text(n)
        t_j = time.time()
        with self._stats_lock:  # this broker's own judge: the deltas are this trial's judge seat
            before = {**{k: v for k, v in self.judge.llm.stats.items()},
                      **{f"rubric_{k}": v for k, v in self.judge.rubric.stats.items()}}
        v = self.judge.verdicts(self.task, patch, its, exec_mode="run", label=f"trial ask-{n}") if its else {"_meta": {}}
        after = {**self.judge.llm.stats, **{f"rubric_{k}": v_ for k, v_ in self.judge.rubric.stats.items()}}
        delta = {k: after[k] - before.get(k, 0) for k in after}
        with self._stats_lock:  # each fresh tagger call is counted once (futures are re-read every turn)
            fresh = [(m["call_index"], t) for m, t in zip(self.delivered_sim, tags)
                     if not t.get("cached") and m["call_index"] not in self._tags_counted]
            self._tags_counted.update(ci for ci, _ in fresh)
        tag_usage = [t.get("usage") or {} for _, t in fresh]
        stats_delta = {"calls": delta.get("calls", 0) + delta.get("rubric_calls", 0) + len(tag_usage),
                       "input_tokens": delta.get("input_tokens", 0) + delta.get("rubric_input_tokens", 0)
                       + sum(u.get("input_tokens") or 0 for u in tag_usage),
                       "output_tokens": delta.get("output_tokens", 0) + delta.get("rubric_output_tokens", 0)
                       + sum(u.get("output_tokens") or 0 for u in tag_usage),
                       "detail": delta}
        meta = v.pop("_meta", {})
        labels = {k: x["verdict"] for k, x in v.items()}
        applicable = [k for k, lab in labels.items() if lab != "not_applicable"]
        all_sat = bool(applicable) and all(labels[k] == "satisfied" for k in applicable)
        rec = {"call_index": n, "n_sim_messages": len(self.delivered_sim), "revealed": rev,
               "judged": [i["key"] for i in its], "labels": labels,
               "judges": {k: x.get("judge") for k, x in v.items()},
               "evidence": {k: (x.get("evidence") or "")[:300] for k, x in v.items()},
               "all_revealed_satisfied": all_sat, "headline": tagger.headline(doc),
               "snapshot_chars": len(patch or ""), "n_satisfied": sum(1 for k in applicable if labels[k] == "satisfied"),
               "texts": {i["key"]: i["text"][:300] for i in its}, "exec": meta.get("exec"),
               "stats_delta": stats_delta, "trigger_judge": mode,
               "tag_sec": round(t_j - t0, 2), "judge_sec": round(time.time() - t_j, 2),
               "started_at": t0, "finished_at": time.time(), "intent_digest": doc["digest"],
               "max_overlays": MAX_OVERLAYS}
        _atomic_write(self.user_sim / "overlay_judge" / f"ask-{n}.json", rec)
        return rec
