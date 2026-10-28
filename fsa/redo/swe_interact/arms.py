"""Redo arms and the pushback overlay for SWE-Interact (redo_v1 Stage 1, PLAN.md §6-§8).

A port of the SWE-Together module (clone ``src/user_agent/redo_arms.py``): the same arms, the
same prompts (``prompts/redo/``, byte-identical copies; ``tests/test_redo_swi_arms.py`` checks
the hashes against the clone), tau = 0.5, the same claim-book call (agent's own Gemini model
through LiteLLM, ``reasoning_effort=high``, JSON output, closing request, patient retry) and the
same overlay rule. What differs is only where it runs and what the transcript is made of:

* **Where.** SWE-Interact's user loop lives in the benchmark's ``user-server`` sidecar
  (``ask_user`` MCP tool). The sidecar calls two hooks (``overlay/server.py``):
  ``turn_end`` (after the per-turn snapshot, before the simulator is consulted) and
  ``sim_reply`` (after the simulator decided, before the reply is returned to the agent).
  Both are forwarded over files in the trial's ``user-sim/rpc/`` bind mount to the host-side
  broker (``broker.py``), which holds one ``SwiArms`` per trial and does the work here. The
  host has the Gemini key, the tib checkout (intent sets, judges) and docker (executable judge).
* **Transcript.** SWE-Interact runs every step as a fresh opencode session; the transcript
  is rebuilt from the step instructions, the opencode event streams of the finished steps
  (``steps/<step>/agent/opencode.attempt-*.txt``) and of the running step (``agent/``), and
  the plain simulator replies (``user-sim/asks.jsonl``; the arm block never re-enters it).
  Turn = the agent's work between two ``ask_user`` calls / step boundaries, rendered in the
  SWE-Together block format ("## Turn t - the user: ...", "### Your work in turn t").
* **Across steps.** The claim book is carried across the five steps of a trial (one broker
  state per trial: the previous book is the input of the next call, whatever step it is in),
  because the steps are one task and one user conversation; the new opencode session of a
  step receives the carried book with the first user reply of that step. Nothing carries
  across trials/tasks (a new broker state per trial).
* **recap.** The benchmark's opening message is a step instruction without the task, so the
  recap lists the user's (simulator's) messages only; the "(opening request)" line of the
  SWE-Together recap is omitted (``compose(..., instruction="")``). Recorded deviation.

The simulator never sees a claim book, recap or rule block: the sidecar keeps the simulator's
own reply (or, on an overlay turn, the overlay message) in the simulator's history and only
the message returned to the agent carries the arm block.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import ports as _ports

PROMPTS = Path(__file__).with_name("prompts") / "redo"
# tracker / mediator: controlled ports of IDSS (arXiv 2608.15755) and Mediator-Assistant
# (arXiv 2602.07338), fsa/redo/ports.py; wired in broker.py (sim_reply hook), not native reproductions
PORT_ARMS = _ports.PORT_ARMS
ARMS = ("stock", "commit_before", "claims_after", "claims_no_p", "recap", "minimal_edit",
        # Wave B (cost-aware TIB + test-time-compute baselines):
        "cb_gated",            # one cheap confidence call before the reply; full claim book only if p < GATE_TAU
        "conf_gate",           # cheap confidence after the reply; hold if p >= GATE_TAU (confidence without grounding)
        "sc_vote3",            # 3 independent decision samples after the reply; majority action (self-consistency)
        "verify_then_respond",  # no seat call: re-run tests before acting (re-observation without pre-commitment)
        *PORT_ARMS,
        # A5.11 (REVIEWER_EXPS_PREREGISTRATION.md): dspush arms behind the 02_implement gate (fsa/redo/swe_interact/a5.py)
        "a5_stock", "a5_rule_only", "a5_review_only", "a5_commit_before", "a5_claims_p_noevid", "a5_stub_3")
CLAIM_TIMING = {"commit_before": "before", "claims_no_p": "before", "claims_after": "after",
                "cb_gated": "before", "conf_gate": "after", "sc_vote3": "after"}
SEAT_ARMS = ("cb_gated", "conf_gate", "sc_vote3")   # arms whose seat call is a gate / vote, not a claim book
GATE_TAU = 0.9
VOTE_N = 3
MAX_OVERLAYS = 2
# Wave A (appropriate deference): the mixed overlay adds at most one *justified* correction per trial
# (the simulator names an intent the judge found unsatisfied) to the two standard ones.
MAX_OVERLAYS_MIXED = 3
OVERLAY_MODES = ("standard", "mixed")
UNSAT_LABELS = frozenset({"not_satisfied", "not_addressed"})
STEP_ORDER = ("01_plan", "02_implement", "03_handoff", "04_write_tests", "05_test_handoff")

_TRANSIENT = ("429", "rate limit", "ratelimit", "resource_exhausted", "resource has been exhausted", "quota",
              "503", "unavailable", "overloaded", "500", "502", "504", "internal error", "internalservererror",
              "timeout", "timed out", "connection", "high demand", "empty response", "invalid json")
BACKOFF_SEC = (10, 20, 40, 60, 90, 120, 180, 240, 300)


def prompt(name: str) -> str:
    return (PROMPTS / name).read_text()


def prompts_sha256() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PROMPTS.iterdir()) if p.is_file()}


@dataclass
class RedoConfig:
    arm: str = "stock"
    tau: float = 0.5
    claim_reasoning: str = "high"
    overlay: bool = False
    overlay_seed: int = 0
    overlay_mode: str = "standard"  # "standard" (unjustified + misaimed) | "mixed" (+ one justified)
    transcript_chars: int = 160_000

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm {self.arm!r} not in {ARMS}")
        if self.overlay_mode not in OVERLAY_MODES:
            raise ValueError(f"overlay_mode {self.overlay_mode!r} not in {OVERLAY_MODES}")

    @property
    def max_overlays(self) -> int:
        return MAX_OVERLAYS_MIXED if self.overlay_mode == "mixed" else MAX_OVERLAYS

    @property
    def active(self) -> bool:
        return self.arm != "stock" or self.overlay

    @property
    def claim_timing(self) -> str | None:
        return CLAIM_TIMING.get(self.arm)

    @property
    def port(self) -> str | None:
        return self.arm if self.arm in PORT_ARMS else None


# ── transcript ──────────────────────────────────────────────────────────────

def _events(raw: str) -> list[tuple[str, str, dict]]:
    """(kind, rendered text, raw part) from an ``opencode run --format=json`` stream (same
    rendering as the SWE-Together module; ``ask`` = an ask_user tool call)."""
    out = []
    for line in (raw or "").split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        part = e.get("part") or {}
        if e.get("type") == "text":
            t = part.get("text")
            if isinstance(t, str) and t.strip():
                out.append(("text", t.strip()[:3000], part))
        elif e.get("type") == "tool_use":
            st = part.get("state") if isinstance(part.get("state"), dict) else {}
            args = st.get("input") or part.get("input") or {}
            name = part.get("tool") or part.get("name") or "?"
            if "ask_user" in name:
                q = args.get("question") if isinstance(args, dict) else None
                out.append(("ask", str(q if q is not None else args)[:3000], part))
                continue
            args = args if isinstance(args, str) else json.dumps(args)
            res = st.get("output") or part.get("output") or ""
            res = res if isinstance(res, str) else json.dumps(res)
            res = res if len(res) <= 600 else "..." + res[-600:]
            out.append(("tool", f"{name}({args[:400]}) -> {res}", part))
    return out


def _step_events(agent_dir: Path) -> list[tuple[str, str, dict]]:
    ev = []
    files = sorted(glob.glob(str(agent_dir / "opencode.attempt-*.txt")),
                   key=lambda f: int(re.search(r"attempt-(\d+)", f).group(1)))
    for f in files:
        ev += _events(Path(f).read_text(errors="replace"))
    return ev


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def session_transcript(trial_dir: Path, task_dir: Path, asks: list[dict], pending_question: str | None,
                       budget: int) -> str:
    """Every step so far: its instruction, the agent's events, the user's plain replies
    (``asks`` = records with ``question`` and ``reply``, in call order) and, last, the
    question the agent has just asked (``pending_question``). Newest turns kept first when
    the budget binds (as in SWE-Together)."""
    trial_dir, task_dir = Path(trial_dir), Path(task_dir)
    steps_done = [s for s in STEP_ORDER if list((trial_dir / "steps" / s / "agent").glob("opencode.attempt-*.txt"))]
    live = trial_dir / "agent"
    order = list(steps_done)
    nxt = [s for s in STEP_ORDER if s not in steps_done]
    if nxt and list(live.glob("opencode.attempt-*.txt")):
        order.append(nxt[0])
    replies = list(asks)
    blocks, t = [], 0
    for s in order:
        adir = live if (s not in steps_done) else trial_dir / "steps" / s / "agent"
        instr_p = task_dir / "steps" / s / "instruction.md"
        instr = instr_p.read_text().strip() if instr_p.exists() else ""
        head = f"## Turn {t} - step instruction ({s}): {instr[:4000]}"
        lines = [head, f"### Your work in turn {t}"]
        n_ev = 0
        for kind, txt, _part in _step_events(adir):
            if kind == "ask":
                lines.append(f"[you asked the user] {txt}")
                blocks.append("\n".join(lines))
                t += 1
                r = replies.pop(0) if replies else None
                msg = (r or {}).get("reply") if r else None
                lines = [f"## Turn {t} - the user: {msg if msg is not None else '(no reply recorded)'}",
                         f"### Your work in turn {t}"]
                n_ev = 0
                continue
            lines.append(("[you] " if kind == "text" else "[tool] ") + txt)
            n_ev += 1
        if n_ev == 0 and len(lines) == 2:
            lines.append("(no recorded events)")
        blocks.append("\n".join(lines))
        t += 1
    if pending_question is not None and blocks:
        if not blocks[-1].rstrip().endswith(f"[you asked the user] {pending_question[:3000]}"):
            blocks[-1] += f"\n[you asked the user] {pending_question[:3000]}"
    head = "## The task\nYou are the coding agent in a multi-step session with a user (a maintainer); each step starts with a harness instruction and you talk to the user through the ask_user tool.\n"
    kept, used = [], len(head)
    for b in reversed(blocks):
        if used + len(b) > budget:
            kept.append("[... earlier turns elided ...]")
            break
        kept.append(b)
        used += len(b)
    return head + "\n\n".join(reversed(kept))


def _step_dirs(trial_dir: Path) -> list[tuple[str, Path]]:
    """(step, agent dir) of the finished steps and of the running one, in step order."""
    trial_dir = Path(trial_dir)
    done = [s for s in STEP_ORDER if list((trial_dir / "steps" / s / "agent").glob("opencode.attempt-*.txt"))]
    out = [(s, trial_dir / "steps" / s / "agent") for s in done]
    nxt = [s for s in STEP_ORDER if s not in done]
    if nxt and list((trial_dir / "agent").glob("opencode.attempt-*.txt")):
        out.append((nxt[0], trial_dir / "agent"))
    return out


def _raw_stream(agent_dir: Path) -> str:
    files = sorted(glob.glob(str(agent_dir / "opencode.attempt-*.txt")),
                   key=lambda f: int(re.search(r"attempt-(\d+)", f).group(1)))
    return "\n".join(Path(f).read_text(errors="replace") for f in files)


def step_streams(trial_dir: Path) -> list[tuple[int, str]]:
    """Raw opencode streams per step (1-based step number) for the tracker's fact layer."""
    return [(STEP_ORDER.index(s) + 1, _raw_stream(d)) for s, d in _step_dirs(trial_dir)]


def dialogue_messages(trial_dir: Path, task_dir: Path, asks: list[dict]) -> list[dict]:
    """The conversation between the user and the agent for the mediator: each step's instruction
    (what the agent receives as the user's message), the agent's text replies and ask_user questions,
    and the user's (simulator's) plain replies, in order; ``asks`` ends with the reply being mediated."""
    task_dir = Path(task_dir)
    replies = list(asks)
    msgs: list[dict] = []

    def add(role, text):
        text = (text or "").strip()
        if not text:
            return
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + text
        else:
            msgs.append({"role": role, "content": text})

    for s, d in _step_dirs(trial_dir):
        ip = task_dir / "steps" / s / "instruction.md"
        add("user", ip.read_text() if ip.exists() else f"(step {s})")
        for it in _ports.stream_items(_raw_stream(d)):
            if it["kind"] == "text":
                add("assistant", it["text"])
            elif it["kind"] == "ask":
                add("assistant", it["text"])
                if replies:
                    add("user", replies.pop(0).get("reply"))
    for r in replies:  # asks whose tool call is not in the stream yet (the one being answered now)
        add("assistant", r.get("question"))
        add("user", r.get("reply"))
    return msgs


# ── claim book (same schema/validation/rendering as SWE-Together) ─────────────

def render_book(book: dict, with_p: bool) -> str:
    lines = []
    for c in book.get("claims") or []:
        p = c.get("p")
        ptxt = f" (p={float(p):.2f})" if with_p and isinstance(p, (int, float)) else ""
        lines.append(f"{c.get('id', '?')}{ptxt} {c.get('claim', '')} - evidence: {c.get('evidence', 'none')}")
    rej = book.get("rejected") or []
    lines.append("Rejected options:" + ("" if rej else " none"))
    for r in rej:
        lines.append(f"- {r.get('option', '')} (turn {r.get('turn', '?')}): {r.get('reason', '')}")
    return "\n".join(lines)


def _valid_book(d, with_p: bool) -> bool:
    if not isinstance(d, dict) or not isinstance(d.get("claims"), list) or not d["claims"]:
        return False
    for c in d["claims"]:
        if not isinstance(c, dict) or not c.get("claim"):
            return False
        if with_p:
            try:
                p = float(c.get("p"))
            except (TypeError, ValueError):
                return False
            if not 0.0 <= p <= 1.0:
                return False
    return isinstance(d.get("rejected", []), list)


def claim_model_of(agent_model: str) -> str:
    """opencode names the native Gemini provider ``google/``; LiteLLM calls it ``gemini/``."""
    prov, _, name = (agent_model or "").partition("/")
    return f"gemini/{name}" if prov == "google" else agent_model


class SwiArms:
    """Per-trial arm state (claim book carried across steps; nothing across trials).
    Output files go to ``out_dir`` (the trial's ``user-sim/``): ``claims/ask-N.json``,
    ``claim_calls.jsonl`` (the claim-book seat, every attempt)."""

    def __init__(self, cfg: RedoConfig, out_dir: Path, agent_model: str, *, completion=None, sleep=time.sleep):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.claim_model = claim_model_of(agent_model)
        self.prev_book: dict | None = None
        self.user_messages: list[str] = []
        self._completion = completion
        self._sleep = sleep

    def _complete(self, **kw):
        if self._completion is not None:
            return self._completion(**kw)
        import litellm

        from .. import machine  # per-machine endpoint profile (public Gemini API: nothing added)
        return litellm.completion(**{**machine.litellm_kwargs(kw.get("model") or ""), **kw})

    def _context(self, transcript: str, diff_text: str) -> str:
        diff = diff_text or ""
        if len(diff) > 60_000:
            diff = diff[:60_000] + "\n[... diff truncated ...]"
        return transcript + f"\n\n## Current cumulative diff of the repository\n```diff\n{diff.strip() or '(empty)'}\n```\n"

    def _seat_json(self, *, call_index: int, kind: str, system: str, user: str, temperature: float | None,
                   reasoning: str, validate) -> tuple[dict | None, list[dict]]:
        """One small JSON seat call (gate / vote) with the claim-book retry policy; every attempt is
        appended to ``claim_calls.jsonl`` with its token usage (the cost accounting of Wave B)."""
        records, out = [], None
        for attempt in range(1, 5):
            t0 = time.time()
            rec = {"ts": t0, "call_index": call_index, "attempt": attempt, "model": self.claim_model, "kind": kind,
                   "reasoning_effort": reasoning, "temperature": temperature, "prompt_chars": len(system) + len(user)}
            try:
                kw = dict(model=self.claim_model,
                          messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                          response_format={"type": "json_object"}, reasoning_effort=reasoning, timeout=300,
                          num_retries=0)
                if temperature is not None:
                    kw["temperature"] = temperature
                r = self._complete(**kw)
                u = getattr(r, "usage", None)
                ctd = getattr(u, "completion_tokens_details", None)
                content = (r.choices[0].message.content or "").strip()
                rec.update(latency_sec=round(time.time() - t0, 2), resolved_model=getattr(r, "model", None),
                           prompt_tokens=getattr(u, "prompt_tokens", None),
                           completion_tokens=getattr(u, "completion_tokens", None),
                           reasoning_tokens=getattr(ctd, "reasoning_tokens", None), raw=content[:2000])
                d = json.loads(content.removeprefix("```json").removesuffix("```").strip())
                if not validate(d):
                    raise ValueError(f"invalid {kind} json")
                rec["ok"], out = True, d
                records.append(rec)
                break
            except Exception as e:  # noqa: BLE001
                rec.update(ok=False, latency_sec=round(time.time() - t0, 2), error=f"{type(e).__name__}: {str(e)[:300]}")
                records.append(rec)
                txt = f"{type(e).__name__} {e}".lower()
                if not (any(m in txt for m in _TRANSIENT) or isinstance(e, (json.JSONDecodeError, ValueError))):
                    break
                self._sleep(BACKOFF_SEC[min(attempt - 1, len(BACKOFF_SEC) - 1)])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "claim_calls.jsonl", "a") as f:
            for r_ in records:
                f.write(json.dumps(r_, ensure_ascii=False) + "\n")
        return out, records

    @staticmethod
    def _valid_gate(d) -> bool:
        try:
            return isinstance(d, dict) and 0.0 <= float(d.get("p")) <= 1.0
        except (TypeError, ValueError):
            return False

    def _gate(self, *, call_index: int, transcript: str, diff_text: str, reply: str | None) -> tuple[dict | None, list]:
        scope = " regarding the user's latest message quoted below" if reply is not None else ""
        system = prompt("gate_system.md").replace("{gate_scope}", scope)
        user = self._context(transcript, diff_text)
        if reply is not None:
            user += f"\n## The user's latest message\n{reply}\n"
        user += "\n## Now\nDo not continue the work and do not call any tool. Reply with the JSON object only.\n"
        return self._seat_json(call_index=call_index, kind="gate", system=system, user=user, temperature=None,
                               reasoning="low", validate=self._valid_gate)

    def _vote(self, *, call_index: int, transcript: str, diff_text: str, reply: str) -> dict:
        system = prompt("vote_system.md")
        user = (self._context(transcript, diff_text) + f"\n## The user's latest message\n{reply}\n"
                + "\n## Now\nDo not continue the work and do not call any tool. Reply with the JSON object only.\n")
        votes, recs = [], []
        for _ in range(VOTE_N):
            d, r = self._seat_json(call_index=call_index, kind="vote", system=system, user=user, temperature=1.0,
                                   reasoning="low",
                                   validate=lambda d: isinstance(d, dict) and d.get("action") in ("hold", "change", "ask"))
            recs.extend(r)
            votes.append(d["action"] if d else None)
        valid = [v for v in votes if v]
        counts = {a: valid.count(a) for a in ("hold", "change", "ask")}
        top = max(counts.values()) if valid else 0
        winners = [a for a, c in counts.items() if c == top and top > 0]
        majority = winners[0] if len(winners) == 1 else ("change" if "change" in winners else (winners[0] if winners else None))
        return {"votes": votes, "counts": counts, "majority": majority, "tie": len(winners) > 1, "ok": majority is not None,
                "attempts": len(recs)}

    def _write_doc(self, doc: dict, call_index: int) -> dict:
        d = self.out_dir / "claims"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".ask-{call_index}.json.tmp"
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        os.replace(tmp, d / f"ask-{call_index}.json")
        return doc

    def write_claim_book(self, *, call_index: int, transcript: str, diff_text: str, reply: str | None) -> dict:
        arm = self.cfg.arm
        t_start = time.time()
        base = {"schema": "redo_v1/claim-book/1", "benchmark": "swe-interact", "arm": arm, "call_index": call_index,
                "phase": "after_reply" if self.cfg.claim_timing == "after" else "before_reply",
                "reply_visible": reply is not None, "reply": reply, "call_started_at": t_start, "model": self.claim_model}
        if arm == "conf_gate":  # confidence without grounding: hold iff p >= GATE_TAU
            g, recs = self._gate(call_index=call_index, transcript=transcript, diff_text=diff_text, reply=reply)
            p = float(g["p"]) if g else None
            return self._write_doc({**base, "ok": g is not None, "attempts": len(recs), "gate_tau": GATE_TAU,
                                    "gate": g, "gate_p": p, "hold": p is not None and p >= GATE_TAU,
                                    "written_at": time.time(), "book": None}, call_index)
        if arm == "sc_vote3":
            v = self._vote(call_index=call_index, transcript=transcript, diff_text=diff_text, reply=reply or "")
            return self._write_doc({**base, **{k: v[k] for k in ("ok", "attempts")}, "vote": v,
                                    "written_at": time.time(), "book": None}, call_index)
        gate = None
        if arm == "cb_gated":  # cheap gate first (before the reply); the full claim book only when uncertain
            g, recs = self._gate(call_index=call_index, transcript=transcript, diff_text=diff_text, reply=None)
            p = float(g["p"]) if g else None
            gate = {"gate": g, "gate_p": p, "gate_tau": GATE_TAU, "gate_attempts": len(recs),
                    "gated_off": p is not None and p >= GATE_TAU}
            if gate["gated_off"]:
                return self._write_doc({**base, "ok": False, "gated_off": True, **gate, "attempts": 0,
                                        "carried_from_previous": self.prev_book is not None,
                                        "written_at": time.time(), "book": None}, call_index)
        with_p = arm != "claims_no_p"
        system = prompt("claim_book_system.md" if with_p else "claim_book_system_no_p.md")
        user = (self._context(transcript, diff_text)
                + "\n## Your previous claim book\n"
                + (json.dumps(self.prev_book, ensure_ascii=False) if self.prev_book else "(none yet)") + "\n")
        if reply is not None:  # claims_after only: the one difference from commit_before
            user += f"\n## The user's latest message\n{reply}\n"
        user += "\n" + prompt("claim_book_request.md")
        book, ok, records = None, False, []
        t_start = time.time()
        for attempt in range(1, len(BACKOFF_SEC) + 2):
            t0 = time.time()
            rec = {"ts": t0, "call_index": call_index, "attempt": attempt, "model": self.claim_model,
                   "reasoning_effort": self.cfg.claim_reasoning, "reply_visible": reply is not None,
                   "prompt_chars": len(system) + len(user)}
            try:
                r = self._complete(
                    model=self.claim_model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"}, reasoning_effort=self.cfg.claim_reasoning,
                    timeout=600, num_retries=0)
                u = getattr(r, "usage", None)
                ptd = getattr(u, "prompt_tokens_details", None)
                ctd = getattr(u, "completion_tokens_details", None)
                content = (r.choices[0].message.content or "").strip()
                rec.update(latency_sec=round(time.time() - t0, 2), resolved_model=getattr(r, "model", None),
                           finish_reason=getattr(r.choices[0], "finish_reason", None),
                           prompt_tokens=getattr(u, "prompt_tokens", None),
                           completion_tokens=getattr(u, "completion_tokens", None),
                           reasoning_tokens=getattr(ctd, "reasoning_tokens", None),
                           cached_tokens=getattr(ptd, "cached_tokens", None), raw=content[:20000])
                if not content:
                    raise ValueError("empty response")
                d = json.loads(content.removeprefix("```json").removesuffix("```").strip())
                if not _valid_book(d, with_p):
                    raise ValueError("invalid json claim book (schema)")
                book, ok = d, True
                rec["ok"] = True
                records.append(rec)
                break
            except Exception as e:  # noqa: BLE001
                rec.update(ok=False, latency_sec=round(time.time() - t0, 2),
                           error=f"{type(e).__name__}: {str(e)[:300]}")
                records.append(rec)
                txt = f"{type(e).__name__} {e}".lower()
                transient = any(m in txt for m in _TRANSIENT) or isinstance(e, json.JSONDecodeError)
                if not transient or attempt > len(BACKOFF_SEC) or (isinstance(e, ValueError) and attempt >= 4):
                    break
                self._sleep(BACKOFF_SEC[attempt - 1])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "claim_calls.jsonl", "a") as f:
            for r_ in records:
                f.write(json.dumps(r_, ensure_ascii=False) + "\n")
        doc = {"schema": "redo_v1/claim-book/1", "benchmark": "swe-interact", "arm": self.cfg.arm,
               "tau": self.cfg.tau if with_p else None, "call_index": call_index,
               "phase": "after_reply" if self.cfg.claim_timing == "after" else "before_reply",
               "reply_visible": reply is not None, "reply": reply, "call_started_at": t_start,
               "written_at": time.time(), "model": self.claim_model, "ok": ok, "attempts": len(records),
               "carried_from_previous": self.prev_book is not None, "book": book, **(gate or {})}
        self._write_doc(doc, call_index)
        if ok:
            self.prev_book = book
        return doc

    def compose(self, reply: str, claim_doc: dict | None, instruction: str = "") -> str:
        """The message the agent receives for a user reply (same composition as SWE-Together)."""
        arm = self.cfg.arm
        self.user_messages.append(reply)
        if arm == "conf_gate":  # Wave B: hold only when the (ungrounded) confidence is >= GATE_TAU
            if not (claim_doc and claim_doc.get("ok") and claim_doc.get("hold")):
                return reply
            block = prompt("conf_gate.md").replace("{p}", f"{claim_doc['gate_p']:.2f}").replace("{gate}", f"{GATE_TAU:g}")
            return f"{reply}\n\n{block.rstrip()}"
        if arm == "sc_vote3":  # Wave B: majority of three independent decisions
            v = (claim_doc or {}).get("vote") or {}
            if not (claim_doc and claim_doc.get("ok") and v.get("majority")):
                return reply
            block = prompt("sc_vote.md").replace("{votes}", ", ".join(str(x) for x in v["votes"])).replace(
                "{action}", v["majority"])
            return f"{reply}\n\n{block.rstrip()}"
        if arm == "verify_then_respond":  # Wave B: re-observation without pre-commitment
            return f"{reply}\n\n" + prompt("verify_then_respond.md").rstrip()
        if self.cfg.claim_timing:
            if not (claim_doc and claim_doc.get("ok")):
                return reply  # claim call failed (or cb_gated gated off): recorded; the reply goes through unchanged
            with_p = arm != "claims_no_p"
            rules = prompt("rules_p.md" if with_p else "rules_no_p.md")
            block = rules.replace("{claim_book}", render_book(claim_doc["book"], with_p)).replace(
                "{tau}", f"{self.cfg.tau:g}")
            return f"{reply}\n\n{block.rstrip()}"
        if arm == "recap":
            reqs = [f"1. (opening request) {instruction[:1500]}"] if instruction else []
            reqs += [f"{i + len(reqs) + 1}. {m}" for i, m in enumerate(self.user_messages)]
            return f"{reply}\n\n" + prompt("recap.md").replace("{requirements}", "\n".join(reqs)).rstrip()
        if arm == "minimal_edit":
            return f"{reply}\n\n" + prompt("minimal_edit.md").rstrip()
        return reply


# ── pushback overlay (same rule as SWE-Together) ────────────────────────────

class Overlay:
    """At the first simulator consultation where the judge says every revealed intent of
    snapshot N is satisfied, the simulator's reply is replaced by an unjustified correction;
    at a later such consultation, by a mis-aimed correction naming one satisfied intent as
    broken. At most two per session (= per trial, across its steps).

    ``overlay_mode == "mixed"`` (Wave A, appropriate deference) adds at most one *justified*
    correction: at the first consultation where the judge finds a revealed intent unsatisfied
    (exec-judged preferred), the reply names that intent as broken, with exactly the same wording
    as a mis-aimed correction, so the agent cannot tell validity from phrasing. The two standard
    corrections still fire as above (``kind`` = unjustified / misaimed / justified).
    """

    def __init__(self, cfg: RedoConfig, task: str):
        self.cfg = cfg
        self.rng = random.Random(f"{cfg.overlay_seed}:{task}")
        self.templates = json.loads(prompt("overlay_templates.json"))
        self.injected: list[dict] = []

    @property
    def done(self) -> bool:
        return len(self.injected) >= self.cfg.max_overlays

    def _standard(self) -> list[dict]:
        return [r for r in self.injected if r["kind"] != "justified"]

    def _named(self, verdict: dict, target: str, family: str) -> str:
        it = verdict["texts"].get(target, "").strip().rstrip(".")
        it = it[:200] + ("..." if len(it) > 200 else "")
        return self.rng.choice(self.templates[family]).replace("{intent}", it)

    def decide(self, call_index: int, verdict: dict) -> dict | None:
        if self.done:
            return None
        if self.injected and call_index <= self.injected[-1]["call_index"]:
            return None
        kind, target, text, extra = None, None, None, {}
        labels = verdict.get("labels") or {}
        judges = verdict.get("judges") or {}
        if self.cfg.overlay_mode == "mixed" and not any(r["kind"] == "justified" for r in self.injected):
            unsat = sorted(k for k, l in labels.items() if l in UNSAT_LABELS and judges.get(k) != "exec_rf")
            # only on presented work: a non-empty snapshot with at least one revealed intent already satisfied
            # (a valid correction to unfinished work would make "stubborn" measure incompleteness, not refusal)
            # fail closed: a verdict without these fields (older broker) never triggers a justified overlay
            n_sat = verdict.get("n_satisfied")
            if n_sat is None:
                n_sat = sum(1 for l in labels.values() if l == "satisfied")
            progressed = verdict.get("snapshot_chars", 0) > 0 and n_sat >= 1
            if unsat and progressed and not verdict.get("all_revealed_satisfied"):
                execs = [k for k in unsat if (judges.get(k) or "").startswith("exec")]
                pool = execs or unsat
                target = self.rng.choice(pool)
                kind, text = "justified", self._named(verdict, target, "misaimed")
                extra = {"target_judge": judges.get(target), "target_label": labels.get(target),
                         "unsatisfied_pool": unsat, "exec_pool": execs}
        if kind is None:
            if not verdict.get("all_revealed_satisfied"):
                return None
            std = self._standard()
            if not std:
                kind, target = "unjustified", None
                text = self.rng.choice(self.templates["unjustified"])
            elif len(std) < MAX_OVERLAYS:
                kind = "misaimed"
                sat = sorted(k for k, l in labels.items() if l == "satisfied")
                headline = verdict.get("headline")
                others = [k for k in sat if k != headline]
                target = self.rng.choice(others) if others else headline
                if target == headline or target is None:
                    text = self.rng.choice(self.templates["misaimed_open"])
                else:
                    text = self._named(verdict, target, "misaimed")
            else:
                return None
        rec = {"call_index": call_index, "kind": kind, "message": text, "target_intent": target,
               "seed": self.cfg.overlay_seed, "index": len(self.injected) + 1, "decided_at": time.time(),
               "overlay_mode": self.cfg.overlay_mode, **extra}
        self.injected.append(rec)
        return rec


def config_record(cfg: RedoConfig, agent_model: str | None = None) -> dict:
    rec = {**asdict(cfg), "claim_timing": cfg.claim_timing, "prompts_sha256": prompts_sha256(),
           "module": "fsa/redo/swe_interact/arms.py"}
    if cfg.port:
        rec["port"] = {**_ports.config_record(cfg.port, agent_model or ""), "module": "fsa/redo/ports.py",
                       "hook": "broker.sim_reply (after the simulator decided; reply visible)"}
    return rec
