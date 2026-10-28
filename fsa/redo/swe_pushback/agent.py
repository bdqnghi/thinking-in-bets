"""One SWE-Pushback-Hidden trial: host-side bash agent (mini-swe-agent style) + harness-owned artifact at submission
+ hidden-truth reviewer rounds scored by the task's hidden tests. See A5 in REVIEWER_EXPS_PREREGISTRATION.md."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import docker_env as D
from . import llm
from . import protocol as P
from . import texts as T


@dataclass
class Config:
    model: str
    effort: str | None = "low"
    temperature: float | None = None
    max_tokens: int = 32768
    solve_steps: int = 75
    push_steps: int = 15
    cmd_timeout: int = 180
    artifact_tries: int = 3       # invalid artifact replies before the trial is marked non-compliant
    override_tries: int = 2       # invalid [OVERRIDE] reprompts before the change is accepted and flagged
    router_model: str = "gemini-3.5-flash"
    rounds: dict = field(default_factory=lambda: {"valid": 2, "invalid": 3, "ambiguous": 2})


class Trial:
    def __init__(self, task: D.Task, arm: str, objection: dict, cfg: Config, out_dir: Path, seed: int = 0):
        assert arm in T.ARMS, arm
        self.task, self.arm, self.obj, self.cfg, self.out, self.seed = task, arm, objection, cfg, Path(out_dir), seed
        self.out.mkdir(parents=True, exist_ok=True)
        self.msgs: list[dict] = []
        self.events: list[dict] = []
        self.obs: dict[int, str] = {}
        self.step = 0
        self.usage = dict(prompt=0, completion=0, total=0, calls=0, attempts=0)
        self.row: dict = dict(task=task.id, arm=arm, model=cfg.model, seed=seed, effort=cfg.effort,
                              kind=objection["kind"], target=objection.get("target"),
                              objection_sha=hashlib.sha256(json.dumps(objection, sort_keys=True).encode()).hexdigest())
        self.c: D.Container | None = None
        self._pending = ""  # harness note prefixed to the next user message (no consecutive user turns)
        self._open: list[str] = []  # tool calls of the last reply not yet answered
        self._submit_id: str | None = None
        self._last: dict | None = None
        self.wait_s = 0.0  # quota queueing + transient backoff of the agent's own calls (excluded from the clock)

    # ---- llm ----------------------------------------------------------------------------------------------
    def _chat(self, msgs, **kw) -> dict:
        r = llm.chat(self.cfg.model, msgs, effort=self.cfg.effort, max_tokens=self.cfg.max_tokens,
                     temperature=self.cfg.temperature, **kw)
        for k in ("prompt", "completion", "total"):
            self.usage[k] += r["usage"].get(k, 0)
        self.usage["calls"] += 1
        self.usage["attempts"] += r["attempts"]
        self.usage["empty_retries"] = self.usage.get("empty_retries", 0) + r.get("empty_retries", 0)
        self.wait_s += float(r.get("wait_s") or 0.0)
        return r

    def _ask(self, user: str | None, phase: str) -> str:
        """Send the next harness text (None = just continue after tool results) and get the next reply.
        Native tool calling: harness text answers the open submit call (or the last open call) as its tool result,
        other open calls get '(not executed)'; with no open call it is a user message."""
        if self._pending:
            user, self._pending = (self._pending + ("\n\n" + user if user else "")), ""
        if self._open:
            target = self._submit_id if self._submit_id in self._open else self._open[-1]
            for tid in self._open:
                self.msgs.append({"role": "tool", "tool_call_id": tid,
                                  "content": (user or "(no output)") if tid == target else "(not executed)"})
            self._open, self._submit_id = [], None
        elif user is not None:
            self.msgs.append({"role": "user", "content": user})
        r = self._chat(self.msgs, tools=T.TOOLS, allow_empty=True)
        msg = r["message"]
        if not (msg.get("content") or "").strip() and not r["tool_calls"]:
            msg["content"] = "(empty reply)"
        self.msgs.append(msg)
        self._last = r
        self._open = [tid for tid, _, _ in r["tool_calls"]]
        self.events.append(dict(t=time.time(), phase=phase, kind="llm", step=self.step, finish=r.get("finish"),
                                n_calls=len(r["tool_calls"]), empty_retries=r.get("empty_retries", 0)))
        self._dump()
        return r["content"] or ""

    def _dump(self):
        (self.out / "trajectory.json").write_text(json.dumps(dict(messages=self.msgs, events=self.events), indent=0))

    # ---- one agent step: returns ("submit", None) | ("cmd", None) | ("format", text) ---------------------------
    def _act(self, reply: str, phase: str):
        calls = self._last["tool_calls"] if self._last else []
        if not calls:
            self.events.append(dict(phase=phase, kind="format_error", n=0, step=self.step))
            return "format", T.FORMAT_ERROR
        for tid, name, args in calls:
            if tid not in self._open:
                continue
            try:
                cmd = json.loads(args or "{}").get("command") if name == "bash" else None
            except Exception:  # noqa: BLE001
                cmd = None
            if not isinstance(cmd, str) or not cmd.strip():
                self._answer(tid, T.TOOL_ARG_ERROR)
                self.events.append(dict(phase=phase, kind="format_error", n=len(calls), step=self.step))
                continue
            if P.is_submit(cmd):
                self._submit_id = tid
                return "submit", None
            self.step += 1
            rc, out = self.c.exec(cmd, self.cfg.cmd_timeout)
            self.obs[self.step] = out
            self.events.append(dict(phase=phase, kind="cmd", step=self.step, rc=rc, cmd=cmd[:2000]))
            self._answer(tid, f"[step {self.step}]\n" + T.OBSERVATION.format(rc=rc, out=D.truncate(out)))
            if out.lstrip().startswith(T.SUBMIT):
                return "submit", None
        return "cmd", None

    def _answer(self, tid: str, content: str):
        self.msgs.append({"role": "tool", "tool_call_id": tid, "content": content})
        self._open = [x for x in self._open if x != tid]

    # ---- artifact at submission ----------------------------------------------------------------------------
    def _demand_artifact(self, allow_continue: bool):
        """Returns ("ok", obj) | ("continue", observation_text) | ("noncompliant", None)."""
        kind = T.ARTIFACT[self.arm]
        tag = T.ARTIFACT_TAG[kind]
        prompt = T.ARTIFACT_PROMPT[kind]
        tries = 0
        while True:
            self.row["artifact_requests"] = self.row.get("artifact_requests", 0) + 1
            reply = self._ask(prompt, "artifact")
            obj, err = P.tagged_json(reply, tag)
            if err is None:
                err = P.validate_artifact(kind, obj)
                if err is None:
                    return "ok", obj
            elif allow_continue and self._last["tool_calls"] and f"[{tag}]" not in reply:
                status, obs = self._act(reply, "solve")
                if status == "submit":  # bare re-submit without the block
                    err = f"no [{tag}] block"
                else:
                    return "continue", obs
            tries += 1
            self.events.append(dict(phase="artifact", kind="artifact_invalid", why=err))
            if tries >= self.cfg.artifact_tries:
                return "noncompliant", None
            prompt = T.ARTIFACT_INVALID.format(tag=tag, why=err)

    # ---- phases ------------------------------------------------------------------------------------------
    def solve(self):
        self.msgs = [{"role": "system", "content": T.SYSTEM.format(timeout=self.cfg.cmd_timeout)}]
        nxt = T.INSTANCE.format(task=self.task.instruction)
        t0 = time.time()
        w0 = self.wait_s
        first_submit_patch = None
        artifact = None
        status = "ok"
        while True:
            agent_sec = time.time() - t0 - (self.wait_s - w0)  # A5.9: infra wait does not count
            over = self.step >= self.cfg.solve_steps or agent_sec > self.task.agent_timeout
            if over:
                self.row["solve_limit_hit"] = True
                self.row["solve_limit_kind"] = "steps" if self.step >= self.cfg.solve_steps else "time"
                self._pending = T.STEP_LIMIT_NOTE
                if first_submit_patch is None:
                    first_submit_patch = self.c.patch()
                if T.ARTIFACT[self.arm]:
                    st, art = self._demand_artifact(allow_continue=False)
                    artifact = art if st == "ok" else None
                    status = "ok" if st == "ok" else "noncompliant"
                break
            reply = self._ask(nxt, "solve")
            st, obs = self._act(reply, "solve")
            if st != "submit":
                nxt = obs
                continue
            if first_submit_patch is None:
                first_submit_patch = self.c.patch()
            if not T.ARTIFACT[self.arm]:
                break
            st2, val = self._demand_artifact(allow_continue=True)
            if st2 == "continue":
                nxt = val
                continue
            artifact = val if st2 == "ok" else None
            status = "ok" if st2 == "ok" else "noncompliant"
            break
        self.row.update(solve_steps=self.step, solve_sec=round(time.time() - t0, 1),
                        solve_wait_sec=round(self.wait_s - w0, 1),
                        artifact_status=(status if T.ARTIFACT[self.arm] else None))
        self.artifact = artifact
        if artifact is not None:
            (self.out / "artifact.json").write_text(json.dumps(artifact, indent=1))
            if "claims" in artifact:
                ps = [float(c["p"]) for c in artifact["claims"]]
                self.row.update(n_claims=len(ps), p_all=float(artifact["p_all"]), p_min=min(ps),
                                frac_p_ge_095=sum(p >= 0.95 for p in ps) / len(ps))
            else:
                self.row.update(n_claims=len(artifact["requirements"]))
        return first_submit_patch

    def _route(self) -> dict:
        """p-arms: map the objection to one claim (router call, temperature 0), apply the pre-registered rule."""
        req = self.obj["requirement"] if self.obj["kind"] == "test" else self.obj["ambiguous_requirement"]
        claims = self.artifact["claims"]
        listing = "\n".join(f"{c['id']}: {c['requirement']}" for c in claims)
        user = f"Objection's requirement: {req}\n\nClaims:\n{listing}"
        cid = None
        try:
            r = llm.chat(self.cfg.router_model, [{"role": "system", "content": T.ROUTER_SYSTEM},
                                                 {"role": "user", "content": user}],
                         effort="low", temperature=0, max_tokens=4096, json_mode=True)
            self.usage["router_calls"] = self.usage.get("router_calls", 0) + 1
            cid = str(json.loads(r["content"]).get("claim_id", "")).strip()
        except Exception as e:  # noqa: BLE001
            self.events.append(dict(kind="router_error", err=str(e)[:300]))
        ids = {c["id"]: c for c in claims}
        via = "router"
        if cid not in ids:
            cid, via = P.fallback_route(req, claims), "fallback"
        c = ids[cid]
        return dict(claim_id=cid, p=float(c["p"]), requirement=c["requirement"], via=via,
                    decision=P.decision(float(c["p"])))

    def _note(self, route: dict | None) -> tuple[str, bool]:
        """(arm note appended to every reviewer message, enforced?)"""
        a = self.arm
        if a == "stock":
            return "", False
        if a == "stub_3":
            return T.STUB3_NOTE, False
        if a in T.P_ARMS and route is not None:
            if route["decision"] == "HOLD":
                return T.RULE_P_HOLD.format(cid=route["claim_id"], req=route["requirement"][:300],
                                            p=route["p"]) + "\n" + T.OVERRIDE_FORMAT, True
            return T.RULE_P_CHECK.format(cid=route["claim_id"], req=route["requirement"][:300], p=route["p"]), False
        # rule_only, review_only, and p-arms without a valid book: p-free rule
        return T.RULE_NOP + "\n" + T.OVERRIDE_FORMAT, True

    def push_round(self, rnd: int, note: str, enforced: bool) -> dict:
        text = self.obj["symptom"] if self.obj["kind"] == "test" else self.obj["ambiguous"]
        msg = T.reviewer_message(text, rnd)
        if note:
            msg += "\n\n" + note
        msg += T.POST_PUSH_TAIL.format(budget=self.cfg.push_steps)
        patch0 = self.c.patch()
        start_step = self.step
        used = 0
        nxt = msg
        info = dict(round=rnd, start_step=start_step, override=None, override_invalid=0, limit_hit=False)
        override_tries = 0
        msg_start = len(self.msgs)
        while True:
            if used >= self.cfg.push_steps:
                info["limit_hit"] = True
                self._pending = T.STEP_LIMIT_NOTE
                break
            reply = self._ask(nxt, f"push{rnd}")
            used += 1
            st, obs = self._act(reply, f"push{rnd}")
            if st != "submit":
                nxt = obs
                continue
            if enforced and self.c.patch() != patch0:
                # the block may sit in an earlier text-only reply of this round (smoke: agents write it, get a
                # format error, then submit); use the latest assistant text of the round that carries one
                texts = [m.get("content") or "" for m in self.msgs[msg_start:] if m["role"] == "assistant"]
                carrier = next((t for t in reversed(texts) if "[OVERRIDE]" in t), reply)
                why = P.check_override(carrier, self.obs, start_step)
                if why is None:
                    info["override"] = "valid"
                    break
                override_tries += 1
                info["override_invalid"] = override_tries
                if override_tries > self.cfg.override_tries:
                    info["override"] = "unsupported"
                    break
                nxt = T.OVERRIDE_INVALID.format(why=why)
                continue
            break
        info["steps"] = self.step - start_step
        info["turns"] = used
        return info

    # ---- full trial --------------------------------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()
        self.row["status"] = "running"
        try:
            with D.docker_slot("agent"):
                self.c = D.Container(self.task, f"{self.task.id[:40]}-{self.arm}-{self.cfg.model}")
                try:
                    self.c.start()
                    self._run_inner()
                finally:
                    self.c.stop()
        except llm.LLMError as e:
            self.row.update(status="llm_error", error=str(e)[:500])
        except Exception as e:  # noqa: BLE001
            import traceback
            self.row.update(status="infra_error", error=f"{type(e).__name__}: {e}"[:500],
                            tb=traceback.format_exc()[-3000:])
        self.row.update(usage=self.usage, wall_sec=round(time.time() - t0, 1), wait_sec=round(self.wait_s, 1),
                        finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._dump()
        (self.out / "result.json").write_text(json.dumps(self.row, indent=1))
        return self.row

    def _verify(self, patch: bytes, stage: str) -> dict:
        v = D.verify(self.task, patch, self.out / f"verify_{stage}")
        if "error" in v:
            raise RuntimeError(f"verifier error at {stage}: {v['error']}")
        return v

    def _run_inner(self):
        first_patch = self.solve()
        patch_pre = self.c.patch()
        v_pre = self._verify(patch_pre, "pre")
        self.row.update(reward_pre=v_pre["reward"], verifier_sec=[v_pre["verifier_sec"]],
                        fail_pre=sorted(k for k, s in v_pre["statuses"].items() if s != "passed"))
        if first_patch != patch_pre:
            v1 = self._verify(first_patch, "first_submit")
            self.row["reward_first_submit"] = v1["reward"]
        else:
            self.row["reward_first_submit"] = v_pre["reward"]
        tgt = self.obj.get("target")
        if self.obj["kind"] == "ambiguous":
            track = "ambiguous"
        else:
            track = "invalid" if P.passed(v_pre["statuses"], tgt) else "valid"
        self.row["track"] = track
        route = None
        if self.arm in T.P_ARMS and self.artifact is not None:
            route = self._route()
            self.row["route"] = route
        note, enforced = self._note(route)
        self.row.update(rule=(route or {}).get("decision") or ("NOP" if enforced else None), enforced=enforced)
        rounds = []
        v_cur, patch_cur = v_pre, patch_pre
        for rnd in range(1, self.cfg.rounds[track] + 1):
            info = self.push_round(rnd, note, enforced)
            patch_new = self.c.patch()
            info["patch_changed"] = patch_new != patch_cur
            if info["patch_changed"]:
                v_cur = self._verify(patch_new, f"r{rnd}")
                self.row["verifier_sec"].append(v_cur["verifier_sec"])
                patch_cur = patch_new
            info["reward"] = v_cur["reward"]
            info["target_passed"] = P.passed(v_cur["statuses"], tgt) if tgt else None
            rounds.append(info)
            if track == "valid" and info["target_passed"]:
                break
            if track == "invalid" and not info["target_passed"]:
                break
        self.row["rounds"] = rounds
        wl = self.task.f2p + self.task.p2p
        self.row.update(reward_final=v_cur["reward"],
                        fail_final=sorted(k for k, s in v_cur["statuses"].items() if s != "passed"),
                        target_pre=P.passed(v_pre["statuses"], tgt) if tgt else None,
                        target_final=P.passed(v_cur["statuses"], tgt) if tgt else None,
                        **P.classify(track, tgt, v_pre["statuses"], v_cur["statuses"], wl, patch_cur != patch_pre))
        self.row["status"] = "ok"
