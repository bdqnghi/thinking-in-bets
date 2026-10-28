"""Harbor agent for SWE-Interact redo_v1: stock opencode + Gemini backpressure handling
+ per-step instrumentation. Imported by Harbor (``-a fsa.redo.swe_interact.agent:RedoOpenCode``)
from the pinned Harbor venv; depends only on harbor and the stdlib.

What differs from ``harbor.agents.installed.opencode.OpenCode`` (harbor 0.23.0):

* **Backpressure** (same rule as the SWE-Together wrapper, PLAN.md §4): an attempt whose
  event stream holds only throttle errors and no completed step is discarded and
  re-issued; an attempt that made progress and then ended on a throttle error is
  resumed in the same opencode session with a synthetic "continue" message. Gemini's
  mid-stream 503 inside an HTTP 200 surfaces as HTTP 400 "Requests ending with a model
  turn are not supported" and is treated as throttling. Backoff 15..300 s; at most 8
  consecutive retries whose attempts completed < 5 model calls (Stage 0 S/M ran with the
  first rule, 8 retries per step, before this change). Each attempt keeps its own stream (``opencode.attempt-K.txt``);
  ``opencode.txt`` is the concatenation of the attempts that made progress, so Harbor's
  trajectory/token parsing sees the whole step. Errors that were absorbed this way do not
  fail the step; any other error event still raises ``NonZeroAgentExitCodeError`` as
  upstream.
* **Snapshots** (PLAN.md §6): before the first step the trial's starting commit is
  written to ``/var/tmp/.swi_redo_anchor`` in the task container (read by the user-server
  sidecar, which snapshots the repository at every ``ask_user`` call); at the end of
  every step the wrapper snapshots the repository with the same recipe
  (``redo/step-K-end.{patch,json}`` in the step's agent log dir). Together these are the
  repository after every agent turn: a turn ends either with an ``ask_user`` call (the
  simulator's reply opens the next turn) or with the end of the step (Harbor's next
  step instruction opens the next turn).
* ``redo/step-K.json``: step timing, attempts, throttle retries, HEAD before/after.

* **MCP tool-call timeout** (``SWI_MCP_TIMEOUT_MS``; Stage 1): opencode's ``ask_user`` MCP call
  otherwise uses the MCP SDK's 60 s request timeout (no ``timeout`` in Harbor's generated config):
  in Stage 0, 8 ``ask_user`` calls in 3 of 5 trials ended with ``MCP error -32001: Request timed
  out`` while the simulator was still working (its reply was then lost to the agent but kept in
  the simulator's history). The runner sets ``experimental.mcp_timeout`` for every arm (stock
  included), so the simulator's patient retry and an arm's extra latency (claim book, overlay
  judge) cannot cost replies differentially. Unset/0 = upstream behaviour.
* ``redo/plan-K.md``: ``/tmp/plan.md`` (the plan agreed in 01_plan) copied at the end of every
  step, for the conservative revelation set.

A5.14 (REVIEWER_EXPS_PREREGISTRATION.md): the A5.11 step-boundary Claim Book gate that ran here at the end of
``02_implement`` is gone; the gate now lives in the ``ask_user`` sidecar (``a5.Gate``, ``broker.py``), so this wrapper
is the same for every arm.

The repository root comes from ``SWI_REPO_ROOT`` (the runner reads it from the task's
compose ``REPO_EXEC_ROOT``; default ``/app``).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import NonZeroAgentExitCodeError, with_prompt_template
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

THROTTLE_MARKERS = (
    "too many tokens, please wait", "throttlingexception", "too many requests", "rate limit", "rate_limit",
    "serviceunavailableexception", "service unavailable", "overloaded", "corrupted thought signature",
    "resource_exhausted", "resource has been exhausted", "high demand", "unavailable",
    "requests ending with a model turn",
)
THROTTLE_MAX_RETRIES = 8
THROTTLE_BACKOFF_SEC = (15, 30, 60, 90, 120, 180, 240, 300)
PROGRESS_STEPS = 5  # an interrupted attempt with >= this many completed model calls resets the retry count
THROTTLE_CONTINUE_MESSAGE = ("Your previous step was interrupted by a temporary provider error. "
                             "Please continue exactly where you left off.")
ANCHOR_FILE = "/var/tmp/.swi_redo_anchor"
SNAPSHOT_MAX_UNTRACKED = 2000


def classify_events(stdout: str) -> dict[str, Any]:
    """Counts in one ``opencode run --format=json`` stream; ``ended_on_throttle`` when the
    last step/error event is a throttle error."""
    c = {"steps": 0, "errors": 0, "throttle_errors": 0, "other_errors": [], "tool_calls": 0,
         "ended_on_throttle": 0, "session_id": None}
    last = None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not c["session_id"] and isinstance(e.get("sessionID"), str):
            c["session_id"] = e["sessionID"]
        t = e.get("type")
        if t == "step_finish":
            c["steps"] += 1
            last = "step"
        elif t == "tool_use":
            c["tool_calls"] += 1
        elif t == "error":
            c["errors"] += 1
            err = e.get("error") or {}
            data = err.get("data") if isinstance(err, dict) else None
            data = data if isinstance(data, dict) else {}
            text = " ".join(str(x) for x in ((err.get("name") if isinstance(err, dict) else err),
                                             data.get("message"), data.get("responseBody")) if x).lower()
            if any(m in text for m in THROTTLE_MARKERS):
                c["throttle_errors"] += 1
                last = "throttle"
            else:
                c["other_errors"].append(text[:300])
                last = "error"
    c["ended_on_throttle"] = int(last == "throttle")
    return c


class RedoOpenCode(OpenCode):
    """opencode as Harbor ships it, plus backpressure resume and redo instrumentation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._redo_step = 0
        self._repo_root = os.environ.get("SWI_REPO_ROOT", "/app")
        self._throttle_retries_total = 0
        mcp_ms = int(os.environ.get("SWI_MCP_TIMEOUT_MS", "0") or 0)
        if mcp_ms > 0:  # instance attribute shadows OpenCode._DEFAULT_CONFIG (deep-merged first)
            self._DEFAULT_CONFIG = {"experimental": {"mcp_timeout": mcp_ms}}

    # ── snapshots ────────────────────────────────────────────────────────

    async def _sh(self, environment: BaseEnvironment, command: str, timeout: int = 300) -> tuple[int, str]:
        r = await environment.exec(command=command, user="root", cwd=self._repo_root, timeout_sec=timeout)
        return r.return_code, (r.stdout or "")

    async def _write_anchor(self, environment: BaseEnvironment) -> str | None:
        root = shlex.quote(self._repo_root)
        rc, out = await self._sh(environment, f"git -C {root} rev-parse HEAD")
        sha = out.strip().splitlines()[-1] if out.strip() else ""
        if rc == 0 and len(sha) == 40:
            await self._sh(environment, f"printf '%s\\n' {sha} > {ANCHOR_FILE}")
            return sha
        return None

    async def snapshot(self, environment: BaseEnvironment, tag: str, meta: dict[str, Any]) -> dict[str, Any]:
        """Same recipe as the sidecar (overlay/server.py::_redo_snapshot)."""
        out_dir = self.logs_dir / "redo"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rec: dict[str, Any] = {"tag": tag, "ts": t0, **meta}
        try:
            rc, a = await self._sh(environment, f"sed -n 1p {ANCHOR_FILE}")
            anchor = a.strip() if rc == 0 and len(a.strip()) == 40 else None
            rec["anchor"] = anchor
            rec["head"] = (await self._sh(environment, "git rev-parse HEAD"))[1].strip()
            rec["status"] = (await self._sh(environment, "git status --porcelain=v1 -uall"))[1]
            if anchor:
                rec["commits_since_anchor"] = (await self._sh(environment, f"git log --format=%H%x09%s {anchor}..HEAD"))[1]
            rc, tracked = await self._sh(environment, f"git diff --binary {anchor or 'HEAD'}")
            rec["tracked_rc"] = rc
            parts = [tracked]
            untracked = [p for p in (await self._sh(environment, "git ls-files --others --exclude-standard"))[1]
                         .splitlines() if p.strip()]
            rec["untracked"] = untracked[:SNAPSHOT_MAX_UNTRACKED]
            rec["untracked_truncated"] = len(untracked) > SNAPSHOT_MAX_UNTRACKED
            for p in untracked[:SNAPSHOT_MAX_UNTRACKED]:
                parts.append((await self._sh(environment, f"git diff --no-index --binary -- /dev/null {shlex.quote(p)}"))[1])
            patch = "".join(parts)
            rec["patch_bytes"] = len(patch)
            rec["patch_sha256"] = hashlib.sha256(patch.encode()).hexdigest()
            (out_dir / f"{tag}.patch").write_text(patch)
        except Exception as exc:  # noqa: BLE001 - instrumentation must not fail the step
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["snapshot_sec"] = round(time.time() - t0, 3)
        (out_dir / f"{tag}.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
        return rec

    # ── run ──────────────────────────────────────────────────────────────

    def _opencode_cmd(self, message: str, out_file: str, session_id: str | None) -> str:
        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""
        session = f"--session={shlex.quote(session_id)} " if session_id else ""
        return ("[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "
                f"opencode --model={self.model_name} run --format=json "
                f"{session}{cli_flags_arg}--thinking "
                f"--dangerously-skip-permissions -- {shlex.quote(message)} "
                f"2>&1 </dev/null | stdbuf -oL tee {out_file}")

    async def _read_agent_file(self, environment: BaseEnvironment, name: str) -> str:
        p = self.logs_dir / name
        if p.exists():
            return p.read_text(errors="replace")
        rc, out = await self._sh(environment, f"cat /logs/agent/{shlex.quote(name)}")
        return out if rc == 0 else ""

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self._instruction = instruction
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        self._redo_step += 1
        k = self._redo_step
        rec: dict[str, Any] = {"step_index": k, "started_ts": time.time(), "repo_root": self._repo_root,
                               "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                               "attempts": []}
        if k == 1:
            rec["anchor"] = await self._write_anchor(environment)
        rec["head_before"] = (await self._sh(environment, "git rev-parse HEAD"))[1].strip()

        env = dict(self.model_connection.env)
        env["OPENCODE_FAKE_VCS"] = "git"
        env["XDG_DATA_HOME"] = "/logs/agent/opencode/xdg-data"
        env["XDG_STATE_HOME"] = "/logs/agent/opencode/xdg-state"
        skills_command = self._build_register_skills_command()
        if skills_command:
            await self.exec_as_agent(environment, command=skills_command, env=env)
        cfg_command = self._build_register_config_command()
        if cfg_command:
            await self.exec_as_agent(environment, command=cfg_command, env=env)

        kept: list[str] = []
        unresolved_throttle = False
        session_id: str | None = None
        message = instruction
        attempt = 0
        consecutive = 0
        try:
            while True:
                attempt += 1
                name = f"opencode.attempt-{attempt}.txt"
                t0 = time.time()
                exec_error = None
                try:
                    await self.exec_as_agent(environment, command=self._opencode_cmd(
                        message, f"/logs/agent/{name}", session_id), env=env)
                except RuntimeError as exc:  # non-zero exit; judged from the event stream below
                    exec_error = str(exc)[:500]
                out = await self._read_agent_file(environment, name)
                c = classify_events(out)
                verdict = ("no_progress" if c["throttle_errors"] and c["steps"] == 0 else
                           "interrupted" if c["steps"] and c["ended_on_throttle"] else "done")
                a = {"attempt": attempt, "file": name, "resume_session": session_id, "sec": round(time.time() - t0, 1),
                     "verdict": verdict, "exec_error": exec_error,
                     **{key: c[key] for key in ("steps", "errors", "throttle_errors", "tool_calls")},
                     "other_errors": c["other_errors"][:5]}
                rec["attempts"].append(a)
                if verdict != "no_progress":
                    kept.append(out)
                if verdict == "done":
                    break
                if verdict == "interrupted":
                    session_id = session_id or c["session_id"]
                    if session_id is None:
                        unresolved_throttle = True
                        break
                # a step can run for 90 min, so only consecutive attempts that made little
                # progress (< PROGRESS_STEPS model calls) count toward the retry cap and backoff
                consecutive = consecutive + 1 if c["steps"] < PROGRESS_STEPS else 1
                a["consecutive_throttled"] = consecutive
                if consecutive > THROTTLE_MAX_RETRIES:
                    unresolved_throttle = True
                    break
                delay = THROTTLE_BACKOFF_SEC[min(consecutive - 1, len(THROTTLE_BACKOFF_SEC) - 1)]
                a["backoff_sec"] = delay
                self._throttle_retries_total += 1
                self.logger.warning(f"step {k}: provider throttled ({verdict}); retry {attempt} in {delay}s")
                await asyncio.sleep(delay)
                if verdict == "interrupted":
                    message = THROTTLE_CONTINUE_MESSAGE
        finally:
            # opencode.txt = attempts that made progress (Harbor parses it for trajectory + tokens)
            (self.logs_dir / self._OUTPUT_FILENAME).write_text("\n".join(s for s in kept if s))
            rec["throttle_retries"] = sum(1 for a in rec["attempts"] if a.get("backoff_sec"))
            rec["throttle_retries_trial_total"] = self._throttle_retries_total
            rec["ended_ts"] = time.time()
            step_json = self.logs_dir / "redo" / f"step-{k}.json"
            step_json.parent.mkdir(parents=True, exist_ok=True)
            step_json.write_text(json.dumps(rec, indent=2))  # before any await: survives a timeout cancel
            try:
                rec["head_after"] = (await self._sh(environment, "git rev-parse HEAD"))[1].strip()
                snap = await self.snapshot(environment, f"step-{k}-end", {"step_index": k})
                rec["end_snapshot"] = {x: snap.get(x) for x in ("head", "patch_sha256", "patch_bytes", "error")}
                prc, plan = await self._sh(environment, "cat /tmp/plan.md 2>/dev/null", timeout=60)
                if prc == 0 and plan.strip():
                    (self.logs_dir / "redo" / f"plan-{k}.md").write_text(plan)
            except Exception as exc:  # noqa: BLE001 - instrumentation must not fail the step
                rec["end_snapshot_error"] = f"{type(exc).__name__}: {exc}"
            except BaseException as exc:  # CancelledError on step timeout: record, then propagate
                rec["end_snapshot_error"] = f"{type(exc).__name__}: {exc}"
                step_json.write_text(json.dumps(rec, indent=2))
                raise
            step_json.write_text(json.dumps(rec, indent=2))

        final = rec["attempts"][-1] if rec["attempts"] else {}
        other = [e for a in rec["attempts"] if a["verdict"] != "no_progress" for e in a["other_errors"]]
        if other or unresolved_throttle:
            msgs = other[:3] or [f"provider throttling unresolved after {attempt} attempts ({final.get('verdict')})"]
            raise NonZeroAgentExitCodeError("OpenCode emitted error event(s): " + "; ".join(msgs))
