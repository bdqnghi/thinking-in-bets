"""Agentic snapshot judge v3 for SWE-Together (redo_v1; docs/results/INTENT_JUDGE_V3_VALIDATION_PLAN.md).

"Is unit u satisfied at snapshot t" (PLAN.md §6: regression, caving, final pass and the overlay trigger
depend on it) is decided the way the benchmark's own correctness judge decides it, but on the Gemini
judge seat and without any dialogue:

1. a FRESH container of the task's native image (``exec_judge.image``: the local arm64 rebuild
   ``bdqnghi/swe-together:<task>`` on arm64 hosts, the official task image on amd64 hosts) is started
   (as root, with the task's cpus / memory; never under emulation) and the snapshot (the cumulative diff vs the
   base commit, ``agent/snapshots/turn-N.patch``) is applied with the same recipe as the verifier
   (``exec_judge.APPLY_SH``: the clone's ``patch_normalize`` candidates, ``git apply``, ``chmod``);
   the task's ``tests/`` are copied to ``/tests`` (the model may run the verifier as evidence);
2. a Gemini model gets a system prompt (``prompts/agentic_judge_v3_system.md``), the unit list and
   the diff, and works with two tools - ``bash`` (a command in the container, output clipped) and
   ``submit_verdict`` (one label + evidence per unit id; validated, re-asked on a malformed call) -
   within a step and wall-clock budget; at the budget it is asked to submit;
3. the container is removed; the verdicts, the full transcript, tokens and timings are cached by
   ``sha256(judge version, verdict-relevant config, image id, task, unit-set digest, patch)``.

Units (``unit`` in the config):

* ``goals`` - the benchmark's frozen ``canonical_goals.json`` goals of the task, except anti-gaming /
  anti-effort floor guards, zero-weight goals and the four chat-only goals (``CHAT_ONLY_GOALS``: they
  are scored from the agent's chat, not the repository). The goal's turn of origin
  (``source_turn``) is taken from the reviewed map (``configs/intents/swe-together/exec_map.json``).
* ``intents`` - the opening request and the oracle intents (``fsa.redo.intent_judge``); the extra
  label ``not_applicable`` is offered.

Every snapshot is judged on the task's whole unit set (so identical snapshots of different trials
share one cached judgement); "revealed by t" only restricts which units enter regression / caving.

CLI:

    python -u -m fsa.redo.swe_together.agentic_judge snapshot --task T --patch-file P [--model M]
    python -u -m fsa.redo.swe_together.agentic_judge trials <run_dir> [...]   # the frozen v3 judge
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import intent_judge as v1
from .. import llm_seats
from . import exec_judge
from .config import CLONE, RESULTS, ROOT

JUDGE_VERSION = "v3"
SCHEMA = "redo_v1/swt-agentic-judge/1"
TRIAL_SCHEMA = "redo_v1/intent-judge-trial/3"
PROMPT_FILE = Path(__file__).with_name("prompts") / "agentic_judge_v3_system.md"
FREEZE_FILE = ROOT / "configs" / "intents" / "swe-together" / "JUDGE_FREEZE.json"
TRIGGER_FREEZE_FILE = ROOT / "configs" / "intents" / "swe-together" / "OVERLAY_TRIGGER_FREEZE.json"
MAP_FILE = ROOT / "configs" / "intents" / "swe-together" / "exec_map.json"
CACHE = RESULTS / "agentic_judge_cache"
LABELS = ("satisfied", "not_satisfied", "not_addressed")
INTENT_LABELS = LABELS + ("not_applicable",)
UNSATISFIED = frozenset({"not_satisfied", "not_addressed"})
EXCLUDED_TIERS = ("anti_gaming", "anti_effort")
# goals scored from the agent's chat turns, not from the repository (read from the goal texts,
# before any v3 judgement): they cannot be judged from a snapshot and are not units
CHAT_ONLY_GOALS = {"sd-scripts-reg-image-dedup": ("goal_1", "goal_3"),
                   "sd-scripts-fp8-lumina": ("goal_7", "goal_8")}
# no-regression goals that the unchanged base satisfies by construction (read from the goal texts,
# before any v3 judgement); excluded only from the base-check denominator (validation plan (a))
PRESERVE_GOALS = {"cli-task-cd4662": ("goal_preserve_hook_rerun_idempotency",),
                  "cli-task-33e050": ("goal_3", "goal_4"),
                  "cli-task-c425e4": ("goal_3", "goal_4"),
                  "sd-scripts-reg-image-dedup": ("goal_6",),
                  "pi-mono-extensions-event-refactor": ("goal_6",)}

DEFAULTS = {
    "version": "v3-dev", "judge": "agentic_v3", "unit": "goals",
    "model": "gemini-3.5-flash", "reasoning_effort": "high",
    "max_steps": 40,            # model calls per snapshot (the last ones may only submit)
    "max_wall_sec": 1500,       # after this the model is asked to submit
    "cmd_timeout_sec": 600,     # per bash call (killed inside the container)
    "tool_output_chars": 12000,  # bash output shown to the model (head + tail)
    "inline_diff_chars": 30000,  # the diff is inlined in the first message up to this size
    "show_rationale": True,     # the frozen goal's rationale is shown beside the goal
    "refute": False,            # after a draft verdict with any `satisfied`, a refutation pass (then final submit)
    "refute_steps": 16,         # extra model steps for the refutation pass
    "slots": 3,                 # concurrent judge containers (host-wide file-lock slots)
    "output_subdir": "agentic_judge_v3",
}
VERDICT_KEYS = ("version", "judge", "unit", "model", "reasoning_effort", "max_steps", "max_wall_sec", "cmd_timeout_sec",
                "tool_output_chars", "inline_diff_chars", "show_rationale", "refute", "refute_steps")

TRANSIENT_BACKOFF_SEC = llm_seats.TRANSIENT_BACKOFF_SEC
OVERLOAD_BACKOFF_SEC = (3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 300, 300, 300)


def _overload(e: BaseException) -> bool:
    t = f"{type(e).__name__} {e}".lower()
    return ("503" in t or "overloaded" in t or "high demand" in t or "unavailable" in t) and "429" not in t
REFUTE_MESSAGE = (
    "Your verdict is recorded as a DRAFT. Before the final verdict, act as a skeptical second reviewer of the units you "
    "marked satisfied ({units}). For each of them, try hard to REFUTE it: re-read the unit text word by word (every "
    "case, path, input form and part it names); look for a code path, caller, state or input for which the described "
    "behaviour still fails; and construct and run a concrete probe of the exact scenario the unit describes (a "
    "throw-away script or test that sets up that situation and checks the outcome). A unit stays satisfied only if "
    "the probe demonstrates the behaviour and you found no failing path. Then call submit_verdict again with the "
    "FINAL label for every unit id.")


def _sha(b: str | bytes) -> str:
    return hashlib.sha256(b.encode("utf-8", "surrogateescape") if isinstance(b, str) else b).hexdigest()


def prompt_sha(path: Path = PROMPT_FILE) -> str:
    return _sha(Path(path).read_bytes())


# ── units ───────────────────────────────────────────────────────────────────

@dataclass
class Unit:
    key: str
    text: str
    source_turn: int
    kind: str                  # "goal" | "intent:<kind>"
    tier: str | None = None
    weight: float | None = None
    rationale: str | None = None


_MAP: dict | None = None


def _goal_turns(task: str) -> dict[str, int]:
    global _MAP
    if _MAP is None:
        _MAP = json.loads(MAP_FILE.read_text()) if MAP_FILE.exists() else {"tasks": {}}
    t = (_MAP.get("tasks") or {}).get(task) or {}
    return {g["id"]: int(g["source_turn"] or 0) for g in t.get("goals", []) if g.get("source_turn") is not None}


def all_goals(task: str) -> list[dict]:
    return json.loads((CLONE / "tasks" / task / "canonical_goals.json").read_text())["completeness_goals"]


def goal_units(task: str) -> list[Unit]:
    turns = _goal_turns(task)
    out = []
    for g in all_goals(task):
        if g.get("tier") in EXCLUDED_TIERS or not (g.get("weight") or 0) > 0 or g["id"] in CHAT_ONLY_GOALS.get(task, ()):
            continue
        out.append(Unit(g["id"], g["goal"].strip(), turns.get(g["id"], 0), "goal", g.get("tier"), g.get("weight"),
                        (g.get("rationale") or "").strip() or None))
    return out


def intent_units(task: str) -> list[Unit]:
    return [Unit(i.key, i.text, i.source_turn, f"intent:{i.kind}") for i in v1.load_task_intents(CLONE / "tasks" / task)]


def task_units(task: str, unit: str) -> list[Unit]:
    return goal_units(task) if unit == "goals" else intent_units(task)


def units_digest(units: list[Unit]) -> str:
    return _sha(json.dumps([asdict(u) for u in units], sort_keys=True, ensure_ascii=False))


def revealed(units: list[Unit], n_sim_messages: int) -> list[Unit]:
    return [u for u in units if u.source_turn <= n_sim_messages]


# ── tools ───────────────────────────────────────────────────────────────────

def tool_specs(unit_ids: list[str], labels: tuple[str, ...]) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": "bash",
            "description": "Run one bash command in the task container (as root, cwd = repository root). "
                           "Returns the exit code and the combined stdout/stderr (long output is clipped in the "
                           "middle). Each call has a time limit.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "description": "the bash command"},
                "timeout_sec": {"type": "integer", "description": "optional time limit in seconds (capped)"}},
                "required": ["command"]}}},
        {"type": "function", "function": {
            "name": "submit_verdict",
            "description": "Submit the final verdict: exactly one entry per unit id. Call once, at the end.",
            "parameters": {"type": "object", "properties": {
                "verdicts": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string", "enum": unit_ids},
                    "label": {"type": "string", "enum": list(labels)},
                    "evidence": {"type": "string", "description": "concrete evidence: file:line, command + result"}},
                    "required": ["id", "label", "evidence"]}},
                "notes": {"type": "string"}},
                "required": ["verdicts"]}}},
    ]


def tools_sha() -> str:
    return _sha(json.dumps(tool_specs(["<id>"], INTENT_LABELS), sort_keys=True))


def clip(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    h = n * 2 // 3
    return text[:h] + f"\n[... {len(text) - n} characters clipped ...]\n" + text[-(n - h):]


# ── container ───────────────────────────────────────────────────────────────

class Slot:
    """One of ``n`` host-wide judge-container slots (flock files; shared by every judge process)."""

    def __init__(self, n: int, cache: Path = CACHE):
        # SWT_JUDGE3_SLOTS lowers the host-wide cap (e.g. while labellers hold containers); not verdict-relevant
        self.n = max(1, int(os.environ.get("SWT_JUDGE3_SLOTS") or n))
        self.dir = cache / ".slots"
        self.f = None

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        while True:
            for i in range(self.n):
                f = open(self.dir / f"slot{i}.lock", "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.f = f
                    return self
                except BlockingIOError:
                    f.close()
            time.sleep(2 + random.random() * 3)

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


class Sandbox:
    """A fresh container of the task image with the snapshot applied (``exec_judge``'s recipe)."""

    def __init__(self, task: str):
        self.task = task
        self.name = f"swt-judge3-{re.sub(r'[^a-z0-9]+', '-', task.lower())[:36]}-{uuid.uuid4().hex[:8]}"
        self.repo: str | None = None
        self.started = False

    def start(self) -> None:
        iid, arch = exec_judge.image_id(self.task)
        host = exec_judge.machine.host_arch()
        if arch != host:
            raise RuntimeError(f"image {exec_judge.image(self.task)} is {arch}, not {host} (never run under emulation)")
        cpus, mem, _ = exec_judge.task_limits(self.task)
        run = ["run", "-d", "--name", self.name, "--label", "tib.swt_judge3=1", "--platform", f"linux/{host}"]
        if cpus:
            run += ["--cpus", cpus]
        if mem:
            run += ["--memory", mem]
        r = exec_judge._docker(*run, exec_judge.image(self.task), "sh", "-c", "sleep infinity", timeout=300)
        if r.returncode:
            raise RuntimeError(f"docker run: {r.stderr.decode(errors='replace')[-400:]}")
        self.started = True

    def apply(self, patch: str) -> tuple[bool, str]:
        cands = exec_judge.apply_candidates(patch) if patch.strip() else []
        with tempfile.TemporaryDirectory() as td:
            for i, c in enumerate(cands):
                p = Path(td) / f"agent.patch.{i}"
                p.write_text(c, encoding="utf-8", errors="surrogateescape")
                c_ = exec_judge._docker("cp", str(p), f"{self.name}:/tmp/agent.patch.{i}", timeout=300)
                if c_.returncode:
                    raise RuntimeError(f"docker cp patch: {c_.stderr.decode(errors='replace')[-300:]}")
        script = exec_judge.APPLY_SH % {"hint": shlex.quote(exec_judge.main_repo_path(patch) or "") if patch.strip()
                                        else "''", "n": len(cands), "last": max(len(cands) - 1, 0)}
        a = exec_judge._docker("exec", "-u", "root", self.name, "bash", "-c", script, timeout=600)
        log = a.stdout.decode(errors="replace") + a.stderr.decode(errors="replace")
        self.repo = exec_judge._docker("exec", self.name, "cat", "/tmp/swt_repo", timeout=60).stdout.decode().strip() or None
        if a.returncode or not self.repo:
            return False, log
        # the tests (optional evidence) and the snapshot for the model; nothing else is copied in
        exec_judge._docker("exec", "-u", "root", self.name, "bash", "-c",
                           "rm -rf /tests && mkdir -p /logs/verifier /logs/agent /tmp/judge && "
                           "git config --global --add safe.directory '*'", timeout=120)
        c = exec_judge._docker("cp", f"{exec_judge.task_dir(self.task) / 'tests'}/.", f"{self.name}:/tests", timeout=600)
        if c.returncode:
            raise RuntimeError(f"docker cp tests: {c.stderr.decode(errors='replace')[-300:]}")
        exec_judge._docker("exec", "-u", "root", self.name, "bash", "-c",
                           "chmod -R a+rwX /tests /logs && chmod +x /tests/test.sh", timeout=120)
        if cands:
            exec_judge._docker("exec", "-u", "root", self.name, "cp", "/tmp/agent.patch", "/tmp/judge/snapshot.patch",
                               timeout=60)
        return True, log

    def run(self, cmd: str, timeout: int) -> tuple[int | None, str, float]:
        t0 = time.time()
        try:
            r = subprocess.run(["docker", "exec", "-u", "root", "-w", self.repo or "/", self.name, "timeout", "-k", "10",
                                str(int(timeout)), "bash", "-c", cmd], capture_output=True, timeout=timeout + 90)
            out = r.stdout.decode(errors="replace") + (("\n[stderr]\n" + r.stderr.decode(errors="replace"))
                                                        if r.stderr else "")
            rc = r.returncode
            if rc == 124:
                out += f"\n[command timed out after {timeout} s and was killed]"
        except subprocess.TimeoutExpired:
            rc, out = None, f"[command did not return within {timeout + 90} s]"
        return rc, out, time.time() - t0

    def close(self) -> None:
        if self.started:
            exec_judge._docker("rm", "-f", self.name, timeout=300)
            self.started = False


# ── model calls ─────────────────────────────────────────────────────────────

def _usage(u) -> dict:
    d = llm_seats._chat_usage(u)
    return {k: d.get(k) for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "total_tokens")}


def chat(client, model: str, effort: str | None, messages: list, tools: list, log: list, tool_choice=None) -> tuple:
    """One model step with patient retry on transient errors (and on a 200 without choices)."""
    n = 0
    while True:
        t0 = time.time()
        try:
            kw = {"model": model, "messages": messages, "tools": tools}
            if effort:
                kw["reasoning_effort"] = effort
            if tool_choice is not None:
                kw["tool_choice"] = tool_choice
            r = client.chat.completions.create(**kw)
            if not r.choices:
                raise RuntimeError("503 unavailable: response without choices")
            log.append({"ok": True, "latency_sec": round(time.time() - t0, 2), "usage": _usage(r.usage),
                        "resolved_model": getattr(r, "model", model), "finish_reason": r.choices[0].finish_reason})
            return r.choices[0].message, r
        except Exception as e:  # noqa: BLE001
            log.append({"ok": False, "latency_sec": round(time.time() - t0, 2), "error": f"{type(e).__name__}: {str(e)[:400]}"})
            # overload (503 / high demand) is per request on Gemini 3.x Flash: retry quickly; quota (429): patiently
            sched = OVERLOAD_BACKOFF_SEC if _overload(e) else TRANSIENT_BACKOFF_SEC
            if not llm_seats.transient(e) or n >= len(sched):
                raise
            time.sleep(sched[n] * (1 + 0.2 * random.random()))
            n += 1


def _msg_dict(m) -> dict:
    d = m.model_dump(exclude_none=True)
    d["role"] = "assistant"
    if not d.get("tool_calls"):
        d.pop("tool_calls", None)
    if d.get("content") is None:
        d["content"] = ""
    return d


# ── the judge ───────────────────────────────────────────────────────────────

class AgenticJudge:
    def __init__(self, cfg: dict | None = None, cache_dir: Path = CACHE, client=None, verbose: bool = False):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.cache = Path(cache_dir)
        self.client = client
        self.verbose = verbose
        self.psha = prompt_sha()
        self.system = PROMPT_FILE.read_text()
        self.stats = {"judged": 0, "cached": 0, "failed": 0}
        self._lock = threading.Lock()

    # identity ------------------------------------------------------------
    def config_record(self) -> dict:
        return {**self.cfg, "prompt_file": str(PROMPT_FILE.relative_to(ROOT)), "prompt_sha256": self.psha,
                "tools_sha256": tools_sha(), "recipe": exec_judge.RECIPE_VERSION, "schema": SCHEMA}

    def verdict_config_digest(self) -> str:
        c = {k: self.cfg[k] for k in VERDICT_KEYS if k not in ("refute", "refute_steps") or self.cfg.get("refute")}
        c.update(prompt_sha256=self.psha, tools_sha256=tools_sha(), recipe=exec_judge.RECIPE_VERSION)
        return _sha(json.dumps(c, sort_keys=True))

    def units(self, task: str) -> list[Unit]:
        return task_units(task, self.cfg["unit"])

    def key(self, task: str, patch: str, units: list[Unit]) -> str:
        iid = exec_judge.image_id(task)[0] or "?"
        return _sha("\n".join([JUDGE_VERSION, self.verdict_config_digest(), iid, task, units_digest(units)]))[:24] + \
            _sha(patch)[:32]

    def cache_path(self, task: str, key: str) -> Path:
        return self.cache / task / f"{key}.json"

    def cached(self, task: str, patch: str, units: list[Unit] | None = None) -> dict | None:
        units = units or self.units(task)
        p = self.cache_path(task, self.key(task, patch, units))
        if p.exists():
            try:
                rec = json.loads(p.read_text())
                return rec if rec.get("status") == "ok" else None
            except json.JSONDecodeError:
                return None
        return None

    # judging ---------------------------------------------------------------
    def judge_snapshot(self, task: str, patch: str, units: list[Unit] | None = None, label: str = "",
                       use_cache: bool = True) -> dict:
        """``{status, verdicts: {id: {verdict, evidence}}, usage, ...}``; cached when status == ok."""
        units = units or self.units(task)
        key = self.key(task, patch, units)
        out_p = self.cache_path(task, key)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        lock = open(out_p.with_suffix(".lock"), "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if use_cache and out_p.exists():
                try:
                    rec = json.loads(out_p.read_text())
                    if rec.get("status") == "ok":
                        with self._lock:
                            self.stats["cached"] += 1
                        return rec
                except json.JSONDecodeError:
                    pass
            with Slot(self.cfg["slots"], self.cache):
                rec = self._session(task, patch, units, key, label)
            if rec["status"] == "ok":
                tmp = out_p.with_suffix(f".{uuid.uuid4().hex}.tmp")
                tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
                os.replace(tmp, out_p)
            else:  # kept for inspection, never served from the cache
                fail = out_p.parent / "_failed" / f"{key}.{int(time.time())}.json"
                fail.parent.mkdir(exist_ok=True)
                fail.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            with self._lock:
                self.stats["judged" if rec["status"] == "ok" else "failed"] += 1
            return rec
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    def _first_message(self, task: str, sb: Sandbox, patch: str, units: list[Unit], labels: tuple) -> str:
        norm = v1.normalize_patch(patch)
        stat = sb.run("git diff HEAD --stat | tail -40; echo; "
                      "echo '[untracked files]'; git status --porcelain | grep '^??' | head -40",
                      120)[1] if norm.strip() else ""
        parts = [f"Task: `{task}`. Repository root: `{sb.repo}` (base commit = HEAD; the snapshot is applied to the "
                 f"working tree).", ""]
        if not norm.strip():
            parts += ["The snapshot is EMPTY: the working tree equals the base commit. Judge the units against the "
                      "base code as it is.", ""]
        else:
            parts += ["Change summary (`git diff HEAD --stat` and untracked files):", "```", stat.strip()[:6000], "```", ""]
            if len(norm) <= self.cfg["inline_diff_chars"]:
                parts += ["The snapshot diff (also at /tmp/judge/snapshot.patch):", "```diff", norm.rstrip(), "```", ""]
            else:
                parts += [f"The snapshot diff is {len(norm)} characters; read it with git or from "
                          "/tmp/judge/snapshot.patch.", ""]
        parts += [f"Labels allowed: {', '.join(labels)}.", "", f"Units to judge ({len(units)}):", ""]
        for u in units:
            head = f"### `{u.key}`"
            if u.kind == "goal":
                head += f" (tier {u.tier}, weight {u.weight})"
            else:
                head += f" ({u.kind.split(':', 1)[-1]}, from message {u.source_turn})"
            parts += [head, u.text]
            if u.rationale and self.cfg["show_rationale"]:
                parts += ["", f"Rubric rationale: {u.rationale}"]
            parts += [""]
        parts += ["Judge every unit, then call submit_verdict once."]
        return "\n".join(parts)

    def _client(self):
        return self.client if self.client is not None else llm_seats.client_for(self.cfg["model"], timeout=600)

    def _session(self, task: str, patch: str, units: list[Unit], key: str, label: str) -> dict:
        cfg = self.cfg
        labels = INTENT_LABELS if cfg["unit"] == "intents" else LABELS
        ids = [u.key for u in units]
        tools = tool_specs(ids, labels)
        submit_only = [t for t in tools if t["function"]["name"] == "submit_verdict"]
        rec: dict = {"schema": SCHEMA, "judge_version": JUDGE_VERSION, "task": task, "key": key, "label": label,
                     "config": self.config_record(), "config_digest": self.verdict_config_digest(),
                     "image": exec_judge.image(task), "image_id": exec_judge.image_id(task)[0],
                     "patch_sha256": _sha(patch), "patch_bytes": len(patch), "units": [asdict(u) for u in units],
                     "units_digest": units_digest(units), "started_at": time.time()}
        calls: list = []
        tool_log: list = []
        messages: list = []
        sb = Sandbox(task)
        t0 = time.time()
        verdict = None
        try:
            sb.start()
            ok, alog = sb.apply(patch)
            rec["repo"] = sb.repo
            rec["apply_log"] = alog[-1500:]
            if not ok:
                rec["status"] = "apply_failed"
                return rec
            rec["setup_sec"] = round(time.time() - t0, 1)
            messages = [{"role": "system", "content": self.system},
                        {"role": "user", "content": self._first_message(task, sb, patch, units, labels)}]
            client = self._client()
            steps = nudges = bad_submits = 0
            forced = False
            limit, wall_limit = cfg["max_steps"], cfg["max_wall_sec"]
            draft = None
            while verdict is None:
                wall = time.time() - t0
                if not forced and (steps >= limit - 2 or wall >= wall_limit):
                    forced = True
                    messages.append({"role": "user", "content":
                                     "Budget exhausted. Call submit_verdict now with your best judgement for every "
                                     "unit id (use not_satisfied / not_addressed where you could not verify)."})
                if steps >= limit + 3:
                    rec["status"] = "no_verdict"
                    break
                msg, _ = chat(client, cfg["model"], cfg["reasoning_effort"], messages,
                              submit_only if forced else tools, calls)
                steps += 1
                messages.append(_msg_dict(msg))
                if not msg.tool_calls:
                    nudges += 1
                    if nudges > 4:
                        rec["status"] = "no_verdict"
                        break
                    messages.append({"role": "user", "content": "Continue with the tools: run bash commands, or call "
                                     "submit_verdict with one entry per unit id."})
                    continue
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"invalid JSON arguments: {e}"})
                        continue
                    if name == "bash" and not forced:
                        cmd = str(args.get("command") or "")
                        to = min(int(args.get("timeout_sec") or cfg["cmd_timeout_sec"]), cfg["cmd_timeout_sec"])
                        rc, out, dt = sb.run(cmd, max(5, to))
                        tool_log.append({"step": steps, "command": cmd[:4000], "exit": rc, "sec": round(dt, 1),
                                         "output_chars": len(out)})
                        if self.verbose:
                            print(f"  [{task}] $ {cmd[:160]!r} -> {rc} ({dt:.0f}s)", flush=True)
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": f"exit={rc}\n{clip(out, cfg['tool_output_chars'])}"})
                    elif name == "submit_verdict":
                        err, parsed = self._validate(args, ids, labels)
                        if err:
                            bad_submits += 1
                            messages.append({"role": "tool", "tool_call_id": tc.id,
                                             "content": f"submit_verdict rejected: {err}. Call it again with exactly "
                                                        f"one entry per unit id: {ids}."})
                            if bad_submits > 4:
                                rec["status"] = "no_verdict"
                                break
                        elif (cfg.get("refute") and draft is None and
                              any(x["verdict"] == "satisfied" for x in parsed["verdicts"].values())):
                            draft = parsed
                            sat = [k for k, x in parsed["verdicts"].items() if x["verdict"] == "satisfied"]
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "recorded as a DRAFT"})
                            messages.append({"role": "user", "content": REFUTE_MESSAGE.format(units=", ".join(sat))})
                            forced = False
                            limit = steps + cfg["refute_steps"]
                            wall_limit = (time.time() - t0) + cfg["max_wall_sec"] / 2
                        else:
                            verdict = parsed
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "accepted"})
                    else:
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": "only submit_verdict is available now" if forced else
                                         f"unknown tool {name}"})
                if rec.get("status") == "no_verdict":
                    break
            if verdict is not None:
                rec["status"] = "ok"
                rec["verdicts"] = verdict["verdicts"]
                rec["notes"] = verdict.get("notes")
                rec["forced_submit"] = forced
                if draft is not None:
                    rec["draft_verdicts"] = draft["verdicts"]
                    rec["refuted"] = sorted(k for k, x in draft["verdicts"].items()
                                            if x["verdict"] == "satisfied" and verdict["verdicts"][k]["verdict"] != "satisfied")
            rec["steps"] = steps
            rec["nudges"] = nudges
            rec["bad_submits"] = bad_submits
        except Exception as e:  # noqa: BLE001
            rec["status"] = "infra_error"
            rec["error"] = f"{type(e).__name__}: {str(e)[:600]}"
        finally:
            sb.close()
            rec["wall_sec"] = round(time.time() - t0, 1)
            rec["calls"] = calls
            rec["tool_log"] = tool_log
            rec["usage"] = self._sum_usage(calls)
            tp = self.cache_path(task, key).with_suffix(f".transcript.{int(time.time())}.json")
            tp.parent.mkdir(parents=True, exist_ok=True)
            tp.write_text(json.dumps(messages[1:], ensure_ascii=False))
            rec["transcript"] = str(tp.relative_to(RESULTS))
        return rec

    @staticmethod
    def _sum_usage(calls: list) -> dict:
        out = {"calls": len(calls), "ok_calls": sum(1 for c in calls if c.get("ok")),
               "failed_attempts": sum(1 for c in calls if not c.get("ok"))}
        for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "total_tokens"):
            out[k] = sum((c.get("usage") or {}).get(k) or 0 for c in calls if c.get("ok"))
        out["model_latency_sec"] = round(sum(c.get("latency_sec") or 0 for c in calls), 1)
        return out

    @staticmethod
    def _validate(args: dict, ids: list[str], labels: tuple) -> tuple[str | None, dict | None]:
        vs = args.get("verdicts")
        if not isinstance(vs, list):
            return "`verdicts` must be a list", None
        got: dict = {}
        for v in vs:
            if not isinstance(v, dict) or v.get("id") not in ids:
                return f"unknown unit id {v.get('id') if isinstance(v, dict) else v!r}", None
            if v.get("label") not in labels:
                return f"label {v.get('label')!r} for {v['id']} is not one of {labels}", None
            if v["id"] in got:
                return f"unit {v['id']} appears twice", None
            got[v["id"]] = {"verdict": v["label"], "evidence": str(v.get("evidence") or "")[:2000]}
        missing = [i for i in ids if i not in got]
        if missing:
            return f"missing unit ids {missing}", None
        return None, {"verdicts": got, "notes": args.get("notes")}


# ── freeze ──────────────────────────────────────────────────────────────────

def load_freeze(path: Path = FREEZE_FILE, verify: bool = True) -> dict:
    """The frozen v3 configuration; verifies prompt, tool schema and unit sets still match."""
    doc = json.loads(Path(path).read_text())
    cfg = doc["config"]
    if cfg.get("judge") != "agentic_v3":
        raise SystemExit(f"{path} does not freeze the agentic v3 judge")
    if verify:
        if prompt_sha(ROOT / cfg["prompt_file"]) != cfg["prompt_sha256"]:
            raise SystemExit(f"{cfg['prompt_file']} changed since the freeze; record a new judge version")
        if tools_sha() != cfg["tools_sha256"]:
            raise SystemExit("the v3 tool schema changed since the freeze; record a new judge version")
        for task, dig in (doc.get("units_digests") or {}).items():
            if units_digest(task_units(task, cfg["unit"])) != dig:
                raise SystemExit(f"unit set of {task} changed since the freeze; record a new judge version")
    return cfg


def freeze_digest(path: Path = FREEZE_FILE) -> str:
    return _sha(Path(path).read_bytes())


def frozen_judge(**kw) -> tuple[AgenticJudge, dict]:
    cfg = load_freeze()
    meta = {"file": str(FREEZE_FILE.relative_to(ROOT)), "sha256": freeze_digest(), "version": cfg["version"]}
    return AgenticJudge(cfg, **kw), meta


# ── in-trial overlay trigger (frozen cheap configuration) ───────────────────

def load_trigger_freeze(path: Path = TRIGGER_FREEZE_FILE) -> dict:
    doc = json.loads(Path(path).read_text())
    judge_cfg = load_freeze()  # the trigger is only valid beside the frozen judge it was validated with
    if doc.get("judge_freeze_sha256") != freeze_digest():
        raise SystemExit("OVERLAY_TRIGGER_FREEZE.json was made under another judge freeze; record a new trigger freeze")
    cfg = doc["config"]
    if cfg["prompt_sha256"] != judge_cfg["prompt_sha256"] or cfg["tools_sha256"] != tools_sha():
        raise SystemExit("trigger prompt / tools differ from the freeze")
    return cfg


TRIGGER_COST_KEYS = ("calls", "failed_attempts", "input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens",
                     "containers", "wall_sec")


class OverlayTrigger:
    """Drop-in for the ``(judge, ij)`` pair the SWE-Together overlay (``user_agent/redo_arms.py``) uses:
    ``ij.load_task_intents(task_dir)`` -> the task's frozen goal units, ``ij.revealed(units, n)``,
    ``judge.judge_snapshot(patch, units)`` -> ``{key: {verdict, evidence}}`` (the frozen cheap v3 config, run
    in its own container), ``judge.stats``, ``ij.patch_sha``, ``ij.judge_config()``."""

    def __init__(self, task_dir: Path):
        self.task = Path(task_dir).name
        self.cfg = load_trigger_freeze()
        self.judge = AgenticJudge(self.cfg)
        # the judge's judged / cached / failed counters plus the spend of fresh (uncached) consultations, so
        # the overlay's per-turn ``judge_stats_delta`` carries this seat's cost (not verdict-relevant)
        self.stats = self.judge.stats
        self.stats.update({k: 0 for k in TRIGGER_COST_KEYS})

    # module-like surface ("ij")
    def load_task_intents(self, task_dir) -> list[Unit]:
        return self.judge.units(Path(task_dir).name)

    @staticmethod
    def revealed(units: list[Unit], n: int) -> list[Unit]:
        return revealed(units, n)

    @staticmethod
    def patch_sha(text: str) -> str:
        return v1.patch_sha(text)

    def judge_config(self) -> dict:
        return {**self.judge.config_record(), "trigger_freeze_sha256": freeze_digest(TRIGGER_FREEZE_FILE)}

    # judge-like surface
    def judge_snapshot(self, patch: str, units: list[Unit]) -> dict:
        all_units = self.judge.units(self.task)
        cached_before = self.stats["cached"]
        rec = self.judge.judge_snapshot(self.task, patch, all_units, label="overlay-trigger")
        if self.stats["cached"] == cached_before:  # a fresh session (ok or not): its calls were spent
            u = rec.get("usage") or {}
            self.stats["calls"] += u.get("calls") or 0
            self.stats["failed_attempts"] += u.get("failed_attempts") or 0
            for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
                self.stats[k] += u.get(k) or 0
            self.stats["containers"] += 1
            self.stats["wall_sec"] = round(self.stats["wall_sec"] + (rec.get("wall_sec") or 0), 1)
        self.last = {"key": rec.get("key"), "status": rec.get("status"), "cached": self.stats["cached"] > cached_before}
        if rec.get("status") != "ok":
            raise RuntimeError(f"trigger judge {rec.get('status')}: {rec.get('error') or ''}")
        keys = {u.key for u in units}
        return {k: x for k, x in rec["verdicts"].items() if k in keys}


# ── trials ──────────────────────────────────────────────────────────────────

def judge_trial(trial_dir: Path, judge: AgenticJudge, freeze_meta: dict | None = None, pool=None) -> dict:
    """Schema of ``intent_judge.judge_trial`` (read by ``fsa.redo.metrics``), keyed by unit id: every
    executed snapshot on the task's whole unit set, ``revealed`` = units whose turn of origin is <= the
    number of simulator messages by then (turn-999, the final capture: every unit)."""
    trial_dir = Path(trial_dir)
    task = v1.trial_task_dir(trial_dir).name
    units = judge.units(task)
    nsim = v1.sim_messages_by_turn(v1.read_turns(trial_dir))
    ref_patch = exec_judge.reference_patch(task)
    jobs = [("ref", None, ref_patch)] if ref_patch.strip() else []
    for t in v1.executed_snapshots(trial_dir):
        jobs.append(("snap", t, (trial_dir / "agent" / "snapshots" / f"turn-{t}.patch").read_text(errors="surrogateescape")))
    run = (lambda j: judge.judge_snapshot(task, j[2], units, label=f"{trial_dir.name}:{j[1]}"))
    recs = list(pool.map(run, jobs)) if pool is not None else [run(j) for j in jobs]
    ref = None
    per = {}
    for (kind, t, text), r in zip(jobs, recs):
        vs = r.get("verdicts") or {}
        if kind == "ref":
            ref = {k: x["verdict"] for k, x in vs.items()} if r.get("status") == "ok" else None
            continue
        n = max(nsim.values(), default=0) if t == 999 else nsim.get(t, max((v for k, v in nsim.items() if k <= t), default=0))
        rev = units if t == 999 else revealed(units, n)
        per[str(t)] = {"n_sim_messages": n, "revealed": [u.key for u in rev], "patch_sha256": v1.patch_sha(text),
                       "status": r.get("status"), "cache_key": r.get("key"),
                       "verdicts": {k: x["verdict"] for k, x in vs.items()},
                       "evidence": {k: x.get("evidence") for k, x in vs.items()}}
    return {"schema": TRIAL_SCHEMA, "trial": trial_dir.name, "task": task, "judge": judge.config_record(),
            "freeze": freeze_meta, "unit": judge.cfg["unit"], "intents": [asdict(u) for u in units],
            "reference_verdicts": ref,
            "durable": sorted(k for k, x in ref.items() if x == "satisfied") if ref else None,
            "snapshots": per, "complete": all(p["status"] == "ok" for p in per.values()),
            "judged_at": time.time()}


def cmd_trials(args) -> int:
    judge, meta = frozen_judge()
    sub = judge.cfg["output_subdir"]
    jobs = []
    for rd in args.run_dirs:
        rd = Path(rd).resolve()
        (rd / sub).mkdir(exist_ok=True)
        for td in sorted(d for d in (rd / "trials").iterdir() if d.is_dir() and "__" in d.name):
            if not (td / "agent" / "turns.jsonl").exists():
                continue
            if args.completed_only and not (td / "result.json").exists():
                continue
            outp = rd / sub / f"{td.name}.json"
            if outp.exists() and not args.force:
                try:
                    old = json.loads(outp.read_text())
                    if old.get("freeze", {}).get("sha256") == meta["sha256"] and old.get("complete"):
                        continue
                except json.JSONDecodeError:
                    pass
            jobs.append((td, outp))
    print(f"{len(jobs)} trials to judge (v3 {meta['version']}, freeze {meta['sha256'][:12]})", flush=True)
    snap_pool = ThreadPoolExecutor(max(1, judge.cfg["slots"]))

    def one(job):
        td, outp = job
        try:
            res = judge_trial(td, judge, meta, pool=snap_pool)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {td}: {type(e).__name__}: {e}", flush=True)
            return
        if outp.exists():
            (outp.parent / "_superseded").mkdir(exist_ok=True)
            outp.rename(outp.parent / "_superseded" / f"{outp.stem}.{int(time.time())}.json")
        outp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"{outp}: {len(res['snapshots'])} snapshots complete={res['complete']}; {judge.stats}", flush=True)
    with ThreadPoolExecutor(args.trial_workers) as pool:
        list(pool.map(one, jobs))
    snap_pool.shutdown()
    return 0


def cmd_snapshot(args) -> int:
    cfg = {k: v for k, v in (("model", args.model), ("reasoning_effort", args.effort), ("unit", args.unit),
                             ("max_steps", args.max_steps)) if v is not None}
    j = AgenticJudge(cfg, verbose=True)
    patch = Path(args.patch_file).read_text(errors="surrogateescape") if args.patch_file else ""
    rec = j.judge_snapshot(args.task, patch, use_cache=not args.no_cache)
    print(json.dumps({k: rec.get(k) for k in ("status", "verdicts", "usage", "wall_sec", "steps", "forced_submit",
                                               "error", "transcript")}, indent=1, ensure_ascii=False))
    return 0 if rec.get("status") == "ok" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--task", required=True)
    s.add_argument("--patch-file", default=None, help="omit for the unmodified base")
    s.add_argument("--model", default=None)
    s.add_argument("--effort", default=None)
    s.add_argument("--unit", default=None, choices=["goals", "intents"])
    s.add_argument("--max-steps", type=int, default=None)
    s.add_argument("--no-cache", action="store_true")
    t = sub.add_parser("trials")
    t.add_argument("run_dirs", nargs="+")
    t.add_argument("--trial-workers", type=int, default=2)
    t.add_argument("--completed-only", action="store_true")
    t.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return cmd_snapshot(a) if a.cmd == "snapshot" else cmd_trials(a)


if __name__ == "__main__":
    sys.exit(main())
