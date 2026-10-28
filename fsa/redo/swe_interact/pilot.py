"""SWE-Interact pilot launcher (PLAN.md §5.4 Stage 2; mirror of fsa/redo/swe_together/pilot.py).

Matrix (``matrix()``), 380 trials on the 20-task effective subset:

    M gemini-3.6-flash, S gemini-3.8-flash  x  {stock, commit_before, claims_after}
        x {natural r1, natural r2, overlay r1}                        = 18 cells x 20 = 360
    W gemini-3.5-flash-lite x stock x natural r1                      =  1 cell  x 20 =  20

Layout: ``<root>/<rung>/<arm>/<natural_r1|natural_r2|overlay_r1>/`` - each cell is an ordinary
SWE-Interact run dir (``manifest.json``, ``jobs/<task>__<rung>__a<N>/``, ``logs/``,
``launches.jsonl``), so ``metrics judge`` / ``metrics report`` work on it unchanged.

    python -u -m fsa.redo.swe_interact.pilot plan                     # matrix, counts, wall-time estimate
    python -u -m fsa.redo.swe_interact.pilot trials                   # resume-friendly; re-run to resume
    python -u -m fsa.redo.swe_interact.pilot trials --cap W=8,M=4,S=2 # after the SWE-Together pilot ends
    python -u -m fsa.redo.swe_interact.pilot judge                    # per-snapshot intent verdicts
    python -u -m fsa.redo.swe_interact.pilot report                   # metrics over all cells

Concurrency: one ``harbor run`` per trial (thread), gated by a per-rung semaphore, a global cap
(``--max-total``) and an rf cap (``--max-rf``, default 2: rf tasks ask for 16 CPU / 16 GB each).
Default caps = ``rungs.json`` ``stage1_trial_cap`` (W 6, M 1, S 1: the Stage 0 recommendation
while the SWE-Together pilot shares the Gemini key); ``--cap W=8,M=4,S=2`` once it has finished
(``stage1_trial_cap_after_swt``); raise further only if a batch shows no 429. The caps in force are
recorded per launch. ``--caps-file PATH`` (JSON, any subset of ``W``/``M``/``S``/``max_total``/``max_rf``)
overrides them at start and is re-read every ``CAPS_POLL_SEC`` while trials run (a change applies to
trials not yet started; running trials are never stopped).

Resume and retries: a trial is complete when its final step's verifier ran (``run.trial_complete``:
``result.json`` with a verifier result and no trial-level exception) - whatever its reward. Only an
incomplete launch (infrastructure: docker/compose failure, provider throttling unresolved, a crash)
is re-launched, at most ``--max-attempts`` times per invocation, each into a new ``__a<N>`` job dir;
every launch is kept (``launches.jsonl`` records rc, wall time, step exceptions). A trial is never
re-run because of its score. Each launch leases an explicit ``10.213.N.0/24`` subnet (host docker
address pools are exhausted); the anonymous-pull ``DOCKER_CONFIG`` is set only in the Harbor
subprocess environment. The intent sets must be frozen (``intents freeze``) before the pilot runs:
overlay cells refuse to start otherwise, and every manifest records the index digest.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .. import caps as caps_lib
from .. import machine
from . import patchset
from .config import RESULTS, ROOT, family, load_rungs, subset_tasks
from .run import (ANON_DOCKER_CONFIG, judge_seats_record, MCP_TIMEOUT_MS, _gemini_record, base_manifest, build_task_image,
                  harbor_cmd, image_record, public_env, redo_env, redo_record, repo_root, run_one, seat_env,
                  trial_complete, write_manifest)

DEFAULT_ROOT = RESULTS / "stage2_swe_interact"
OVERLAY_SEED = 20260926          # = subset seed; same seed for every arm (overlay templates pair by task)
TAU = 0.5
PILOT_ARMS = ("stock", "commit_before", "claims_after")
ARM_STATUS = (Path(os.environ.get("SWE_BENCHMARK_ARM_DIR", ROOT.parent / "swe_benchmark_arm")) / "swe_interact/status.json")  # read-only


def matrix() -> list[dict]:
    cells = []
    for rung in ("M", "S"):
        for arm in PILOT_ARMS:
            for rep in (1, 2):
                cells.append({"rung": rung, "arm": arm, "overlay": False, "replicate": rep})
            cells.append({"rung": rung, "arm": arm, "overlay": True, "replicate": 1})
    cells.append({"rung": "W", "arm": "stock", "overlay": False, "replicate": 1})
    return cells


def cell_cond(c: dict) -> str:
    """natural | overlay | mixed (Wave A: the mixed-validity overlay, ``arms.Overlay`` mode "mixed")."""
    if not c["overlay"]:
        return "natural"
    return "mixed" if c.get("mode", "standard") == "mixed" else "overlay"


def cell_dir(root: Path, c: dict) -> Path:
    return root / c["rung"] / c["arm"] / f"{cell_cond(c)}_r{c['replicate']}"


def parse_caps(s: str | None) -> dict[str, int]:
    cfg = load_rungs()
    caps = {r: int(v.get("stage1_trial_cap") or 1) for r, v in cfg["rungs"].items()}
    for part in (s or "").split(","):
        if part.strip():
            k, v = part.split("=")
            caps[k.strip()] = int(v)
    return caps


def parse_cells(spec: str) -> list[dict]:
    """``RUNG:ARM:natural|overlay|mixed:REP,...`` (smoke tests / partial reruns / Wave A cells)."""
    out = []
    for part in spec.split(","):
        rung, arm, cond, rep = part.strip().split(":")
        if cond not in ("natural", "overlay", "mixed"):
            raise SystemExit(f"cell condition {cond!r} not in natural|overlay|mixed")
        out.append({"rung": rung, "arm": arm, "overlay": cond != "natural", "replicate": int(rep),
                    "mode": "mixed" if cond == "mixed" else "standard"})
    return out


def select_cells(a) -> list[dict]:
    cells = parse_cells(a.cells) if a.cells else matrix()
    if a.rungs:
        cells = [c for c in cells if c["rung"] in a.rungs.split(",")]
    if a.arms:
        cells = [c for c in cells if c["arm"] in a.arms.split(",")]
    if a.condition:
        cells = [c for c in cells if cell_cond(c) == a.condition]
    return cells


def task_done(cd: Path, task: str, rung: str) -> bool:
    jobs = cd / "jobs"
    return jobs.exists() and any(trial_complete(j) for j in jobs.glob(f"{task[:80]}__{rung}__a*"))


INFRA_CAP = 3  # PLAN.md §5.4: at most two re-runs after the first failed (infrastructure) attempt


def agent_started(job: Path) -> bool | None:
    """Whether the agent ran in this attempt: True if some trial of the job has an agent execution or
    step results, False if every trial failed before (environment or agent setup, e.g. a package
    install), None if unknown (no readable result.json)."""
    from .run import trial_dirs
    seen = False
    for d in trial_dirs(job):
        try:
            res = json.loads((d / "result.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        seen = True
        if res.get("agent_execution") or res.get("step_results"):
            return True
    return False if seen else None


def failed_attempts(cd: Path, task: str, rung: str) -> int:
    """Attempt directories of ``task`` on ``rung`` that did not complete, not counting attempts that
    failed before the agent started (``agent_started`` False): those are not the agent's attempts,
    so they neither use up the re-run cap nor could be scored as its result (2026-09-29)."""
    jobs = cd / "jobs"
    if not jobs.exists():
        return 0
    return sum(1 for j in jobs.glob(f"{task[:80]}__{rung}__a*")
               if j.is_dir() and not trial_complete(j) and agent_started(j) is not False)


def ensure_images(tasks: list[str], pull: bool) -> dict:
    """Task images native to this host (``run.image_record``): prebuilt ones pulled anonymously,
    build-based ones (no task.toml ``docker_image``) built as Harbor builds them, and the amd64
    layer's local images (``amd64.py``; missing or built from another recipe) built from their recipe
    (``run.build_task_image``). The arm64 rebuilds' validation status applies to arm64 hosts only."""
    arch = machine.host_arch()
    status = json.loads(ARM_STATUS.read_text()) if arch == "arm64" and ARM_STATUS.exists() else {}
    out = {}
    for t in tasks:
        rec = image_record(t)
        if not rec["present"] and pull:
            if rec["prebuilt"] and "amd64_recipe" not in rec:
                ANON_DOCKER_CONFIG.mkdir(parents=True, exist_ok=True)
                (ANON_DOCKER_CONFIG / "config.json").write_text("{}")
                subprocess.run(["docker", "pull", "--platform", f"linux/{arch}", rec["image"]], capture_output=True,
                               env={**__import__("os").environ, "DOCKER_CONFIG": str(ANON_DOCKER_CONFIG)})
                rec = {**image_record(t), "pulled_now": True}
            else:
                b = build_task_image(t)
                rec = {**image_record(t), "built_now": b["built_now"],
                       **({} if b["built_now"] else {"build_log_tail": b["build_log_tail"]})}
        st = status.get(t, {})
        out[t] = {**rec, "gold_validated": st.get("validated"), "pushed": st.get("pushed")}
    return out


def step_exceptions(job: Path) -> dict:
    out = {}
    for t in job.iterdir() if job.exists() else []:
        try:
            res = json.loads((t / "result.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError, NotADirectoryError):
            continue
        out["trial"] = (res.get("exception_info") or {}).get("exception_type")
        for s in res.get("step_results") or []:
            if s.get("exception_info"):
                out[s["step_name"]] = s["exception_info"].get("exception_type")
    return out


# ── plan ────────────────────────────────────────────────────────────────────

def estimate(cells: list[dict], n_tasks: int, caps: dict, mins: dict) -> dict:
    rungs = sorted({c["rung"] for c in cells}) or ["W", "M", "S"]
    n = {r: sum(1 for c in cells if c["rung"] == r) * n_tasks for r in rungs}
    hours = {r: round(n[r] * mins.get(r, mins["S"]) / max(caps.get(r, 1), 1) / 60, 1) for r in n}
    return {"trials_per_rung": n, "caps": caps, "hours_per_rung": hours,
            "wall_hours (rungs in parallel)": max(hours.values()),
            "wall_days": round(max(hours.values()) / 24, 1)}


def cmd_plan(a) -> int:
    cells = select_cells(a)
    tasks = a.tasks.split(",") if a.tasks else subset_tasks()
    cfg = load_rungs()
    mins = {"W": a.min_per_trial_w, "M": a.min_per_trial_m, "S": a.min_per_trial_s}  # other rungs: the S value
    print(json.dumps({
        "cells": len(cells), "tasks": len(tasks), "trials": len(cells) * len(tasks),
        "minutes_per_trial_assumed": mins,
        "note": "Stage 0: W 25-55 min, M/S 2-2.7 h per session under contention with the SWE-Together pilot",
        "estimate_default_caps": estimate(cells, len(tasks), parse_caps(a.cap), mins),
        "estimate_after_swt_caps": estimate(cells, len(tasks), {r: int(v.get("stage1_trial_cap_after_swt") or 1)
                                                                for r, v in cfg["rungs"].items()}, mins),
        "max_total": a.max_total, "max_rf": a.max_rf}, indent=1))
    for c in cells:
        print(cell_dir(Path(a.root), c).relative_to(ROOT) if Path(a.root).is_relative_to(ROOT) else cell_dir(Path(a.root), c))
    return 0


# ── trials ──────────────────────────────────────────────────────────────────

def cmd_trials(a) -> int:
    cfg = load_rungs()
    root = Path(a.root).resolve()
    tasks = a.tasks.split(",") if a.tasks else subset_tasks()
    bad = [t for t in tasks if t not in set(subset_tasks())]
    if bad:
        raise SystemExit(f"tasks outside the frozen effective subset: {bad}")
    cells = select_cells(a)
    caps = parse_caps(a.cap)
    max_total, max_rf = a.max_total, a.max_rf
    caps_file = Path(a.caps_file).resolve() if getattr(a, "caps_file", None) else None
    file_caps = None
    if caps_file is not None:
        try:
            file_caps = caps_lib.read_caps_file(caps_file, CAPS_KEYS)
        except ValueError as e:
            raise SystemExit(f"--caps-file {caps_file}: {e}")
        if file_caps:
            caps.update({r: v for r, v in file_caps.items() if r in RUNG_KEYS})
            max_total = file_caps.get("max_total", max_total)
            max_rf = file_caps.get("max_rf", max_rf)
    from .intents import index_digest
    idx = index_digest()
    if any(c["overlay"] for c in cells) and not idx["frozen"]:
        raise SystemExit("intent sets are not frozen (python -m fsa.redo.swe_interact.intents freeze)")
    images = ensure_images(tasks, pull=not a.no_pull)
    missing = [t for t, r in images.items() if not r["present"]]
    if missing:
        raise SystemExit(f"native task images missing locally: {missing}")
    pv = patchset.verify()
    jobs = []
    for c in cells:
        cd = cell_dir(root, c)
        (cd / "jobs").mkdir(parents=True, exist_ok=True)
        rung = cfg["rungs"][c["rung"]]
        redo = redo_record(c["arm"], tau=TAU, overlay=c["overlay"], overlay_seed=OVERLAY_SEED,
                           mcp_timeout_ms=a.mcp_timeout_ms, overlay_wait=a.overlay_wait, agent_model=rung["model"],
                           overlay_mode=c.get("mode", "standard"))
        man = base_manifest(cfg, "trials")
        man.update({
            "rung": c["rung"], "cell": {**c, "pilot": a.pilot_name, "caps": caps, "max_total": max_total,
                                        "max_rf": max_rf, "max_attempts": a.max_attempts,
                                        **({"caps_file": str(caps_file),
                                            "caps_note": "caps at invocation start; the caps file may change "
                                                         "them while trials run - launches.jsonl records the "
                                                         "caps in force when each launch started"}
                                           if caps_file else {})},
            "redo": redo, "intents_index": idx,
            "agent": {"model": rung["model"], "thinking": rung["thinking"], "harness": cfg["harness"],
                      "provider_record": _gemini_record(rung["model"].split("/", 1)[1])},
            "user_sim": {**cfg["seats"]["user_sim"], "provider_record": _gemini_record(cfg["seats"]["user_sim"]["model"])},
            "rubric_grader": {**cfg["seats"]["rubric_grader"],
                              "provider_record": _gemini_record(cfg["seats"]["rubric_grader"]["model"])},
            "judge_seats": judge_seats_record(), "harbor": cfg["harbor"],
            "tasks": {t: {"family": family(t), "repo_root": repo_root(t), **images[t]} for t in tasks},
            "commands": {t: harbor_cmd(cfg, c["rung"], t, cd / "jobs", f"{t[:80]}__{c['rung']}__a<N>") for t in tasks},
            "env": public_env({**seat_env(cfg, tasks[0]), **redo_env(c["arm"], overlay=c["overlay"],
                                                                     mcp_timeout_ms=a.mcp_timeout_ms)}),
        })
        if not a.dry_run:
            write_manifest(cd, man)
        for t in tasks:
            jobs.append({"cell": c, "task": t, "dir": cd})
    order = {("natural", 1): 0, ("overlay", 1): 1, ("mixed", 1): 1, ("natural", 2): 2}
    jobs.sort(key=lambda j: (order.get((cell_cond(j["cell"]), j["cell"]["replicate"]), 3),
                             tasks.index(j["task"]), j["cell"]["rung"], j["cell"]["arm"]))
    todo = [j for j in jobs if not task_done(j["dir"], j["task"], j["cell"]["rung"])]
    print(f"{len(jobs)} trials in {len(cells)} cells; {len(jobs) - len(todo)} already complete; caps {caps}, "
          f"max_total {max_total}, max_rf {max_rf}; patch {pv['patch_sha256'][:12]}"
          + (f"; caps file {caps_file}" if caps_file else ""), flush=True)
    if a.dry_run:
        for j in todo[:40]:
            print("  would run", cell_dir(root, j["cell"]).relative_to(root), j["task"])
        return 0
    slots = Slots(caps, max_total, max_rf)
    stats = {"done": 0, "incomplete": 0, "refused": 0, "drained": 0}
    lock = threading.Lock()
    for i, j in enumerate(todo):
        j["order"] = i

    def run_job(j):
        c, task, cd = j["cell"], j["task"], j["dir"]
        with slots.hold(j["order"], c["rung"], family(task) == "rf"):
            for attempt in range(1, a.max_attempts + 1):
                if task_done(cd, task, c["rung"]):
                    with lock:
                        stats["done"] += 1
                    return
                if failed_attempts(cd, task, c["rung"]) >= INFRA_CAP:
                    print(f"CAPPED {task} on {c['rung']}: {INFRA_CAP} failed attempts; the last is scored as the agent's result", flush=True)
                    with lock:
                        stats["done"] += 1
                    return
                why = machine.drain_requested()
                if why:  # operator/watchdog drain (e.g. credentials about to expire): start nothing new
                    print(f"DRAINED {task} on {c['rung']}: {why}", flush=True)
                    with lock:
                        stats["drained"] += 1
                    return
                try:
                    live = patchset.verify()["patch_sha256"]
                except SystemExit as e:
                    live = f"differs: {e}"
                if live != pv["patch_sha256"]:
                    print(f"REFUSING {task}: clone patch changed since start ({live[:80]})", flush=True)
                    with lock:
                        stats["refused"] += 1
                    return
                short = {"natural": "nat", "overlay": "ov", "mixed": "mx"}[cell_cond(c)]
                tag = f"{c['rung']}/{c['arm']}/{short}{c['replicate']} {task}"
                print(f"[{datetime.now():%H:%M:%S}] start {tag} (attempt {attempt})", flush=True)
                started = datetime.now(timezone.utc).isoformat()
                caps_now = slots.snapshot()
                r = run_one(cfg, c["rung"], task, cd, arm=c["arm"], overlay=c["overlay"], tau=TAU,
                            overlay_seed=OVERLAY_SEED, mcp_timeout_ms=a.mcp_timeout_ms, overlay_wait=a.overlay_wait,
                            overlay_mode=c.get("mode", "standard"))
                rec = {**r, "attempt": attempt, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                       "patch_sha256": live, **caps_now,
                       "step_exceptions": step_exceptions(cd / "jobs" / r["job"]) if r.get("job") else None,
                       "retry_reason": None if r.get("complete") or r.get("skipped") else "incomplete (infrastructure)"}
                with lock, open(cd / "launches.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{datetime.now():%H:%M:%S}] done  {tag} rc={r.get('rc')} complete={r.get('complete')} "
                      f"{r.get('wall_sec', 0):.0f}s", flush=True)
                if r.get("complete") or r.get("skipped"):
                    with lock:
                        stats["done"] += 1
                    return
                time.sleep(30)
            with lock:
                stats["incomplete"] += 1

    stop = threading.Event()
    if caps_file is not None:
        threading.Thread(target=watch_caps, args=(caps_file, slots, file_caps, stop), daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=max(len(todo), 1)) as ex:  # threads wait in Slots, in queue order
            list(ex.map(run_job, todo))
    finally:
        stop.set()
    print(json.dumps(stats), flush=True)
    return 0 if stats["incomplete"] == 0 and stats["refused"] == 0 and stats["drained"] == 0 else 1


class Slots:
    """Per-rung, global and rf concurrency caps. A waiting job starts when its caps allow it and no
    earlier job in the queue order (rep-major, task, rung, arm: both arms of a task run close in
    time) that could start is still waiting."""

    def __init__(self, caps: dict, max_total: int, max_rf: int):
        self.caps, self.max_total, self.max_rf = dict(caps), max_total, max_rf
        self.run = {r: 0 for r in {"W", "M", "S", *caps}}
        self.total = self.rf = 0
        self.waiting: dict[int, tuple[str, bool]] = {}
        self.cv = threading.Condition()

    def _fits(self, rung: str, rf: bool) -> bool:
        return (self.run[rung] < self.caps[rung] and self.total < self.max_total
                and (not rf or self.rf < self.max_rf))

    def hold(self, order: int, rung: str, rf: bool):
        slots = self

        class _Ctx:
            def __enter__(self_):
                with slots.cv:
                    slots.waiting[order] = (rung, rf)
                    while not (slots._fits(rung, rf) and all(
                            o >= order or not slots._fits(*slots.waiting[o]) for o in slots.waiting)):
                        slots.cv.wait(timeout=5)
                    del slots.waiting[order]
                    slots.run[rung] += 1
                    slots.total += 1
                    slots.rf += int(rf)
                    slots.cv.notify_all()

            def __exit__(self_, *exc):
                with slots.cv:
                    slots.run[rung] -= 1
                    slots.total -= 1
                    slots.rf -= int(rf)
                    slots.cv.notify_all()
                return False
        return _Ctx()

    def set_caps(self, spec: dict[str, int]) -> None:
        """Apply caps-file values (``CAPS_KEYS`` subset); running trials are never stopped."""
        with self.cv:
            for r in self.run:
                if r in spec:
                    self.caps[r] = spec[r]
            self.max_total = spec.get("max_total", self.max_total)
            self.max_rf = spec.get("max_rf", self.max_rf)
            self.cv.notify_all()

    def snapshot(self) -> dict:
        """The caps now in force (recorded per launch)."""
        with self.cv:
            return {"caps": dict(self.caps), "max_total": self.max_total, "max_rf": self.max_rf}


RUNG_KEYS = tuple(load_rungs()["rungs"])  # every rung in rungs.json (W, M, S, and any exploratory rung)
CAPS_KEYS = (*RUNG_KEYS, "max_total", "max_rf")
CAPS_POLL_SEC = 60.0


def watch_caps(path: Path, slots: Slots, last: dict | None, stop: threading.Event,
               every: float = CAPS_POLL_SEC) -> None:
    """Poller thread: re-reads the caps file every ``every`` s and applies changes until ``stop``."""
    last_err = None
    while not stop.wait(every):
        try:
            spec = caps_lib.read_caps_file(path, CAPS_KEYS)
        except ValueError as e:
            if str(e) != last_err:
                print(f"[{datetime.now():%H:%M:%S}] CAPS file {path} ignored ({e}); keeping {slots.snapshot()}",
                      flush=True)
            last_err = str(e)
            continue
        last_err = None
        if spec is None or spec == last:
            continue
        slots.set_caps(spec)
        last = spec
        print(f"[{datetime.now():%H:%M:%S}] CAPS {slots.snapshot()} (from {path})", flush=True)


def _cells_present(a) -> list[Path]:
    root = Path(a.root).resolve()
    return [cell_dir(root, c) for c in select_cells(a) if (cell_dir(root, c) / "jobs").exists()]


def cmd_judge(a) -> int:
    cells = [str(c) for c in _cells_present(a)]
    return subprocess.call([sys.executable, "-u", "-m", "fsa.redo.swe_interact.metrics", "judge", *cells,
                            "--rf-exec", a.rf_exec], cwd=ROOT) if cells else 0


SEC8_ARMS = ("tracker", "mediator", "recap", "minimal_edit", "claims_no_p")


def cmd_report(a) -> int:
    root = Path(a.root).resolve()
    pairs = [
        "commit_before:stock",
        "claims_after:stock",
        "commit_before:claims_after",
        *(f"{arm}:stock" for arm in SEC8_ARMS),
        *(f"commit_before:{arm}" for arm in SEC8_ARMS),
        *(f"claims_after:{arm}" for arm in SEC8_ARMS),
    ]
    cmp_args = [x for p in pairs for x in ("--compare", p)]
    return subprocess.call([sys.executable, "-u", "-m", "fsa.redo.swe_interact.metrics", "report", str(root),
                            "--out", str(root / "metrics_report.json"), "--trials-out", str(root / "metrics_trials.json"),
                            *cmp_args], cwd=ROOT)


def main(argv=None) -> int:
    machine.apply_process_env()  # this machine's endpoint profile, for this process and its children
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "trials", "judge", "report"])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--pilot-name", default="stage2_swe_interact")
    ap.add_argument("--cap", default=None, help="per-rung trial caps, e.g. W=8,M=4,S=2 (default: rungs.json stage1_trial_cap)")
    ap.add_argument("--max-total", type=int, default=8, help="global concurrent-trial cap")
    ap.add_argument("--max-rf", type=int, default=2, help="concurrent rf_* trials (16 CPU / 16 GB each)")
    ap.add_argument("--caps-file", default=None,
                    help="trials: JSON caps file (any subset of W, M, S, max_total, max_rf) that overrides "
                         "--cap/--max-total/--max-rf and is re-read while trials run")
    ap.add_argument("--max-attempts", type=int, default=3, help="launches per incomplete trial per invocation")
    ap.add_argument("--mcp-timeout-ms", type=int, default=MCP_TIMEOUT_MS)
    ap.add_argument("--overlay-wait", type=float, default=240.0, help="max wait (s) for the in-trial judge")
    ap.add_argument("--rungs", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--condition", default=None, choices=["natural", "overlay", "mixed"])
    ap.add_argument("--tasks", default=None, help="restrict to these subset tasks (smoke)")
    ap.add_argument("--cells", default=None, help="explicit cells RUNG:ARM:natural|overlay|mixed:REP,...")
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rf-exec", default="all", choices=["all", "steps", "off"])
    ap.add_argument("--min-per-trial-w", type=float, default=45.0)
    ap.add_argument("--min-per-trial-m", type=float, default=150.0)
    ap.add_argument("--min-per-trial-s", type=float, default=150.0)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "trials": cmd_trials, "judge": cmd_judge, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
