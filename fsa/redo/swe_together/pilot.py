"""Stage 1 SWE-Together pilot launcher (PLAN.md §5.4 Stage 1, Gate 1).

Matrix (``matrix()``), 380 trials on the 20-task frozen subset:

    M gemini-3.6-flash, S gemini-3.8-flash  x  {stock, commit_before, claims_after}
        x {natural r1, natural r2, overlay r1}                        = 18 cells x 20 = 360
    W gemini-3.5-flash-lite x stock x natural r1                      =  1 cell  x 20 =  20

Layout: ``<root>/<rung>/<arm>/<natural_r1|natural_r2|overlay_r1>/`` - each cell is an
ordinary run dir (``manifest.json``, ``trials/``, ``logs/``, ``launches.jsonl``), so
``run judge``, ``intent_judge trials`` and ``metrics report`` work on it unchanged.

    python -u -m fsa.redo.swe_together.pilot plan                       # matrix, counts, wall-time estimate
    python -u -m fsa.redo.swe_together.pilot trials [--env-type docker] # resume-friendly; re-run to resume
    python -u -m fsa.redo.swe_together.pilot judge                      # benchmark tagger + intent coverage
                                   # (the codex correctness judge seat is dropped: --skip-correctness)
    python -u -m fsa.redo.swe_together.pilot judge --completed-only --correctness-only
                                   # while trials still run: correctness judge on completed trials only
    python -u -m fsa.redo.swe_together.pilot intents                    # frozen per-intent judge (v2) per cell
    python -u -m fsa.redo.swe_together.pilot report                     # fsa.redo.metrics over all cells

Concurrency: one ``run_eval`` process per trial, gated by a per-rung semaphore and a
global container cap. Defaults come from ``rungs.json`` (``stage1_trial_cap``: W 8,
M 4, S 2 - the Stage 0 recommendation) and ``--max-total`` (14). Raise them with
``--cap W=16,M=7,S=5`` (or edit ``stage1_trial_cap``) once the first batch shows no
429; the caps in force are recorded per launch. ``--caps-file PATH`` (JSON such as
``{"W": 8, "M": 4, "S": 4, "max_total": 14}``, any subset of those keys) overrides them
while the invocation runs: the file is read at start and every ``CAPS_POLL_SEC``; a
change takes effect for trials not yet started (running trials are never stopped;
lowering a cap only delays new starts) and is printed as a ``CAPS`` line.

Clone revisions (2026-09-28): natural cells run on the env-selected pair (unset = the pinned
``third_party/swe-together`` + ``redo_v1.patch``, Stage 1.1). Overlay cells run on the Stage 1.2 copy
(``--overlay-clone-dir`` / ``--overlay-patch-file``, default ``config.TRIGGER_V3``:
``results/redo_v1/trigger_v3_work/swe-together`` + ``redo_v1_trigger.patch``) with the frozen trigger
(``--overlay-trigger agentic_v3``, ``SWT_REDO_TRIGGER``; manifest ``redo.overlay_trigger`` records the
trigger-freeze digest). Every launch re-verifies the revision its own cell uses (``launches.jsonl``
``patch_file`` / ``patch_sha256``). Each trigger consultation holds one judge container, taken from the
host-wide v3 judge slots (``agentic_judge_cache/.slots``, 3; ``SWT_JUDGE3_SLOTS`` lowers it), on top of
``--max-total`` trial containers. ``--trigger-slots N`` (or ``"trigger_slots": N`` in the caps file, live;
0 = the frozen default) gives overlay trials launched from then on ``SWT_JUDGE3_SLOTS=N``: the number of
host-wide slots their trigger may use. It is not verdict-relevant (not in the frozen judge's verdict keys);
it only bounds how long a consultation waits for a container, which counts against the trial's wall-clock
budget. Offline judging (``stage1_finish --slots k``, k <= 3) takes slots 0..k-1, so with N >= k + the
overlay trials' concurrency a consultation never waits. Launch records carry the value in force.

Resume: a trial is skipped when its cell already holds a completed trial for the task
(``result.json`` with verifier rewards and a non-``infra_failed`` sidecar, the
benchmark's own ``--skip-existing`` rule); ``run_eval --skip-existing`` moves failed /
incomplete attempts to ``trials/_failed/`` before re-running. Nothing is deleted.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import caps as caps_lib
from .. import machine
from . import patchset
from .config import DEFAULT, RESULTS, ROOT, TRIGGER_V3, Revision, load_rungs, official_image, subset_tasks
from .run import (DEFAULT_TRIGGER, TRIGGERS, check_trigger_support, redo_env, redo_record, seat_env, task_image_refs,
                  trials_command, trials_manifest, write_manifest)

DEFAULT_ROOT = RESULTS / "stage1"
OVERLAY_SEED = 20260926          # = subset seed; same seed for every arm, so overlay templates pair by task
TAU = 0.5
PILOT_ARMS = ("stock", "commit_before", "claims_after")
ARM_STATUS = (Path(os.environ.get("SWE_BENCHMARK_ARM_DIR", ROOT.parent / "swe_benchmark_arm")) / "swe_together/status.json")  # read-only


def matrix() -> list[dict]:
    cells = []
    for rung in ("M", "S"):
        for arm in PILOT_ARMS:
            for rep in (1, 2):
                cells.append({"rung": rung, "arm": arm, "overlay": False, "replicate": rep})
            cells.append({"rung": rung, "arm": arm, "overlay": True, "replicate": 1})
    cells.append({"rung": "W", "arm": "stock", "overlay": False, "replicate": 1})
    return cells


def cell_dir(root: Path, c: dict) -> Path:
    return root / c["rung"] / c["arm"] / f"{'overlay' if c['overlay'] else 'natural'}_r{c['replicate']}"


def subset_digest() -> dict:
    cfg = load_rungs()
    p = ROOT / cfg["benchmark"]["subset"]
    return {"path": cfg["benchmark"]["subset"], "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "tasks": subset_tasks()}


def parse_caps(s: str | None) -> dict[str, int]:
    cfg = load_rungs()
    caps = {r: int(v.get("stage1_trial_cap") or 1) for r, v in cfg["rungs"].items()}
    for part in (s or "").split(","):
        if part.strip():
            k, v = part.split("=")
            caps[k.strip()] = int(v)
    return caps


CAPS_POLL_SEC = 60.0
CAPS_KEYS = ("W", "M", "S", "max_total", "trigger_slots")
RUNG_KEYS = ("W", "M", "S")


class Knob:
    """A launch setting that the caps file may change while trials run (``trigger_slots``);
    ``n`` None or 0 = not overridden."""

    def __init__(self, n: int | None = None):
        self.n = n or None

    async def set(self, n: int | None) -> None:
        self.n = n or None


def trigger_env(knobs: dict | None, overlay: bool) -> dict[str, str]:
    """The env an overlay launch gets from the live ``trigger_slots`` knob (natural cells: none)."""
    k = (knobs or {}).get("trigger_slots")
    return {"SWT_JUDGE3_SLOTS": str(k.n)} if overlay and k is not None and k.n else {}


class LiveCap:
    """Concurrency limit that can change while trials run (``--caps-file``). Waiters are served
    in arrival order, as with ``asyncio.Semaphore``; lowering the limit never stops a holder."""

    def __init__(self, n: int):
        self.n = int(n)
        self.active = 0
        self._cond = asyncio.Condition()

    async def __aenter__(self) -> "LiveCap":
        async with self._cond:
            await self._cond.wait_for(lambda: self.active < self.n)
            self.active += 1
        return self

    async def __aexit__(self, *exc) -> None:
        async with self._cond:
            self.active -= 1
            self._cond.notify_all()

    async def set(self, n: int) -> None:
        async with self._cond:
            self.n = int(n)
            self._cond.notify_all()


def read_caps_file(path: Path) -> dict[str, int] | None:
    """The caps in ``path`` (``fsa.redo.caps``; keys from ``CAPS_KEYS``); None when absent."""
    return caps_lib.read_caps_file(path, CAPS_KEYS)


def live_caps(sems: dict, total, knobs: dict | None = None) -> dict:
    """{"caps": per-rung limits now in force, "max_total": global limit now in force} plus each knob
    that is set (``trigger_slots``)."""
    return {"caps": {r: s.n for r, s in sems.items()}, "max_total": total.n,
            **{k: v.n for k, v in (knobs or {}).items() if v.n}}


async def apply_caps(spec: dict[str, int], sems: dict, total: LiveCap, knobs: dict | None = None) -> None:
    """Applies the keys present in ``spec``; an absent key leaves its cap or knob unchanged."""
    for r, s in sems.items():
        if r in spec:
            await s.set(spec[r])
    if "max_total" in spec:
        await total.set(spec["max_total"])
    for k, v in (knobs or {}).items():
        if k in spec:
            await v.set(spec[k])


async def watch_caps(path: Path, sems: dict, total: LiveCap, last: dict | None,
                     every: float = CAPS_POLL_SEC, knobs: dict | None = None) -> None:
    """Re-reads ``path`` every ``every`` seconds and applies changed caps (cancelled at the end)."""
    last_err = None
    while True:
        await asyncio.sleep(every)
        try:
            spec = read_caps_file(path)
        except ValueError as e:
            if str(e) != last_err:
                print(f"[{datetime.now():%H:%M:%S}] CAPS file {path} ignored ({e}); keeping "
                      f"{live_caps(sems, total, knobs)}", flush=True)
            last_err = str(e)
            continue
        last_err = None
        if spec is None or spec == last:
            continue
        await apply_caps(spec, sems, total, knobs)
        last = spec
        print(f"[{datetime.now():%H:%M:%S}] CAPS {live_caps(sems, total, knobs)} (from {path})", flush=True)


def trial_complete(trials_dir: Path, task: str) -> bool:
    """Mirror of run_eval._classify_trial_dirs (completed only with a verifier result
    and a non-infra_failed sidecar); unknown sidecar -> let run_eval decide."""
    if not trials_dir.exists():
        return False
    for d in trials_dir.iterdir():
        if not (d.is_dir() and d.name.startswith(task[:32] + "__")):
            continue
        try:
            res = json.loads((d / "result.json").read_text())
            infra = json.loads((d / "trial_infra.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if (res.get("verifier_result") or {}).get("rewards") and infra.get("status") != "infra_failed":
            return True
    return False


# ── images ──────────────────────────────────────────────────────────────────

def ensure_images(tasks: list[str], template: str | None, pull: bool = True) -> dict:
    """Local native image per task (pulled if missing; never removed). arm64 hosts (``template``
    set): the arm64 rebuild, with its gold-validation status from the image-build repo (read-only).
    amd64 hosts (``template`` None): the task's official image from task.toml. An image whose
    architecture is not the host's counts as missing (never run under emulation)."""
    host = machine.host_arch()
    status = json.loads(ARM_STATUS.read_text()) if template and ARM_STATUS.exists() else {}
    out = {}
    for t in tasks:
        img = template.format(task=t) if template else official_image(t)
        r = subprocess.run(["docker", "image", "inspect", img, "--format", "{{.Id}} {{.Architecture}}"],
                           capture_output=True, text=True)
        pulled = False
        if r.returncode != 0 and pull:
            pr = subprocess.run(["docker", "pull", "--platform", f"linux/{host}", img], capture_output=True, text=True)
            pulled = pr.returncode == 0
            r = subprocess.run(["docker", "image", "inspect", img, "--format", "{{.Id}} {{.Architecture}}"],
                               capture_output=True, text=True)
        st = status.get(t, {})
        idarch = r.stdout.strip().split()
        arch = idarch[1] if len(idarch) > 1 else None
        out[t] = {"image": img, "present": r.returncode == 0 and arch == host, "pulled_now": pulled,
                  "image_id": idarch[0] if idarch else None, "arch": arch, "official": not template,
                  "gold_validated": st.get("validated"), "gold_reward": st.get("reward"),
                  "gold_mode": st.get("mode") if template else "official image (reference rewards: swt_exec_cache)",
                  "pushed": st.get("pushed")}
    return out


# ── commands ────────────────────────────────────────────────────────────────

def cmd_plan(args) -> int:
    cells = matrix()
    tasks = subset_tasks()
    caps = parse_caps(args.cap)
    n = {r: sum(1 for c in cells if c["rung"] == r) * len(tasks) for r in ("W", "M", "S")}
    mins = {"W": args.min_per_trial_w, "M": args.min_per_trial, "S": args.min_per_trial}
    est = {r: n[r] * mins[r] / caps[r] / 60 for r in n}
    print(json.dumps({"cells": len(cells), "tasks": len(tasks), "trials": sum(n.values()), "trials_per_rung": n,
                      "caps": caps, "max_total": args.max_total,
                      "minutes_per_trial_assumed": mins,
                      "wall_hours_per_rung": {k: round(v, 1) for k, v in est.items()},
                      "wall_hours_total (rungs in parallel)": round(max(est.values()), 1)}, indent=1))
    for c in cells:
        print(cell_dir(Path(args.root), c).relative_to(ROOT))
    return 0



INFRA_CAP = 3  # PLAN.md §5.4: at most two re-runs after the first infrastructure flag


def infra_flagged_attempts(trials_dir: Path, task: str) -> int:
    """Attempts of ``task`` in ``trials/`` and ``trials/_failed/`` whose harness sidecar says infra_failed
    (``run_eval --skip-existing`` moves a flagged attempt into ``_failed/`` when the next attempt starts, so
    the latest flagged attempt sits in ``trials/``). Trial dir names carry at most 32 characters of the task
    name (as in ``trial_complete``)."""
    n = 0
    for base in (trials_dir, trials_dir / "_failed"):
        if not base.exists():
            continue
        for d in base.glob(f"{task[:32]}__*"):
            try:
                if json.loads((d / "trial_infra.json").read_text()).get("status") == "infra_failed":
                    n += 1
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
    return n


LIVE_POLL_SEC = 30.0


def trial_tag(c: dict, task: str) -> str:
    """``run_eval --tag`` of one trial of cell ``c`` (at most 120 characters)."""
    return f"s1_{c['rung']}_{c['arm']}_{'ov' if c['overlay'] else 'nat'}{c['replicate']}_{task}"[:120]


def _ere(s: str) -> str:
    """``s`` as a literal in a POSIX extended regular expression (pgrep)."""
    return "".join("\\" + ch if ch in ".[]()*+?{}|^$\\" else ch for ch in s)


def live_launch(tag: str) -> bool:
    """True while a ``run_eval`` process with this trial tag runs, e.g. one started by an earlier launcher
    that was drained or stopped. Launching the trial again now would collide with it in the cell
    (``--skip-existing`` moves an unfinished attempt aside)."""
    r = subprocess.run(["pgrep", "-f", "--", f"--tag {_ere(tag)}( |$)"], capture_output=True)
    return r.returncode == 0


async def wait_live_launch(tag: str, label: str, poll: float = LIVE_POLL_SEC) -> float:
    """Waits (the caller holds the slot that run occupies) until no earlier run of ``tag`` is live;
    returns the seconds waited."""
    waited = 0.0
    while live_launch(tag):
        if not waited:
            print(f"[{datetime.now():%H:%M:%S}] wait  {label}: an earlier launcher's run of this trial is "
                  "still going", flush=True)
        await asyncio.sleep(poll)
        waited += poll
    return waited


async def _run_trial(job: dict, sems: dict, total: asyncio.Semaphore, args, cfg: dict, env_base: dict,
                     stats: dict, knobs: dict | None = None) -> None:
    c, task, cd = job["cell"], job["task"], job["dir"]
    trials_dir = cd / "trials"
    if trial_complete(trials_dir, task):
        stats["skipped"] += 1
        return
    if infra_flagged_attempts(trials_dir, task) >= INFRA_CAP:
        print(f"CAPPED {task}: {INFRA_CAP} infrastructure-flagged attempts; the last is scored as the agent's result", flush=True)
        stats["skipped"] += 1
        return
    label = f"{c['rung']}/{c['arm']}/{'ov' if c['overlay'] else 'nat'}{c['replicate']} {task}"
    async with sems[c["rung"]], total:
        for attempt in range(1, args.max_attempts + 1):
            tag = trial_tag(c, task)
            waited = await wait_live_launch(tag, label)
            if trial_complete(trials_dir, task):  # e.g. completed by an earlier launcher's run of it
                stats["skipped"] += 1
                return
            if infra_flagged_attempts(trials_dir, task) >= INFRA_CAP:
                print(f"CAPPED {task}: {INFRA_CAP} infrastructure-flagged attempts", flush=True)
                break
            why = machine.drain_requested()
            if why:  # operator/watchdog drain (e.g. credentials about to expire): start nothing new
                print(f"DRAINED {task}: {why}", flush=True)
                stats["drained"] = stats.get("drained", 0) + 1
                return
            # the clone copy this cell uses must still be exactly the revision recorded when this invocation started
            rev = job.get("rev") or DEFAULT
            try:
                live = patchset.verify(rev)["patch_sha256"]
            except SystemExit as e:
                live = f"differs: {e}"
            if live != job["patch_sha256"]:
                print(f"REFUSING {task}: clone patch changed since start ({live[:80]})", flush=True)
                stats["incomplete"] += 1
                return
            env = dict(env_base)
            env.update(redo_env(c["arm"], tau=TAU, overlay=c["overlay"], overlay_seed=OVERLAY_SEED,
                                trigger=job.get("trigger")))
            env.update(rev.env())
            tenv = trigger_env(knobs, c["overlay"])  # live host-wide judge slots for this trial's trigger
            env.update(tenv)
            cmd = trials_command(cfg, c["rung"], args.env_type, trials_dir, [task], 1, tag, skip_existing=True, rev=rev)
            (cd / "logs").mkdir(parents=True, exist_ok=True)
            log = cd / "logs" / f"{task}.{int(time.time())}.log"
            rec = {"task": task, "attempt": attempt, "started_at": datetime.now(timezone.utc).isoformat(),
                   "patch_sha256": live, "patch_file": rev.patch_file.name,
                   **({"clone_dir": str(rev.clone), "trigger": job.get("trigger")} if not rev.pinned else {}),
                   "log": str(log.relative_to(cd)), **live_caps(sems, total),
                   **({"trigger_slots": int(tenv["SWT_JUDGE3_SLOTS"])} if tenv else {}),
                   **({"waited_for_earlier_launch_sec": waited} if waited else {}),
                   "env_type": args.env_type}
            print(f"[{datetime.now():%H:%M:%S}] start {label} (attempt {attempt})", flush=True)
            t0 = time.time()
            with open(log, "w") as f:
                proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(rev.clone), env=env, stdout=f,
                                                            stderr=subprocess.STDOUT)
                rc = await proc.wait()
            rec.update(rc=rc, finished_at=datetime.now(timezone.utc).isoformat(),
                       wall_sec=round(time.time() - t0, 1), complete=trial_complete(trials_dir, task))
            with open(cd / "launches.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{datetime.now():%H:%M:%S}] done  {label} rc={rc} complete={rec['complete']} "
                  f"{rec['wall_sec']:.0f}s", flush=True)
            if rec["complete"]:
                stats["done"] += 1
                return
        stats["incomplete"] += 1


def parse_cells(spec: str) -> list[dict]:
    """``RUNG:ARM:natural|overlay:REP,...`` (smoke tests / partial reruns)."""
    out = []
    for part in spec.split(","):
        rung, arm, cond, rep = part.strip().split(":")
        out.append({"rung": rung, "arm": arm, "overlay": cond == "overlay", "replicate": int(rep)})
    return out


def select_cells(args) -> list[dict]:
    cells = parse_cells(args.cells) if args.cells else matrix()
    if args.rungs:
        cells = [c for c in cells if c["rung"] in args.rungs.split(",")]
    if args.arms:
        cells = [c for c in cells if c["arm"] in args.arms.split(",")]
    if args.condition:
        cells = [c for c in cells if ("overlay" if c["overlay"] else "natural") == args.condition]
    return cells


async def amain_trials(args) -> int:
    cfg = load_rungs()
    root = Path(args.root).resolve()
    tasks = args.tasks.split(",") if args.tasks else subset_tasks()
    bad = [t for t in tasks if t not in set(subset_tasks())]
    if bad:
        raise SystemExit(f"tasks outside the frozen subset: {bad}")
    cells = select_cells(args)
    caps = parse_caps(args.cap)
    max_total = args.max_total
    trigger_slots = getattr(args, "trigger_slots", None) or None
    caps_file = Path(args.caps_file).resolve() if getattr(args, "caps_file", None) else None
    file_caps = None
    if caps_file is not None:
        try:
            file_caps = read_caps_file(caps_file)
        except ValueError as e:
            raise SystemExit(f"--caps-file {caps_file}: {e}")
        if file_caps:
            caps.update({r: v for r, v in file_caps.items() if r in RUNG_KEYS})
            max_total = file_caps.get("max_total", max_total)
            if "trigger_slots" in file_caps:
                trigger_slots = file_caps["trigger_slots"] or None
    env_base = seat_env(cfg, args.env_type)
    if args.env_type == "docker":
        images = ensure_images(tasks, env_base.get("SWT_DOCKER_IMAGE_TEMPLATE"), pull=not args.no_pull)
        host = machine.host_arch()
        missing = [t for t, r in images.items() if not r["present"]]
        unvalidated = [t for t, r in images.items() if r["present"] and not r["official"] and not r["gold_validated"]]
        if missing and not args.allow_missing_images:
            raise SystemExit(f"no local {host} image for {missing} (pull failed or --no-pull); fix the images "
                             "(e.g. `docker login`, then re-run) or pass --allow-missing-images to skip these tasks")
        if missing:
            print(f"WARNING: no {host} image for {missing}; these trials are skipped (listed in the manifest)", flush=True)
        if unvalidated:
            print(f"WARNING: arm64 image not gold-validated: {unvalidated}", flush=True)
        run_tasks = [t for t in tasks if images[t]["present"]]
    else:
        images = task_image_refs(tasks, "e2b", None)
        run_tasks = tasks
    sub = subset_digest()
    ov_rev, ov_trigger = overlay_revision(args)
    revs = {DEFAULT: None}
    if any(c["overlay"] for c in cells):
        revs[ov_rev] = None
        check_trigger_support(ov_trigger, ov_rev)
    for rv in revs:  # only the revisions this invocation actually uses are verified
        revs[rv] = patchset.verify(rv)["patch_sha256"]
    jobs = []
    for c in cells:
        cd = cell_dir(root, c)
        (cd / "trials").mkdir(parents=True, exist_ok=True)
        rev, trig = (ov_rev, ov_trigger) if c["overlay"] else (DEFAULT, None)
        redo = redo_record(c["arm"], tau=TAU, overlay=c["overlay"], overlay_seed=OVERLAY_SEED, trigger=trig, rev=rev)
        env = {**env_base, **redo_env(c["arm"], tau=TAU, overlay=c["overlay"], overlay_seed=OVERLAY_SEED, trigger=trig),
               **rev.env()}
        cmd = trials_command(cfg, c["rung"], args.env_type, cd / "trials", ["<task>"], 1, "<per-trial tag>", True, rev=rev)
        man = trials_manifest(cfg, args, c["rung"], args.env_type, env, run_tasks, 1, cd / "trials", cmd, redo,
                              cell={**c, "pilot": "stage1", "seats": cfg["seats"], "subset": sub,
                                    "caps": caps, "max_total": max_total,
                                    **({"caps_file": str(caps_file),
                                        "caps_note": "caps at invocation start; the caps file may change them "
                                                     "while trials run - launches.jsonl records the caps "
                                                     "in force at each launch"} if caps_file else {}),
                                    **({"trigger_slots": trigger_slots,
                                        "trigger_slots_note": "SWT_JUDGE3_SLOTS given to this cell's trials at "
                                                              "invocation start (host-wide judge slots the trigger "
                                                              "may use; not verdict-relevant); launches.jsonl "
                                                              "records the value per launch"}
                                       if c["overlay"] and trigger_slots else {}),
                                    "skipped_tasks_no_image": [t for t in tasks if t not in run_tasks]},
                              images=images, rev=rev)
        if not args.dry_run:
            write_manifest(cd, man)
        for t in run_tasks:
            jobs.append({"cell": c, "task": t, "dir": cd, "patch_sha256": revs[rev], "rev": rev, "trigger": trig})
    # rep-major, then task, then arm: both arms of a task run close in time (pairing)
    order = {("natural", 1): 0, ("natural", 2): 2, ("overlay", 1): 1}
    jobs.sort(key=lambda j: (order.get(("overlay" if j["cell"]["overlay"] else "natural", j["cell"]["replicate"]), 3),
                             tasks.index(j["task"]), j["cell"]["arm"]))
    todo = [j for j in jobs if not trial_complete(j["dir"] / "trials", j["task"])]
    print(f"{len(jobs)} trials in {len(cells)} cells; {len(jobs) - len(todo)} already complete; "
          f"caps {caps}, max_total {max_total}, env {args.env_type}"
          + (f", trigger_slots {trigger_slots}" if trigger_slots else "")
          + (f", caps file {caps_file}" if caps_file else ""), flush=True)
    if args.dry_run:
        for j in todo[:40]:
            print("  would run", cell_dir(root, j["cell"]).relative_to(root), j["task"])
        return 0
    sems = {r: LiveCap(caps[r]) for r in RUNG_KEYS}
    total = LiveCap(max_total)
    knobs = {"trigger_slots": Knob(trigger_slots)}
    stats = {"skipped": 0, "done": 0, "incomplete": 0, "drained": 0}
    watcher = (asyncio.create_task(watch_caps(caps_file, sems, total, file_caps, knobs=knobs))
               if caps_file else None)
    try:
        await asyncio.gather(*(_run_trial(j, sems, total, args, cfg, env_base, stats, knobs) for j in todo))
    finally:
        if watcher is not None:
            watcher.cancel()
    print(json.dumps(stats), flush=True)
    return 0 if stats["incomplete"] == 0 and stats["drained"] == 0 else 1


def overlay_revision(args) -> tuple[Revision, str]:
    """(clone revision, trigger) for overlay cells: the Stage 1.2 copy and the frozen trigger by default."""
    clone = getattr(args, "overlay_clone_dir", None)
    patch = getattr(args, "overlay_patch_file", None)
    rev = Revision(Path(clone or TRIGGER_V3.clone).resolve(), Path(patch or TRIGGER_V3.patch_file).resolve())
    return rev, getattr(args, "overlay_trigger", None) or DEFAULT_TRIGGER


def _cells_present(args) -> list[Path]:
    root = Path(args.root).resolve()
    return [cell_dir(root, c) for c in select_cells(args) if (cell_dir(root, c) / "trials").exists()]


def park_failed_verdicts(cell: Path) -> int:
    """Move judge verdicts that carry an ``error`` (e.g. ``verdict_read_failed``) aside
    (``judge_verdict.failed-<ts>.json``, kept) so the idempotent benchmark judge
    re-judges them; the aggregator would otherwise score them 0.0."""
    n = 0
    for v in (cell / "trials").glob("*__*/judge_verdict.json"):
        try:
            bad = "error" in json.loads(v.read_text())
        except json.JSONDecodeError:
            bad = True
        if bad:
            v.rename(v.with_name(f"judge_verdict.failed-{int(time.time())}.json"))
            n += 1
    return n


def stage_completed(cell: Path) -> Path | None:
    """``<cell>/judge_staging/trials/``: a fresh dir of symlinks to the cell's COMPLETED trials
    only (``trial_complete``'s rule: verifier rewards and a non-infra_failed sidecar). The
    benchmark judge discovers every ``<task>__<id>`` dir under its trials root, so pointing it
    here keeps it away from trials that are still running (no verdict, coverage or tag file
    is written into a live trial dir). None when nothing is complete."""
    stage = cell / "judge_staging" / "trials"
    if stage.exists():
        for link in stage.iterdir():
            if link.is_symlink():
                link.unlink()
    stage.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted((cell / "trials").iterdir()):
        if not (d.is_dir() and "__" in d.name):
            continue
        try:
            res = json.loads((d / "result.json").read_text())
            infra = json.loads((d / "trial_infra.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if (res.get("verifier_result") or {}).get("rewards") and infra.get("status") != "infra_failed":
            (stage / d.name).symlink_to(d.resolve())
            n += 1
    return stage if n else None


def cmd_judge(args) -> int:
    rc = 0
    for cd in _cells_present(args):
        extra = []
        if args.completed_only:
            stage = stage_completed(cd)
            if stage is None:
                print(f"{cd}: no completed trial yet", flush=True)
                continue
            extra += ["--trials-root", str(stage)]
        if args.correctness_only:
            extra += ["--skip-intent-coverage", "--skip-tag-messages"]
        rc_i = 0
        for attempt in range(2):
            parked = park_failed_verdicts(cd)  # also clears failures left by an earlier invocation
            if attempt and not parked:
                break
            if parked:
                print(f"re-judging {parked} failed verdict(s) in {cd}", flush=True)
            rc_i = subprocess.call([sys.executable, "-u", "-m", "fsa.redo.swe_together.run", "judge", "--run-dir",
                                    str(cd), "--denom-tasks", "20", *extra], cwd=ROOT)
        rc |= rc_i
    return rc


def cmd_intents(args) -> int:
    """The frozen judge (``configs/intents/swe-together/JUDGE_FREEZE.json``: the agentic v3 judge when the freeze
    says ``judge: agentic_v3``, else the v2 per-intent judge) on every cell; ``--completed-only`` skips trials
    that are still running."""
    cells = [str(c) for c in _cells_present(args)]
    extra = ["--completed-only"] if args.completed_only else []
    fz = ROOT / "configs" / "intents" / "swe-together" / "JUDGE_FREEZE.json"
    if not fz.exists():
        raise SystemExit(f"{fz} does not exist: no validated SWE-Together judge is frozen")
    agentic = json.loads(fz.read_text()).get("config", {}).get("judge") == "agentic_v3"
    mod = "fsa.redo.swe_together.agentic_judge" if agentic else "fsa.redo.swe_together.intent_judge_v2"
    return subprocess.call([sys.executable, "-u", "-m", mod, "trials", *cells, *extra], cwd=ROOT) if cells else 0


SEC8_ARMS = ("tracker", "mediator", "recap", "minimal_edit", "claims_no_p")


def cmd_report(args) -> int:
    root = Path(args.root).resolve()
    pairs = [
        "commit_before:stock",
        "claims_after:stock",
        "commit_before:claims_after",
        *(f"{arm}:stock" for arm in SEC8_ARMS),
        *(f"commit_before:{arm}" for arm in SEC8_ARMS),
        *(f"claims_after:{arm}" for arm in SEC8_ARMS),
    ]
    cmp_args = [x for p in pairs for x in ("--compare", p)]
    return subprocess.call([sys.executable, "-u", "-m", "fsa.redo.metrics", "report", str(root),
                            "--out", str(root / "metrics_report.json"),
                            "--trials-out", str(root / "metrics_trials.json"),
                            *cmp_args], cwd=ROOT)


def main(argv=None) -> int:
    machine.apply_process_env()  # this machine's endpoint profile, for this process and its children
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "trials", "judge", "intents", "report"])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--env-type", default="docker", choices=["docker", "e2b"])
    ap.add_argument("--cap", default=None, help="per-rung trial caps, e.g. W=16,M=7,S=5 (default: rungs.json)")
    ap.add_argument("--max-total", type=int, default=14, help="global concurrent-trial (container) cap")
    ap.add_argument("--caps-file", default=None,
                    help="trials: JSON caps file ({\"W\": 8, \"M\": 4, \"S\": 4, \"max_total\": 14, \"trigger_slots\": 11}, "
                         "any subset) that overrides --cap/--max-total/--trigger-slots and is re-read while trials run")
    ap.add_argument("--trigger-slots", type=int, default=None,
                    help="trials: SWT_JUDGE3_SLOTS for overlay trials (host-wide judge slots their trigger may use; "
                         "default: the frozen 3)")
    ap.add_argument("--max-attempts", type=int, default=INFRA_CAP, help="launches per trial per invocation")
    ap.add_argument("--rungs", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--condition", default=None, choices=["natural", "overlay"])
    ap.add_argument("--tasks", default=None, help="restrict to these subset tasks (smoke)")
    ap.add_argument("--cells", default=None, help="explicit cells RUNG:ARM:natural|overlay:REP,... instead of the matrix")
    ap.add_argument("--overlay-clone-dir", default=None,
                    help="clone copy for overlay cells (default: config.TRIGGER_V3, the Stage 1.2 copy)")
    ap.add_argument("--overlay-patch-file", default=None,
                    help="recorded patch revision that copy must equal (default: redo_v1_trigger.patch)")
    ap.add_argument("--overlay-trigger", default=DEFAULT_TRIGGER, choices=list(TRIGGERS),
                    help="overlay trigger (default: the frozen agentic_v3)")
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--allow-missing-images", action="store_true",
                    help="skip tasks whose native image is unavailable instead of refusing to start")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--completed-only", action="store_true",
                    help="judge: only trials already complete (symlink staging dir); never touches running trials")
    ap.add_argument("--correctness-only", action="store_true",
                    help="judge: benchmark correctness judge only (skip intent coverage and message tagging, "
                         "which share the simulator's gemini-3.1-pro quota); run the full judge after the pilot")
    ap.add_argument("--min-per-trial", type=float, default=15.0, help="plan: minutes per M/S trial")
    ap.add_argument("--min-per-trial-w", type=float, default=10.0, help="plan: minutes per W trial")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd == "trials":
        return asyncio.run(amain_trials(a))
    return {"judge": cmd_judge, "intents": cmd_intents, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
