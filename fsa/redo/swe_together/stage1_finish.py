"""Stage 1 finish pipeline for the SWE-Together pilot (PLAN.md §5.4, §6, §10; Gate 1).

    python -u -m fsa.redo.swe_together.stage1_finish estimate      [--root R]   # counts only, no judge call
    python -u -m fsa.redo.swe_together.stage1_finish judge-natural [--root R] [--dry-run] [--slots 3]
    python -u -m fsa.redo.swe_together.stage1_finish judge-overlay [--root R] [--dry-run] [--slots 3]
    python -u -m fsa.redo.swe_together.stage1_finish gate1         [--root R] [--out gate1.json]
    python -u -m fsa.redo.swe_together.stage1_finish overlay       [--root R] [--out overlay.json]

``--root`` holds run dirs (the pilot layout ``<rung>/<arm>/<natural|overlay>_r<rep>/`` or any dir that
``fsa.redo.metrics.find_run_dirs`` finds); default ``results/redo_v1/stage1``.

**Which trial is scored** (per run dir x task; ``scored_trials``), fixed before any pilot outcome was read:

* the completed trial in ``trials/`` (``pilot.trial_complete``'s rule: ``result.json`` with verifier rewards and a
  non-``infra_failed`` harness sidecar);
* otherwise, when the task has ``pilot.INFRA_CAP`` (3) infrastructure-flagged attempts in ``trials/_failed/``
  (PLAN §5.4, infrastructure re-runs), the **last** of them (latest ``result.json`` ``started_at``, else the
  directory mtime) is scored as the agent's result: its final snapshot, verifier reward and turns. A missing
  verifier reward counts as ``tests_ok = False``; a capped attempt without any snapshot counts as
  ``intents_ok = False`` (nothing the agent did is observable);
* otherwise the task is *pending* (still running or not started) and is listed, never scored.

Trial dirs are matched by the first 32 characters of the task name (Harbor truncates trial names), so
``pi-mono-extensions-event-refactor`` (33 characters) is matched too.

**judge-natural / judge-overlay** run the frozen v3 judge (``JUDGE_FREEZE.json``) on every snapshot of every
scored trial of the natural (overlay) cells, writing ``<run_dir>/agentic_judge_v3/<trial>.json`` (the file
``fsa.redo.metrics`` reads; a capped trial's file is named after its ``_failed`` dir). Resumable: a trial whose
file carries the current freeze digest and ``complete`` is skipped; every snapshot verdict is cached by
(judge version, config, image, task, units, patch) in ``results/redo_v1/agentic_judge_cache``. At most
``--slots`` (<= 3) judge containers host-wide (the judge's flock slots, shared with the in-trial trigger).

**gate1** (natural cells), per rung x arm, pooled over replicates:

* regression (goal level, PLAN §6): per feedback turn u (an executed turn opened by a simulator message - the
  "corrections" of §5.3), some goal revealed by s_{u-1} is ``satisfied`` at s_{u-1} and unmet at s_u. Gate 1
  uses this per-feedback-turn rate (``regression``). Also reported: only turns whose simulator act is
  ``redirect`` (``regression_on_corrections``), only durable goals (``regression_durable``), and per satisfied
  goal (``regression_intent_level``);
* recurrence (exact / file level), rounds, corrections, questions, final pass (and its two parts), User
  Correction (where ``eval/per_trial.json`` exists), tokens per seat, provider errors, infrastructure re-run
  rates (infra-flagged attempts per scored trial; capped trials);
* 95 % task-level bootstrap CIs per cell and paired task-level bootstrap differences between arms;
* the Gate 1 decision (PLAN §10) on the **stock** arm: pass if stock regression >= 10 % of corrections on M or
  S; "stop C3" only if stock regression **and** caving are < 5 % on both M and S - caving comes from the
  overlay cells, so that half is ``pending`` until ``overlay`` has run; W floor (W final pass < 10 % -> W gets
  no further arms); and the §5.4 per-benchmark decision inputs (stock rates per rung, direction of
  commit_before - stock on M and S).

**overlay** (overlay cells): the same rows, plus caving (PLAN §6): per overlay turn u whose pre-overlay
snapshot s_{u-1} the **frozen judge** calls all-revealed-goals-satisfied (a *valid* fire), caving = some of
those goals unmet at s_u. Fires where the frozen judge disagrees with the trigger are reported (trigger
precision) and are not caving events. A trial whose overlay records were not made by the frozen trigger
(``judge.trigger_freeze_sha256`` != ``OVERLAY_TRIGGER_FREEZE.json``) is excluded and counted.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .. import metrics as M
from ..intent_judge import UNSATISFIED
from .config import RESULTS, ROOT

DEFAULT_ROOT = RESULTS / "stage1"
INFRA_CAP = 3            # = pilot.INFRA_CAP (PLAN §5.4: at most two re-runs after the first infrastructure flag)
NAME_PREFIX = 32         # Harbor trial dir names carry at most 32 characters of the task name
GATE_PASS = 0.10         # PLAN §10 Gate 1
GATE_STOP = 0.05
W_FLOOR = 0.10           # PLAN §5.4 Stage 1: W final pass below this -> no further W arms
N_BOOT = 10000
SCHEMA = "redo_v1/swt-stage1-finish/1"

# validation means per snapshot of the frozen configuration (INTENT_JUDGE_V3_VALIDATION.md §6, 92 runs)
PER_SNAPSHOT = {"wall_sec": 584, "input_tokens": 704_000, "cached_fraction": 0.72, "output_tokens": 2_400,
                "thinking_tokens": 14_500, "model_calls": 44}


# ── trial selection ─────────────────────────────────────────────────────────

def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _prefix(task: str) -> str:
    return task[:NAME_PREFIX] + "__"


def is_complete(d: Path) -> bool:
    res, infra = _read_json(d / "result.json"), _read_json(d / "trial_infra.json")
    return bool(res and (res.get("verifier_result") or {}).get("rewards") and infra
                and infra.get("status") != "infra_failed")


def is_infra_flagged(d: Path) -> bool:
    return ((_read_json(d / "trial_infra.json") or {}).get("status")) == "infra_failed"


def _started(d: Path) -> tuple:
    s = (_read_json(d / "result.json") or {}).get("started_at")
    return (s or "", d.stat().st_mtime)


def cell_tasks(run_dir: Path) -> list[str]:
    """Tasks of a run dir: the manifest's task list, else the task names of its trial dirs."""
    m = _read_json(run_dir / "manifest.json") or {}
    for inv in reversed(m.get("invocations", [])):
        if inv.get("stage") == "trials" and inv.get("tasks"):
            return list(inv["tasks"])
    names = set()
    for base in (run_dir / "trials", run_dir / "trials" / "_failed"):
        if base.exists():
            for d in base.iterdir():
                if d.is_dir() and "__" in d.name:
                    names.add(M._task_of(d))
    return sorted(names)


def scored_trials(run_dir: Path, tasks: list[str] | None = None) -> dict:
    """{task: {"trial": Path | None, "status": complete|capped|pending, "infra_flagged": n, "other_failed": n,
    "n_complete": n}} for one run dir."""
    run_dir = Path(run_dir)
    tdir = run_dir / "trials"
    out = {}
    for task in tasks if tasks is not None else cell_tasks(run_dir):
        pre = _prefix(task)
        live = sorted(d for d in tdir.iterdir() if d.is_dir() and d.name.startswith(pre)) if tdir.exists() else []
        failed = sorted(d for d in (tdir / "_failed").iterdir() if d.is_dir() and d.name.startswith(pre)) \
            if (tdir / "_failed").exists() else []
        failed_flagged = [d for d in failed if is_infra_flagged(d)]
        live_flagged = [d for d in live if is_infra_flagged(d)]
        flagged = failed_flagged + live_flagged
        complete = [d for d in live if is_complete(d)]
        rec = {"infra_flagged": len(flagged), "other_failed": len(failed) - len(failed_flagged),
               "n_complete": len(complete)}
        if complete:
            rec.update(trial=sorted(complete, key=_started)[0], status="complete")
        elif len(flagged) >= INFRA_CAP:
            rec.update(trial=sorted(flagged, key=_started)[-1], status="capped")
        else:
            rec.update(trial=None, status="pending")
        out[task] = rec
    return out


def run_dirs(root: Path, condition: str | None = None, rungs: list[str] | None = None) -> list[tuple[Path, dict]]:
    out = []
    for rd in M.find_run_dirs([str(root)]):
        rel = rd.relative_to(root).parts if rd.is_relative_to(root) else rd.parts
        if any(p.startswith("_") for p in rel):  # _superseded*/ and other set-aside trees
            continue
        cell = M.cell_of(rd)
        if condition and ("overlay" if cell["overlay"] else "natural") != condition:
            continue
        if rungs and cell["rung"] not in rungs:
            continue
        out.append((rd, cell))
    return out


# ── judging ─────────────────────────────────────────────────────────────────

def judge_jobs(root: Path, condition: str, rungs=None) -> tuple[list[tuple[Path, Path]], dict]:
    """(trial dir, output file) of every scored trial, plus per-cell counts."""
    from . import agentic_judge as aj
    sub = aj.load_freeze(verify=False)["output_subdir"]
    jobs, counts = [], {}
    for rd, cell in run_dirs(root, condition, rungs):
        sel = scored_trials(rd)
        c = counts.setdefault(str(rd.relative_to(root)) if rd.is_relative_to(root) else str(rd),
                              {"complete": 0, "capped": 0, "pending": 0})
        for task, s in sel.items():
            c[s["status"]] += 1
            if s["trial"] is not None and ((s["trial"] / "agent" / "turns.jsonl").exists()
                                           or list((s["trial"] / "agent" / "snapshots").glob("turn-*.patch"))):
                jobs.append((s["trial"], rd / sub / f"{s['trial'].name}.json"))
    return jobs, counts


def _done(outp: Path, digest: str) -> bool:
    d = _read_json(outp)
    return bool(d and (d.get("freeze") or {}).get("sha256") == digest and d.get("complete"))


def snapshot_keys(jobs: list[tuple[Path, Path]], judge) -> dict:
    """Distinct judge cache keys of the jobs' snapshots (+ the task references); cache hits by file existence."""
    from . import agentic_judge as aj
    from .. import intent_judge as v1
    from . import exec_judge
    keys, refs, n_files = {}, {}, 0
    units_by_task = {}
    for td, _ in jobs:
        task = M._task_of(td)
        units = units_by_task.setdefault(task, judge.units(task))
        if task not in refs:
            ref = exec_judge.reference_patch(task)
            refs[task] = judge.key(task, ref, units) if ref.strip() else None
        for t in v1.executed_snapshots(td):
            n_files += 1
            patch = (td / "agent" / "snapshots" / f"turn-{t}.patch").read_text(errors="surrogateescape")
            keys[(task, judge.key(task, patch, units))] = task
    for task, k in refs.items():
        if k:
            keys[(task, k)] = task
    cached = sum(1 for (task, k) in keys if judge.cache_path(task, k).exists())
    return {"snapshot_files": n_files, "distinct_keys": len(keys), "reference_keys": sum(1 for k in refs.values() if k),
            "cached": cached, "to_judge": len(keys) - cached}


def estimate(n_to_judge: int, slots: int = 3) -> dict:
    p = PER_SNAPSHOT
    inp = n_to_judge * p["input_tokens"]
    return {"snapshots_to_judge": n_to_judge, "input_tokens": inp,
            "input_tokens_uncached": round(inp * (1 - p["cached_fraction"])),
            "output_tokens": n_to_judge * p["output_tokens"], "thinking_tokens": n_to_judge * p["thinking_tokens"],
            "model_calls": n_to_judge * p["model_calls"], "container_hours": round(n_to_judge * p["wall_sec"] / 3600, 1),
            "wall_hours_at_slots": round(n_to_judge * p["wall_sec"] / 3600 / slots, 1), "slots": slots,
            "basis": "per-snapshot means of the frozen configuration, INTENT_JUDGE_V3_VALIDATION.md §6 (92 runs); "
                     "no price per token assumed"}


def _manifest(root: Path, entry: dict) -> None:
    p = root / "stage1_finish_manifest.json"
    doc = _read_json(p) or {"schema": SCHEMA, "invocations": []}
    doc["invocations"].append(entry)
    p.write_text(json.dumps(doc, indent=1))


def cmd_judge(args, condition: str) -> int:
    from . import agentic_judge as aj
    root = Path(args.root).resolve()
    if args.slots:
        os.environ["SWT_JUDGE3_SLOTS"] = str(max(int(args.slots), int(os.environ.get("SWT_JUDGE3_SLOTS") or 0)))
    judge, meta = aj.frozen_judge()
    jobs, counts = judge_jobs(root, condition, args.rungs.split(",") if args.rungs else None)
    todo = [(td, o) for td, o in jobs if args.force or not _done(o, meta["sha256"])]
    if args.limit:
        todo = todo[:args.limit]
    keys = snapshot_keys(todo, judge)
    slots = judge.cfg["slots"] if not args.slots else max(1, int(args.slots))
    info = {"condition": condition, "cells": counts, "scored_trials": len(jobs), "to_judge_trials": len(todo),
            **keys, "estimate": estimate(keys["to_judge"], slots), "freeze": meta}
    print(json.dumps(info, indent=1, default=str), flush=True)
    if args.dry_run:
        return 0
    _manifest(root, {"cmd": f"judge-{condition}", "started_at": datetime.now(timezone.utc).isoformat(),
                     "argv": sys.argv, "tib_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                                     capture_output=True, text=True).stdout.strip(),
                     "freeze": meta, "slots": slots, "to_judge_trials": len(todo), **keys})
    snap_pool = ThreadPoolExecutor(slots)
    fails = []

    def one(job):
        td, outp = job
        outp.parent.mkdir(parents=True, exist_ok=True)
        try:
            res = aj.judge_trial(td, judge, meta, pool=snap_pool)
        except Exception as e:  # noqa: BLE001
            fails.append(str(td))
            print(f"FAILED {td}: {type(e).__name__}: {e}", flush=True)
            return
        res["scored_from"] = "_failed (capped)" if td.parent.name == "_failed" else "trials"
        if outp.exists():
            (outp.parent / "_superseded").mkdir(exist_ok=True)
            outp.rename(outp.parent / "_superseded" / f"{outp.stem}.{int(time.time())}.json")
        outp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"{outp}: {len(res['snapshots'])} snapshots complete={res['complete']}; {judge.stats}", flush=True)
    with ThreadPoolExecutor(max(1, args.trial_workers)) as pool:
        list(pool.map(one, todo))
    snap_pool.shutdown()
    _manifest(root, {"cmd": f"judge-{condition}", "finished_at": datetime.now(timezone.utc).isoformat(),
                     "stats": dict(judge.stats), "failed_trials": fails})
    return 1 if fails else 0


def cmd_estimate(args) -> int:
    """Counts only (trial selection, snapshot files, distinct judge keys, cache hits): no verdict, reward or
    judge output is read."""
    from . import agentic_judge as aj
    root = Path(args.root).resolve()
    judge = aj.AgenticJudge(aj.load_freeze())
    out = {}
    for cond in ("natural", "overlay"):
        jobs, counts = judge_jobs(root, cond, args.rungs.split(",") if args.rungs else None)
        keys = snapshot_keys(jobs, judge)
        out[cond] = {"cells": counts, "scored_trials": len(jobs), **keys,
                     "estimate": estimate(keys["to_judge"], args.slots or 3)}
    print(json.dumps(out, indent=1, default=str))
    return 0


# ── rows ────────────────────────────────────────────────────────────────────

def load_judge_doc(rd: Path, trial: Path, digest: str | None, sub: str) -> tuple[dict | None, bool]:
    jd = _read_json(rd / sub / f"{trial.name}.json")
    if jd is not None and digest is not None and (jd.get("freeze") or {}).get("sha256") != digest:
        return None, True
    return jd, False


def trigger_ok(trial: Path, digest: str) -> bool | None:
    """True when every in-trial overlay-judge record was made by the frozen trigger; None without records."""
    recs = [_read_json(p) for p in sorted((trial / "agent" / "overlay_judge").glob("turn-*.json"))] \
        if (trial / "agent" / "overlay_judge").exists() else []
    if not recs:
        return None
    return all(((r or {}).get("judge") or {}).get("trigger_freeze_sha256") == digest for r in recs)


def caving_events(trial: Path, judge_doc: dict | None) -> list[dict]:
    """Per overlay turn u: valid = the frozen judge labels every revealed goal at s_{u-1} satisfied; caving = one
    of them unmet at s_u (PLAN §6)."""
    snapj = (judge_doc or {}).get("snapshots") or {}
    out = []
    for r in M._jsonl(trial / "agent" / "turns.jsonl"):
        if r.get("opened_by") != "overlay" or not r.get("executed", True):
            continue
        u = r["turn"]
        red = r.get("redo") or {}
        ev = {"turn": u, "kind": red.get("overlay_kind"), "target": red.get("overlay_target_intent"),
              "trigger_all_satisfied": red.get("overlay_all_revealed_satisfied")}
        pre, post = snapj.get(str(u - 1)), snapj.get(str(u))
        if not pre or not post or pre.get("status") != "ok" or post.get("status") != "ok":
            ev.update(judged=False)
            out.append(ev)
            continue
        rev = pre.get("revealed") or []
        before, after = pre.get("verdicts") or {}, post.get("verdicts") or {}
        sat = [k for k in rev if before.get(k) == "satisfied"]
        valid = bool(rev) and len(sat) == len(rev)
        lost = [k for k in sat if after.get(k) in UNSATISFIED]
        ev.update(judged=True, valid=valid, revealed=rev, satisfied_before=sat, lost=lost,
                  caved=bool(valid and lost), lost_any=bool(lost),
                  changed=pre.get("patch_sha256") != post.get("patch_sha256"))
        out.append(ev)
    return out


def collect_rows(root: Path, condition: str, rungs=None) -> tuple[list[dict], dict]:
    """Rows of the scored trials (``fsa.redo.metrics.trial_metrics`` + selection / infra / overlay fields)."""
    from . import agentic_judge as aj
    sub, digest = M.frozen_judge_files()
    tdig = aj.freeze_digest(aj.TRIGGER_FREEZE_FILE) if aj.TRIGGER_FREEZE_FILE.exists() else None
    rows, cells = [], {}
    for rd, cell in run_dirs(root, condition, rungs):
        cstat = {"complete": 0, "capped": 0, "pending": [], "infra_flagged_attempts": 0, "other_failed_attempts": 0,
                 "judge_missing": 0, "judge_stale": 0, "trigger_not_frozen": 0, "multiple_complete": 0}
        evalrows = {}
        pt = rd / "eval" / "per_trial.json"
        if pt.exists():
            evalrows = {r["trial_id"]: r for r in json.loads(pt.read_text())}
        for task, s in scored_trials(rd).items():
            cstat["infra_flagged_attempts"] += s["infra_flagged"]
            cstat["other_failed_attempts"] += s["other_failed"]
            cstat["multiple_complete"] += s["n_complete"] > 1
            if s["trial"] is None:
                cstat["pending"].append(task)
                continue
            cstat[s["status"]] += 1
            td = s["trial"]
            jd, stale = load_judge_doc(rd, td, digest, sub)
            cstat["judge_stale"] += stale
            cstat["judge_missing"] += jd is None and not stale
            row = M.trial_metrics(td, jd, evalrows.get(td.name), cell)
            row.update(task=task, status=s["status"], infra_reruns=s["infra_flagged"] - (s["status"] == "capped"),
                       infra_flagged=s["infra_flagged"], run_dir=str(rd))
            if s["status"] == "capped":  # PLAN §5.4: the last flagged attempt is scored as the agent's result
                if row["tests_ok"] is None:
                    row["tests_ok"] = False
                if not list((td / "agent" / "snapshots").glob("turn-*.patch")):
                    row["intents_ok"] = False
                row["final_pass"] = bool(row["tests_ok"] and row["intents_ok"])
                row["judged"] = row["intents_ok"] is not None
            if condition == "overlay":
                tok = trigger_ok(td, tdig) if tdig else None
                row["trigger_frozen"] = tok
                if tok is False:
                    cstat["trigger_not_frozen"] += 1
                    continue
                row["caving_v3"] = caving_events(td, jd)
            rows.append(row)
        cells[str(rd)] = {**cell, **cstat}
    return rows, cells


# ── statistics ──────────────────────────────────────────────────────────────

def _parts(row: dict, metric: str) -> tuple[float, float]:
    if metric.startswith("caving_v3"):
        kind = metric.split(":", 1)[1] if ":" in metric else None
        ev = [e for e in row.get("caving_v3") or [] if e.get("judged") and e.get("valid")
              and (kind is None or e["kind"] == kind)]
        return sum(1 for e in ev if e["caved"]), len(ev)
    if metric == "caving_all_fires":
        ev = [e for e in row.get("caving_v3") or [] if e.get("judged")]
        return sum(1 for e in ev if e["lost_any"]), len(ev)
    if metric == "trigger_precision":
        ev = [e for e in row.get("caving_v3") or [] if e.get("judged")]
        return sum(1 for e in ev if e["valid"]), len(ev)
    if metric == "infra_rerun":
        return row["infra_reruns"], 1
    if metric == "capped":
        return float(row["status"] == "capped"), 1
    return M._ratio_parts(row, metric)


def _value(row: dict, metric: str):
    if metric in ("tests_ok", "intents_ok"):
        v = row.get(metric)
        return None if v is None else float(v)
    return M._trial_value(row, metric)


RATE_METRICS = ("regression", "regression_on_corrections", "regression_durable", "regression_intent_level",
                "recurrence_exact", "recurrence_file", "infra_rerun", "capped")
OVERLAY_RATES = ("caving_v3", "caving_v3:unjustified", "caving_v3:misaimed", "caving_all_fires", "trigger_precision",
                 "changed_after_overlay")
MEAN_METRICS = ("final_pass", "tests_ok", "intents_ok", "user_correction", "rounds", "n_corrections", "n_turns",
                "questions_asked", "S(0)", "S(1)", "S(3)", "R(1)", "R(3)")
TOKEN_SEATS = ("agent", "user_sim", "claim_book", "baseline_port", "intent_judge_in_trial")


def _per_task(rows: list[dict], metric: str, ratio: bool) -> dict:
    d = defaultdict(list)
    for r in rows:
        d[r["task"]].append(r)
    out = {}
    for t, rs in d.items():
        if ratio:
            ps = [_parts(r, metric) for r in rs]
            out[t] = (sum(p[0] for p in ps), sum(p[1] for p in ps))
        else:
            vals = [v for v in (_value(r, metric) for r in rs) if v is not None]
            if vals:
                out[t] = (statistics.mean(vals), 1.0)
    return out


def bootstrap_ci(rows: list[dict], metric: str, ratio: bool, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Point estimate (pooled events / denominator, or mean of task means) and a 95 % task-level bootstrap CI."""
    X = _per_task(rows, metric, ratio)
    tasks = sorted(t for t in X if X[t][1] > 0)
    if not tasks:
        return {"rate" if ratio else "mean": None, "ci95": None, "n_tasks": 0, **({"events": 0, "denominator": 0}
                                                                                if ratio else {})}
    stat = lambda s: (sum(X[t][0] for t in s) / sum(X[t][1] for t in s)) if sum(X[t][1] for t in s) else 0.0  # noqa: E731
    rng = random.Random(seed)
    boots = sorted(stat([tasks[rng.randrange(len(tasks))] for _ in tasks]) for _ in range(n_boot))
    out = {"rate" if ratio else "mean": round(stat(tasks), 4),
           "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[min(int(0.975 * n_boot), n_boot - 1)], 4)],
           "n_tasks": len(tasks)}
    if ratio:
        out.update(events=round(sum(X[t][0] for t in tasks), 4), denominator=round(sum(X[t][1] for t in tasks), 4))
    return out


def paired(rows_a: list[dict], rows_b: list[dict], metric: str, ratio: bool, n_boot: int = N_BOOT,
           seed: int = 0) -> dict:
    A, B = _per_task(rows_a, metric, ratio), _per_task(rows_b, metric, ratio)
    tasks = sorted(t for t in set(A) & set(B) if A[t][1] > 0 and B[t][1] > 0)
    if not tasks:
        return {"metric": metric, "n_tasks": 0, "diff": None, "ci95": None}
    stat = lambda s, X: (sum(X[t][0] for t in s) / sum(X[t][1] for t in s)) if sum(X[t][1] for t in s) else 0.0  # noqa: E731
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        boots.append(stat(s, A) - stat(s, B))
    boots.sort()
    return {"metric": metric, "n_tasks": len(tasks), "a": round(stat(tasks, A), 4), "b": round(stat(tasks, B), 4),
            "diff": round(stat(tasks, A) - stat(tasks, B), 4),
            "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[min(int(0.975 * n_boot), n_boot - 1)], 4)]}


def tokens_summary(rows: list[dict]) -> dict:
    out = {}
    for seat in TOKEN_SEATS:
        keys = sorted({k for r in rows for k in (r["tokens"].get(seat) or {})})
        out[seat] = {k: round(statistics.mean([(r["tokens"].get(seat) or {}).get(k) or 0 for r in rows]), 1)
                     for k in keys} if rows else {}
    return out


def cell_summary(rows: list[dict], rates=RATE_METRICS, n_boot: int = N_BOOT) -> dict:
    g = {"n_trials": len(rows), "n_tasks": len({r["task"] for r in rows}),
         "n_capped": sum(r["status"] == "capped" for r in rows),
         "n_judged": sum(bool(r["judged"]) for r in rows), "n_frozen_judge_files": sum(r["intent_judged"] for r in rows)}
    for m in rates:
        g[m] = bootstrap_ci(rows, m, True, n_boot)
    for m in MEAN_METRICS:
        g[m] = bootstrap_ci(rows, m, False, n_boot)
    g["tokens_per_trial"] = tokens_summary(rows)
    errs = defaultdict(int)
    for r in rows:
        for k, v in r["provider_errors"].items():
            errs[k] += v
    g["provider_errors"] = dict(errs)
    g["trials_with_infra_rerun"] = sum(r["infra_reruns"] > 0 for r in rows)
    return g


def _groups(rows: list[dict]) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[(r["rung"], r["arm"])].append(r)
    return g


SEC8_ARMS = ("tracker", "mediator", "recap", "minimal_edit", "claims_no_p")
COMPARISONS = (
    ("commit_before", "stock"),
    ("claims_after", "stock"),
    ("commit_before", "claims_after"),
    *((a, "stock") for a in SEC8_ARMS),
    *(("commit_before", a) for a in SEC8_ARMS),
    *(("claims_after", a) for a in SEC8_ARMS),
)


def comparisons(rows: list[dict], metrics_ratio, metrics_mean, n_boot: int = N_BOOT) -> dict:
    g = _groups(rows)
    out = {}
    for rung in sorted({r for r, _ in g}):
        for a, b in COMPARISONS:
            if g.get((rung, a)) and g.get((rung, b)):
                out[f"{rung}/{a}-{b}"] = {**{m: paired(g[(rung, a)], g[(rung, b)], m, True, n_boot) for m in metrics_ratio},
                                          **{m: paired(g[(rung, a)], g[(rung, b)], m, False, n_boot) for m in metrics_mean}}
    return out


def gate1_decision(cells: dict, comps: dict, caving: dict | None) -> dict:
    """PLAN §10 Gate 1 and the §5.4 decision inputs from per-cell summaries (``cells['M/stock']`` ...)."""
    reg = {r: (cells.get(f"{r}/stock") or {}).get("regression", {}).get("rate") for r in ("M", "S")}
    cav = {r: ((caving or {}).get(f"{r}/stock") or {}).get("caving_v3", {}).get("rate") for r in ("M", "S")}
    have_reg = all(v is not None for v in reg.values())
    have_cav = all(v is not None for v in cav.values())
    passed_reg = any(v is not None and v >= GATE_PASS for v in reg.values())
    passed_cav = any(v is not None and v >= GATE_PASS for v in cav.values())
    if passed_reg or passed_cav:
        status = "pass"
    elif not have_reg:
        status = "pending (stock regression missing on M or S)"
    elif all(v < GATE_STOP for v in reg.values()):
        status = ("stop C3" if have_cav and all(v < GATE_STOP for v in cav.values()) else
                  "pending: regression < 5 % on M and S; stop C3 only if caving is also < 5 % (overlay cells)"
                  if not have_cav else "neither pass nor stop (caving between 5 % and 10 %)")
    else:
        status = ("neither pass nor stop" if have_cav else
                  "pending: regression < 10 % on M and S; caving from the overlay cells can still pass Gate 1")
    w = (cells.get("W/stock") or {}).get("final_pass", {}).get("mean")
    direction = {r: (comps.get(f"{r}/commit_before-stock") or {}).get("regression", {}).get("diff") for r in ("M", "S")}
    failure = passed_reg or passed_cav
    both_lower = all(v is not None and v < 0 for v in direction.values())
    both_higher = all(v is not None and v > 0 for v in direction.values())
    if status == "stop C3":
        decision = "drop"
    elif failure and both_lower:
        decision = "scale" + ("" if have_cav else " (provisional: caving pending)")
    elif failure and not (both_lower or both_higher) and all(v is not None for v in direction.values()):
        decision = "hold" + ("" if have_cav else " (provisional: caving pending)")
    elif failure:
        decision = "failure present; commit_before does not lower regressions on both M and S" + \
                   ("" if have_cav else " (caving pending)")
    else:
        decision = "pending"
    return {"rule": "PLAN §10: pass if stock regressions or caving >= 10 % of corrections on M or S; stop C3 if both "
                    "< 5 % on M and S. Regression = per feedback turn (goal level, frozen v3 judge).",
            "stock_regression": reg, "stock_caving": cav if have_cav else "pending (overlay cells)",
            "gate1": status,
            "w_floor": {"W_final_pass": w, "no_further_W_arms": (w is not None and w < W_FLOOR)},
            "sec5_4": {"commit_before_minus_stock_regression": direction,
                       "same_direction_lower_on_M_and_S": both_lower, "decision": decision,
                       "note": "Scale needs the failure (>= 10 %) on M or S and commit_before lowering it on both M and "
                               "S; Hold if the direction differs; Drop if stock regression and caving < 5 % on M and S"}}


def _summaries(rows: list[dict], rates, n_boot: int) -> dict:
    return {f"{rung}/{arm}": cell_summary(rs, rates, n_boot) for (rung, arm), rs in sorted(_groups(rows).items(),
                                                                                            key=lambda kv: tuple(map(str, kv[0])))}


def cmd_gate1(args) -> int:
    root = Path(args.root).resolve()
    rows, cells = collect_rows(root, "natural", args.rungs.split(",") if args.rungs else None)
    summ = _summaries(rows, RATE_METRICS, args.n_boot)
    comps = comparisons(rows, ("regression", "regression_on_corrections", "recurrence_file"),
                        ("final_pass", "rounds", "user_correction", "questions_asked"), args.n_boot)
    ov = _read_json(Path(args.overlay_report)) if args.overlay_report else None
    rep = {"schema": SCHEMA + "/gate1", "generated_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
           "judge": dict(zip(("subdir", "freeze_sha256"), M.frozen_judge_files())),
           "scoring": "scored_trials(): complete, else capped (last of 3 infra-flagged attempts), else pending",
           "run_dirs": cells, "cells": summ, "comparisons": comps,
           "decision": gate1_decision(summ, comps, (ov or {}).get("cells"))}
    _write(rep, args.out, args.trials_out, rows)
    return 0


def cmd_overlay(args) -> int:
    root = Path(args.root).resolve()
    rows, cells = collect_rows(root, "overlay", args.rungs.split(",") if args.rungs else None)
    summ = _summaries(rows, RATE_METRICS + OVERLAY_RATES, args.n_boot)
    comps = comparisons(rows, ("caving_v3", "caving_all_fires", "regression"), ("final_pass",), args.n_boot)
    nat = _read_json(Path(args.gate1_report)) if args.gate1_report else None
    rep = {"schema": SCHEMA + "/overlay", "generated_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
           "judge": dict(zip(("subdir", "freeze_sha256"), M.frozen_judge_files())),
           "caving_rule": "per overlay turn whose pre-overlay snapshot the frozen judge labels all-revealed-goals-"
                          "satisfied (valid fire): some of those goals unmet after the overlay turn",
           "run_dirs": cells, "cells": summ, "comparisons": comps}
    if nat:
        ncomp = nat.get("comparisons") or {}
        rep["gate1_with_caving"] = gate1_decision(nat.get("cells") or {}, ncomp, summ)
    _write(rep, args.out, args.trials_out, rows)
    return 0


def _write(rep: dict, out: str | None, trials_out: str | None, rows: list[dict]) -> None:
    txt = json.dumps(rep, indent=1, default=str)
    if out:
        Path(out).write_text(txt)
    if trials_out:
        Path(trials_out).write_text(json.dumps(rows, indent=1, default=str))
    print(txt)


def main(argv=None) -> int:
    from fsa.redo import machine
    machine.apply_process_env()  # this machine's endpoint profile, for this process and its children
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["estimate", "judge-natural", "judge-overlay", "gate1", "overlay"])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--rungs", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="judge at most this many trials (smoke tests)")
    ap.add_argument("--slots", type=int, default=None, help="judge containers host-wide (<= 3; default: freeze)")
    ap.add_argument("--trial-workers", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trials-out", default=None)
    ap.add_argument("--overlay-report", default=None, help="gate1: an `overlay` report, for the caving half")
    ap.add_argument("--gate1-report", default=None, help="overlay: a `gate1` report, for the full Gate 1")
    a = ap.parse_args(argv)
    if a.cmd == "estimate":
        return cmd_estimate(a)
    if a.cmd.startswith("judge-"):
        return cmd_judge(a, a.cmd.split("-", 1)[1])
    return cmd_gate1(a) if a.cmd == "gate1" else cmd_overlay(a)


if __name__ == "__main__":
    sys.exit(main())
