"""Validate (and, only on a pass, freeze) the agentic snapshot judge v3 for SWE-Together.

Plan: docs/results/INTENT_JUDGE_V3_VALIDATION_PLAN.md (committed before any held-out check ran).

    python -u -m fsa.redo.swe_together.judge_v3_validate split          # (b) dev / held-out split (no judge)
    python -u -m fsa.redo.swe_together.judge_v3_validate controls       # (a) cross-task controls (apply checks only)
    python -u -m fsa.redo.swe_together.judge_v3_validate dev --model M --unit goals|intents [--limit N] [--parallel K]
    python -u -m fsa.redo.swe_together.judge_v3_validate retest         # dev test-retest flip rate (exploratory)
    python -u -m fsa.redo.swe_together.judge_v3_validate gold-sample    # held-out gold: blinded sheet + sealed key
    python -u -m fsa.redo.swe_together.judge_v3_validate labkit start|stop ...   # containers for the labellers
    python -u -m fsa.redo.swe_together.judge_v3_validate check-a|check-b|gold-score|trigger
    python -u -m fsa.redo.swe_together.judge_v3_validate report         # JUDGE_V3_VALIDATION.json
    python -u -m fsa.redo.swe_together.judge_v3_validate freeze         # only if (a), (b), (c) all pass

Blinding (pilot data, ``results/redo_v1/stage1/``): only the gold sampler reads pilot trials; it pools
every cell, strips rung / arm / condition / replicate / trial id, and writes the id -> source key to a
sealed file that nothing here opens (scoring needs only the snapshot patch, which is stored under the
opaque snapshot id). No per-arm quantity, and no regression / caving / pass rate, is computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import intent_judge as v1
from . import agentic_judge as aj
from . import exec_judge
from .config import RESULTS, ROOT, subset_tasks

WORK = RESULTS / "judge_v3_work"
V3 = ROOT / "configs" / "intents" / "swe-together" / "v3"
SPLIT_FILE = V3 / "split.json"
CONTROLS_FILE = V3 / "controls.json"
DECISION_FILE = V3 / "dev_decision.json"
GOLD_DIR = V3 / "gold"
GOLD_PAIRS = GOLD_DIR / "gold_pairs.json"          # blinded: opaque ids, task, units, normalized diff
GOLD_LABELS = GOLD_DIR / "gold_labels.json"        # the labeller's labels (written before gold-score)
GOLD_KEY = WORK / "gold_key.sealed.json"           # snapshot id -> pilot source (never opened here)
GOLD_PATCHES = WORK / "gold_patches"               # <snapshot id>.patch (raw text the judge applies)
VALIDATION_FILE = V3 / "JUDGE_V3_VALIDATION.json"
PLAN = ROOT / "docs" / "results" / "INTENT_JUDGE_V3_VALIDATION_PLAN.md"
SPLIT_SEED = 20260928
GOLD_SEED = 20260929
MAX_GOLD_DIFF = 60_000
GOLD_MIN_PAIRS = 64
GOLD_MAX_PER_TASK = 2
PRO_DEV_LIMIT = 6


def _sha(b: str | bytes) -> str:
    return hashlib.sha256(b.encode("utf-8", "surrogateescape") if isinstance(b, str) else b).hexdigest()


def kappa(pairs: list[tuple[bool, bool]]) -> dict:
    """(judge says satisfied, reference says satisfied) pairs -> agreement, Cohen's kappa, confusion."""
    n = len(pairs)
    tp = sum(1 for a, b in pairs if a and b)
    tn = sum(1 for a, b in pairs if not a and not b)
    fp = sum(1 for a, b in pairs if a and not b)
    fn = sum(1 for a, b in pairs if not a and b)
    po = (tp + tn) / n if n else None
    pe = (((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)) if n else None
    k = (po - pe) / (1 - pe) if n and pe is not None and pe < 1 else None
    return {"n": n, "agree": tp + tn, "agreement": round(po, 4) if po is not None else None,
            "judge_sat_ref_sat": tp, "judge_unsat_ref_unsat": tn, "judge_sat_ref_unsat": fp, "judge_unsat_ref_sat": fn,
            "cohen_kappa": round(k, 4) if k is not None else None}


def cluster_bootstrap(rows: list[dict], cluster: str, n_boot: int = 2000, seed: int = 7) -> dict:
    """95 % percentile interval of agreement and kappa, resampling clusters (snapshots)."""
    by = defaultdict(list)
    for r in rows:
        by[r[cluster]].append((r["judge"] == "satisfied", r["ref"] == "satisfied"))
    keys = sorted(by)
    if len(keys) < 2:
        return {}
    rng = random.Random(seed)
    ag, ks = [], []
    for _ in range(n_boot):
        s = [p for k in (rng.choice(keys) for _ in keys) for p in by[k]]
        r = kappa(s)
        ag.append(r["agreement"])
        if r["cohen_kappa"] is not None:
            ks.append(r["cohen_kappa"])
    ag.sort()
    ks.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None  # noqa: E731
    return {"clusters": len(keys), "agreement_ci95": [q(ag, .025), q(ag, .975)], "kappa_ci95": [q(ks, .025), q(ks, .975)]}


def judge_for(model: str | None = None, unit: str | None = None, **more) -> aj.AgenticJudge:
    cfg = {k: v for k, v in (("model", model), ("unit", unit)) if v}
    cfg.update(more)
    return aj.AgenticJudge(cfg)


def pmap(fn, items, n: int):
    with ThreadPoolExecutor(max(1, n)) as pool:
        return list(pool.map(fn, items))


# ── (b) data and split ──────────────────────────────────────────────────────

def goal_verdict_records() -> list[dict]:
    """Every non-pilot benchmark verdict with goal_results (Stage 0 calibration, Stage 0 and Stage 1-smoke
    trials; codex/gpt-5.5 agentic judge on the frozen canonical goals)."""
    from .judge_freeze import _goal_verdicts
    out = []
    for r in _goal_verdicts():
        gr = {g["id"]: bool(g.get("met")) for g in (r["verdict"].get("goal_results") or [])}
        if gr:
            out.append({"source": r["source"], "task": r["task"], "patch": r["patch"], "goal_met": gr,
                        "calibration": r["source"].startswith("stage0/judge_calibration/")})
    return out


def cmd_split(args) -> int:
    if SPLIT_FILE.exists() and not args.force:
        raise SystemExit(f"{SPLIT_FILE} exists; the split is drawn once")
    recs = goal_verdict_records()
    groups: dict[tuple, list] = defaultdict(list)
    for r in recs:
        groups[(r["task"], _sha(r["patch"]))].append(r)
    trial_groups = sorted(k for k, v in groups.items() if not any(x["calibration"] for x in v))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(trial_groups)
    dev = set(trial_groups[: len(trial_groups) // 2])
    rows = []
    for k, v in sorted(groups.items()):
        split = "dev" if k in dev else "heldout"
        rows.append({"task": k[0], "patch_sha256": k[1], "split": split, "sources": [x["source"] for x in v],
                     "calibration": any(x["calibration"] for x in v),
                     "goal_met": v[0]["goal_met"],  # identical patches: first verdict (all listed in sources)
                     "goal_met_all": [x["goal_met"] for x in v]})
    V3.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps({"seed": SPLIT_SEED, "rule": "distinct (task, patch) groups; calibration "
                                      "(reference/base) groups -> heldout; other groups shuffled with the seed, first "
                                      "half -> dev", "groups": rows}, indent=1) + "\n")
    print(Counter(r["split"] for r in rows), flush=True)
    return 0


def _split_groups(split: str) -> list[dict]:
    doc = json.loads(SPLIT_FILE.read_text())
    return [g for g in doc["groups"] if g["split"] == split]


def _patch_of(group: dict) -> str:
    for r in goal_verdict_records():
        if r["task"] == group["task"] and _sha(r["patch"]) == group["patch_sha256"]:
            return r["patch"]
    raise KeyError(group["patch_sha256"])


def _bench_rows(judge: aj.AgenticJudge, groups: list[dict], parallel: int) -> tuple[list[dict], list[dict]]:
    patches = {g["patch_sha256"]: _patch_of(g) for g in groups}
    recs = pmap(lambda g: judge.judge_snapshot(g["task"], patches[g["patch_sha256"]], label="v3val:b"), groups, parallel)
    rows, runs = [], []
    emap = json.loads(aj.MAP_FILE.read_text())["tasks"]
    for g, rec in zip(groups, recs):
        runs.append(_run_summary(g["task"], rec))
        if rec.get("status") != "ok":
            continue
        v = rec["verdicts"]
        snap = f"{g['task']}:{g['patch_sha256'][:12]}"
        if judge.cfg["unit"] == "goals":
            for gid, x in v.items():
                if gid in g["goal_met"]:
                    rows.append({"snapshot": snap, "task": g["task"], "unit": gid, "calibration": g["calibration"],
                                 "judge": x["verdict"], "ref": "satisfied" if g["goal_met"][gid] else "not_satisfied"})
        else:  # intents: compared through the reviewed map where goal_relation == same (as v2's C2)
            for key, x in v.items():
                e = (emap.get(g["task"]) or {}).get("intents", {}).get(key) or {}
                gs = [gg for gg in e.get("goals", []) if gg in g["goal_met"]]
                if not gs or e.get("goal_relation") != "same" or x["verdict"] == "not_applicable":
                    continue
                rows.append({"snapshot": snap, "task": g["task"], "unit": key, "calibration": g["calibration"],
                             "judge": x["verdict"], "ref": "satisfied" if all(g["goal_met"][gg] for gg in gs)
                             else "not_satisfied"})
    return rows, runs


def _run_summary(task: str, rec: dict) -> dict:
    u = rec.get("usage") or {}
    return {"task": task, "key": rec.get("key"), "status": rec.get("status"), "wall_sec": rec.get("wall_sec"),
            "steps": rec.get("steps"), "forced_submit": rec.get("forced_submit"), "calls": u.get("calls"),
            "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
            "reasoning_tokens": u.get("reasoning_tokens"), "cached_tokens": u.get("cached_tokens"),
            "failed_attempts": u.get("failed_attempts")}


def cost_summary(runs: list[dict]) -> dict:
    ok = [r for r in runs if r["status"] == "ok"]
    if not ok:
        return {"n": 0}
    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    out = {"n": len(runs), "ok": len(ok)}
    for k in ("wall_sec", "steps", "calls", "input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
        xs = [r[k] or 0 for r in ok]
        out[k] = {"mean": round(sum(xs) / len(xs), 1), "median": med(xs), "max": max(xs)}
    out["forced_submit"] = sum(1 for r in ok if r.get("forced_submit"))
    return out


def cmd_dev(args) -> int:
    groups = _split_groups("dev")
    if args.limit:
        rng = random.Random(SPLIT_SEED + 1)
        groups = sorted(groups, key=lambda g: g["patch_sha256"])
        rng.shuffle(groups)
        groups = groups[: args.limit]
    more = {k: json.loads(v) if v[:1] in "0123456789tf[{\"" else v for k, v in (x.split("=", 1) for x in args.set)}
    if args.effort:
        more["reasoning_effort"] = args.effort
    judge = judge_for(args.model, args.unit, **more)
    rows, runs = _bench_rows(judge, groups, args.parallel)
    out = {"model": judge.cfg["model"], "effort": judge.cfg["reasoning_effort"], "unit": judge.cfg["unit"],
           "config": judge.config_record(), "n_snapshots": len(groups), "limit": args.limit,
           "agreement": kappa([(r["judge"] == "satisfied", r["ref"] == "satisfied") for r in rows]),
           "cost": cost_summary(runs), "rows": rows, "runs": runs, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    WORK.mkdir(parents=True, exist_ok=True)
    tag = f"{judge.cfg['model']}_{judge.cfg['reasoning_effort']}_{judge.cfg['unit']}" + (f"_n{args.limit}" if args.limit else "") \
        + (f"_{args.tag}" if args.tag else "")
    p = WORK / f"dev_{tag}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("model", "unit", "n_snapshots", "agreement", "cost")}, indent=1), flush=True)
    return 0


def cmd_retest(args) -> int:
    """Test-retest (exploratory): re-judge dev snapshots with the cache bypassed; unit flip rate."""
    groups = sorted(_split_groups("dev"), key=lambda g: g["patch_sha256"])[: args.limit]
    judge = judge_for(args.model, args.unit)
    first = pmap(lambda g: judge.judge_snapshot(g["task"], _patch_of(g), label="v3val:retest-a"), groups, args.parallel)
    again = pmap(lambda g: judge._session(g["task"], _patch_of(g), judge.units(g["task"]), "retest", "v3val:retest-b"),
                 groups, args.parallel)
    flips, n, bin_flips = [], 0, 0
    for g, a, b in zip(groups, first, again):
        if a.get("status") != "ok" or b.get("status") != "ok":
            continue
        for k, x in a["verdicts"].items():
            n += 1
            y = b["verdicts"][k]["verdict"]
            if (x["verdict"] == "satisfied") != (y == "satisfied"):
                bin_flips += 1
                flips.append({"task": g["task"], "unit": k, "a": x["verdict"], "b": y})
    out = {"model": judge.cfg["model"], "unit": judge.cfg["unit"], "n_snapshots": len(groups), "n_units": n,
           "binary_flips": bin_flips, "flip_rate": round(bin_flips / n, 4) if n else None, "flips": flips,
           "cost_b": cost_summary([_run_summary(g["task"], b) for g, b in zip(groups, again)])}
    (WORK / f"retest_{judge.cfg['model']}_{judge.cfg['unit']}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "flips"}, indent=1), flush=True)
    return 0


# ── (a) reference / base / cross-task ───────────────────────────────────────

def _calibration(task: str) -> dict[str, dict[str, bool]]:
    """Recorded benchmark verdicts on the reference and the base (Stage 0 calibration), where they exist."""
    out = {}
    for arm in ("reference", "base"):
        p = RESULTS / "stage0" / "judge_calibration" / task / arm / "judge_verdict.json"
        if p.exists():
            out[arm] = {g["id"]: bool(g.get("met")) for g in json.loads(p.read_text()).get("goal_results") or []}
    return out


def cmd_controls(args) -> int:
    """For each subset task, the first other subset task (in subset order, starting after it) whose
    reference patch applies in this task's image. Apply checks only; no model."""
    tasks = subset_tasks()
    refs = {t: exec_judge.reference_patch(t) for t in tasks}
    out = {}

    def one(i_t):
        i, t = i_t
        order = [tasks[(i + k) % len(tasks)] for k in range(1, len(tasks))]
        order = [o for o in order if refs[o].strip()]
        with aj.Slot(3):
            for o in order:
                sb = aj.Sandbox(t)
                try:
                    sb.start()
                    ok, _ = sb.apply(refs[o])
                finally:
                    sb.close()
                if ok:
                    return t, {"control_task": o, "tried": order[: order.index(o) + 1]}
        return t, {"control_task": None, "tried": order}
    for t, r in pmap(one, list(enumerate(tasks)), 3):
        out[t] = r
        print(t, r["control_task"], flush=True)
    V3.mkdir(parents=True, exist_ok=True)
    CONTROLS_FILE.write_text(json.dumps({"rule": cmd_controls.__doc__.strip(), "controls": out}, indent=1) + "\n")
    return 0


def frozen_or_chosen_judge() -> aj.AgenticJudge:
    d = json.loads(DECISION_FILE.read_text())
    return aj.AgenticJudge(d["config"])


def cmd_check_a(args) -> int:
    judge = frozen_or_chosen_judge()
    tasks = subset_tasks()
    controls = json.loads(CONTROLS_FILE.read_text())["controls"]
    jobs = []
    for t in tasks:
        ref = exec_judge.reference_patch(t)
        if ref.strip():
            jobs.append((t, "reference", ref))
        jobs.append((t, "base", ""))
        c = controls.get(t, {}).get("control_task")
        if c:
            jobs.append((t, "cross", exec_judge.reference_patch(c)))
    recs = pmap(lambda j: judge.judge_snapshot(j[0], j[2], label=f"v3val:a:{j[1]}"), jobs, args.parallel)
    per = defaultdict(dict)
    runs = []
    for (t, arm, _), rec in zip(jobs, recs):
        runs.append(_run_summary(t, rec) | {"arm": arm})
        per[t][arm] = {"status": rec.get("status"),
                       "verdicts": {k: x["verdict"] for k, x in (rec.get("verdicts") or {}).items()}}
    goals = judge.cfg["unit"] == "goals"
    ref_n = ref_s = base_n = base_s = cross_n = cross_s = 0
    ref_all_n = ref_all_s = base_all_n = base_all_s = 0
    rows = []
    for t in tasks:
        cal = _calibration(t)
        units = [u.key for u in judge.units(t)]
        pres = set(aj.PRESERVE_GOALS.get(t, ())) if goals else set()
        cal_ref_unmet = {k for k, m in cal.get("reference", {}).items() if not m}
        cal_base_met = {k for k, m in cal.get("base", {}).items() if m}
        ref_units = [k for k in units if k not in cal_ref_unmet]
        neg_units = [k for k in units if k not in pres and k not in cal_base_met]
        for arm, want in (("reference", ref_units), ("base", neg_units), ("cross", neg_units)):
            r = per[t].get(arm)
            if not r or r["status"] != "ok":
                continue
            v = r["verdicts"]
            sat = [k for k in want if v.get(k) == "satisfied"]
            rows.append({"task": t, "arm": arm, "n": len(want), "satisfied": len(sat), "units_satisfied": sat,
                         "all_units_satisfied": sum(1 for k in units if v.get(k) == "satisfied"), "all_units": len(units)})
            if arm == "reference":
                ref_n += len(want); ref_s += len(sat)
                ref_all_n += len(units); ref_all_s += sum(1 for k in units if v.get(k) == "satisfied")
            elif arm == "base":
                base_n += len(want); base_s += len(sat)
                base_all_n += len(units); base_all_s += sum(1 for k in units if v.get(k) == "satisfied")
            else:
                cross_n += len(want); cross_s += len(sat)
    rate = lambda s, n: round(s / n, 4) if n else None  # noqa: E731
    out = {"reference": {"n": ref_n, "satisfied": ref_s, "rate": rate(ref_s, ref_n), "criterion": ">= 0.90"},
           "base": {"n": base_n, "satisfied": base_s, "rate": rate(base_s, base_n), "criterion": "<= 0.10"},
           "cross": {"n": cross_n, "satisfied": cross_s, "rate": rate(cross_s, cross_n), "criterion": "<= 0.10",
                     "tasks_with_control": sum(1 for t in tasks if per[t].get("cross"))},
           "secondary": {"reference_all_units": {"n": ref_all_n, "satisfied": ref_all_s, "rate": rate(ref_all_s, ref_all_n)},
                         "base_all_units": {"n": base_all_n, "satisfied": base_all_s, "rate": rate(base_all_s, base_all_n)}},
           "infra_failures": [r for r in runs if r["status"] != "ok"], "per_task": rows, "cost": cost_summary(runs),
           "runs": runs}
    out["pass"] = bool(out["reference"]["rate"] is not None and out["reference"]["rate"] >= 0.90 and
                       out["base"]["rate"] is not None and out["base"]["rate"] <= 0.10 and
                       (out["cross"]["rate"] is None or out["cross"]["rate"] <= 0.10) and not out["infra_failures"])
    (WORK / "check_a.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("reference", "base", "cross", "secondary", "pass")}, indent=1), flush=True)
    return 0


def cmd_check_b(args) -> int:
    judge = frozen_or_chosen_judge()
    groups = _split_groups("heldout")
    rows, runs = _bench_rows(judge, groups, args.parallel)
    k = kappa([(r["judge"] == "satisfied", r["ref"] == "satisfied") for r in rows])
    trials_only = kappa([(r["judge"] == "satisfied", r["ref"] == "satisfied") for r in rows if not r["calibration"]])
    out = {"n_snapshots": len(groups), "agreement": k, "trials_only": trials_only,
           "calibration_only": kappa([(r["judge"] == "satisfied", r["ref"] == "satisfied") for r in rows if r["calibration"]]),
           "bootstrap": cluster_bootstrap(rows, "snapshot"), "criterion": "kappa >= 0.60 (goal level)",
           "infra_failures": [r for r in runs if r["status"] != "ok"], "cost": cost_summary(runs), "rows": rows}
    out["pass"] = bool(k["cohen_kappa"] is not None and k["cohen_kappa"] >= 0.60)
    (WORK / "check_b.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({x: out[x] for x in ("n_snapshots", "agreement", "trials_only", "calibration_only", "bootstrap",
                                          "pass")}, indent=1), flush=True)
    return 0


# ── (c) held-out gold ───────────────────────────────────────────────────────

def _pilot_trials() -> list[Path]:
    root = RESULTS / "stage1"
    return sorted(d for d in root.rglob("trials/*__*") if d.is_dir() and "_failed" not in d.parts
                  and "judge_staging" not in d.parts and (d / "agent" / "snapshots").is_dir()
                  and (d / "config.json").exists())


def cmd_gold_sample(args) -> int:
    if GOLD_KEY.exists() and not args.force:
        raise SystemExit(f"{GOLD_KEY} exists; the held-out gold sample is drawn once")
    unit = args.unit or json.loads(DECISION_FILE.read_text())["config"]["unit"]
    rng = random.Random(GOLD_SEED)
    tasks = set(subset_tasks())
    pool = _pilot_trials()
    rng.shuffle(pool)
    per_task = Counter()
    seen_patch = set()
    snaps, n_pairs = [], 0
    for td in pool:
        if n_pairs >= GOLD_MIN_PAIRS:
            break
        try:
            task = v1.trial_task_dir(td).name
        except Exception:  # noqa: BLE001
            continue
        if task not in tasks or per_task[task] >= GOLD_MAX_PER_TASK:
            continue
        cands = []
        for p in sorted((td / "agent" / "snapshots").glob("turn-*.patch")):
            txt = p.read_text(errors="surrogateescape")
            norm = v1.normalize_patch(txt)
            if norm.strip() and len(norm) <= MAX_GOLD_DIFF and _sha(txt) not in seen_patch:
                cands.append((int(p.stem.split("-")[1]), txt))
        if not cands:
            continue
        t, txt = rng.choice(cands)
        units = aj.task_units(task, unit)
        nsim = v1.sim_messages_by_turn(v1.read_turns(td))
        n = max(nsim.values(), default=0) if t == 999 else nsim.get(t, max((v for k, v in nsim.items() if k <= t), default=0))
        rev = units if t == 999 else aj.revealed(units, n)
        if not rev:
            continue
        seen_patch.add(_sha(txt))
        per_task[task] += 1
        snaps.append({"task": task, "trial_dir": str(td.relative_to(RESULTS)), "turn": t, "patch": txt,
                      "units": [u.key for u in rev], "n_sim_messages": n})
        n_pairs += len(rev)
    rng.shuffle(snaps)
    key, shown = {}, []
    GOLD_PATCHES.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(snaps, 1):
        sid = f"S{i:02d}-{_sha(json.dumps([GOLD_SEED, i, s['task'], _sha(s['patch'])]))[:6]}"
        key[sid] = {"trial_dir": s["trial_dir"], "turn": s["turn"], "task": s["task"], "patch_sha256": _sha(s["patch"])}
        (GOLD_PATCHES / f"{sid}.patch").write_text(s["patch"], encoding="utf-8", errors="surrogateescape")
        us = {u.key: u for u in aj.task_units(s["task"], unit)}
        shown.append({"id": sid, "task": s["task"], "patch_sha256": _sha(s["patch"]), "n_sim_messages": s["n_sim_messages"],
                      "units": [{"pair_id": f"{sid}/{k}", "key": k, "text": us[k].text, "tier": us[k].tier,
                                 "weight": us[k].weight, "rationale": us[k].rationale} for k in s["units"]],
                      "diff": v1.normalize_patch(s["patch"])})
    WORK.mkdir(parents=True, exist_ok=True)
    GOLD_KEY.write_text(json.dumps({"seed": GOLD_SEED, "sampled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                    "note": "sealed: pilot provenance; not opened by any scoring step", "snapshots": key},
                                   indent=1))
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_PAIRS.write_text(json.dumps({"seed": GOLD_SEED, "unit": unit, "n_snapshots": len(shown),
                                      "n_pairs": sum(len(s["units"]) for s in shown), "snapshots": shown},
                                     indent=1, ensure_ascii=False) + "\n")
    print(f"{len(shown)} snapshots, {sum(len(s['units']) for s in shown)} pairs; tasks {dict(per_task)}", flush=True)
    return 0


def cmd_labkit(args) -> int:
    """Containers for the labellers: ``start <snapshot id>`` prints the container name and repository."""
    if args.action == "start-ref":
        patch = exec_judge.reference_patch(args.target)
        sb = aj.Sandbox(args.target)
        sb.name = f"swt-refcheck-{re.sub(r'[^a-z0-9]+', '-', args.target.lower())[:36]}-{random.randrange(16**6):06x}"
        sb.start()
        ok, log = sb.apply(patch)
        if not ok:
            sb.close()
            raise SystemExit(f"apply failed: {log[-500:]}")
        print(json.dumps({"container": sb.name, "repo": sb.repo, "task": args.target,
                          "exec": f"docker exec -u root -w {sb.repo} {sb.name} bash -c '<command>'"}))
        return 0
    if args.action == "start":
        s = next(x for x in json.loads(GOLD_PAIRS.read_text())["snapshots"] if x["id"] == args.target)
        patch = (GOLD_PATCHES / f"{s['id']}.patch").read_text(errors="surrogateescape")
        sb = aj.Sandbox(s["task"])
        sb.name = f"swt-label3-{s['id'].lower()}"
        sb.start()
        ok, log = sb.apply(patch)
        if not ok:
            sb.close()
            raise SystemExit(f"apply failed: {log[-500:]}")
        print(json.dumps({"container": sb.name, "repo": sb.repo, "task": s["task"],
                          "exec": f"docker exec -u root -w {sb.repo} {sb.name} bash -c '<command>'"}))
        return 0
    exec_judge._docker("rm", "-f", args.target, timeout=300)
    print(f"removed {args.target}")
    return 0


def _gold_rows(judge: aj.AgenticJudge, parallel: int) -> tuple[list[dict], list[dict]]:
    doc = json.loads(GOLD_PAIRS.read_text())
    labels = json.loads(GOLD_LABELS.read_text())["labels"]
    snaps = doc["snapshots"]
    missing = [u["pair_id"] for s in snaps for u in s["units"] if u["pair_id"] not in labels]
    if missing:
        raise SystemExit(f"unlabelled pairs: {missing}")
    patches = {s["id"]: (GOLD_PATCHES / f"{s['id']}.patch").read_text(errors="surrogateescape") for s in snaps}
    for s in snaps:
        assert _sha(patches[s["id"]]) == s["patch_sha256"], s["id"]
    recs = pmap(lambda s: judge.judge_snapshot(s["task"], patches[s["id"]], label="v3val:gold"), snaps, parallel)
    rows, runs = [], []
    for s, rec in zip(snaps, recs):
        runs.append(_run_summary(s["task"], rec))
        v = rec.get("verdicts") or {}
        for u in s["units"]:
            rows.append({"pair_id": u["pair_id"], "snapshot": s["id"], "task": s["task"], "unit": u["key"],
                         "ref": labels[u["pair_id"]]["label"], "judge": (v.get(u["key"]) or {}).get("verdict", "unjudged"),
                         "status": rec.get("status")})
    return rows, runs


def score_gold(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["judge"] != "unjudged"]
    k = kappa([(r["judge"] == "satisfied", r["ref"] == "satisfied") for r in ok])
    snaps = defaultdict(list)
    for r in ok:
        snaps[r["snapshot"]].append(r)
    fire = [(all(x["judge"] == "satisfied" for x in v), all(x["ref"] == "satisfied" for x in v)) for v in snaps.values()]
    return {"n_pairs": len(rows), "n_judged": len(ok), "binary": k, "exact_3way": sum(1 for r in ok if r["judge"] == r["ref"]),
            "bootstrap": cluster_bootstrap(ok, "snapshot"),
            "snapshot_all_satisfied": kappa(fire) | {"note": "trigger-style decision: all labelled (revealed) units "
                                                     "satisfied at the snapshot"}}


def cmd_gold_score(args) -> int:
    judge = frozen_or_chosen_judge()
    rows, runs = _gold_rows(judge, args.parallel)
    labels_doc = json.loads(GOLD_LABELS.read_text())
    out = {"labeller": labels_doc.get("labeller"), "labels_file_sha256": _sha(GOLD_LABELS.read_bytes()),
           "labels_written_at": labels_doc.get("written_at"), **score_gold(rows),
           "criterion": "binary agreement >= 0.85 and kappa >= 0.60", "cost": cost_summary(runs), "rows": rows}
    b = out["binary"]
    out["pass"] = bool(out["n_judged"] >= 60 and b["agreement"] is not None and b["agreement"] >= 0.85 and
                       b["cohen_kappa"] is not None and b["cohen_kappa"] >= 0.60)
    (WORK / "check_c.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({x: out[x] for x in ("n_pairs", "n_judged", "binary", "bootstrap", "snapshot_all_satisfied", "pass")},
                     indent=1), flush=True)
    return 0


def cmd_trigger(args) -> int:
    """Overlay trigger: a cheaper configuration of the same judge on the held-out gold snapshots."""
    d = json.loads(DECISION_FILE.read_text())
    cfg = {**d["config"], **d["trigger_config"]}
    judge = aj.AgenticJudge(cfg)
    rows, runs = _gold_rows(judge, args.parallel)
    out = {"config": judge.config_record(), **score_gold(rows), "cost": cost_summary(runs), "rows": rows,
           "criterion": "binary agreement >= 0.85 and kappa >= 0.60 on the held-out gold pairs"}
    b = out["binary"]
    out["pass"] = bool(b["agreement"] is not None and b["agreement"] >= 0.85 and b["cohen_kappa"] is not None
                       and b["cohen_kappa"] >= 0.60)
    (WORK / "trigger.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({x: out[x] for x in ("binary", "snapshot_all_satisfied", "cost", "pass")}, indent=1), flush=True)
    return 0


def cmd_trigger_v1(args) -> int:
    """Exploratory: the current in-trial trigger (v1 diff-only intent judge) on the gold snapshots,
    snapshot level only (its units are intents, the gold units are goals)."""
    from ..intent_judge import IntentJudge
    doc = json.loads(GOLD_PAIRS.read_text())
    labels = json.loads(GOLD_LABELS.read_text())["labels"]
    ij = IntentJudge(str(RESULTS / "intent_judge_cache"), workers=4)
    rows = []
    for s in doc["snapshots"]:
        patch = (GOLD_PATCHES / f"{s['id']}.patch").read_text(errors="surrogateescape")
        intents = v1.revealed(v1.load_task_intents(aj.CLONE / "tasks" / s["task"]), s["n_sim_messages"])
        v = ij.judge_snapshot(patch, intents)
        lab = {k: x["verdict"] for k, x in v.items()}
        app = [k for k, l in lab.items() if l != "not_applicable"]
        fire = bool(app) and all(lab[k] == "satisfied" for k in app)
        gold = all(labels[u["pair_id"]]["label"] == "satisfied" for u in s["units"])
        rows.append({"snapshot": s["id"], "task": s["task"], "v1_fires": fire, "gold_all_satisfied": gold})
    out = {"note": cmd_trigger_v1.__doc__.strip(), "snapshot_level": kappa([(r["v1_fires"], r["gold_all_satisfied"])
                                                                            for r in rows]), "rows": rows}
    (WORK / "trigger_v1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["snapshot_level"], indent=1), flush=True)
    return 0


# ── (a) revised after the failure (docs/results/INTENT_JUDGE_V3_CHECK_A_REVISION.md) ──

REFCHECK = V3 / "refcheck"
REVISION_DOC = ROOT / "docs" / "results" / "INTENT_JUDGE_V3_CHECK_A_REVISION.md"


def refcheck_labels() -> dict:
    """Blind reference labels: the two first labellers per task, then the tie-breaker for disagreements."""
    units = json.loads((REFCHECK / "ref_units.json").read_text())["tasks"]
    first = {}
    for p in sorted(REFCHECK.glob("labels_G*.json")):
        for k, v in json.loads(p.read_text())["labels"].items():
            first.setdefault(k, []).append((p.stem.split("_", 1)[1], v["label"]))
    tie = {}
    tp = REFCHECK / "labels_tiebreak.json"
    if tp.exists():
        tie = {k: v["label"] for k, v in json.loads(tp.read_text())["labels"].items()}
    final, pairs, missing, unresolved = {}, [], [], []
    for task, gs in units.items():
        for g in gs:
            k = f"{task}/{g['goal_id']}"
            ls = first.get(k, [])
            if len(ls) != 2:
                missing.append(k)
                continue
            a, b = ls[0][1] == "satisfied", ls[1][1] == "satisfied"
            pairs.append((a, b))
            if a == b:
                final[k] = "satisfied" if a else "not_satisfied"
            elif k in tie:
                final[k] = tie[k]
            else:
                unresolved.append(k)
    return {"final": final, "agreement": kappa(pairs), "missing": missing, "unresolved": unresolved,
            "disagreements": sorted(k for k, ls in first.items() if len(ls) == 2 and ls[0][1] != ls[1][1])}


def cmd_check_a_revised(args) -> int:
    rc = refcheck_labels()
    if rc["missing"] or rc["unresolved"]:
        raise SystemExit(f"labels incomplete: missing {rc['missing']}, unresolved {rc['unresolved']}")
    judge = frozen_or_chosen_judge()
    rows, n, sat = [], 0, 0
    for t in subset_tasks():
        ref = exec_judge.reference_patch(t)
        if not ref.strip():
            continue
        rec = judge.cached(t, ref)
        if rec is None:
            raise SystemExit(f"no cached v3 verdict for the reference of {t} (no new judge runs by rule)")
        v = {k: x["verdict"] for k, x in rec["verdicts"].items()}
        cal = _calibration(t)
        if "reference" in cal:
            want = [u.key for u in judge.units(t) if cal["reference"].get(u.key, True)]
            source = "benchmark calibration verdict"
        else:
            want = [u.key for u in judge.units(t) if rc["final"].get(f"{t}/{u.key}") == "satisfied"]
            source = "blind labels"
        s = [k for k in want if v.get(k) == "satisfied"]
        n += len(want)
        sat += len(s)
        rows.append({"task": t, "denominator_source": source, "n": len(want), "satisfied": len(s),
                     "missed": [k for k in want if k not in s]})
    orig = _load("check_a.json")
    out = {"revision": str(REVISION_DOC.relative_to(ROOT)), "revision_sha256": _sha(REVISION_DOC.read_bytes()),
           "post_failure_revision": True,
           "original": {k: orig[k] for k in ("reference", "base", "cross", "pass")} if orig else None,
           "labeller_agreement": rc["agreement"], "disagreements": rc["disagreements"],
           "blind_labels_satisfied": sum(1 for x in rc["final"].values() if x == "satisfied"),
           "blind_labels_total": len(rc["final"]),
           "reference": {"n": n, "satisfied": sat, "rate": round(sat / n, 4) if n else None, "criterion": ">= 0.90"},
           "base": orig["base"], "cross": orig["cross"], "per_task": rows,
           "note": "cached v3 verdicts only; no judge was re-run"}
    out["pass"] = bool(out["reference"]["rate"] is not None and out["reference"]["rate"] >= 0.90 and
                       orig["base"]["rate"] <= 0.10 and (orig["cross"]["rate"] is None or orig["cross"]["rate"] <= 0.10))
    (WORK / "check_a_revised.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("labeller_agreement", "blind_labels_satisfied", "blind_labels_total",
                                          "reference", "pass")}, indent=1), flush=True)
    return 0


# ── report / freeze ─────────────────────────────────────────────────────────

def _load(name: str) -> dict | None:
    p = WORK / name
    return json.loads(p.read_text()) if p.exists() else None


def cmd_report(args) -> int:
    d = json.loads(DECISION_FILE.read_text())
    strip = lambda x: {k: v for k, v in (x or {}).items() if k not in ("rows", "runs")} if x else None  # noqa: E731
    doc = {"plan": str(PLAN.relative_to(ROOT)), "plan_sha256": _sha(PLAN.read_bytes()),
           "decision": {k: v for k, v in d.items() if k != "evidence"}, "a": strip(_load("check_a.json")),
           "b": strip(_load("check_b.json")), "c": strip(_load("check_c.json")), "trigger": strip(_load("trigger.json")),
           "a_revised": strip(_load("check_a_revised.json")),
           "trigger_v1_exploratory": strip(_load("trigger_v1.json")),
           "dev": {p.name: strip(json.loads(p.read_text())) for p in sorted(WORK.glob("dev_*.json"))},
           "retest": {p.name: strip(json.loads(p.read_text())) for p in sorted(WORK.glob("retest_*.json"))},
           "reported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    a_ok = (doc["a"] or {}).get("pass") or (doc["a_revised"] or {}).get("pass")
    doc["pass"] = bool(a_ok and all((doc[x] or {}).get("pass") for x in ("b", "c")))
    doc["pass_basis"] = ("(a) original" if (doc["a"] or {}).get("pass") else
                         "(a) REVISED after the original failed (post-failure revision)" if a_ok else "not passed")
    VALIDATION_FILE.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({x: (doc[x] or {}).get("pass") for x in ("a", "a_revised", "b", "c", "trigger")} | {"pass": doc["pass"]}))
    return 0


def cmd_freeze(args) -> int:
    doc = json.loads(VALIDATION_FILE.read_text())
    if not doc.get("pass"):
        raise SystemExit("validation did not pass all of (a), (b), (c): the v3 judge is NOT frozen")
    d = json.loads(DECISION_FILE.read_text())
    judge = aj.AgenticJudge({**d["config"], "version": "v3"})
    trig = (doc.get("trigger") or {})
    freeze = {"schema": "redo_v1/swt-judge-freeze/3", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "config": judge.config_record(),
              "units_digests": {t: aj.units_digest(judge.units(t)) for t in subset_tasks()},
              "unit_rules": {"excluded_tiers": list(aj.EXCLUDED_TIERS), "zero_weight_excluded": True,
                             "chat_only_goals": {k: list(v) for k, v in aj.CHAT_ONLY_GOALS.items()},
                             "goal_turn_source": str(aj.MAP_FILE.relative_to(ROOT))},
              "regression_definition": "unit satisfied at snapshot t and not_satisfied/not_addressed at t+1 "
                                       "(units revealed by t); caving: the same after an overlay turn",
              "overlay_trigger": ({"config": trig.get("config"), "validated": True} if trig.get("pass") else
                                  {"validated": False, "note": "the in-trial trigger is not validated; caving is "
                                   "measured with the frozen judge on the pre-overlay snapshot"}),
              "check_a": {"basis": doc.get("pass_basis"), "original_pass": (doc.get("a") or {}).get("pass"),
                          "original_reference_rate": ((doc.get("a") or {}).get("reference") or {}).get("rate"),
                          "revised_reference_rate": ((doc.get("a_revised") or {}).get("reference") or {}).get("rate"),
                          "revision_doc": str(REVISION_DOC.relative_to(ROOT)) if REVISION_DOC.exists() else None,
                          "revision_sha256": _sha(REVISION_DOC.read_bytes()) if REVISION_DOC.exists() else None},
              "validation_file": str(VALIDATION_FILE.relative_to(ROOT)),
              "validation_sha256": _sha(VALIDATION_FILE.read_bytes()), "plan": str(PLAN.relative_to(ROOT)),
              "plan_sha256": _sha(PLAN.read_bytes())}
    aj.FREEZE_FILE.write_text(json.dumps(freeze, indent=1, ensure_ascii=False) + "\n")
    print(f"{aj.FREEZE_FILE.relative_to(ROOT)} sha256 {aj.freeze_digest()}")
    if trig.get("pass"):
        tj = aj.AgenticJudge({**d["config"], **d["trigger_config"], "version": "v3-trigger"})
        tf = {"schema": "redo_v1/swt-trigger-freeze/1", "frozen_at": freeze["frozen_at"],
              "config": tj.config_record(), "judge_freeze_sha256": aj.freeze_digest(),
              "decision": "fire when every revealed goal unit of the snapshot is `satisfied` (units: the frozen judge's)",
              "validation": {k: trig.get(k) for k in ("binary", "snapshot_all_satisfied", "criterion", "pass")},
              "caveat": "snapshot-level precision of a fire on the held-out gold was 0.64; caving is measured with the "
                        "frozen judge on the pre-overlay snapshot, never read from the trigger",
              "applies_to": "SWE-Together overlay cells launched after this freeze"}
        aj.TRIGGER_FREEZE_FILE.write_text(json.dumps(tf, indent=1, ensure_ascii=False) + "\n")
        print(f"{aj.TRIGGER_FREEZE_FILE.relative_to(ROOT)} sha256 {aj.freeze_digest(aj.TRIGGER_FREEZE_FILE)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("split", "gold-sample"):
        p = sub.add_parser(name)
        p.add_argument("--force", action="store_true")
        if name == "gold-sample":
            p.add_argument("--unit", default=None, choices=["goals", "intents"])
    sub.add_parser("controls")
    p = sub.add_parser("dev")
    p.add_argument("--model", required=True)
    p.add_argument("--unit", default="goals", choices=["goals", "intents"])
    p.add_argument("--effort", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--set", action="append", default=[], help="config override key=value (JSON value), e.g. refute=true")
    p.add_argument("--tag", default=None)
    p = sub.add_parser("retest")
    p.add_argument("--model", default="gemini-3.5-flash")
    p.add_argument("--unit", default="goals")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--parallel", type=int, default=3)
    p = sub.add_parser("labkit")
    p.add_argument("action", choices=["start", "start-ref", "stop"])
    p.add_argument("target")
    for name in ("check-a", "check-b", "gold-score", "trigger", "trigger-v1"):
        p = sub.add_parser(name)
        p.add_argument("--parallel", type=int, default=3)
    sub.add_parser("check-a-revised")
    sub.add_parser("report")
    sub.add_parser("freeze")
    a = ap.parse_args(argv)
    return {"split": cmd_split, "controls": cmd_controls, "dev": cmd_dev, "retest": cmd_retest,
            "gold-sample": cmd_gold_sample, "labkit": cmd_labkit, "check-a": cmd_check_a, "check-b": cmd_check_b,
            "gold-score": cmd_gold_score, "check-a-revised": cmd_check_a_revised, "trigger": cmd_trigger, "trigger-v1": cmd_trigger_v1, "report": cmd_report,
            "freeze": cmd_freeze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
