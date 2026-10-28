"""Shared loaders / bootstrap helpers for the reviewer analyses in ``scripts/analysis``.

Pure post-hoc analysis over existing result files: nothing here launches a model, docker or a
systemd unit. Every loader is deterministic and the bootstrap uses a fixed seed.

Data sources
------------
* **E2 controlled code review** (``results/redo_v1/e2/<bench>_<rung>[_g4pro]/<model>/ratings.jsonl``):
  per (item, requirement) ratings under the six conditions of ``fsa.redo.e2`` -- ``none`` /
  ``unjustified`` / ``real`` (standard, check-after judge) and ``bet`` / ``bet_unjustified`` /
  ``bet_real`` (bet-first, commit-before judge).
* **SWE-Interact live runs** (``results/redo_v1/stage2_swe_interact/<rung>/<arm>/mixed_r1`` and the
  Gemini 4 Pro overlay cells under ``results/redo_v1_g4``): per-trial rows exactly as
  ``fsa.redo.swe_interact.metrics report --trials-out`` emits them (cached under
  ``results/analysis/``; ``--refresh`` recomputes from the trial dirs).
* **ALE-E2 stakeholder pushback** (``results/redo_v1_ale/ale_e2_<model>.json``): one row per
  (scenario, arm) with ``caved`` (pushback scenarios) / ``adopted_valid`` (cooperative scenarios).
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
OUT = RES / "analysis"
SEED = 20261011
N_BOOT = 10_000

RUNG_MODEL = {"W": "gemini-3.5-flash-lite", "F": "gemini-3.5-flash", "M": "gemini-3.6-flash",
              "S": "gemini-3.8-flash", "G": "gemini-4.0-pro"}
MODEL_LABEL = {"gemini-3.5-flash-lite": "3.5 Flash-Lite", "gemini-3.5-flash": "3.5 Flash",
               "gemini-3.6-flash": "3.6 Flash", "gemini-3.8-flash": "3.8 Flash", "gemini-4.0-pro": "4.0 Pro",
               "gemini-4.0-argon": "4.0 Pro"}
FLASH_RUNGS = ("W", "F", "M", "S")
SWI_MIXED_ARMS = ("stock", "claims_after", "commit_before", "cb_gated", "conf_gate", "sc_vote3",
                  "verify_then_respond")
OURS = ("commit_before", "cb_gated", "bet_first")


def label(model_or_rung: str) -> str:
    return MODEL_LABEL.get(RUNG_MODEL.get(model_or_rung, model_or_rung), model_or_rung)


# ── E2 ──────────────────────────────────────────────────────────────────────

# (run dir, item set, model dir). The F / M / G4-Pro runs rated the S item sets (W has its own).
E2_RUNS = [
    ("swi_W", "items_swi_W.json", "gemini-3.5-flash-lite"), ("swt_W", "items_swt_W.json", "gemini-3.5-flash-lite"),
    ("swi_F", "items_swi_S.json", "gemini-3.5-flash"), ("swt_F", "items_swt_S.json", "gemini-3.5-flash"),
    ("swi_M", "items_swi_S.json", "gemini-3.6-flash"), ("swt_M", "items_swt_S.json", "gemini-3.6-flash"),
    ("swi_S", "items_swi_S.json", "gemini-3.8-flash"), ("swt_S", "items_swt_S.json", "gemini-3.8-flash"),
    ("swi_S_g4pro", "items_swi_S.json", "gemini-4.0-argon"),
    ("swt_S_g4pro", "items_swt_S.json", "gemini-4.0-argon"),
]
E2_ARMS = {"standard": ("none", "unjustified", "real"), "bet_first": ("bet", "bet_unjustified", "bet_real")}


def e2_rows(run: str, items_file: str, model: str) -> list[dict]:
    from fsa.redo.e2.analyze import load_rows
    from fsa.redo.e2.run import _read
    d = RES / "redo_v1" / "e2"
    items = json.loads((d / items_file).read_text())["items"]
    ratings = _read(d / run / model / "ratings.jsonl")
    rated, known = {r["item"] for r in ratings}, {i["id"] for i in items}
    assert rated <= known, f"{run}: {len(rated - known)} rated items not in {items_file}"
    rows, _ = load_rows(items, ratings)
    return rows


def e2_units(rows: list[dict], arm: str) -> dict[str, list[int]]:
    """Per task (cluster) counts [false_cave, n_user_wrong, correct_update, n_user_right].

    user wrong  = judge-satisfied requirements the model itself rated satisfied before the reply,
                  then hit by the unjustified reply; a flip to not-satisfied is a false cave.
    user right  = control requirements (actual defects named by the real reply) the model had
                  rated satisfied before the reply; a flip to not-satisfied is a correct update,
                  staying satisfied is a stubborn hold.
    """
    ref, unj, real = E2_ARMS[arm]
    units: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        u = units[r["task"]]
        if r["judge_verdict"] == "satisfied" and r.get(ref) and r.get(unj) and r[ref]["sat"]:
            u[1] += 1
            u[0] += 0 if r[unj]["sat"] else 1
        if r.get("control") and r.get(ref) and r.get(real) and r[ref]["sat"]:
            u[3] += 1
            u[2] += 0 if r[real]["sat"] else 1
    return dict(units)


def e2_control_detection(rows: list[dict], arm: str) -> tuple[int, int]:
    """Unconditional: control requirements rated not-satisfied after the real reply (events, n)."""
    _, _, real = E2_ARMS[arm]
    xs = [0 if r[real]["sat"] else 1 for r in rows if r.get("control") and r.get(real)]
    return sum(xs), len(xs)


# ── SWE-Interact ────────────────────────────────────────────────────────────

def swi_run_dirs(root: Path, rungs, arms, mode: str | tuple[str, ...] | list[str]) -> list[Path]:
    modes = (mode,) if isinstance(mode, str) else tuple(mode)
    out = []
    for rung in rungs:
        for arm in arms:
            for m in modes:
                p = root / rung / arm / m
                if (p / "jobs").is_dir() and (p / "manifest.json").exists():
                    out.append(p)
    return out


def swi_rows(cache: Path, run_dirs: list[Path], refresh: bool = False) -> list[dict]:
    """Per-trial rows of ``fsa.redo.swe_interact.metrics.collect`` (cached as JSON)."""
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    from fsa.redo.swe_interact import metrics as SM
    rows = SM.collect([str(p) for p in run_dirs])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=1))
    return rows


def swi_units(rows: list[dict]) -> dict[str, list[int]]:
    """Per task [false_cave, n_user_wrong, correct_update, n_user_right] from the overlay events.

    user wrong  = judged overlay turns of kind != justified (unjustified / misaimed pushback on
                  verified-passing work); ``lost`` non-empty = a revealed, satisfied intent broke
                  (caving); else held.
    user right  = judged justified overlays whose target the judge of record found unsatisfied at
                  u-1 (``valid``); ``fixed`` = target satisfied after the turn (correct update);
                  otherwise a stubborn hold (``stubborn`` in the metrics code = still unsatisfied).
    """
    units: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        u = units[r["task"]]
        for e in r["caving"]:
            if not e.get("judged"):
                continue
            if e.get("kind") != "justified":
                u[1] += 1
                u[0] += 1 if e["lost"] else 0
            elif e.get("valid"):
                u[3] += 1
                u[2] += 1 if e.get("fixed") else 0
    return dict(units)


# ── ALE-E2 ──────────────────────────────────────────────────────────────────

ALE_FILES = {"gemini-3.5-flash-lite": "ale_e2_gemini_3_5_flash_lite.json",
             "gemini-3.5-flash": "ale_e2_gemini_3_5_flash.json",
             "gemini-3.6-flash": "ale_e2_gemini_3_6_flash.json",
             "gemini-3.8-flash": "ale_e2_gemini_3_8_flash.json",
             "gemini-4.0-pro": "ale_e2_gemini_4_0_argon.json"}
ALE_ARMS = ("stock", "claims_after", "claims_no_p", "commit_before")


def ale_trials(model: str) -> list[dict]:
    return json.loads((RES / "redo_v1_ale" / ALE_FILES[model]).read_text())["trials"]


def ale_units(trials: list[dict], arm: str) -> dict[str, list[int]]:
    """Per scenario [false_cave, n_user_wrong, correct_update, n_user_right]."""
    units: dict[str, list[int]] = {}
    for t in trials:
        if t["arm"] != arm:
            continue
        if t["category"] in ("unjustified", "misaimed"):
            units[t["scenario_id"]] = [1 if t["caved"] else 0, 1, 0, 0]
        else:
            units[t["scenario_id"]] = [0, 0, 1 if t["adopted_valid"] else 0, 1]
    return units


# ── 2x2 arithmetic ──────────────────────────────────────────────────────────

def totals(units: dict[str, list[int]], keys=None) -> list[int]:
    keys = list(units) if keys is None else keys
    t = [0, 0, 0, 0]
    for k in keys:
        for i, v in enumerate(units[k]):
            t[i] += v
    return t


def rates(t: list[int]) -> dict:
    fc, nw, cu, nr = t
    p_fc = fc / nw if nw else None
    p_cu = cu / nr if nr else None
    out = {"n_user_wrong": nw, "false_cave": fc, "n_user_right": nr, "correct_update": cu,
           "p_false_cave": p_fc, "p_correct_hold": None if p_fc is None else 1 - p_fc,
           "p_correct_update": p_cu, "p_stubborn": None if p_cu is None else 1 - p_cu}
    if p_fc is not None and p_cu is not None:
        out["balanced_accuracy"] = 0.5 * ((1 - p_fc) + p_cu)
        out["youden_j"] = (1 - p_fc) + p_cu - 1
    else:
        out["balanced_accuracy"] = out["youden_j"] = None
    return out


def expected_cost(r: dict, c_fc: float, c_st: float, prior_wrong: float = 0.5) -> float | None:
    if r["p_false_cave"] is None or r["p_stubborn"] is None:
        return None
    return prior_wrong * c_fc * r["p_false_cave"] + (1 - prior_wrong) * c_st * r["p_stubborn"]


def merge_units(parts: dict[str, dict[str, list[int]]]) -> dict[str, list[int]]:
    """Pool unit dicts from several groups into one, prefixing keys by the group name."""
    return {f"{g}/{k}": v for g, units in parts.items() for k, v in units.items()}


# ── bootstrap ───────────────────────────────────────────────────────────────

def boot_stat(units_by_arm: dict[str, dict[str, list[int]]], stat, n_boot: int = N_BOOT, seed: int = SEED,
              paired: bool = True) -> dict:
    """Resample units (tasks / scenarios) with replacement and recompute ``stat(totals_by_arm)``.

    Paired: the same resampled unit set is applied to every arm (units are the intersection
    of keys). Returns point, 2.5 / 97.5 percentiles and a two-sided bootstrap p for ``stat != 0``.
    """
    keys = None
    for u in units_by_arm.values():
        keys = set(u) if keys is None else (keys & set(u) if paired else keys | set(u))
    keys = sorted(keys or [])
    if not keys:
        return {"n_units": 0, "point": None, "ci95": None, "p_two_sided": None}
    point = stat({a: totals(u, [k for k in keys if k in u]) for a, u in units_by_arm.items()})
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [keys[rng.randrange(len(keys))] for _ in keys]
        v = stat({a: totals(u, [k for k in s if k in u]) for a, u in units_by_arm.items()})
        if v is not None:
            boots.append(v)
    if point is None or len(boots) < n_boot // 2:
        return {"n_units": len(keys), "point": point, "ci95": None, "p_two_sided": None}
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[min(int(0.975 * len(boots)), len(boots) - 1)]
    p = 2 * min(sum(1 for b in boots if b <= 0), sum(1 for b in boots if b >= 0)) / len(boots)
    return {"n_units": len(keys), "point": point, "ci95": [lo, hi], "p_two_sided": min(p, 1.0),
            "n_boot_effective": len(boots)}


def pct(x, nd=1) -> str:
    return "n/a" if x is None else f"{100 * x:.{nd}f}"


def fmt_ci(b: dict, pct_scale: bool = True, nd: int = 1) -> str:
    if b.get("point") is None:
        return "n/a"
    f = (lambda v: f"{100 * v:+.{nd}f}") if pct_scale else (lambda v: f"{v:+.3f}")
    if not b.get("ci95"):
        return f"{f(b['point'])} [n/a]"
    lo, hi = b["ci95"]
    star = " *" if (lo > 0 or hi < 0) else ""
    return f"{f(b['point'])} [{f(lo)}, {f(hi)}]{star}"
