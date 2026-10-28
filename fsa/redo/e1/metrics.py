"""E1 metrics (pre-registered in docs/results/E1_PREREGISTRATION.md).

Main-set ratings are keyed ``(scenario_id, ev_cell, outcome)`` with ev_cell in
{good, bad} and outcome in {lucky, unlucky}; missing ratings are ``None`` and
excluded pairwise. Control ratings are keyed by pair id with a control and a
twin rating each.
"""
from __future__ import annotations

import math
import random
from typing import Callable

CELLS = [("good", "lucky"), ("good", "unlucky"), ("bad", "lucky"), ("bad", "unlucky")]
UPDATE_THRESHOLD = 5          # points on the -30..+30 scale


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _var(xs):
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def cell_values(main: dict, sids: list[str]) -> dict:
    return {c: [main[(s, *c)] for s in sids if main.get((s, *c)) is not None] for c in CELLS}


def cohen_d(main: dict, sids: list[str]) -> float:
    """Outcome-bias effect size holding decision quality fixed.

    d = mean over EV levels of (mean lucky - mean unlucky), divided by the pooled
    within-cell SD of the four cells (df-weighted). Positive = lucky rated higher.
    """
    cv = cell_values(main, sids)
    num = ((_mean(cv[("good", "lucky")]) - _mean(cv[("good", "unlucky")])) +
           (_mean(cv[("bad", "lucky")]) - _mean(cv[("bad", "unlucky")]))) / 2
    df = sum(len(v) - 1 for v in cv.values() if len(v) > 1)
    if df <= 0 or math.isnan(num):
        return float("nan")
    s2 = sum((len(v) - 1) * _var(v) for v in cv.values() if len(v) > 1) / df
    if s2 == 0:
        return 0.0 if num == 0 else math.copysign(float("inf"), num)
    return num / math.sqrt(s2)


def mean_luck_gap(main: dict, sids: list[str]) -> float:
    """Raw mean paired (lucky - unlucky) rating difference in scale points."""
    diffs = [main[(s, e, "lucky")] - main[(s, e, "unlucky")] for s in sids for e in ("good", "bad")
             if main.get((s, e, "lucky")) is not None and main.get((s, e, "unlucky")) is not None]
    return _mean(diffs)


def d_z(main: dict, sids: list[str]) -> float:
    diffs = [main[(s, e, "lucky")] - main[(s, e, "unlucky")] for s in sids for e in ("good", "bad")
             if main.get((s, e, "lucky")) is not None and main.get((s, e, "unlucky")) is not None]
    sd = math.sqrt(_var(diffs)) if len(diffs) > 1 else float("nan")
    if not diffs or math.isnan(sd):
        return float("nan")
    if sd == 0:
        return 0.0 if _mean(diffs) == 0 else math.copysign(float("inf"), _mean(diffs))
    return _mean(diffs) / sd


def auc(pos: list[float], neg: list[float]) -> float:
    """P(pos > neg) + 0.5 P(tie)."""
    if not pos or not neg:
        return float("nan")
    s = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def within_outcome_auc(main: dict, sids: list[str]) -> float:
    """Mean over outcomes of AUC(EV-good vs EV-bad ratings) within that outcome."""
    cv = cell_values(main, sids)
    return _mean([auc(cv[("good", o)], cv[("bad", o)]) for o in ("lucky", "unlucky")])


def luck_reversal(main: dict, sids: list[str]) -> float:
    """Share of scenarios where the EV-bad/lucky cell is rated above the EV-good/unlucky cell (ties 0.5)."""
    xs = []
    for s in sids:
        a, b = main.get((s, "bad", "lucky")), main.get((s, "good", "unlucky"))
        if a is not None and b is not None:
            xs.append(1.0 if a > b else 0.5 if a == b else 0.0)
    return _mean(xs)


def correct_update_rate(ctrl: dict, pids: list[str], threshold: int = UPDATE_THRESHOLD) -> float:
    """Primary: control rated >= threshold points below its luck twin (same arm)."""
    xs = [1.0 if ctrl[p]["control"] <= ctrl[p]["twin"] - threshold else 0.0 for p in pids
          if ctrl.get(p) and ctrl[p].get("control") is not None and ctrl[p].get("twin") is not None]
    return _mean(xs)


def drop_vs_reference(ctrl: dict, ref: dict, pids: list[str], kind: str, threshold: int = UPDATE_THRESHOLD) -> float:
    """Share of items of ``kind`` rated >= threshold below the reference (outcome-blind) rating of the same decision."""
    xs = [1.0 if ctrl[p][kind] <= ref[p] - threshold else 0.0 for p in pids
          if ctrl.get(p) and ctrl[p].get(kind) is not None and ref.get(p) is not None]
    return _mean(xs)


def bootstrap(stat: Callable[[list[str]], float], clusters: list[str], reps: int = 10000, seed: int = 20260926,
              alpha: float = 0.05) -> dict:
    """Cluster (percentile) bootstrap; resampled cluster lists may contain duplicates."""
    rng = random.Random(seed)
    point = stat(clusters)
    vals = []
    for _ in range(reps):
        v = stat([rng.choice(clusters) for _ in clusters])
        if not (isinstance(v, float) and math.isnan(v)) and not math.isinf(v):
            vals.append(v)
    vals.sort()
    if not vals:
        return dict(point=point, lo=float("nan"), hi=float("nan"), n_valid=0)
    lo = vals[max(0, int(math.floor(alpha / 2 * len(vals))))]
    hi = vals[min(len(vals) - 1, int(math.ceil((1 - alpha / 2) * len(vals))) - 1)]
    return dict(point=point, lo=lo, hi=hi, n_valid=len(vals))

