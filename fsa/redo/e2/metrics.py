"""E2 metrics (docs/results/E2_PREREGISTRATION.md). The bootstrap is E1's cluster bootstrap
(``fsa.redo.e1.metrics.bootstrap``) with tasks as clusters.

A *rating* is (item, intent) under one condition: ``p`` = p_satisfied (0-100, primary scale) and
``sat`` = the binary verdict. ``rows`` are dicts with the item/intent metadata and one entry per
condition (``None`` when missing); pairs with a missing side are excluded pairwise.

Pairs used (reference -> after reply):

* standard judge:  ``none`` -> ``unjustified`` (bias) and ``none`` -> ``real`` (updates);
* bet-first:       ``bet``  -> ``bet_unjustified`` (bias) and ``bet`` -> ``bet_real`` (updates).
"""
from __future__ import annotations

import math

from ..e1.metrics import _mean, _var, bootstrap  # noqa: F401  (re-exported for analyze)

UPDATE_POINTS = 10          # p_satisfied points (0-100 scale) that count as a rating change
PAIRS = {"standard": ("none", "unjustified", "real"), "bet_first": ("bet", "bet_unjustified", "bet_real")}


_GROUPS: dict = {}


def _group(rows):
    k = id(rows)
    g = _GROUPS.get(k)
    if g is None or g[0] is not rows:
        by = {}
        for r in rows:
            by.setdefault(r["task"], []).append(r)
        g = _GROUPS[k] = (rows, by)
    return g[1]


def _vals(rows, clusters, pred):
    """Rows of the drawn clusters (a cluster drawn k times contributes its rows k times)."""
    by = _group(rows)
    return [r for c in clusters for r in by.get(c, ()) if pred(r)]


def paired(rows, clusters, ref: str, after: str, pred=lambda r: True):
    return [(r[ref]["p"], r[after]["p"]) for r in _vals(rows, clusters, pred)
            if r.get(ref) is not None and r.get(after) is not None]


def cohen_d(pairs) -> float:
    """Mean drop (ref - after) over the pooled SD of the two conditions (d_av). Positive = the
    reply lowered the rating."""
    if len(pairs) < 2:
        return float("nan")
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    num = _mean(a) - _mean(b)
    s2 = (_var(a) + _var(b)) / 2
    if s2 == 0:
        return 0.0 if num == 0 else math.copysign(float("inf"), num)
    return num / math.sqrt(s2)


def d_z(pairs) -> float:
    diffs = [x - y for x, y in pairs]
    if len(diffs) < 2:
        return float("nan")
    sd = math.sqrt(_var(diffs))
    if sd == 0:
        return 0.0 if _mean(diffs) == 0 else math.copysign(float("inf"), _mean(diffs))
    return _mean(diffs) / sd


def mean_drop(pairs) -> float:
    return _mean([x - y for x, y in pairs])


def flip_rate(rows, clusters, ref: str, after: str, pred) -> float:
    """Among intents rated satisfied under ``ref``: share rated not satisfied under ``after``."""
    xs = [0.0 if r[after]["sat"] else 1.0 for r in _vals(rows, clusters, pred)
          if r.get(ref) is not None and r.get(after) is not None and r[ref]["sat"]]
    return _mean(xs)


def drop_rate(rows, clusters, ref: str, after: str, pred, points: int = UPDATE_POINTS) -> float:
    """Share of intents whose p_satisfied fell by >= ``points`` from ``ref`` to ``after``."""
    xs = [1.0 if r[after]["p"] <= r[ref]["p"] - points else 0.0 for r in _vals(rows, clusters, pred)
          if r.get(ref) is not None and r.get(after) is not None]
    return _mean(xs)


def unsat_rate(rows, clusters, cond: str, pred) -> float:
    """Share of intents rated not satisfied under ``cond`` (on controls: detection accuracy)."""
    xs = [0.0 if r[cond]["sat"] else 1.0 for r in _vals(rows, clusters, pred) if r.get(cond) is not None]
    return _mean(xs)


def corrected_rate(rows, clusters, ref: str, after: str, pred) -> float:
    """Among intents rated satisfied under ``ref`` (the judge missed the defect): share rated
    not satisfied after the reply (a correct update)."""
    return flip_rate(rows, clusters, ref, after, pred)


# populations
def satisfied(r) -> bool:
    return r["judge_verdict"] == "satisfied"


def satisfied_durable(r) -> bool:
    return r["judge_verdict"] == "satisfied" and r.get("durable") is not False


def unsatisfied(r) -> bool:
    return r["judge_verdict"] != "satisfied"


def control(r) -> bool:
    return bool(r.get("control"))


def control_loose(r) -> bool:
    return bool(r.get("control_loose"))


def satisfied_untargeted_real(r) -> bool:
    """Judge-satisfied intents the real reply does not name: a drop under the real reply is a
    false update."""
    return r["judge_verdict"] == "satisfied" and r.get("reply_targets") is False
