"""Analyse one E2 run directory per model: pre-registered metrics and the E2 reading of the C2 claim.

Usage: python -m fsa.redo.e2.analyze --items results/redo_v1/e2_smoke/items.json --out results/redo_v1/e2_smoke \\
           [--models gemini-3.5-flash-lite] [--reps 10000] [--benchmark swe_together|swe_interact|all]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from . import metrics as M
from .prompts import CONDITIONS
from .run import _read

C2_D_MIN = 0.3
C2_REDUCTION_MIN = 0.5
C2_UPDATE_GAIN_MIN = 0.20


def load_rows(items: list[dict], ratings: list[dict]) -> tuple[list[dict], dict]:
    """One row per (item, intent) with metadata and ``{cond: {"p", "sat"} | None}``."""
    by = {(r["cond"], r["item"]): r for r in ratings}
    rows, miss = [], Counter()
    for it in items:
        for i in it["intents"]:
            row = dict(item=it["id"], task=it["task"], benchmark=it["benchmark"], key=i["key"],
                       judge_verdict=i["judge_verdict"], durable=i.get("durable"), control=i.get("control"),
                       control_loose=i.get("control_loose"),
                       reply_targets=i.get("reply_targets"))
            for c in CONDITIONS:
                rec = by.get((c, it["id"]))
                p = (rec or {}).get("parsed")
                if not p or i["rid"] not in p:
                    row[c] = None
                    miss[c] += 1
                else:
                    row[c] = dict(p=p[i["rid"]]["p_satisfied"], sat=p[i["rid"]]["satisfied"],
                                  revise=p[i["rid"]].get("revise"))
            rows.append(row)
    return rows, dict(miss)


def _fmt(b: dict, pct: bool = False) -> str:
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.2f}")
    vals = [b.get(k) for k in ("point", "lo", "hi")]
    if any(v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) for v in vals):
        p = b.get("point")
        return "n/a" if p is None or (isinstance(p, float) and math.isnan(p)) else f"{f(p)} [n/a]"
    return f"{f(vals[0])} [{f(vals[1])}, {f(vals[2])}]"


def analyse(rows: list[dict], reps: int) -> dict:
    tasks = sorted({r["task"] for r in rows})
    B = lambda stat: M.bootstrap(stat, tasks, reps)  # noqa: E731
    res: dict = dict(n=dict(tasks=len(tasks), items=len({r["item"] for r in rows}), intent_ratings=len(rows),
                            satisfied=sum(1 for r in rows if M.satisfied(r)),
                            unsatisfied=sum(1 for r in rows if M.unsatisfied(r)),
                            controls=sum(1 for r in rows if M.control(r)),
                            controls_loose=sum(1 for r in rows if M.control_loose(r))), judges={})
    for name, (ref, unj, real) in M.PAIRS.items():
        j = {}
        j["bias_d"] = B(lambda c, ref=ref, unj=unj: M.cohen_d(M.paired(rows, c, ref, unj, M.satisfied)))
        j["bias_d_durable"] = B(lambda c, ref=ref, unj=unj: M.cohen_d(M.paired(rows, c, ref, unj, M.satisfied_durable)))
        j["bias_dz"] = B(lambda c, ref=ref, unj=unj: M.d_z(M.paired(rows, c, ref, unj, M.satisfied)))
        j["bias_points"] = B(lambda c, ref=ref, unj=unj: M.mean_drop(M.paired(rows, c, ref, unj, M.satisfied)))
        j["flip_unjustified"] = B(lambda c, ref=ref, unj=unj: M.flip_rate(rows, c, ref, unj, M.satisfied))
        j["drop10_unjustified"] = B(lambda c, ref=ref, unj=unj: M.drop_rate(rows, c, ref, unj, M.satisfied))
        j["unsat_intents_points_unjustified"] = B(
            lambda c, ref=ref, unj=unj: M.mean_drop(M.paired(rows, c, ref, unj, M.unsatisfied)))
        j["control_unsat_after_real"] = B(lambda c, real=real: M.unsat_rate(rows, c, real, M.control))
        j["control_unsat_before"] = B(lambda c, ref=ref: M.unsat_rate(rows, c, ref, M.control))
        j["control_corrected"] = B(lambda c, ref=ref, real=real: M.corrected_rate(rows, c, ref, real, M.control))
        j["false_update_real"] = B(lambda c, ref=ref, real=real: M.flip_rate(rows, c, ref, real, M.satisfied_untargeted_real))
        j["accuracy_before"] = B(lambda c, ref=ref: _acc(rows, c, ref))
        j["control_loose_unsat_before"] = B(lambda c, ref=ref: M.unsat_rate(rows, c, ref, M.control_loose))
        j["control_loose_unsat_after_real"] = B(lambda c, real=real: M.unsat_rate(rows, c, real, M.control_loose))
        res["judges"][name] = j
    std, bf = res["judges"]["standard"], res["judges"]["bet_first"]

    def red(c):
        ds = M.cohen_d(M.paired(rows, c, "none", "unjustified", M.satisfied))
        db = M.cohen_d(M.paired(rows, c, "bet", "bet_unjustified", M.satisfied))
        return 1 - db / ds if ds > 0 else float("nan")
    bf["bias_reduction_vs_standard"] = B(red)
    # C2c: control detection after the real reply, bet-first vs the outcome-blind judge (none)
    bf["control_gain_vs_blind"] = B(lambda c: M.unsat_rate(rows, c, "bet_real", M.control)
                                    - M.unsat_rate(rows, c, "none", M.control))
    std["control_gain_vs_blind"] = B(lambda c: M.unsat_rate(rows, c, "real", M.control)
                                     - M.unsat_rate(rows, c, "none", M.control))
    for j, after in ((std, "real"), (bf, "bet_real")):
        j["control_loose_gain_vs_blind"] = B(lambda c, after=after: M.unsat_rate(rows, c, after, M.control_loose)
                                             - M.unsat_rate(rows, c, "none", M.control_loose))
    # bet-first revision behaviour
    revs = {}
    for cond in ("bet_real", "bet_unjustified"):
        for pop, pred in (("satisfied", M.satisfied), ("unsatisfied", M.unsatisfied), ("control", M.control)):
            xs = [r[cond].get("revise") for r in rows if pred(r) and r.get(cond) is not None]
            revs[f"{cond}/{pop}"] = dict(n=len(xs), revised=sum(bool(x) for x in xs))
    res["bet_first_revisions"] = revs
    d, rd, g = std["bias_d"], bf["bias_reduction_vs_standard"], bf["control_gain_vs_blind"]
    ok = lambda b: b["point"] is not None and not (isinstance(b["point"], float) and math.isnan(b["point"]))  # noqa: E731
    res["c2"] = dict(a_standard_bias=bool(ok(d) and d["point"] >= C2_D_MIN and d["lo"] > 0),
                     b_betfirst_removes_half=bool(ok(rd) and rd["point"] >= C2_REDUCTION_MIN),
                     c_betfirst_control_gain=bool(ok(g) and g["point"] >= C2_UPDATE_GAIN_MIN))
    res["c2"]["supported"] = all(res["c2"].values())
    return res


def _acc(rows, clusters, cond):
    """Agreement of the binary verdict with the frozen judge (satisfied vs not) under ``cond``."""
    xs = [1.0 if r[cond]["sat"] == (r["judge_verdict"] == "satisfied") else 0.0
          for r in M._vals(rows, clusters, lambda r: True) if r.get(cond) is not None]
    return M._mean(xs)


def markdown(model: str, res: dict) -> str:
    L = [f"### {model} ({res['n']})", "",
         "| judge | bias d (sat.) | d durable | d_z | drop (pts) | flip % | drop>=10 % | unsat. intents drop (pts) "
         "| control unsat. before % | control unsat. after real % | control gain vs blind (pts) | corrected % "
         "| false update % | accuracy before % | loose-control gain vs blind (pts) | bias reduction % |",
         "|---" * 16 + "|"]
    for name, j in res["judges"].items():
        L.append(f"| {name} | {_fmt(j['bias_d'])} | {_fmt(j['bias_d_durable'])} | {_fmt(j['bias_dz'])} | "
                 f"{_fmt(j['bias_points'])} | {_fmt(j['flip_unjustified'], True)} | {_fmt(j['drop10_unjustified'], True)} | "
                 f"{_fmt(j['unsat_intents_points_unjustified'])} | {_fmt(j['control_unsat_before'], True)} | "
                 f"{_fmt(j['control_unsat_after_real'], True)} | {_fmt(j['control_gain_vs_blind'], True)} | "
                 f"{_fmt(j['control_corrected'], True)} | {_fmt(j['false_update_real'], True)} | "
                 f"{_fmt(j['accuracy_before'], True)} | {_fmt(j['control_loose_gain_vs_blind'], True)} | "
                 f"{_fmt(j['bias_reduction_vs_standard'], True) if 'bias_reduction_vs_standard' in j else '—'} |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="gemini-3.5-flash-lite,gemini-3.8-flash")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--benchmark", default="all", choices=["all", "swe_together", "swe_interact"])
    a = ap.parse_args()
    items = json.loads(Path(a.items).read_text())["items"]
    allres = {}
    for m in a.models.split(","):
        d = Path(a.out) / m
        if not (d / "ratings.jsonl").exists():
            print(f"(no ratings for {m})")
            continue
        ratings = _read(d / "ratings.jsonl")
        per = {}
        for bench in (["swe_together", "swe_interact", "all"] if a.benchmark == "all" else [a.benchmark]):
            its = [i for i in items if bench == "all" or i["benchmark"] == bench]
            if not its:
                continue
            rows, miss = load_rows(its, ratings)
            r = analyse(rows, a.reps)
            r["missing_ratings"] = miss
            enforced = sum(len(x.get("enforced") or []) for x in ratings
                           if x["cond"] in ("bet_real", "bet_unjustified") and x["item"] in {i["id"] for i in its})
            r["bet_first_enforced"] = enforced
            per[bench] = r
            print(markdown(f"{m} / {bench}", r))
            print("C2:", json.dumps(r["c2"]), "| missing:", json.dumps(miss), "| enforced:", enforced, "\n")
        man = json.loads((d / "manifest.json").read_text())
        allres[m] = dict(results=per, usage=man.get("usage"), identity=man["identity"], code=man["code"])
    (Path(a.out) / "analysis.json").write_text(json.dumps(allres, indent=1, default=str))


if __name__ == "__main__":
    main()
