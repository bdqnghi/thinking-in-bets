"""Analyse one E1 run directory per model and emit the pre-registered metrics and C2 verdict.

Usage: python -m fsa.redo.e1.analyze --phase main [--models gemini-3.5-flash-lite,gemini-3.8-flash] [--reps 10000]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import metrics as M
from .prompts import ARMS
from .run import BASE, _read

C2_D_MIN = 0.3
C2_REDUCTION_MIN = 0.5
C2_UPDATE_GAIN_MIN = 0.20


def load(phase: str, model: str):
    body = json.loads((BASE / "items.json").read_text())
    items = {i["id"]: i for i in body["items"]}
    out = BASE / phase / model
    recs = _read(out / "ratings.jsonl")
    # the analysis uses the final rating per arm: bet_first -> stage 2; everything else stage 1
    final: dict[str, dict[str, int | None]] = {a: {} for a in ARMS}
    bets: dict[str, int | None] = {}
    revised: dict[str, bool | None] = {}
    units: dict[str, list] = {a: [] for a in ARMS}
    for r in recs:
        if r["arm"] == "bet_first" and r["stage"] == 1:
            for iid in r["item_ids"]:
                bets[iid] = r["rating"]
            units["bet_first_stage1"] = units.get("bet_first_stage1", []) + [r["rating"]]
            continue
        units[r["arm"]].append(r["rating"])
        for iid in r["item_ids"]:
            final[r["arm"]][iid] = r["rating"]
            if r["arm"] == "bet_first":
                revised[iid] = None if r.get("parsed") is None else bool(r["parsed"]["revise"])
    return body, items, final, bets, revised, units, out


def tables(items: dict, ratings: dict):
    main, ctrl, ref = {}, {}, {}
    for iid, it in items.items():
        v = ratings.get(iid)
        if it["kind"] == "main":
            main[(it["scenario_id"], it["ev_cell"], it["outcome"])] = v
        else:
            ctrl.setdefault(it["pair_id"], {})[it["kind"]] = v
    return main, ctrl


def _fmt(b: dict, pct: bool = False) -> str:
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.2f}")
    if any(isinstance(b[k], float) and (math.isnan(b[k]) or math.isinf(b[k])) for k in ("point", "lo", "hi")):
        return f"{b['point']} [n/a]"
    return f"{f(b['point'])} [{f(b['lo'])}, {f(b['hi'])}]"


def analyse(phase: str, model: str, reps: int) -> dict:
    body, items, final, bets, revised, units, out = load(phase, model)
    sids = sorted({i["scenario_id"] for i in items.values() if i["kind"] == "main"})
    pids = sorted({i["pair_id"] for i in items.values() if i["kind"] != "main"})
    T = {a: tables(items, final[a]) for a in ARMS}
    T["bet_first_stage1"] = tables(items, bets)
    blind_ref = {p: T["outcome_blind"][1][p].get("control") for p in pids}
    res: dict = dict(model=model, phase=phase, items_sha256=body["items_sha256"], arms={})
    for a in list(ARMS) + ["bet_first_stage1"]:
        main, ctrl = T[a]
        n_items = sum(1 for i in items if i in final.get(a, bets))
        miss_items = sum(1 for v in (final[a] if a in final else bets).values() if v is None)
        u = units.get(a, [])
        cv = M.cell_values(main, sids)
        res["arms"][a] = dict(
            missing=dict(units=len(u), units_missing=sum(1 for x in u if x is None), items=n_items, items_missing=miss_items),
            cell_means={f"{e}/{o}": M._mean(cv[(e, o)]) for e, o in M.CELLS},
            cohen_d=M.bootstrap(lambda s, m=main: M.cohen_d(m, s), sids, reps),
            d_z=M.bootstrap(lambda s, m=main: M.d_z(m, s), sids, reps),
            luck_gap_points=M.bootstrap(lambda s, m=main: M.mean_luck_gap(m, s), sids, reps),
            auc_within_outcome=M.bootstrap(lambda s, m=main: M.within_outcome_auc(m, s), sids, reps),
            luck_reversal=M.bootstrap(lambda s, m=main: M.luck_reversal(m, s), sids, reps),
            correct_update=M.bootstrap(lambda p, c=ctrl: M.correct_update_rate(c, p), pids, reps),
            control_drop_vs_blind=M.bootstrap(lambda p, c=ctrl: M.drop_vs_reference(c, blind_ref, p, "control"), pids, reps),
            twin_drop_vs_blind=M.bootstrap(lambda p, c=ctrl: M.drop_vs_reference(c, blind_ref, p, "twin"), pids, reps),
        )
    std_main = T["standard"][0]
    for a in ARMS:
        if a == "standard":
            continue
        am = T[a][0]
        res["arms"][a]["bias_reduction_vs_standard"] = M.bootstrap(
            lambda s, am=am: (1 - M.cohen_d(am, s) / M.cohen_d(std_main, s)) if M.cohen_d(std_main, s) > 0 else float("nan"),
            sids, reps)
    blind_ctrl = T["outcome_blind"][1]
    for a in ARMS:
        ac = T[a][1]
        res["arms"][a]["update_gain_vs_blind"] = M.bootstrap(
            lambda p, ac=ac: M.correct_update_rate(ac, p) - M.correct_update_rate(blind_ctrl, p), pids, reps)
    # bet-first revision behaviour by item kind
    rev = {}
    for kind in ("main", "control", "twin"):
        xs = [revised[i] for i, it in items.items() if it["kind"] == kind and revised.get(i) is not None]
        rev[kind] = dict(n=len(xs), revised=sum(xs), rate=M._mean([float(x) for x in xs]))
    res["bet_first_revisions"] = rev
    enforced = [r for r in _read(out / "ratings.jsonl") if r["arm"] == "bet_first" and r["stage"] == 2 and r.get("enforced")]
    res["bet_first_enforced_no_revise"] = len(enforced)
    # C2 verdict
    d = res["arms"]["standard"]["cohen_d"]
    red = res["arms"]["bet_first"]["bias_reduction_vs_standard"]
    gain = res["arms"]["bet_first"]["update_gain_vs_blind"]
    c2a = d["point"] >= C2_D_MIN and d["lo"] > 0
    c2b = (not math.isnan(red["point"])) and red["point"] >= C2_REDUCTION_MIN
    c2c = gain["point"] >= C2_UPDATE_GAIN_MIN
    res["c2"] = dict(a_standard_bias=c2a, b_betfirst_removes_half=c2b, c_betfirst_update_gain=c2c,
                     supported=bool(c2a and c2b and c2c),
                     strong=dict(a=d["lo"] >= C2_D_MIN, b=red["lo"] >= C2_REDUCTION_MIN, c=gain["lo"] >= C2_UPDATE_GAIN_MIN))
    manifest = json.loads((out / "manifest.json").read_text())
    res["usage"] = manifest.get("usage")
    res["manifest"] = dict(identity=manifest["identity"], code=manifest["code"])
    return res


def markdown(res: dict) -> str:
    lines = [f"### {res['model']}", "",
             "| arm | Cohen's d (lucky−unlucky) | luck gap (pts) | within-outcome AUC | luck reversal % | correct update % | update gain vs blind (pts) | bias reduction vs standard % | missing units |",
             "|---|---|---|---|---|---|---|---|---|"]
    for a, r in res["arms"].items():
        red = r.get("bias_reduction_vs_standard")
        gain = r.get("update_gain_vs_blind")
        lines.append(f"| {a} | {_fmt(r['cohen_d'])} | {_fmt(r['luck_gap_points'])} | {_fmt(r['auc_within_outcome'])} | "
                     f"{_fmt(r['luck_reversal'], True)} | {_fmt(r['correct_update'], True)} | "
                     f"{_fmt(gain, True) if gain else '—'} | {_fmt(red, True) if red else '—'} | "
                     f"{r['missing']['units_missing']}/{r['missing']['units']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="main")
    ap.add_argument("--models", default="gemini-3.5-flash-lite,gemini-3.8-flash")
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args()
    allres = {}
    for m in a.models.split(","):
        if not (BASE / a.phase / m / "ratings.jsonl").exists():
            print(f"(no ratings for {m})")
            continue
        r = analyse(a.phase, m, a.reps)
        allres[m] = r
        print(markdown(r))
        print("C2:", json.dumps(r["c2"]), "| bet-first revisions:", json.dumps(r["bet_first_revisions"]), "\n")
    (BASE / a.phase / "analysis.json").write_text(json.dumps(allres, indent=1, default=str))


if __name__ == "__main__":
    main()
