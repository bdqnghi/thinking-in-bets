#!/usr/bin/env python3
"""TIB gains split by task difficulty, with a leave-one-model-out (LOO) guard against regression to the mean.

Pure post-hoc analysis (no model runs). Re-runnable with fixed seeds:

    python3 scripts/analysis/difficulty_split.py            # ~1-2 min, writes results/analysis/difficulty_split.{json,md}
                                                            # and docs/figures/analysis/difficulty_*.png

Difficulty of task t for an evaluated cell (benchmark b, model m, arm a) is the mean **stock** success of t over the
stock cells listed in ``tib_data.DIFF_SOURCE[b]`` (never a TIB arm; each (source bench, model) cell is one vote;
ALE-Claw uses the continuous score, everything else binary pass):

* ``loo`` (primary): drop every cell of model m. The evaluated stock outcome never enters its own difficulty, so a
  task cannot look "hard" because the very stock run we compare against happened to fail.
* ``all`` (contrast): keep all models, including m's own stock run. This is the naive definition; it is biased
  towards "TIB helps on hard tasks, hurts on easy ones" by construction (regression to the mean). The size of that
  artefact is estimated with a within-task arm-label permutation null (labels swapped -> difficulty recomputed).

Bins: hard = d == 0 (stock solves on none of the other models), easy = d > 0.5 (most), medium = otherwise.
Per bin and cell: n tasks, stock / arm rate, delta (pp), W/T/L on (task, replicate) pairs, paired task bootstrap
95% CI (10,000 resamples) and a sign-flip permutation p. Pooled rows are task-weighted, bootstrap stratified by cell.
"""
from __future__ import annotations

import argparse
import json
import zlib
from collections import defaultdict

import numpy as np

import tib_data as T

BINS = ("easy", "medium", "hard")
N_BOOT = 10_000
N_NULL = 1_000


def rng_for(*key) -> np.random.Generator:
    return np.random.default_rng(T.SEED + zlib.crc32("|".join(map(str, key)).encode()))


def boot_ci(x: np.ndarray, rng, n_boot=N_BOOT):
    if len(x) == 0:
        return None
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def signflip_p(x: np.ndarray, rng, n=20_000):
    x = x[x != 0]
    if len(x) == 0:
        return 1.0
    obs = abs(x.mean())
    if len(x) <= 16:  # exact
        signs = ((np.arange(2 ** len(x))[:, None] >> np.arange(len(x))) & 1) * 2 - 1
        return float(np.mean(np.abs((signs * x).mean(1)) >= obs - 1e-12))
    s = rng.choice([-1, 1], (n, len(x)))
    return float((np.sum(np.abs((s * x).mean(1)) >= obs - 1e-12) + 1) / (n + 1))


def task_values(cell, outcome):
    """Per task (stock mean, arm mean) over replicates for outcome 'y' or 'score'."""
    ks, ka = ("stock", "arm") if outcome == "y" else ("stock_s", "arm_s")
    tasks = sorted(cell)
    s = np.array([T.mean(cell[t][ks]) if T.mean(cell[t][ks]) is not None else np.nan for t in tasks], float)
    a = np.array([T.mean(cell[t][ka]) if T.mean(cell[t][ka]) is not None else np.nan for t in tasks], float)
    return tasks, s, a


def wtl(cell, tasks, outcome):
    key = "pairs" if outcome == "y" else "pairs_s"
    w = t_ = l = 0
    for t in tasks:
        for s, a in cell[t][key]:
            if s is None or a is None:
                continue
            if a > s + 1e-9:
                w += 1
            elif a < s - 1e-9:
                l += 1
            else:
                t_ += 1
    return [w, t_, l]


def diff_components(st, bench, task, model, mode):
    """(sum, count) of difficulty votes that do not depend on the evaluated cell's own stock outcome, and whether
    the evaluated cell's own stock outcome is a vote (mode=all and bench in its own DIFF_SOURCE)."""
    tot = cnt = 0.0
    own = False
    for sb in T.DIFF_SOURCE[bench]:
        for m, v in st.get(sb, {}).get(task, {}).items():
            if m == model:
                if mode == "loo":
                    continue
                if sb == bench:
                    own = True
                    continue
            tot += v
            cnt += 1
    return tot, cnt, own


def bins_from(base_sum, base_cnt, own_flag, own_val):
    d = np.where(own_flag, (base_sum + np.nan_to_num(own_val)) / np.maximum(base_cnt + 1, 1),
                 np.where(base_cnt > 0, base_sum / np.maximum(base_cnt, 1), np.nan))
    d = np.where((base_cnt + own_flag) > 0, d, np.nan)
    b = np.full(len(d), "", dtype=object)
    b[d == 0] = "hard"
    b[(d > 0) & (d <= 0.5)] = "medium"
    b[d > 0.5] = "easy"
    return d, b


def analyse_cell(key, cell, st, mode, outcome):
    bench, model, arm = key
    tasks, s, a = task_values(cell, outcome)
    ok = ~np.isnan(s) & ~np.isnan(a)
    comps = [diff_components(st, bench, t, model, mode) for t in tasks]
    base_sum = np.array([c[0] for c in comps]); base_cnt = np.array([c[1] for c in comps])
    own_flag = np.array([c[2] for c in comps])
    # own stock value as used in the difficulty table (stock mean over replicates; score for SCORE_DIFF)
    if bench in T.SCORE_DIFF:
        own_s = np.array([T.mean(cell[t]["stock_s"]) for t in tasks], float)
        own_a = np.array([T.mean(cell[t]["arm_s"]) for t in tasks], float)
    else:
        own_s = np.array([T.mean(cell[t]["stock"]) for t in tasks], float)
        own_a = np.array([T.mean(cell[t]["arm"]) for t in tasks], float)
    d, b = bins_from(base_sum, base_cnt, own_flag, own_s)
    rng = rng_for(key, mode, outcome)
    res = {}
    diff = a - s
    for bn in BINS:
        m = ok & (b == bn)
        x = diff[m]
        sel = [t for t, mm in zip(tasks, m) if mm]
        res[bn] = {"n": int(m.sum()), "stock": float(s[m].mean()) if m.any() else None,
                   "arm": float(a[m].mean()) if m.any() else None,
                   "delta": float(x.mean()) if m.any() else None,
                   "ci95": boot_ci(x, rng) if m.any() else None,
                   "p": signflip_p(x, rng) if m.any() else None,
                   "wtl": wtl(cell, sel, outcome)}
    # permutation null: swap arm labels within task, recompute bins (only matters when own stock is a vote)
    null = {bn: [] for bn in BINS}
    if own_flag.any():
        for _ in range(N_NULL):
            sw = rng.random(len(tasks)) < 0.5
            s2, a2 = np.where(sw, a, s), np.where(sw, s, a)
            own2 = np.where(sw, own_a, own_s)
            _, b2 = bins_from(base_sum, base_cnt, own_flag, own2)
            for bn in BINS:
                m = ok & (b2 == bn)
                if m.any():
                    null[bn].append(float((a2[m] - s2[m]).mean()))
        for bn in BINS:
            res[bn]["null_mean"] = float(np.mean(null[bn])) if null[bn] else None
    else:
        for bn in BINS:
            res[bn]["null_mean"] = 0.0  # LOO / external difficulty: swapping labels leaves bins unchanged
    res["_d"] = {t: (None if np.isnan(dd) else float(dd)) for t, dd in zip(tasks, d)}
    res["_diff"] = {t: (None if np.isnan(x) else float(x)) for t, x in zip(tasks, diff)}
    res["_bin"] = dict(zip(tasks, b.tolist()))
    return res


def pooled(cell_results, keys, mode, outcome, label):
    """Task-weighted pooled delta per bin over cells; bootstrap stratified by cell."""
    out = {}
    for bn in BINS:
        xs = []
        for k in keys:
            r = cell_results[(k, mode, outcome)]
            x = np.array([r["_diff"][t] for t in r["_bin"] if r["_bin"][t] == bn and r["_diff"][t] is not None])
            if len(x):
                xs.append(x)
        if not xs:
            out[bn] = {"n": 0}
            continue
        n = sum(len(x) for x in xs)
        rng = rng_for("pooled", label, mode, outcome, bn)
        boots = np.zeros(N_BOOT)
        for x in xs:
            idx = rng.integers(0, len(x), (N_BOOT, len(x)))
            boots += x[idx].sum(1)
        boots /= n
        allx = np.concatenate(xs)
        w = sum(cell_results[(k, mode, outcome)][bn]["wtl"][0] for k in keys)
        ti = sum(cell_results[(k, mode, outcome)][bn]["wtl"][1] for k in keys)
        lo = sum(cell_results[(k, mode, outcome)][bn]["wtl"][2] for k in keys)
        nm = [cell_results[(k, mode, outcome)][bn].get("null_mean") for k in keys]
        nn = [cell_results[(k, mode, outcome)][bn]["n"] for k in keys]
        null = sum((v or 0) * c for v, c in zip(nm, nn)) / max(sum(nn), 1)
        out[bn] = {"n": int(n), "n_cells": len(xs), "delta": float(allx.mean()),
                   "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                   "p": signflip_p(allx, rng), "wtl": [w, ti, lo], "null_mean": float(null)}
    return out


def fmt(r, scale=100, nd=1):
    if not r or not r.get("n"):
        return "—"
    ci = r.get("ci95")
    star = " *" if ci and (ci[0] > 0 or ci[1] < 0) else ""
    s = f"{scale * r['delta']:+.{nd}f} [{scale * ci[0]:+.{nd}f}, {scale * ci[1]:+.{nd}f}]{star}" if ci else f"{scale * r['delta']:+.{nd}f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    recs = T.load_all()
    st = T.stock_table(recs)
    cells = T.paired_cells(recs)
    R = {}
    for key, cell in cells.items():
        for mode in ("loo", "all"):
            for outcome in ("y", "score"):
                if outcome == "score" and key[0] not in T.SCORE_BENCH:
                    continue
                R[(key, mode, outcome)] = analyse_cell(key, cell, st, mode, outcome)

    ours_keys = sorted(k for k in cells if k[2] in T.OURS)
    base_keys = sorted(k for k in cells if k[2] not in T.OURS)
    variants = {"cb_draftfirst": [k for k in ours_keys if k[2] == "cb_draftfirst"],
                "commit_before": [k for k in ours_keys if k[2] == "commit_before"],
                "cb_gated": [k for k in ours_keys if k[2] == "cb_gated"],
                "all_ours": ours_keys,
                "all_ours_excl_hle_ss": [k for k in ours_keys if k[0] != "hle_ss"],
                "baselines (claims_after, claims_no_p, conf_gate, sc_vote3, verify_then_respond)": base_keys}
    P = {}
    for mode in ("loo", "all"):
        for v, ks in variants.items():
            P[(v, mode)] = pooled(R, ks, mode, "y", v)
        # by model strength, all ours
        for m in T.RANK:
            ks = [k for k in ours_keys if k[1] == m]
            if ks:
                P[(f"all_ours | model {m}", mode)] = pooled(R, ks, mode, "y", f"m{m}")
            ks = [k for k in base_keys if k[1] == m]
            if ks:
                P[(f"baselines | model {m}", mode)] = pooled(R, ks, mode, "y", f"bm{m}")
        for v in ("cb_draftfirst", "commit_before", "cb_gated"):
            for m in T.RANK:
                ks = [k for k in variants[v] if k[1] == m]
                if ks:
                    P[(f"{v} | model {m}", mode)] = pooled(R, ks, mode, "y", f"{v}{m}")
    # "truly hard": no stock success on ANY model (incl. own) in any difficulty source; does TIB ever solve them?
    unlock = []
    for k in ours_keys:
        r = R[(k, "all", "y")]
        bench, model, arm = k
        n_hard = n_solved = 0
        solved = []
        for t, dd in r["_d"].items():
            # all-model difficulty is 0 and the evaluated stock failed -> no stock run anywhere solved it
            same_bench_stock = st.get(bench, {}).get(t, {})
            if dd == 0 and all(v == 0 for v in same_bench_stock.values()):
                n_hard += 1
                a = T.mean(cells[k][t]["arm"])
                if a and a > 0:
                    n_solved += 1
                    solved.append(t)
        unlock.append({"bench": bench, "model": model, "arm": arm, "n_never_solved_by_stock": n_hard,
                       "tib_solves": n_solved, "tasks": solved[:20]})

    # push-type split for the 2-turn pushback benchmarks (difficulty is confounded with push type there:
    # hard questions are mostly wrong at turn 1 -> constructive hint; easy ones right -> false pushback)
    ptype = {(r["bench"], r["model"], r["task"]): r.get("push_type") for r in recs if r["bench"] in ("hle_mt", "fin_mt")}
    push = []
    for bench in ("hle_mt", "fin_mt"):
        for arm in ("commit_before", "claims_after", "claims_no_p"):
            for pt in ("false_pushback", "constructive_hint"):
                for bn in BINS:
                    xs, w, ti, lo = [], 0, 0, 0
                    for k in cells:
                        if k[0] != bench or k[2] != arm:
                            continue
                        r = R[(k, "loo", "y")]
                        for t in r["_bin"]:
                            if r["_bin"][t] == bn and ptype.get((bench, k[1], t)) == pt and r["_diff"][t] is not None:
                                xs.append(r["_diff"][t])
                                w += r["_diff"][t] > 0; lo += r["_diff"][t] < 0; ti += r["_diff"][t] == 0
                    x = np.array(xs)
                    rng = rng_for("push", bench, arm, pt, bn)
                    push.append({"bench": bench, "arm": arm, "push_type": pt, "bin": bn, "n": int(len(x)),
                                 "delta": float(x.mean()) if len(x) else None, "ci95": boot_ci(x, rng) if len(x) else None,
                                 "wtl": [int(w), int(ti), int(lo)]})

    # ── JSON ──
    def clean(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}
    js = {"meta": {"seed": T.SEED, "n_boot": N_BOOT, "n_null": N_NULL,
                   "bins": "hard: d==0; medium: 0<d<=0.5; easy: d>0.5 (d = mean stock success over difficulty-source cells)",
                   "difficulty_source": T.DIFF_SOURCE},
          "cells": [{"bench": k[0], "model": k[1], "arm": k[2], "mode": mode, "outcome": oc,
                     "ours": k[2] in T.OURS, "bins": {bn: clean(R[(k, mode, oc)][bn]) for bn in BINS}}
                    for (k, mode, oc) in sorted(R)],
          "pooled": [{"group": g, "mode": mode, "bins": P[(g, mode)]} for (g, mode) in P],
          "unlock_never_solved": unlock, "push_type_split": push,
          "task_difficulty_loo": {f"{k[0]}|{k[1]}|{k[2]}": R[(k, "loo", "y")]["_d"] for k in ours_keys}}
    (T.OUT / "difficulty_split.json").write_text(json.dumps(js, indent=1))

    # ── Markdown ──
    L = ["# TIB vs stock by task difficulty (auto-generated by `scripts/analysis/difficulty_split.py`)", "",
         f"Seed {T.SEED}; {N_BOOT:,} paired task bootstrap resamples; permutation null {N_NULL} label swaps. "
         "Δ in percentage points (TIB − stock), `*` = 95% CI excludes 0. Bins: hard = stock solves on none of the "
         "difficulty-source cells, easy = on more than half, medium = in between. `null` = expected Δ in that bin "
         "under no TIB effect (non-zero only for the naive `all` definition: regression-to-the-mean artefact).", ""]
    for mode in ("loo", "all"):
        L += [f"## Pooled ({'leave-one-model-out, primary' if mode == 'loo' else 'all models incl. own stock, biased contrast'})", "",
              "| group | easy Δ [CI] (n; W/T/L; null) | medium Δ [CI] (n; W/T/L; null) | hard Δ [CI] (n; W/T/L; null) |",
              "|---|---|---|---|"]
        for (g, md), pr in P.items():
            if md != mode:
                continue
            row = [g]
            for bn in BINS:
                r = pr[bn]
                row.append("—" if not r.get("n") else f"{fmt(r)} (n={r['n']}; {r['wtl'][0]}/{r['wtl'][1]}/{r['wtl'][2]}; null {100 * r['null_mean']:+.1f})")
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    L += ["## Per cell, leave-one-model-out difficulty (binary pass; score rows for ALE / Finance)", "",
          "| bench | model | arm | outcome | easy: stock→arm, Δ [CI] (n; W/T/L) | medium | hard |", "|---|---|---|---|---|---|---|"]
    for (k, mode, oc) in sorted(R):
        if mode != "loo":
            continue
        row = [T.BENCH_LABEL[k[0]], T.MODEL_LABEL[k[1]], ("**" + k[2] + "**") if k[2] in T.OURS else k[2], "pass" if oc == "y" else "score"]
        for bn in BINS:
            r = R[(k, mode, oc)][bn]
            row.append("—" if not r["n"] else f"{100 * r['stock']:.0f}→{100 * r['arm']:.0f}, {fmt(r)} (n={r['n']}; {r['wtl'][0]}/{r['wtl'][1]}/{r['wtl'][2]})")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## Tasks no stock run solved on any model (all-model d = 0 and no stock success on this benchmark): does TIB unlock them?", "",
          "Caveat: the evaluated stock run is part of the definition, so some of these 'unlocks' are single-sample noise; "
          "the unbiased version of this question is the LOO hard-bin row above.", "",
          "| bench | model | arm | never solved by stock | TIB solves |", "|---|---|---|---|---|"]
    for u in unlock:
        L.append(f"| {T.BENCH_LABEL[u['bench']]} | {T.MODEL_LABEL[u['model']]} | {u['arm']} | {u['n_never_solved_by_stock']} | {u['tib_solves']} |")
    L += ["", "## 2-turn pushback benchmarks split by push type (LOO difficulty, pooled over models)", "",
          "| bench | arm | push type | easy Δ [CI] (n; W/T/L) | medium | hard |", "|---|---|---|---|---|---|"]
    for bench in ("hle_mt", "fin_mt"):
        for arm in ("commit_before", "claims_after", "claims_no_p"):
            for pt in ("false_pushback", "constructive_hint"):
                row = [T.BENCH_LABEL[bench], arm, pt]
                for bn in BINS:
                    p = next(p for p in push if p["bench"] == bench and p["arm"] == arm and p["push_type"] == pt and p["bin"] == bn)
                    row.append("—" if not p["n"] else f"{fmt(p)} (n={p['n']}; {p['wtl'][0]}/{p['wtl'][1]}/{p['wtl'][2]})")
                L.append("| " + " | ".join(row) + " |")
    (T.OUT / "difficulty_split.md").write_text("\n".join(L) + "\n")

    if not args.no_fig:
        figures(R, P, cells, ours_keys)
    # console summary
    for mode in ("loo", "all"):
        print(f"== pooled {mode}")
        for (g, md), pr in P.items():
            if md == mode and ("|" not in g or g.startswith("all_ours")):
                print(f"  {g:40s} " + "  ".join(f"{bn}: {fmt(pr[bn])} n={pr[bn].get('n', 0)} null={100 * pr[bn].get('null_mean', 0):+.1f}" for bn in BINS))


def figures(R, P, cells, ours_keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T.FIG.mkdir(parents=True, exist_ok=True)
    # Fig 1: per-bench x model, LOO, ours
    benches = sorted({k[0] for k in ours_keys}, key=list(T.BENCH_LABEL).index)
    ncol = 4
    nrow = int(np.ceil(len(benches) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    colors = {"W": "#9e9e9e", "F": "#64b5f6", "M": "#4db6ac", "S": "#1565c0", "G": "#c62828"}
    for ax, b in zip(axes.flat, benches):
        ks = [k for k in ours_keys if k[0] == b]
        width = 0.8 / max(len(ks), 1)
        for i, k in enumerate(sorted(ks, key=lambda k: (T.RANK[k[1]], k[2]))):
            oc = "score" if b in ("ale_t300", "ale_s1") else "y"
            r = R[(k, "loo", oc)]
            for j, bn in enumerate(BINS):
                rr = r[bn]
                if not rr["n"]:
                    continue
                x = j + (i - (len(ks) - 1) / 2) * width
                ax.bar(x, 100 * rr["delta"], width * 0.9, color=colors[k[1]], alpha=0.85 if k[2] != "cb_gated" else 0.5,
                       label=f"{T.MODEL_LABEL[k[1]]} {k[2]}" if j == 0 else None)
                ax.errorbar(x, 100 * rr["delta"], yerr=[[100 * (rr["delta"] - rr["ci95"][0])], [100 * (rr["ci95"][1] - rr["delta"])]],
                            color="k", lw=0.7, capsize=1.5)
                ax.text(x, ax.get_ylim()[0] if False else 0, f"{rr['n']}", ha="center", va="bottom", fontsize=5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(3)); ax.set_xticklabels(BINS)
        ax.set_title(T.BENCH_LABEL[b] + (" (score)" if b.startswith("ale_d") or b in ("ale_t300", "ale_s1") else ""), fontsize=8)
        ax.set_ylabel("Δ pp (TIB − stock)", fontsize=7)
        ax.legend(fontsize=5, loc="best")
    for ax in list(axes.flat)[len(benches):]:
        ax.axis("off")
    fig.suptitle("TIB − stock by leave-one-model-out task difficulty (bar label = n tasks; 95% paired bootstrap CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(T.FIG / "difficulty_by_bench_loo.png", dpi=150)
    plt.close(fig)
    # Fig 2: pooled LOO vs all (RTM artefact)
    groups = ["cb_draftfirst", "commit_before", "cb_gated", "all_ours"]
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.2), sharey=True)
    for ax, g in zip(axes, groups):
        for j, bn in enumerate(BINS):
            for o, (mode, c) in enumerate((("loo", "#1565c0"), ("all", "#ef6c00"))):
                r = P[(g, mode)][bn]
                if not r.get("n"):
                    continue
                x = j + (o - 0.5) * 0.35
                ax.bar(x, 100 * r["delta"], 0.33, color=c, label=("LOO (primary)" if mode == "loo" else "all-model (naive)") if j == 0 else None)
                ax.errorbar(x, 100 * r["delta"], yerr=[[100 * (r["delta"] - r["ci95"][0])], [100 * (r["ci95"][1] - r["delta"])]], color="k", lw=0.7, capsize=2)
                if mode == "all":
                    ax.plot([x - 0.16, x + 0.16], [100 * r["null_mean"]] * 2, color="k", ls=":", lw=1.2,
                            label="permutation null (RTM artefact)" if j == 0 else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(3)); ax.set_xticklabels(BINS)
        ax.set_title(g, fontsize=9)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("pooled Δ pass (pp)")
    fig.suptitle("Pooled TIB − stock pass by difficulty: leave-one-model-out vs naive all-model bins", fontsize=10)
    fig.tight_layout()
    fig.savefig(T.FIG / "difficulty_pooled_loo_vs_all.png", dpi=150)
    plt.close(fig)
    # Fig 3: ours vs baselines by model strength x bin (LOO)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, bn in zip(axes, BINS):
        for o, (pref, c) in enumerate((("all_ours", "#1565c0"), ("baselines", "#757575"))):
            xs, ys, lo, hi = [], [], [], []
            for m in T.RANK:
                r = P.get((f"{pref} | model {m}", "loo"), {}).get(bn)
                if r and r.get("n"):
                    xs.append(T.RANK[m] + (o - 0.5) * 0.15); ys.append(100 * r["delta"])
                    lo.append(100 * (r["delta"] - r["ci95"][0])); hi.append(100 * (r["ci95"][1] - r["delta"]))
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=c, capsize=2, label=pref)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(list(T.RANK.values())); ax.set_xticklabels([T.MODEL_LABEL[m].replace(" ", "\n") for m in T.RANK], fontsize=7)
        ax.set_title(f"{bn} tasks (LOO)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("pooled Δ pass vs stock (pp)")
    fig.tight_layout()
    fig.savefig(T.FIG / "difficulty_by_model_strength.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
