#!/usr/bin/env python3
"""One model for the capability threshold: does the TIB effect grow with model capability, and where does it cross 0?

Pure post-hoc analysis (no model runs). Needs statsmodels, which is not installed in the system python; run with
the scratch venv (``python3 -m venv --system-site-packages <dir> && <dir>/bin/pip install statsmodels``):

    VENV=.venv
    $VENV/bin/python scripts/analysis/capability_model.py         # ~5-10 min; results/analysis/capability_model.{json,md}

Data: every paired (benchmark, model, task, replicate) outcome of a TIB arm and its stock partner (``tib_data``).
Outcome y = binary pass (ALE-Claw: score == 1; HLE / Finance multi-turn: turn-2 correct). Treatment ``tib`` = 1 for
the TIB arm. Fits, separately for each TIB variant (cb_draftfirst / commit_before / cb_gated) and pooled:

1. Bayesian mixed logistic (statsmodels ``BinomialBayesMixedGLM``, variational Bayes):
   (a) requested spec   y ~ tib * cap + (1|task) + (1|benchmark)
   (b) extended spec    y ~ tib * cap + (1|task) + (1|benchmark:model) + (0 + tib|benchmark)
       (b) lets the TIB effect differ by benchmark, so the interaction is not driven by one benchmark's offset.
2. GEE logistic, exchangeable working correlation within task, robust SEs:
   y ~ C(benchmark:model) + tib + tib:cap   [(+ tib:difficulty)]
3. Hierarchical bootstrap of the fixed-effects logistic in 2 (resample benchmarks, then tasks within benchmark;
   and a task-only bootstrap with benchmarks fixed) -> percentile CIs for the interaction and the crossover
   capability cap* = -b_tib / b_tib:cap.

Capability measures ``cap``: ``rank`` (W=1, F=2, M=3, S=4, G=5; centred at 3) and ``stockpass`` (stock pass rate of
that model on that benchmark, centred at 0.5; mildly circular because stock outcomes are also on the left-hand side).
Difficulty x arm: task difficulty = leave-one-model-out stock success (``tib_data.difficulty``), centred at 0.5.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

import tib_data as T

warnings.filterwarnings("ignore")
VARIANTS = ("cb_draftfirst", "commit_before", "cb_gated", "pooled", "cb_draftfirst_noHLE", "pooled_noHLE")
BAYES_MAX_ROWS = 3500  # statsmodels VB: ~1 min at 2.1k rows, >40 min per fit at >=5.3k rows (not run there)


def build_frame() -> pd.DataFrame:
    recs = T.load_all()
    st = T.stock_table(recs)
    cells = T.paired_cells(recs)
    # stock pass rate of each (bench, model) over all its stock records
    acc = defaultdict(list)
    for r in recs:
        if r["arm"] == "stock":
            acc[(r["bench"], r["model"])].append(r["y"])
    sp = {k: float(np.mean(v)) for k, v in acc.items()}
    rows, seen_stock = [], set()
    for (b, m, arm), cell in cells.items():
        if arm not in T.OURS:
            continue
        for t, v in cell.items():
            d, _ = T.difficulty(st, b, t, m, "loo")
            base = {"bench": b, "model": m, "task": f"{b}|{t}", "rank": T.RANK[m], "stockpass": sp[(b, m)],
                    "d_loo": d}
            for rep_i, y in zip(v["reps_s"], v["stock"]):
                rows.append({**base, "arm": "stock", "variant": arm, "tib": 0, "y": y, "rep": rep_i})
            for rep_i, y in zip(v["reps_a"], v["arm"]):
                rows.append({**base, "arm": arm, "variant": arm, "tib": 1, "y": y, "rep": rep_i})
    df = pd.DataFrame(rows)
    df["rank_c"] = df["rank"] - 3.0
    df["stockpass_c"] = df["stockpass"] - 0.5
    df["d_c"] = df["d_loo"] - 0.5
    df["bench_model"] = df["bench"] + ":" + df["model"]
    return df


def frame_for(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant.endswith("_noHLE"):
        d = frame_for(df, variant[:-len("_noHLE")])
        return d[d["bench"] != "hle_ss"].copy()
    if variant != "pooled":
        return df[df["variant"] == variant].copy()
    # pooled: every ours arm with tib=1; stock rows de-duplicated across variants sharing a cell (SWI mixed)
    s = df[df["tib"] == 0].drop_duplicates(["bench", "model", "task", "rep"])
    return pd.concat([s, df[df["tib"] == 1]], ignore_index=True)


# ── fast fixed-effects logistic (IRLS) ─────────────────────────────────────

def logit_irls(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, iters=50, ridge=1e-6):
    n, p = X.shape
    w = np.ones(n) if w is None else w
    beta = np.zeros(p)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = w * mu * (1 - mu) + 1e-10
        z = eta + (y - mu) / (mu * (1 - mu) + 1e-10)
        A = X.T @ (W[:, None] * X) + ridge * np.eye(p)
        nb = np.linalg.solve(A, X.T @ (W * z))
        if np.max(np.abs(nb - beta)) < 1e-8:
            beta = nb
            break
        beta = nb
    return beta


def design(d: pd.DataFrame, cap: str, with_diff: bool):
    bm = pd.get_dummies(d["bench_model"], dtype=float)
    cols = [bm.values, d[["tib"]].values.astype(float), (d["tib"] * d[cap]).values[:, None]]
    names = list(bm.columns) + ["tib", f"tib:{cap}"]
    if with_diff:
        dd = d["d_c"].fillna(0).values
        cols += [dd[:, None], (d["tib"].values * dd)[:, None]]
        names += ["d_c", "tib:d_c"]
    return np.hstack(cols), names


def hier_boot(d: pd.DataFrame, cap: str, with_diff: bool, n_boot: int, levels: str, seed: int):
    """levels='bench_task': resample benchmarks then tasks within; 'task': resample tasks within each benchmark."""
    rng = np.random.default_rng(seed)
    d = d.reset_index(drop=True)
    benches = sorted(d["bench"].unique())
    tasks_by_b = {b: sorted(d.loc[d["bench"] == b, "task"].unique()) for b in benches}
    idx_by_task = {t: np.flatnonzero(d["task"].values == t) for t in d["task"].unique()}
    X, names = design(d, cap, with_diff)
    y = d["y"].values.astype(float)
    ji, jx = names.index("tib"), names.index(f"tib:{cap}")
    jd = names.index("tib:d_c") if with_diff else None
    full = logit_irls(X, y)
    out = []
    for _ in range(n_boot):
        bs = rng.choice(benches, len(benches), replace=True) if levels == "bench_task" else benches
        w = np.zeros(len(d))
        for b in bs:
            ts = tasks_by_b[b]
            for t in rng.choice(ts, len(ts), replace=True):
                w[idx_by_task[t]] += 1
        keep = w > 0
        # columns of bench_model dummies absent from the resample are dropped
        Xk = X[keep]
        colok = (np.abs(Xk).sum(0) > 0)
        if not colok[jx]:
            continue
        bk = logit_irls(Xk[:, colok], y[keep], w[keep])
        full_b = np.full(X.shape[1], np.nan)
        full_b[colok] = bk
        out.append((full_b[ji], full_b[jx], full_b[jd] if jd is not None else np.nan))
    out = np.array(out)
    res = {"point": {"tib": float(full[ji]), "interaction": float(full[jx])}, "n_boot": int(len(out))}
    if with_diff:
        res["point"]["tib:d_c"] = float(full[jd])
    def ci(v):
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    res["ci_tib"] = ci(out[:, 0])
    res["ci_interaction"] = ci(out[:, 1])
    res["p_interaction_le0"] = float(np.mean(out[:, 1] <= 0))
    if with_diff:
        res["ci_tib:d_c"] = ci(out[:, 2])
        res["p_tib:d_c_le0"] = float(np.mean(out[:, 2] <= 0))
    cross = -out[:, 0] / out[:, 1]
    sgn = np.sign(full[jx])
    same = np.sign(out[:, 1]) == sgn
    res["crossover_point"] = float(-full[ji] / full[jx]) if full[jx] != 0 else None
    # percentile CI of the crossover over resamples whose interaction has the same sign as the point estimate
    res["crossover_ci_same_sign"] = ci(cross[same]) if same.sum() > 20 else None
    res["frac_same_sign"] = float(same.mean())
    res["frac_positive_slope"] = float((out[:, 1] > 0).mean())
    return res


def gee_fit(d: pd.DataFrame, cap: str, with_diff: bool):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    f = f"y ~ C(bench_model) + tib + tib:{cap}" + (" + d_c + tib:d_c" if with_diff else "")
    dd = d.copy()
    dd["d_c"] = dd["d_c"].fillna(0)
    dd = dd.sort_values("task")
    try:
        m = smf.gee(f, groups="task", data=dd, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    out = {}
    for k in ["tib", f"tib:{cap}"] + (["tib:d_c"] if with_diff else []):
        out[k] = {"coef": float(m.params[k]), "se": float(m.bse[k]), "p": float(m.pvalues[k]),
                  "ci95": [float(m.conf_int().loc[k, 0]), float(m.conf_int().loc[k, 1])]}
    b0, b1 = m.params["tib"], m.params[f"tib:{cap}"]
    out["crossover_point"] = float(-b0 / b1) if b1 != 0 else None
    # delta-method SE for crossover
    V = m.cov_params().loc[["tib", f"tib:{cap}"], ["tib", f"tib:{cap}"]].values
    g = np.array([-1 / b1, b0 / b1 ** 2])
    se = float(np.sqrt(g @ V @ g))
    out["crossover_delta_ci95"] = [out["crossover_point"] - 1.96 * se, out["crossover_point"] + 1.96 * se]
    out["dep_param"] = float(m.cov_struct.dep_params)
    return out


def bayes_fit(d: pd.DataFrame, cap: str, spec: str, with_diff: bool = False):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    dd = d.copy()
    dd["d_c"] = dd["d_c"].fillna(0)
    f = f"y ~ tib * {cap}" + (" + d_c + tib:d_c" if with_diff else "")
    vc = {"task": "0 + C(task)"}
    nb = dd["bench"].nunique()
    if spec == "a":
        if nb > 1:
            vc["bench"] = "0 + C(bench)"
    else:
        vc["bench_model"] = "0 + C(bench_model)"
        if nb > 1:
            vc["tib_bench"] = "0 + C(bench):tib"
    try:
        mod = BinomialBayesMixedGLM.from_formula(f, vc, dd)
        r = mod.fit_vb()
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    names = mod.exog_names
    out = {}
    for k in names:
        i = names.index(k)
        out[k] = {"mean": float(r.fe_mean[i]), "sd": float(r.fe_sd[i]),
                  "ci95": [float(r.fe_mean[i] - 1.96 * r.fe_sd[i]), float(r.fe_mean[i] + 1.96 * r.fe_sd[i])]}
    out["vc_sd"] = {n: float(np.exp(m)) for n, m in zip(mod.vcp_names, r.vcp_mean)}
    it = f"tib:{cap}"
    b0, b1 = out["tib"]["mean"], out[it]["mean"]
    rng = np.random.default_rng(T.SEED)
    s0 = rng.normal(b0, out["tib"]["sd"], 20000)
    s1 = rng.normal(b1, out[it]["sd"], 20000)
    same = np.sign(s1) == np.sign(b1)
    cr = -s0[same] / s1[same]
    out["crossover_point"] = float(-b0 / b1) if b1 != 0 else None
    out["crossover_ci95_meanfield"] = [float(np.percentile(cr, 2.5)), float(np.percentile(cr, 97.5))] if same.sum() > 100 else None
    out["post_prob_interaction_gt0"] = float((s1 > 0).mean())
    return out


def cap_to_label(cap: str, x):
    if x is None or not np.isfinite(x):
        return "n/a"
    if cap == "rank_c":
        r = x + 3
        names = ["W", "F", "M", "S", "G"]
        lo = int(np.clip(np.floor(r), 1, 5))
        return f"rank {r:.2f} (between {names[lo - 1]} and {names[min(lo, 4)]})" if 1 <= r <= 5 else f"rank {r:.2f} (outside W..G)"
    return f"stock pass {100 * (x + 0.5):.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--skip-bayes", action="store_true")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma list; partial runs write capability_model_<v>.json")
    ap.add_argument("--merge", action="store_true", help="merge capability_model_<v>.json parts into capability_model.{json,md}")
    args = ap.parse_args()
    if args.merge:
        res = {"meta": {"seed": T.SEED}, "fits": {}}
        for v in VARIANTS:
            p = T.OUT / f"capability_model_{v}.json"
            if p.exists():
                part = json.loads(p.read_text())
                res["meta"].update(part["meta"])
                res["fits"][v] = part["fits"][v]
        (T.OUT / "capability_model.json").write_text(json.dumps(res, indent=1))
        write_md(res)
        return
    variants = [v for v in args.variants.split(",") if v]
    df = build_frame()
    res = {"meta": {"seed": T.SEED, "n_boot": args.n_boot,
                    "n_rows": int(len(df))}, "fits": {}}
    print(df.groupby(["variant", "bench"]).size())
    for v in variants:
        d = frame_for(df, v)
        R = {"n_rows": int(len(d)), "n_tasks": int(d["task"].nunique()), "benches": sorted(d["bench"].unique()),
             "cells": sorted(d["bench_model"].unique())}
        for cap in ("rank_c", "stockpass_c"):
            key = cap
            R[key] = {}
            if not args.skip_bayes and len(d) <= BAYES_MAX_ROWS:
                R[key]["bayes_a"] = bayes_fit(d, cap, "a")
                if cap == "rank_c":
                    R[key]["bayes_b"] = bayes_fit(d, cap, "b")
            R[key]["gee"] = gee_fit(d, cap, False)
            R[key]["boot_bench_task"] = hier_boot(d, cap, False, args.n_boot, "bench_task", T.SEED + 1) if d["bench"].nunique() > 1 else None
            R[key]["boot_task"] = hier_boot(d, cap, False, args.n_boot, "task", T.SEED + 2)
            print(v, cap, "gee", {k: (round(x["coef"], 3), [round(c, 3) for c in x["ci95"]]) for k, x in R[key]["gee"].items() if isinstance(x, dict) and "coef" in x},
                  "cross", R[key]["gee"].get("crossover_point"))
        # difficulty x arm (rank as capability)
        R["difficulty"] = {"gee": gee_fit(d, "rank_c", True),
                           "boot_task": hier_boot(d, "rank_c", True, args.n_boot, "task", T.SEED + 3)}
        if not args.skip_bayes and len(d) <= BAYES_MAX_ROWS:
            R["difficulty"]["bayes_b"] = bayes_fit(d, "rank_c", "b", with_diff=True)
        print(v, "difficulty", R["difficulty"]["gee"].get("tib:d_c"))
        res["fits"][v] = R
        if len(variants) < len(VARIANTS):
            (T.OUT / f"capability_model_{v}.json").write_text(json.dumps({"meta": res["meta"], "fits": {v: R}}, indent=1))
    if len(variants) == len(VARIANTS):
        (T.OUT / "capability_model.json").write_text(json.dumps(res, indent=1))
        write_md(res)


def write_md(res):
    L = ["# Capability-threshold model (auto-generated by `scripts/analysis/capability_model.py`)", "",
         "Logit scale. `tib` = TIB effect at the centring point (rank 3 = 3.6 Flash; or stock pass 50%); "
         "`tib:cap` = change in the TIB log-odds ratio per capability unit (per model rank step; or per 100 pp of "
         "stock pass rate). Crossover = capability at which the TIB log-odds ratio is 0; with a negative interaction "
         "TIB helps *below* it and hurts above it. Crossover CIs use only resamples / draws whose interaction has the "
         "same sign as the point estimate (report `P(int>0)` alongside). `boot_bench_task` resamples benchmarks then "
         "tasks (the honest CI when generalising to new benchmarks); `boot_task` keeps benchmarks fixed.", ""]
    for v, R in res["fits"].items():
        L += [f"## {v}", "", f"rows {R['n_rows']}, tasks {R['n_tasks']}, benchmarks {', '.join(R['benches'])}", "",
              "| capability | method | tib (at centre) [95% CI] | tib:cap [95% CI] | P(int>0) / p | crossover [CI] |", "|---|---|---|---|---|---|"]
        for cap in ("rank_c", "stockpass_c"):
            C = R[cap]
            for meth in ("bayes_a", "bayes_b", "gee", "boot_bench_task", "boot_task"):
                r = C.get(meth)
                if not r or "error" in r:
                    if r:
                        L.append(f"| {cap} | {meth} | error: {r['error'][:60]} | | | |")
                    continue
                it = f"tib:{cap}"
                if meth.startswith("bayes"):
                    a, b = r["tib"], r[it]
                    cr = r.get("crossover_ci95_meanfield")
                    L.append(f"| {cap} | {meth} | {a['mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | {b['mean']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | "
                             f"{r['post_prob_interaction_gt0']:.3f} | {cap_to_label(cap, r['crossover_point'])} "
                             f"[{cap_to_label(cap, cr[0]) if cr else 'n/a'}; {cap_to_label(cap, cr[1]) if cr else 'n/a'}] |")
                elif meth == "gee":
                    a, b = r["tib"], r[it]
                    cr = r.get("crossover_delta_ci95")
                    L.append(f"| {cap} | gee (exch., robust) | {a['coef']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | {b['coef']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | "
                             f"p={b['p']:.3g} | {cap_to_label(cap, r['crossover_point'])} [{cap_to_label(cap, cr[0])}; {cap_to_label(cap, cr[1])}] |")
                else:
                    cr = r.get("crossover_ci_same_sign")
                    L.append(f"| {cap} | {meth} (n={r['n_boot']}) | {r['point']['tib']:+.3f} [{r['ci_tib'][0]:+.3f}, {r['ci_tib'][1]:+.3f}] | "
                             f"{r['point']['interaction']:+.3f} [{r['ci_interaction'][0]:+.3f}, {r['ci_interaction'][1]:+.3f}] | "
                             f"{1 - r['p_interaction_le0']:.3f} | {cap_to_label(cap, r['crossover_point'])} "
                             f"[{cap_to_label(cap, cr[0]) if cr else 'n/a'}; {cap_to_label(cap, cr[1]) if cr else 'n/a'}] |")
        D = R["difficulty"]
        g = D["gee"]
        if "error" not in g:
            L += ["", f"Difficulty x arm (GEE, rank capability + LOO difficulty, centred at 0.5): tib:d_c = "
                  f"{g['tib:d_c']['coef']:+.3f} [{g['tib:d_c']['ci95'][0]:+.3f}, {g['tib:d_c']['ci95'][1]:+.3f}] (p={g['tib:d_c']['p']:.3g}); "
                  f"tib:rank = {g['tib:rank_c']['coef']:+.3f} [{g['tib:rank_c']['ci95'][0]:+.3f}, {g['tib:rank_c']['ci95'][1]:+.3f}]. "
                  f"Task bootstrap tib:d_c CI {D['boot_task'].get('ci_tib:d_c')}."]
            if D.get("bayes_b") and "error" not in D["bayes_b"]:
                bb = D["bayes_b"]["tib:d_c"]
                L.append(f"Bayes (spec b) tib:d_c = {bb['mean']:+.3f} [{bb['ci95'][0]:+.3f}, {bb['ci95'][1]:+.3f}].")
        L.append("")
    (T.OUT / "capability_model.md").write_text("\n".join(L) + "\n")
    try:
        figure(res)
    except Exception as e:  # noqa: BLE001
        print("figure failed:", e)


def figure(res):
    """Per-cell pass delta (TIB - stock, pp) vs model rank, one panel per variant, with the GEE (rank) curve
    evaluated at the mean stock pass rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = build_frame()
    T.FIG.mkdir(parents=True, exist_ok=True)
    vs = [v for v in VARIANTS if v in res["fits"]]
    fig, axes = plt.subplots(1, len(vs), figsize=(4.3 * len(vs), 3.6), sharey=True, squeeze=False)
    cmap = plt.get_cmap("tab20")
    benches = list(T.BENCH_LABEL)
    for ax, v in zip(axes[0], vs):
        d = frame_for(df, v)
        g = d.groupby(["bench", "model", "tib"])["y"].mean().unstack("tib").dropna()
        for (b, m), row in g.iterrows():
            ax.scatter(T.RANK[m] + (benches.index(b) - 6) * 0.025, 100 * (row[1] - row[0]), color=cmap(benches.index(b)),
                       s=22, label=T.BENCH_LABEL[b])
        ge = res["fits"][v]["rank_c"].get("gee", {})
        if "tib" in ge:
            p0 = float(d.loc[d.tib == 0, "y"].mean())
            lo = np.log(p0 / (1 - p0))
            xs = np.linspace(1, 5, 50)
            lor = ge["tib"]["coef"] + ge["tib:rank_c"]["coef"] * (xs - 3)
            ax.plot(xs, 100 * (1 / (1 + np.exp(-(lo + lor))) - p0), "k-", lw=1.5, label="GEE fit (at mean stock pass)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(list(T.RANK.values())); ax.set_xticklabels([T.MODEL_LABEL[m].replace(" ", "\n") for m in T.RANK], fontsize=7)
        ax.set_title(v)
    axes[0][0].set_ylabel("Δ pass, TIB − stock (pp), per benchmark × model cell")
    h, lab = axes[0][-1].get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    fig.legend(uniq.values(), uniq.keys(), fontsize=6, loc="lower center", ncol=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(T.FIG / "capability_interaction.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
