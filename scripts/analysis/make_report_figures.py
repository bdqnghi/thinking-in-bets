#!/usr/bin/env python3
"""Figures for docs/results/ANALYSIS_REPORT.md (paper quality, PDF + PNG).

Plotting only. Reads the analysis outputs in results/analysis/*.json (written by the other analysis
scripts) and, for the per-cell TIB-vs-stock effects in Fig. 1 / Fig. 3, the raw per-task result files
through the shared read-only loader ``tib_data.load_all()`` (results/harbor/*/per_task.json, results/hle,
results/finance_agent*, results/redo_v1_ale, SWE-Interact trial caches). No model runs.

    python3 scripts/analysis/make_report_figures.py            # all figures
    python3 scripts/analysis/make_report_figures.py --only 1 4  # a subset
    python3 scripts/analysis/make_report_figures.py --slides    # sans-serif variant (Roboto)

Outputs: docs/figures/report/figNN_<name>.{pdf,png}, docs/figures/report/report_numbers.json
(every derived number the report quotes that is not already in an analysis JSON).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402

ROOT = fs.ROOT
A = ROOT / "results" / "analysis"
SEED = 20261014
NUMBERS: dict = {}


def J(name: str):
    return json.loads((A / name).read_text())


def pct(x):
    return None if x is None else 100.0 * x


# ═══════════════════════════════════════════════════════════════════════════
# Per-cell TIB vs stock effects (paired task bootstrap)
# ═══════════════════════════════════════════════════════════════════════════

CLAIM_ID = {
    ("ds11", "G", "cb_draftfirst"): "HARBOR-ds11-G", ("ds11", "S", "cb_draftfirst"): "UNREP-ds11-S",
    ("ds11", "F", "cb_draftfirst"): "UNREP-ds11-F", ("ds11m", "G", "cb_draftfirst"): "HARBOR-ds11m-G",
    ("tb4", "S", "cb_draftfirst"): "HARBOR-tb4-S", ("tb4", "G", "cb_draftfirst"): "UNREP-tb4-G",
    ("tb4", "W", "cb_draftfirst"): "UNREP-tb4-W", ("tb4", "F", "cb_draftfirst"): "UNREP-tb4-F",
    ("swi_overlay", "G", "commit_before"): "SWI-pass-commit_before-G-r1r2",
    ("swi_overlay", "S", "commit_before"): "SWI-pure-pass-commit_before-S",
    ("swi_mixed", "S", "commit_before"): "SWI-pass-commit_before-S-r1r2",
    ("swi_mixed", "S", "cb_gated"): "SWI-pass-cb_gated-S-r1r2",
}
for _m in fs.MODEL_ORDER:
    CLAIM_ID[("hle_ss", _m, "cb_draftfirst")] = f"HLE-acc-{_m}"
    CLAIM_ID[("hle_mt", _m, "commit_before")] = f"HLE-mt-t2acc-{_m}"
    CLAIM_ID[("fin_tools", _m, "cb_draftfirst")] = f"FIN-tools-{_m}"
    CLAIM_ID[("fin_closed", _m, "cb_draftfirst")] = f"FIN-closed-{_m}"
    CLAIM_ID[("fin_mt", _m, "commit_before")] = f"UNREP-fin_mt-{_m}"
    CLAIM_ID[("ale_t300", _m, "cb_draftfirst")] = f"ALE-docker-ale_t300-{_m}"
    CLAIM_ID[("ale_s1", _m, "cb_draftfirst")] = f"ALE-docker-ale_s1-{_m}"
    CLAIM_ID[("ale_e2", _m, "commit_before")] = f"ALE-E2-pass-{_m}"
    if _m in "WFM":
        CLAIM_ID[("swi_mixed", _m, "commit_before")] = f"SWI-pass-commit_before-{_m}"
        CLAIM_ID[("swi_mixed", _m, "cb_gated")] = f"SWI-pass-cb_gated-{_m}"

SCORE_PRIMARY = {"ale_t300", "ale_s1", "fin_tools", "fin_closed"}  # headline metric is the score
BENCH_ORDER = ["swi_mixed", "swi_overlay", "ale_e2", "hle_mt", "fin_mt", "hle_ss", "fin_tools", "fin_closed",
               "ds11", "ds11m", "tb4", "ale_t300", "ale_s1"]
ARM_ORDER = {"commit_before": 0, "cb_gated": 1, "cb_draftfirst": 2}

_EFFECTS = None


def cell_effects(n_boot: int = 5000):
    """[{bench, model, arm, n, stock, arm_rate, delta, lo, hi, (score versions), claim info}] for TIB arms."""
    global _EFFECTS
    if _EFFECTS is not None:
        return _EFFECTS
    import tib_data as td  # read-only shared loader
    recs = td.load_all()
    cells = td.paired_cells(recs, min_n=10)
    mc = {c["id"]: c for c in J("multiple_comparisons.json")["claims"]}
    rng = np.random.default_rng(SEED)
    out = []
    for (b, m, arm), cell in cells.items():
        if arm not in fs.OURS:
            continue
        tasks = sorted(cell)
        s_y = np.array([np.mean(cell[t]["stock"]) for t in tasks])
        a_y = np.array([np.mean(cell[t]["arm"]) for t in tasks])

        def _sc(key, fb):
            v = []
            for t in tasks:
                xs = [x for x in cell[t][key] if x is not None]
                v.append(np.mean(xs) if xs else np.nan)
            v = np.array(v, float)
            return v if not np.isnan(v).any() else None
        s_s, a_s = _sc("stock_s", "stock"), _sc("arm_s", "arm")
        idx = rng.integers(0, len(tasks), size=(n_boot, len(tasks)))

        def boot(s, a):
            d = a - s
            bs = d[idx].mean(axis=1)
            return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
        dy, ly, hy = boot(s_y, a_y)
        rec = {"bench": b, "model": m, "arm": arm, "n": len(tasks), "stock": 100 * s_y.mean(),
               "arm_rate": 100 * a_y.mean(), "delta": dy, "lo": ly, "hi": hy,
               "w": int(((a_y - s_y) > 0).sum()), "l": int(((a_y - s_y) < 0).sum())}
        if b in SCORE_PRIMARY and s_s is not None and a_s is not None:
            ds_, ls_, hs_ = boot(s_s, a_s)
            rec.update(score_delta=ds_, score_lo=ls_, score_hi=hs_, stock_score=100 * s_s.mean(),
                       arm_score=100 * a_s.mean())
        cid = CLAIM_ID.get((b, m, arm))
        c = mc.get(cid) if cid else None
        rec.update(claim_id=cid, label=(c or {}).get("label"), family=(c or {}).get("family"),
                   p=(c or {}).get("p"), holm_ext=(c or {}).get("holm_ext"), holm=(c or {}).get("holm"))
        out.append(rec)
    out.sort(key=lambda r: (BENCH_ORDER.index(r["bench"]) if r["bench"] in BENCH_ORDER else 99,
                            fs.MODEL_RANK[r["model"]], ARM_ORDER[r["arm"]]))
    _EFFECTS = out
    NUMBERS["cell_effects"] = out
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1: headline forest plot
# ═══════════════════════════════════════════════════════════════════════════

def fig01_forest():
    eff = cell_effects()
    rows = []  # (kind, payload)
    for b in BENCH_ORDER:
        rs = [r for r in eff if r["bench"] == b]
        if not rs:
            continue
        unit = "score" if b in SCORE_PRIMARY else "pass"
        rows.append(("hdr", f"{fs.BENCH_LABEL[b]}" + ("  (score)" if unit == "score" else "")))
        rows.extend(("cell", r) for r in rs)
    n_hdr = sum(1 for k, _ in rows if k == "hdr")
    gap = 0.45
    nrow = len(rows) + gap * (n_hdr - 1)
    h = 0.118 * nrow + 0.6
    fig, ax = plt.subplots(figsize=(fs.DCOL, h))
    fig.subplots_adjust(left=0.24, right=0.70)
    multi_arm = {b for b in BENCH_ORDER if len({r["arm"] for r in eff if r["bench"] == b}) > 1}
    short_arm = {"commit_before": "always-on", "cb_gated": "on-demand", "cb_draftfirst": "draft-first"}
    y = nrow - 1
    yt, yl = [], []
    lo_all, hi_all = [], []
    first = True
    for kind, p in rows:
        if kind == "hdr":
            if not first:
                y -= gap
            first = False
            ax.text(-0.345, y, p, transform=mtransforms.blended_transform_factory(ax.transAxes, ax.transData),
                    ha="left", va="center", fontweight="bold", fontsize=6.9)
        else:
            r = p
            sc = r["bench"] in SCORE_PRIMARY and "score_delta" in r
            d, lo, hi = (r["score_delta"], r["score_lo"], r["score_hi"]) if sc else (r["delta"], r["lo"], r["hi"])
            col = fs.ARM_COLOR[r["arm"]]
            surv = r["holm_ext"] is not None and r["holm_ext"] < 0.05
            ax.plot([lo, hi], [y, y], color=col, lw=0.9, solid_capstyle="butt", zorder=2)
            ax.plot(d, y, marker="o", ms=3.6, mec=col, mfc=col if surv else "white", mew=0.9, zorder=3)
            lo_all.append(lo)
            hi_all.append(hi)
            yt.append(y)
            yl.append(fs.MODEL_LABEL[r["model"]] + (f" ({short_arm[r['arm']]})" if r["bench"] in multi_arm else ""))
            tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(1.02, y, f"{d:+.1f} [{lo:+.0f}, {hi:+.0f}]", transform=tr, ha="left", va="center", fontsize=6.4)
            ax.text(1.30, y, f"{r['n']}", transform=tr, ha="right", va="center", fontsize=6.4)
            tag = {"PR": "pre-reg.", "PP": "part. pre-reg.", "EX": "exploratory"}.get(r["label"], "")
            if r["family"] == "unreported":
                tag = "not in summary"
            ax.text(1.34, y, tag, transform=tr, ha="left", va="center", fontsize=6.2,
                    color="#b2182b" if r["family"] == "unreported" else "#333333",
                    style="italic" if r["family"] == "unreported" else "normal")
            ax.text(1.62, y, "\u2713" if surv else "", transform=tr, ha="center", va="center", fontsize=7,
                    fontfamily="DejaVu Sans")
        y -= 1
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=6.3)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.7, nrow - 0.3)
    xmin = min(-60, math.floor(min(lo_all) / 10) * 10)
    xmax = max(60, math.ceil(max(hi_all) / 10) * 10)
    ax.set_xlim(xmin, xmax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#ededed", lw=0.5)
    fs.zero_line(ax, "x")
    ax.set_xlabel("TIB $-$ standard agent (percentage points, 95% paired-task bootstrap CI)")
    top = nrow - 0.3
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for x, s, ha in ((1.02, "\u0394 [95% CI]", "left"), (1.30, "n", "right"), (1.34, "status", "left"),
                     (1.62, "Holm", "center")):
        ax.text(x, top + 0.2, s, transform=tr, ha=ha, va="bottom", fontsize=6.6, fontweight="bold")
    ax.spines["left"].set_visible(False)
    hs = fs.arm_handles(["commit_before", "cb_gated", "cb_draftfirst"], ms=3.6)
    hs += [Line2D([], [], color="k", marker="o", ls="none", ms=3.6, mfc="k", label="survives Holm (94 tests)"),
           Line2D([], [], color="k", marker="o", ls="none", ms=3.6, mfc="white", label="does not survive")]
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 1.0 + 0.25 / h), ncol=5, fontsize=6.4,
              columnspacing=0.9, handletextpad=0.3)
    return fs.save(fig, "fig01_forest")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2: gains by task difficulty
# ═══════════════════════════════════════════════════════════════════════════
BINS = ["easy", "medium", "hard"]
BIN_LABEL = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}


def _pooled(ds, group, mode="loo"):
    for p in ds["pooled"]:
        if p["group"] == group and p["mode"] == mode:
            return p["bins"]
    return None


def _errbar(ax, x, b, color, marker="o", mfc=None, label=None, ms=3.8, lw=0.9, ls="-", connect=True, z=3):
    ys = [pct(b[k]["delta"]) if b.get(k) and b[k]["delta"] is not None else np.nan for k in BINS]
    los = [pct(b[k]["ci95"][0]) if b.get(k) and b[k]["ci95"] else np.nan for k in BINS]
    his = [pct(b[k]["ci95"][1]) if b.get(k) and b[k]["ci95"] else np.nan for k in BINS]
    ys, los, his = map(np.array, (ys, los, his))
    ax.errorbar(x, ys, yerr=[ys - los, his - ys], fmt="none", ecolor=color, elinewidth=lw, zorder=z)
    if connect:
        ax.plot(x, ys, color=color, lw=0.7, ls=ls, alpha=0.8, zorder=z)
    ax.plot(x, ys, marker=marker, ls="none", color=color, mfc=mfc or color, mec=color, ms=ms, mew=0.8,
            label=label, zorder=z + 1)
    return ys, los, his


def fig02_difficulty():
    ds = J("difficulty_split.json")
    fig, axs = plt.subplots(1, 4, figsize=(fs.DCOL, 2.25), gridspec_kw={"wspace": 0.45})
    x0 = np.arange(3)
    # (a) pooled, LOO, by variant
    ax = axs[0]
    groups = [("cb_draftfirst", "cb_draftfirst"), ("commit_before", "commit_before"), ("cb_gated", "cb_gated"),
              ("baselines (claims_after, claims_no_p, conf_gate, sc_vote3, verify_then_respond)", "stock")]
    offs = np.linspace(-0.24, 0.24, len(groups))
    for (g, arm), o in zip(groups, offs):
        b = _pooled(ds, g)
        lab = fs.ARM_LABEL[arm] if arm != "stock" else "Baselines (5 arms)"
        _errbar(ax, x0 + o, b, fs.ARM_COLOR[arm] if arm != "stock" else "#9a9a9a", label=lab,
                marker="o" if arm != "stock" else "s")
    b = _pooled(ds, "cb_draftfirst")
    ax.text(2 - 0.24, pct(b["hard"]["delta"]) + 5, f"{pct(b['hard']['delta']):+.1f}\n{b['hard']['wtl'][0]}W/{b['hard']['wtl'][2]}L",
            fontsize=5.8, color=fs.ARM_COLOR["cb_draftfirst"], ha="center", va="bottom")
    ax.text(0 - 0.24, pct(b["easy"]["delta"]) + 5, f"{pct(b['easy']['delta']):+.1f}", fontsize=5.8,
            color=fs.ARM_COLOR["cb_draftfirst"], ha="center", va="bottom")
    ax.set_ylabel("TIB $-$ standard (pp)")
    fs.ptitle(ax, "a", "Pooled (LOO bins)")
    # (b) regression to the mean: naive vs LOO
    ax = axs[1]
    b_all, b_loo = _pooled(ds, "cb_draftfirst", "all"), _pooled(ds, "cb_draftfirst", "loo")
    _errbar(ax, x0 - 0.12, b_all, "#b9a6d6", label="Naive bins (own stock run)", marker="s")
    nul = [pct(b_all[k]["null_mean"]) for k in BINS]
    ax.plot(x0 - 0.12, nul, ls="none", marker="_", ms=9, mew=1.2, color="#333333", label="Permutation null", zorder=6)
    _errbar(ax, x0 + 0.12, b_loo, fs.ARM_COLOR["cb_draftfirst"], label="LOO bins (null = 0)")
    fs.ptitle(ax, "b", "RTM check")
    # (c) always-on: push type
    ax = axs[2]
    pts = [p for p in ds["push_type_split"] if p["arm"] == "commit_before"]
    styles = {("hle_mt", "false_pushback"): ("HLE, mistaken pushback", "o", None, "-"),
              ("hle_mt", "constructive_hint"): ("HLE, valid hint", "o", "white", "--"),
              ("fin_mt", "constructive_hint"): ("Finance, valid hint", "^", "white", ":")}
    offs = {("hle_mt", "false_pushback"): -0.15, ("hle_mt", "constructive_hint"): 0.0, ("fin_mt", "constructive_hint"): 0.15}
    for key, (lab, mk, mfc, ls) in styles.items():
        bb = {p["bin"]: p for p in pts if (p["bench"], p["push_type"]) == key}
        _errbar(ax, x0 + offs[key], bb, fs.ARM_COLOR["commit_before"], marker=mk, mfc=mfc, label=lab, ls=ls)
    fs.ptitle(ax, "c", "Always-on by push")
    # (d) draft-first by model (LOO)
    ax = axs[3]
    ms = ["W", "F", "S", "G"]
    offs = np.linspace(-0.22, 0.22, len(ms))
    for m, o in zip(ms, offs):
        b = _pooled(ds, f"cb_draftfirst | model {m}")
        _errbar(ax, x0 + o, b, fs.MODEL_COLOR[m], marker=fs.MODEL_MARKER[m], label=fs.MODEL_LABEL[m],
                ms=4.2 if m == "G" else 3.4)
    fs.ptitle(ax, "d", "Draft-first by model")
    for i, ax in enumerate(axs):
        ax.set_xticks(x0)
        ax.set_xticklabels([BIN_LABEL[k] for k in BINS])
        ax.set_xlim(-0.5, 2.5)
        fs.zero_line(ax, "y")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=6.2, handletextpad=0.3,
                  borderaxespad=0)
    axs[0].set_ylim(-48, 32)
    axs[1].set_ylim(-9, 9)
    axs[2].set_ylim(-80, 40)
    axs[3].set_ylim(-9, 15)
    return fs.save(fig, "fig02_difficulty")


def fig02b_difficulty_by_bench():
    ds = J("difficulty_split.json")
    panels = [("ds11", "cb_draftfirst"), ("ds11m", "cb_draftfirst"), ("tb4", "cb_draftfirst"), ("hle_ss", "cb_draftfirst"),
              ("fin_tools", "cb_draftfirst"), ("fin_closed", "cb_draftfirst"), ("ale_t300", "cb_draftfirst"),
              ("ale_s1", "cb_draftfirst"), ("hle_mt", "commit_before"), ("fin_mt", "commit_before"),
              ("ale_e2", "commit_before"), ("swi_overlay", "commit_before"), ("swi_mixed", "commit_before"),
              ("swi_mixed", "cb_gated")]
    fig, axs = plt.subplots(4, 4, figsize=(fs.DCOL, 6.4), gridspec_kw={"hspace": 0.62, "wspace": 0.38})
    axs = axs.ravel()
    x0 = np.arange(3)
    for i, (b, arm) in enumerate(panels):
        ax = axs[i]
        outcome = "score" if b in ("ale_t300", "ale_s1") else "y"
        cs = [c for c in ds["cells"] if c["mode"] == "loo" and c["bench"] == b and c["arm"] == arm and c["outcome"] == outcome]
        cs.sort(key=lambda c: fs.MODEL_RANK[c["model"]])
        offs = np.linspace(-0.2, 0.2, max(len(cs), 1)) if len(cs) > 1 else [0.0]
        for c, o in zip(cs, offs):
            m = c["model"]
            _errbar(ax, x0 + o, c["bins"], fs.MODEL_COLOR[m], marker=fs.MODEL_MARKER[m],
                    ms=4.0 if m == "G" else 3.0, lw=0.6)
        name = fs.BENCH_LABEL[b].replace(", ", "\n", 1) if len(fs.BENCH_LABEL[b]) > 24 else fs.BENCH_LABEL[b]
        ax.set_title(f"{name}\n{fs.ARM_LABEL[arm]}" + (" (score)" if outcome == "score" else ""), fontsize=6.6,
                     color=fs.ARM_COLOR[arm])
        ax.set_xticks(x0)
        ax.set_xticklabels(["E", "M", "H"])
        ax.set_xlim(-0.5, 2.5)
        fs.zero_line(ax, "y")
        ax.tick_params(labelsize=6.4)
        if i % 4 == 0:
            ax.set_ylabel("\u0394 vs standard (pp)", fontsize=6.8)
    for ax in axs[len(panels):]:
        ax.axis("off")
    hs = [Line2D([], [], color=fs.MODEL_COLOR[m], marker=fs.MODEL_MARKER[m], ls="-", lw=0.7, ms=4,
                 label=fs.MODEL_LABEL[m]) for m in fs.MODEL_ORDER]
    axs[len(panels)].legend(handles=hs, loc="center left", title="Model", fontsize=6.8, bbox_to_anchor=(0.1, 0.5))
    axs[len(panels) + 1].text(0.0, 0.5, "E/M/H = easy / medium / hard,\nleave-one-model-out difficulty\n"
                              "(share of other models' stock\nruns that solve the task).\nBars: 95% paired bootstrap CI.",
                              transform=axs[len(panels) + 1].transAxes, fontsize=6.4, va="center")
    return fs.save(fig, "fig02b_difficulty_by_benchmark")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3: capability
# ═══════════════════════════════════════════════════════════════════════════

def fig03_capability():
    cap = J("capability_model.json")["fits"]
    eff = cell_effects()
    fig, axs = plt.subplots(1, 3, figsize=(fs.DCOL, 2.15), gridspec_kw={"wspace": 0.36})
    # (a) fitted TIB log-odds ratio vs model rank (GEE, benchmarks fixed)
    ax = axs[0]
    r = np.linspace(1, 5, 50)
    for key, arm, lab in (("cb_draftfirst", "cb_draftfirst", "Draft-first"), ("commit_before", "commit_before", "Always-on"),
                          ("cb_draftfirst_noHLE", "cb_draftfirst", "Draft-first w/o HLE"),
                          ("cb_gated", "cb_gated", "On-demand (1 bench)"), ("pooled", "stock", "Pooled (13 bench)")):
        if key not in cap:
            continue
        g = cap[key]["rank_c"]["gee"]
        b0, b1 = g["tib"]["coef"], g["tib:rank_c"]["coef"]
        col = fs.ARM_COLOR[arm] if arm != "stock" else "#222222"
        ax.plot(r, b0 + b1 * (r - 3), color=col, lw=1.4 if key == "pooled" else 0.9,
                ls={"cb_gated": ":", "cb_draftfirst_noHLE": "--"}.get(key, "-"), label=lab)
    g = cap["pooled"]["rank_c"]["gee"]
    cx = 3 + g["crossover_point"]
    clo, chi = (3 + v for v in g["crossover_delta_ci95"])
    ax.errorbar([cx], [0], xerr=[[cx - clo], [chi - cx]], fmt="o", color="#222222", ms=3.5, elinewidth=0.9, zorder=5)
    ax.annotate(f"pooled crossover\nrank {cx:.1f} [{clo:.1f}, {chi:.1f}]", xy=(cx, 0), xytext=(3.05, -0.62),
                fontsize=6.0, arrowprops=dict(arrowstyle="-", lw=0.5))
    fs.zero_line(ax, "y")
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([fs.MODEL_SHORT[m] for m in fs.MODEL_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Fitted TIB log-odds ratio")
    ax.set_ylim(-0.78, 0.8)
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.6, borderaxespad=0.1)
    fs.ptitle(ax, "a", "GEE fit, benchmarks fixed")
    # (b) observed per-cell deltas vs model
    ax = axs[1]
    rng = np.random.default_rng(3)
    for arm in ("cb_draftfirst", "commit_before", "cb_gated"):
        rs = [e for e in eff if e["arm"] == arm]
        xs = np.array([fs.MODEL_RANK[e["model"]] for e in rs]) + {"cb_draftfirst": -0.18, "commit_before": 0.0, "cb_gated": 0.18}[arm]
        xs = xs + rng.uniform(-0.05, 0.05, len(xs))
        ax.scatter(xs, [e["delta"] for e in rs], s=[max(4, min(28, e["n"] / 6)) for e in rs], color=fs.ARM_COLOR[arm],
                   alpha=0.75, lw=0, label=fs.ARM_LABEL[arm])
        means = [np.mean([e["delta"] for e in rs if e["model"] == m]) if any(e["model"] == m for e in rs) else np.nan
                 for m in fs.MODEL_ORDER]
        ax.plot(np.arange(1, 6) + {"cb_draftfirst": -0.18, "commit_before": 0.0, "cb_gated": 0.18}[arm], means,
                color=fs.ARM_COLOR[arm], lw=0.8, marker="_", ms=7)
    fs.zero_line(ax, "y")
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([fs.MODEL_SHORT[m] for m in fs.MODEL_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Cell \u0394 pass rate (pp)")
    fs.ptitle(ax, "b", "Observed cells (bar = mean)")
    ax.legend(handles=fs.arm_handles(["cb_draftfirst", "commit_before", "cb_gated"], ms=3.5), loc="lower left",
              fontsize=5.8, handletextpad=0.2, borderaxespad=0.1)
    ax.set_ylim(-62, 72)
    # (c) delta vs stock pass rate
    ax = axs[2]
    for arm in ("cb_draftfirst", "commit_before", "cb_gated"):
        rs = [e for e in eff if e["arm"] == arm]
        ax.scatter([e["stock"] for e in rs], [e["delta"] for e in rs], s=[max(4, min(28, e["n"] / 6)) for e in rs],
                   color=fs.ARM_COLOR[arm], alpha=0.75, lw=0)
    gs = cap["pooled"]["stockpass_c"]["gee"]
    sx = 50 + 100 * gs["crossover_point"]
    slo, shi = (50 + 100 * v for v in gs["crossover_delta_ci95"])
    ax.axvspan(slo, shi, color="#dddddd", alpha=0.6, lw=0, zorder=0)
    ax.axvline(sx, color="#222222", lw=0.8, ls="--", zorder=1)
    ax.text(sx + 2, 62, f"fitted crossover\n{sx:.0f}% [{slo:.0f}, {shi:.0f}]", fontsize=6.2, va="top")
    fs.zero_line(ax, "y")
    ax.set_xlabel("Standard-agent pass rate (%)")
    ax.set_ylabel("Cell \u0394 pass rate (pp)")
    ax.set_ylim(-62, 72)
    ax.set_xlim(0, 100)
    fs.ptitle(ax, "c", "Gain vs headroom")
    NUMBERS["capability"] = {"pooled_crossover_rank": [cx, clo, chi], "pooled_crossover_stockpass": [sx, slo, shi]}
    return fs.save(fig, "fig03_capability")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4: bet calibration
# ═══════════════════════════════════════════════════════════════════════════

def fig04_calibration():
    bc = J("bet_calibration.json")
    cal = bc["e2"]["q1"]["calibration"]
    fig = plt.figure(figsize=(fs.DCOL, 2.35))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.2, 1], width_ratios=[1.25, 1, 1, 1], hspace=0.08, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[1, 0], sharex=ax)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.6, ls="--", zorder=0)
    conds = [("bet", "commit_before", "Bet before reply"), ("none", "none", "Blind rating (no reply)"),
             ("unjustified", "claims_after", "Check after pushback")]
    w = 0.028
    for j, (k, colk, lab) in enumerate(conds):
        c = cal["pooled/all"][k]
        bins = [b for b in c["bins"] if b["n"] and b["n"] >= 20]
        ax.plot([b["mean_p"] for b in bins], [b["frac_true"] for b in bins], marker="o", ms=3.0,
                color=fs.ARM_COLOR[colk], lw=0.9, label=f"{lab}")
        cent = np.array([(b["lo"] + b["hi"]) / 2 for b in c["bins"]])
        share = np.array([b["n"] for b in c["bins"]]) / c["n"]
        axh.bar(cent + (j - 1) * w, share, width=w, color=fs.ARM_COLOR[colk], lw=0)
    ax.set_ylabel("Share actually satisfied")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", fontsize=5.9, handlelength=1.2, handletextpad=0.3)
    b = cal["pooled/all"]["bet"]
    ax.text(0.98, 0.04, f"n = {b['n']:,} ratings\nbet ECE {b['ece']:.2f}, AUROC {b['auroc']:.2f}", ha="right",
            va="bottom", fontsize=5.9, transform=ax.transAxes)
    axh.set_xlabel("Stated probability $p$")
    axh.set_ylabel("Share", fontsize=6.5)
    axh.set_yticks([0, 0.4])
    axh.grid(axis="y", visible=False)
    axh.tick_params(labelsize=6)
    fs.ptitle(ax, "a", "E2 reliability")
    # (b) ECE by model, (c) AUROC by model
    xs = np.arange(5)
    for pi, (metric, ttl, lim) in enumerate((("ece", "E2 calibration error (ECE, lower is better)", (0, 0.5)),
                                            ("auroc", "E2 discrimination (AUROC)", (0.5, 1.0)))):
        ax = fig.add_subplot(gs[:, 1 + pi])
        for j, (k, colk, lab) in enumerate(conds):
            vals = [cal[f"{m}/all"][k][metric] for m in fs.MODEL_ORDER]
            off = (j - 1) * 0.2
            if k == "bet":
                ci = [cal[f"{m}/all"][k][f"{metric}_ci"] for m in fs.MODEL_ORDER]
                ax.errorbar(xs + off, vals, yerr=[[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]],
                            fmt="o", color=fs.ARM_COLOR[colk], ms=3.2, elinewidth=0.8, label=lab, zorder=4)
            else:
                ax.plot(xs + off, vals, ls="none", marker="s" if k == "none" else "^", ms=3.0, color=fs.ARM_COLOR[colk],
                        label=lab, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([fs.MODEL_SHORT[m] for m in fs.MODEL_ORDER], rotation=30, ha="right")
        ax.set_ylim(*lim)
        fs.ptitle(ax, "bc"[pi], "E2 ECE by model" if metric == "ece" else "E2 AUROC by model")
        ax.set_ylabel("ECE (lower is better)" if metric == "ece" else "AUROC (higher is better)")
    # (d) HLE: pre-reply bet vs post-reply claims vs turn-1 confidence
    ax = fig.add_subplot(gs[:, 3])
    mt = bc["hle"]["q1"]["multi_turn"]
    hm = ["W", "F", "S", "G"]
    xs4 = np.arange(len(hm))
    for j, (k, colk, lab, mk) in enumerate((("commit_before_min_p (pre)", "commit_before", "Bet before reply (min $p$)", "o"),
                                            ("t1_confidence", "stock", "Turn-1 confidence", "s"),
                                            ("claims_after_min_p (post)", "claims_after", "Claims after reply", "^"))):
        vals = [mt[m][k]["auroc"] for m in hm]
        ax.plot(xs4 + (j - 1) * 0.18, vals, ls="-", lw=0.6, marker=mk, ms=3.0, color=fs.ARM_COLOR[colk], label=lab)
    ax.axhline(0.5, color="#999999", lw=0.6, ls="--", zorder=0)
    ax.set_xticks(xs4)
    ax.set_xticklabels([fs.MODEL_SHORT[m] for m in hm], rotation=30, ha="right")
    ax.set_ylim(0.35, 0.97)
    ax.set_ylabel("AUROC (turn-1 answer correct)")
    fs.ptitle(ax, "d", "HLE 2-turn")
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2, handletextpad=0.3)
    # shared legend for b/c
    fig.axes[2].legend(loc="upper right", fontsize=5.6, handletextpad=0.3)
    return fs.save(fig, "fig04_calibration")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5: hindsight drop
# ═══════════════════════════════════════════════════════════════════════════

def fig05_hindsight():
    hs = J("bet_calibration.json")["e2"]["q1"]["hindsight"]
    rows = fs.MODEL_ORDER + ["pooled"]
    fig, ax = plt.subplots(figsize=(fs.COL, 2.3))
    ys = np.arange(len(rows))[::-1]
    for y, m in zip(ys, rows):
        ca = hs[f"{m}/all"]["none->unjustified"]
        bt = hs[f"{m}/all"]["bet->bet_unjustified"]
        vals = [(-100 * ca["drop_true"], "claims_after", "o", None, 0.12),
                (-100 * ca["drop_false"], "claims_after", "o", "white", 0.12),
                (-100 * bt["drop_true"], "commit_before", "o", None, -0.08),
                (-100 * bt["drop_false"], "commit_before", "o", "white", -0.28)]
        ax.plot([vals[0][0], vals[1][0]], [y + 0.12] * 2, color=fs.ARM_COLOR["claims_after"], lw=0.7, zorder=1)
        for v, arm, mk, mfc, dy in vals:
            ax.plot(v, y + dy, marker=mk, ms=4.0 if m != "pooled" else 4.6, mec=fs.ARM_COLOR[arm],
                    mfc=mfc or fs.ARM_COLOR[arm], mew=0.9, ls="none", zorder=3)
        if m == "pooled":
            ax.annotate(f"true {vals[0][0]:.1f}, false {vals[1][0]:.1f} pts", xy=(min(vals[0][0], vals[1][0]), y + 0.12),
                        xytext=(-21, y - 0.55), fontsize=6.0, color=fs.ARM_COLOR["claims_after"],
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=fs.ARM_COLOR["claims_after"]))
    ax.axhline(0.5, color="#cccccc", lw=0.5)
    ax.text(-0.6, ys[0] + 0.45, "bets: no change", ha="right", va="center", fontsize=6.0, color=fs.ARM_COLOR["commit_before"])
    ax.set_yticks(ys)
    ax.set_yticklabels([fs.MODEL_LABEL.get(m, "Pooled") for m in rows])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#ededed", lw=0.5)
    fs.zero_line(ax, "x")
    ax.set_xlim(-23, 2)
    ax.set_ylim(-1.0, len(rows) - 0.25)
    ax.set_xlabel("Change in stated confidence after an unjustified\n\u201cthis is wrong\u201d reply (percentage points)")
    hsl = [Line2D([], [], color=fs.ARM_COLOR["claims_after"], marker="o", ls="none", ms=4, label="Check-after, true claim"),
           Line2D([], [], color=fs.ARM_COLOR["claims_after"], marker="o", mfc="white", ls="none", ms=4, label="Check-after, false claim"),
           Line2D([], [], color=fs.ARM_COLOR["commit_before"], marker="o", ls="none", ms=4, label="Bet, true claim"),
           Line2D([], [], color=fs.ARM_COLOR["commit_before"], marker="o", mfc="white", ls="none", ms=4, label="Bet, false claim")]
    ax.legend(handles=hsl, loc="upper center", bbox_to_anchor=(0.45, 1.2), ncol=2, fontsize=6.0, columnspacing=0.8,
              handletextpad=0.2)
    return fs.save(fig, "fig05_hindsight")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6: does the agent follow its bets?
# ═══════════════════════════════════════════════════════════════════════════

def fig06_bet_following():
    bc = J("bet_calibration.json")
    e2 = bc["e2"]["q2"]
    hle = bc["hle"]["q2"]
    fig, axs = plt.subplots(1, 3, figsize=(fs.DCOL, 2.1), gridspec_kw={"wspace": 0.38})
    xs = np.arange(5)
    hm = ["W", "F", "S", "G"]
    hx = np.array([fs.MODEL_ORDER.index(m) for m in hm])
    # (a) AUROC p -> hold
    ax = axs[0]
    v = [e2[m]["bet_first/real_reply_targeted"]["auroc_p_predicts_hold"] for m in fs.MODEL_ORDER]
    ax.plot(xs - 0.08, v, marker="o", ms=3.4, color=fs.ARM_COLOR["commit_before"], lw=0.8, label="E2 code review ($p \\geq$ 0.75 rule)")
    v2 = [hle[m]["min_p"]["auroc_p_predicts_hold"] for m in hm]
    ax.plot(hx + 0.08, v2, marker="s", ms=3.2, color="#08306b", lw=0.8, ls="--", label="HLE 2-turn (min $p \\geq$ 0.85 rule)")
    ax.axhline(0.5, color="#999999", lw=0.6, ls="--", zorder=0)
    ax.text(2.0, 0.49, "chance: decision ignores $p$", fontsize=5.8, color="#777777", va="top", ha="center")
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("AUROC, pre-reply $p \\rightarrow$ hold")
    fs.ptitle(ax, "a", "Do decisions track the bet?")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=5.9, handletextpad=0.3)
    # (b) bad outcome when violating vs following (E2, t = 0.75)
    ax = axs[1]
    bv = [e2[m]["bet_first/real_reply_targeted"]["by_t"]["0.75"]["bad_rate_if_violation"] for m in fs.MODEL_ORDER]
    bcn = [e2[m]["bet_first/real_reply_targeted"]["by_t"]["0.75"]["bad_rate_if_consistent"] for m in fs.MODEL_ORDER]
    nv = [e2[m]["bet_first/real_reply_targeted"]["by_t"]["0.75"]["violations"] for m in fs.MODEL_ORDER]
    w = 0.36
    ax.bar(xs - w / 2, 100 * np.array(bv), width=w, color="#c0392b", label="Dropped a claim it bet $\\geq$ 0.75 on")
    ax.bar(xs + w / 2, 100 * np.array(bcn), width=w, color="#bdbdbd", label="Acted in line with the bet")
    for x, val, n in zip(xs, bv, nv):
        ax.text(x - w / 2, 100 * val + 1.5, f"n={n}", ha="center", fontsize=5.6, rotation=90, va="bottom")
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Decision was wrong (%)")
    fs.ptitle(ax, "b", "E2: cost of ignoring the bet")
    ax.legend(loc="upper left", fontsize=5.9, handletextpad=0.3)
    # (c) decision accuracy actual vs following the rule
    ax = axs[2]
    act = [e2[m]["bet_first/real_reply_targeted"]["decision_accuracy"] for m in fs.MODEL_ORDER]
    rul = [e2[m]["bet_first/real_reply_targeted"]["by_t"]["0.75"]["rule_decision_accuracy"] for m in fs.MODEL_ORDER]
    for x, a, r in zip(xs, act, rul):
        ax.annotate("", xy=(x - 0.1, 100 * r), xytext=(x - 0.1, 100 * a),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=fs.ARM_COLOR["commit_before"], mutation_scale=6))
    ax.plot(xs - 0.1, 100 * np.array(act), "o", ms=3.2, mfc="white", mec=fs.ARM_COLOR["commit_before"], label="E2 actual")
    ax.plot(xs - 0.1, 100 * np.array(rul), "o", ms=3.2, color=fs.ARM_COLOR["commit_before"], label="E2 if rule obeyed")
    ha = [hle[m]["min_p"]["decision_accuracy"] for m in hm]
    hr = [hle[m]["min_p"]["by_t"]["0.85"]["rule_decision_accuracy"] for m in hm]
    for x, a, r in zip(hx, ha, hr):
        ax.annotate("", xy=(x + 0.12, 100 * r), xytext=(x + 0.12, 100 * a),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#08306b", mutation_scale=6, ls="--"))
    ax.plot(hx + 0.12, 100 * np.array(ha), "s", ms=3.0, mfc="white", mec="#08306b", label="HLE actual")
    ax.plot(hx + 0.12, 100 * np.array(hr), "s", ms=3.0, color="#08306b", label="HLE if rule obeyed")
    ax.set_ylim(40, 108)
    ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax.set_ylabel("Decision accuracy (%)")
    fs.ptitle(ax, "c", "If the bet were obeyed")
    ax.legend(loc="upper left", fontsize=5.6, ncol=2, handletextpad=0.2, columnspacing=0.6)
    for i, ax in enumerate(axs):
        ax.set_xticks(xs)
        ax.set_xticklabels([fs.MODEL_SHORT[m] for m in fs.MODEL_ORDER], rotation=30, ha="right")
        ax.set_xlim(-0.6, 4.6)
    return fs.save(fig, "fig06_bet_following")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 7: deference operating points and expected cost
# ═══════════════════════════════════════════════════════════════════════════

def fig07_deference():
    dc = J("deference_costs.json")
    S = dc["settings"]
    panels = [("e2", "pooled", "E2 code review\n5 models pooled"), ("swi", "pooled", "SWE-Interact mixed\n4 Flash models (r1)"),
              ("swi_r1r2", "3.8 Flash (r1+r2, n=40)", "SWE-Interact mixed\n3.8 Flash (r1+r2)"),
              ("ale", "pooled", "ALE-E2 reviewer pushback\n5 models pooled")]
    mats = list(dc["cost_matrices"].keys())
    fig, axs = plt.subplots(2, 4, figsize=(fs.DCOL, 4.3), gridspec_kw={"hspace": 0.75, "wspace": 0.42,
                                                                       "height_ratios": [1.15, 1]})
    for j, (sk, ck, ttl) in enumerate(panels):
        cell = S[sk]["cells"][ck]
        arms = [a for a in S[sk]["arms"] if a in cell]
        ax = axs[0, j]
        for ba in (0.9, 0.8, 0.7, 0.6):
            c = 2 * (1 - ba) * 100
            ax.plot([0, c], [c, 0], color="#e0e0e0", lw=0.6, zorder=0)
            if c <= 70:
                ax.text(c / 2 + 1, c / 2 + 1, f"BA {ba:.1f}", fontsize=5.2, color="#a0a0a0", rotation=-45, ha="left")
        for a in arms:
            r = cell[a]
            ax.plot(100 * r["p_false_cave"], 100 * r["p_stubborn"], marker="o" if a in fs.OURS or a == "bet_first" else "s",
                    ms=4.4 if a in fs.OURS or a == "bet_first" else 3.4, color=fs.ARM_COLOR[a], ls="none",
                    mec="k" if a in fs.OURS or a == "bet_first" else fs.ARM_COLOR[a], mew=0.4, zorder=3)
        ref = S[sk]["reference_arm"]
        ax.set_xlim(-3, max(60, 100 * max(cell[a]["p_false_cave"] for a in arms) + 6))
        ax.set_ylim(-3, max(75, 100 * max(cell[a]["p_stubborn"] for a in arms) + 6))
        ax.set_xlabel("False caves (%)\n(user wrong, agent gave in)", fontsize=6.6)
        if j == 0:
            ax.set_ylabel("Stubborn holds (%)\n(user right, agent ignored)", fontsize=6.6)
        fs.ptitle(ax, "abcd"[j], ttl, fontsize=6.9)
        ax.grid(axis="y", visible=False)
        nw = cell[ref]["n_user_wrong"]
        nr = cell[ref]["n_user_right"]
        ax.text(0.97, 0.97, f"n = {nw:,} / {nr} events\n(user wrong / right)", ha="right", va="top", fontsize=5.4,
                transform=ax.transAxes, color="#555555")
        # costs
        ax = axs[1, j]
        w = 0.8 / len(arms)
        for i, a in enumerate(arms):
            vals = [cell[a]["costs"][mtx] for mtx in mats]
            ax.bar(np.arange(len(mats)) - 0.4 + w / 2 + i * w, vals, width=w, color=fs.ARM_COLOR[a],
                   edgecolor="k" if a in fs.OURS or a == "bet_first" else "none", lw=0.4)
        best = S[sk]["best_arm_by_cost"].get(ck) if isinstance(S[sk]["best_arm_by_cost"], dict) else None
        ax.set_xticks(np.arange(len(mats)))
        ax.set_xticklabels([m.split()[-1] for m in mats])
        ax.set_xlabel("Cost ratio  false cave : stubborn", fontsize=6.6)
        if j == 0:
            ax.set_ylabel("Expected cost per pushback\n(lower is better)", fontsize=6.6)
        fs.ptitle(ax, "efgh"[j], "expected cost", fontsize=6.9)
    allarms = ["stock", "standard", "claims_after", "conf_gate", "sc_vote3", "verify_then_respond", "bet_first",
               "commit_before", "cb_gated", "claims_no_p"]
    lab = dict(fs.ARM_LABEL)
    lab["standard"] = "E2 check-after (reference)"
    lab["bet_first"] = "E2 bet before reply"
    hs = [Line2D([], [], color=fs.ARM_COLOR[a], marker="o" if a in fs.OURS or a == "bet_first" else "s", ls="none",
                 ms=4, mec="k" if a in fs.OURS or a == "bet_first" else fs.ARM_COLOR[a], mew=0.4, label=lab[a])
          for a in allarms if a != "standard"]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, fontsize=6.3, columnspacing=1.0,
               handletextpad=0.3)
    return fs.save(fig, "fig07_deference")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 8: cost vs accuracy (Pareto)
# ═══════════════════════════════════════════════════════════════════════════

def fig08_cost():
    c = J("cost_per_point.json")
    cells = c["cells"]
    panels = [("ds11m", "DeepSWE (mini-swe-agent)*"), ("ds11", "DeepSWE (opencode)"), ("tb4", "Terminal-Bench 4.0"),
              ("swi_overlay", "SWE-Interact, pure pushback"), ("swi_mixed", "SWE-Interact, mixed"),
              ("ale_claw_t300", "ALE-Claw, 300 turns"), ("ale_claw_s1", "ALE-Claw, 80 turns"),
              ("hle_single", "HLE single shot"), ("hle_2turn", "HLE 2-turn pushback"),
              ("finance_tools", "Finance Agent, live tools"), ("finance_closed", "Finance Agent, closed book")]
    score_y = {"ale_claw_t300", "ale_claw_s1", "finance_tools", "finance_closed"}
    fig, axs = plt.subplots(3, 4, figsize=(fs.DCOL, 5.2), gridspec_kw={"hspace": 0.62, "wspace": 0.42})
    axs = axs.ravel()
    for i, (b, ttl) in enumerate(panels):
        ax = axs[i]
        bc = [e for e in cells if e["bench"] == b]
        for m in fs.MODEL_ORDER:
            st = [e for e in bc if e["rung"] == m and e["arm"] == "stock"]
            if not st:
                continue
            st = st[0]

            fa = {(x["bench"], x["model"]): x for x in cell_effects() if x["arm"] == "cb_draftfirst"}

            def yv(e):
                # Harbor: pass rates from first-attempt per_task.json (paired tasks); tokens from retained trials
                if b in ("ds11", "ds11m", "tb4") and (b, m) in fa:
                    return fa[(b, m)]["stock" if e["arm"] == "stock" else "arm_rate"]
                if b in score_y and e.get("mean_score") is not None:
                    return 100 * e["mean_score"]
                return 100 * e["pass_rate"]
            x0, y0 = st["total_tokens_mean"], yv(st)
            ax.plot(x0, y0, marker=fs.MODEL_MARKER[m], color=fs.ARM_COLOR["stock"], ms=4.2 if m == "G" else 3.2,
                    ls="none", zorder=4)
            for e in bc:
                if e["rung"] != m or e["arm"] == "stock":
                    continue
                arm = "cb_draftfirst" if e["arm"] == "tib" else e["arm"]
                if arm not in fs.OURS:
                    continue
                x1, y1 = e["total_tokens_mean"], yv(e)
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>", lw=0.7, color=fs.ARM_COLOR[arm], mutation_scale=5,
                                            shrinkA=1.5, shrinkB=1.5))
                ax.plot(x1, y1, marker=fs.MODEL_MARKER[m], color=fs.ARM_COLOR[arm], ms=4.2 if m == "G" else 3.2,
                        ls="none", zorder=5)
        ax.set_xscale("log")
        fs.log_tokens_axis(ax)
        ax.set_title(ttl, fontsize=6.9)
        ax.tick_params(labelsize=6.2)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(("Score" if b in score_y else ("First-attempt pass" if b in ("ds11", "ds11m", "tb4") else "Pass rate")) + " (%)",
                      fontsize=6.6)
        if i >= 7:
            ax.set_xlabel("Tokens per task (log scale)", fontsize=6.6)
    ax = axs[len(panels)]
    ax.axis("off")
    hs = [Line2D([], [], color="#555555", marker=fs.MODEL_MARKER[m], ls="none", ms=4, label=fs.MODEL_LABEL[m])
          for m in fs.MODEL_ORDER]
    hs += [Line2D([], [], color=fs.ARM_COLOR[a], lw=0.9, marker=">", ms=3, label=("Standard agent" if a == "stock" else fs.ARM_LABEL[a]))
           for a in ("stock", "commit_before", "cb_gated", "cb_draftfirst")]
    ax.legend(handles=hs, loc="center", fontsize=6.3, handletextpad=0.3, borderaxespad=0)
    return fs.save(fig, "fig08_cost_pareto")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 9: why TIB lost
# ═══════════════════════════════════════════════════════════════════════════
LOSS_LABEL = {
    "timeout_api_starved": "Timed out (API outage)",
    "timeout_budget_exhausted": "Timed out (budget)",
    "terminated_timeout": "Timed out (budget)",
    "turn_budget_exhausted": "Ran out of turns",
    "no_committed_patch": "No committed patch",
    "empty_or_safety_blocked_responses": "Empty / safety-blocked replies",
    "claim_book_skipped": "Skipped the audit (no Bet Card)",
    "post_audit_edits_still_wrong": "Edited after audit, still wrong",
    "overconfident_audit_all_p_ge_0.95": "Confident audit, still wrong (all $p \\geq$ 0.95)",
    "other_wrong_answer": "Other wrong answer / lower score",
    "other_lower_score": "Other wrong answer / lower score",
}
LOSS_ORDER = ["Timed out (API outage)", "Timed out (budget)", "Ran out of turns", "No committed patch",
              "Empty / safety-blocked replies", "Skipped the audit (no Bet Card)", "Edited after audit, still wrong",
              "Confident audit, still wrong (all $p \\geq$ 0.95)", "Other wrong answer / lower score"]
LOSS_COLOR = dict(zip(LOSS_ORDER, ["#d62728", "#f4a582", "#fddbc7", "#8c564b", "#c49c94", "#9467bd",
                                   "#c5b0d5", "#1f77b4", "#c7c7c7"]))


def fig09_losses():
    L = J("log_mining.json")
    rows = []
    bn = {"ds11": "DeepSWE (opencode)", "ds11m": "DeepSWE (mini-swe)", "tb4": "Terminal-Bench"}
    for l in L["losses"]:
        rows.append((f"{bn[l['bench']]}, {fs.MODEL_LABEL[l['rung']]}", l["W"], l["L"], l["loss_categories"]))
    for l in L["ale_t300"]:
        rows.append((f"ALE-Claw t300, {fs.MODEL_LABEL[l['rung']]}", l["W"], l["L"], l["loss_categories"]))
    fig, ax = plt.subplots(figsize=(fs.DCOL, 2.6))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, w, lo, cats) in zip(ys, rows):
        agg = Counter()
        for k, v in cats.items():
            agg[LOSS_LABEL.get(k, k)] += v
        left = 0
        for cname in LOSS_ORDER:
            v = agg.get(cname, 0)
            if v:
                ax.barh(y, v, left=left, color=LOSS_COLOR[cname], height=0.68, lw=0.3, edgecolor="white")
                if v >= 3:
                    ax.text(left + v / 2, y, str(v), ha="center", va="center", fontsize=5.8,
                            color="white" if cname in ("Timed out (API outage)", "Confident audit, still wrong (all $p \\geq$ 0.95)",
                                                       "Skipped the audit (no Bet Card)", "No committed patch") else "k")
                left += v
        ax.text(left + 0.6, y, f"{w} W / {lo} L", va="center", fontsize=6.2, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    ax.axhline(ys[len(L["losses"]) - 1] - 0.5, color="#bbbbbb", lw=0.5)
    ax.set_xlabel("Tasks where the standard agent won and TIB (draft-first) lost")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#ededed", lw=0.5)
    ax.set_xlim(0, 44)
    hs = [Patch(color=LOSS_COLOR[c], label=c) for c in LOSS_ORDER]
    ax.legend(handles=hs, loc="lower right", fontsize=6.0, ncol=1, handlelength=1.0, handletextpad=0.4,
              bbox_to_anchor=(1.0, 0.0))
    return fs.save(fig, "fig09_loss_taxonomy")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 10: scoring audit (Harbor)
# ═══════════════════════════════════════════════════════════════════════════

def fig10_scoring_audit():
    L = J("log_mining.json")
    pr = [p for p in L["retry_audit"]["paired"] if p["counted"]["n"] >= 20]
    order = {("ds11", "G"): 0, ("ds11m", "G"): 1, ("tb4", "S"): 2, ("tb4", "W"): 3, ("tb4", "G"): 4, ("ds11", "S"): 5}
    pr.sort(key=lambda p: order.get((p["bench"], p["rung"]), 9))
    fig, axs = plt.subplots(1, 3, figsize=(fs.DCOL, 2.25), gridspec_kw={"wspace": 0.55, "width_ratios": [1.4, 1.15, 0.9]})
    ax = axs[0]
    bn = {"ds11": "DeepSWE oc", "ds11m": "DeepSWE mini", "tb4": "T-Bench"}
    rules = [("counted", "Counted (prefers a passing retry)", "o", "#999999"),
             ("itt_excuse_infra", "First attempt, infra crashes excused", "D", "#5e3c99"),
             ("itt_first_scored", "First scored attempt", "s", "#1b9e77")]
    ys = np.arange(len(pr))[::-1]
    for y, p in zip(ys, pr):
        vals = [p[k]["delta_pts"] for k, *_ in rules]
        ax.plot([min(vals), max(vals)], [y, y], color="#cccccc", lw=1.4, zorder=1, solid_capstyle="round")
        for (k, lab, mk, col), v in zip(rules, vals):
            ax.plot(v, y, marker=mk, color=col, ms=3.6, ls="none", zorder=3)
        ax.text(31, y, f"n={p['counted']['n']}", fontsize=5.8, va="center", color="#555555")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{bn[p['bench']]}, {fs.MODEL_SHORT[p['rung']]}" for p in pr], fontsize=6.6)
    fs.zero_line(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#ededed", lw=0.5)
    ax.set_xlim(-26, 36)
    ax.set_xlabel("TIB $-$ standard (pp)")
    ax.legend(handles=[Line2D([], [], marker=mk, color=col, ls="none", ms=3.6, label=lab) for k, lab, mk, col in rules],
              loc="upper center", bbox_to_anchor=(0.4, -0.2), fontsize=5.6, ncol=2, handletextpad=0.2,
              columnspacing=0.6)
    fs.ptitle(ax, "a", "Harbor: three scoring rules", x=-0.45)
    # (b) four-way, same 58 tasks
    ax = axs[1]
    fw = L["harness"]["four_way"]
    vals = [100 * fw["opencode"]["stock"], 100 * fw["opencode"]["arm"], 100 * fw["mswea"]["stock"], 100 * fw["mswea"]["arm"]]
    xs = np.array([0, 0.85, 2.1, 2.95])
    cols = [fs.ARM_COLOR["stock"], fs.ARM_COLOR["cb_draftfirst"]] * 2
    ax.bar(xs, vals, width=0.8, color=cols)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.2)
    ax.set_xticks([0.425, 2.525])
    ax.set_xticklabels(["opencode\nno commit rule", "mini-swe-agent\n+ commit rule"], fontsize=5.6)
    ax.set_ylabel("Pass rate, same %d tasks (%%)" % fw["opencode"]["n"], fontsize=6.6)
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(handles=[Patch(color=fs.ARM_COLOR["stock"], label="Standard"), Patch(color=fs.ARM_COLOR["cb_draftfirst"], label="Draft-first")],
              loc="upper center", fontsize=5.8, ncol=2, handlelength=1.0, columnspacing=0.8)
    for x0, k in ((0.425, "opencode"), (2.525, "mswea")):
        f = fw[k]
        top = 100 * max(f["stock"], f["arm"])
        ax.text(x0, top + 6, f"{f['delta_pts']:+.1f} pp\n{f['W']}W / {f['L']}L", ha="center", va="bottom", fontsize=5.6)
    fs.ptitle(ax, "b", "DeepSWE, 4.0 Pro")
    # (c) what the mini-swe-agent wins were
    ax = axs[2]
    wk = L["harness"]["mswea_wins_stock_failure_kind"]
    prof = L["harness"]["profile"]
    t_out = wk.get("stock_timed_out", 0)
    near = wk.get("stock_near_miss_partial>=0.95", 0)
    lw = fw["mswea"]["L"]
    ax.bar([0], [t_out], color="#d62728", width=0.6, label="Standard timed out with\nempty patch (no commits)")
    ax.bar([0], [near], bottom=[t_out], color="#fdae61", width=0.6, label="Standard near-miss\n(partial score $\\geq$ 0.95)")
    ax.bar([1], [lw], color="#9467bd", width=0.6, label="TIB skipped audit")
    ax.text(0, t_out / 2, str(t_out), ha="center", va="center", fontsize=6.4, color="white")
    ax.text(0, t_out + near / 2, str(near), ha="center", va="center", fontsize=6.4)
    ax.text(1, lw + 0.2, str(lw), ha="center", fontsize=6.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["TIB wins", "TIB losses"])
    ax.set_ylabel("Tasks (mini-swe-agent, 4.0 Pro)", fontsize=6.6)
    ax.set_ylim(0, 17)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.legend(loc="upper right", fontsize=5.4, handlelength=0.9, borderaxespad=0.1)
    es, et = prof["ds11m_G_stock"]["empty_patch"], prof["ds11m_G_tib"]["empty_patch"]
    ax.text(1, 3.0, f"empty patch\nstd {es}/{prof['ds11m_G_stock']['n']}\nTIB {et}/{prof['ds11m_G_tib']['n']}",
            fontsize=5.6, ha="center", color="#555555")
    ax.grid(axis="y", color="#ededed")
    fs.ptitle(ax, "c", "Where the wins came from")
    return fs.save(fig, "fig10_scoring_audit")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 11: claim-book compliance and informativeness
# ═══════════════════════════════════════════════════════════════════════════

def fig11_claimbooks():
    L = J("log_mining.json")
    bc = J("bet_calibration.json")
    fig, axs = plt.subplots(1, 3, figsize=(fs.DCOL, 2.2), gridspec_kw={"wspace": 0.5, "width_ratios": [1.1, 1.3, 1]})
    # (a) compliance
    ax = axs[0]
    items = []
    for k, lab in (("ds11_G_tib", "DeepSWE oc, 4.0 Pro"), ("ds11m_G_tib", "DeepSWE mini, 4.0 Pro")):
        pf = L["harness"]["profile"][k]
        items.append((lab, pf["claim_book_written"], pf["claim_book_with_p"]))
    for l in sorted(L["ale_t300"], key=lambda r: fs.MODEL_RANK[r["rung"]]):
        items.append((f"ALE-Claw t300, {fs.MODEL_SHORT[l['rung']]}", l["claim_book_rate_tib"], None))
    ys = np.arange(len(items))[::-1]
    for y, (lab, w, wp) in zip(ys, items):
        ax.barh(y, 100 * w, color=fs.ARM_COLOR["cb_draftfirst"], height=0.62, lw=0)
        if wp is not None:
            ax.barh(y, 100 * wp, color="#c9b8e3", height=0.26, lw=0)
        ax.text(100 * w + 2, y, f"{100 * w:.0f}%", va="center", fontsize=6.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=6.4)
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Trajectories with a Bet Card (%)", fontsize=6.6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#ededed")
    ax.legend(handles=[Patch(color=fs.ARM_COLOR["cb_draftfirst"], label="Bet Card written"),
                       Patch(color="#b9a6d6", label="... with parsable $p$")], loc="lower right", fontsize=5.6,
              bbox_to_anchor=(1.02, 0.12))
    fs.ptitle(ax, "a", "Bet Card compliance", x=-0.6)
    # (b) all-high bets vs actual pass
    ax = axs[1]
    bcal = L["bet_calibration"]
    keys = [k for k in ("ds11_G_tib", "ds11m_G_tib", "ds11_S_tib", "tb4_G_tib", "tb4_S_tib") if k in bcal]
    names = {"ds11_G_tib": "DS-oc\n4.0 Pro", "ds11m_G_tib": "DS-mini\n4.0 Pro", "ds11_S_tib": "DS-oc\n3.8 F",
             "tb4_G_tib": "TB4\n4.0 Pro", "tb4_S_tib": "TB4\n3.8 F"}
    xs = np.arange(len(keys))
    allhi = [100 * bcal[k]["frac_min_p_ge_0.95"] for k in keys]
    meanp = [100 * bcal[k]["mean_of_mean_p"] for k in keys]
    pas = [100 * bcal[k]["pass_rate"] for k in keys]
    ax.bar(xs, allhi, width=0.62, color="#d9d0ea", label="Bet Cards with every $p \\geq$ 0.95")
    ax.plot(xs, meanp, ls="none", marker="_", ms=10, mew=1.3, color=fs.ARM_COLOR["cb_draftfirst"], label="Mean stated $p$")
    ax.plot(xs, pas, ls="none", marker="o", ms=4, color="#d62728", label="Actual pass rate")
    for x, k, pa in zip(xs, keys, pas):
        ax.text(x, 3, f"AUROC\n{bcal[k]['auroc_min_p_predicts_pass']:.2f}", ha="center", fontsize=5.4, color="#333333")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{names[k]}\nn={bcal[k]['n']}" for k in keys], fontsize=5.7)
    ax.set_ylim(0, 138)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Share / rate (%)", fontsize=6.6)
    ax.legend(loc="upper center", fontsize=5.6, ncol=2, handletextpad=0.2, columnspacing=0.6, borderaxespad=0.1)
    fs.ptitle(ax, "b", "Agentic Bet Cards: all-high $p$")
    # (c) SWE-Interact live claims: distribution of p
    ax = axs[2]
    pdist = bc["swi"]["q1"]["p_distribution"]
    ms = [m for m in fs.MODEL_ORDER if f"commit_before/{m}" in pdist]
    xs = np.arange(len(ms))
    eq1 = [100 * pdist[f"commit_before/{m}"]["share_p_eq_1"] for m in ms]
    ge9 = [100 * pdist[f"commit_before/{m}"]["share_p_ge_0.9"] for m in ms]
    lt5 = [100 * pdist[f"commit_before/{m}"]["share_p_lt_0.5"] for m in ms]
    ax.bar(xs, eq1, color=fs.ARM_COLOR["commit_before"], width=0.62, label="$p$ = 1.0")
    ax.bar(xs, np.array(ge9) - np.array(eq1), bottom=eq1, color="#9ecae1", width=0.62, label="0.9 $\\leq p$ < 1")
    ax.bar(xs, 100 - np.array(ge9), bottom=ge9, color="#eeeeee", width=0.62, label="$p$ < 0.9")
    for x, m in zip(xs, ms):
        ax.text(x, 102, f"{pdist[f'commit_before/{m}']['n_claims']:,}", ha="center", fontsize=5.0, rotation=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([fs.MODEL_SHORT[m] for m in ms], rotation=30, ha="right")
    ax.set_ylim(0, 138)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Share of claims (%)", fontsize=6.6)
    ax.legend(loc="upper center", fontsize=5.6, ncol=3, handletextpad=0.2, columnspacing=0.5, borderaxespad=0.1,
              handlelength=1.0)
    fs.ptitle(ax, "c", "SWE-Interact live claims")
    return fs.save(fig, "fig11_claimbooks")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 12: judge robustness (preliminary)
# ═══════════════════════════════════════════════════════════════════════════

def fig12_judge():
    s = J("second_judge.json")
    ev = s["events"]
    fig, axs = plt.subplots(1, 4, figsize=(fs.DCOL, 2.15),
                            gridspec_kw={"wspace": 0.7, "width_ratios": [1.3, 1.0, 1.0, 1.0]})
    # (a) agreement per cell x arm
    ax = axs[0]
    cnt = Counter((e["cell"], e["arm"], e["label_second"] == e["label_primary"]) for e in ev)
    order_c = ["S_mixed_r1r2", "G_overlay_r1r2", "WFM_mixed_r1"]
    order_a = ["stock", "claims_after", "commit_before", "cb_gated"]
    keys = sorted({(e["cell"], e["arm"]) for e in ev}, key=lambda k: (order_c.index(k[0]), order_a.index(k[1])))
    cname = {"S_mixed_r1r2": "3.8 F mixed", "G_overlay_r1r2": "4.0 Pro pushback", "WFM_mixed_r1": "W/F/M mixed"}
    short = {"stock": "standard", "claims_after": "check-after", "commit_before": "always-on", "cb_gated": "on-demand"}
    ys = np.arange(len(keys))[::-1]
    for y, k in zip(ys, keys):
        ag, dis = cnt[(k[0], k[1], True)], cnt[(k[0], k[1], False)]
        ax.barh(y, ag, color=fs.ARM_COLOR[k[1]], height=0.62)
        if dis:
            ax.barh(y, dis, left=ag, color="#d62728", height=0.62)
        ax.text(ag + dis + 1.5, y, f"{ag}/{ag + dis}", va="center", fontsize=5.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{cname[k[0]]}, {short[k[1]]}" for k in keys], fontsize=5.8)
    ax.set_xlim(0, 155)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Pushback turns re-judged", fontsize=6.6)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color="#888888", label="Same label"), Patch(color="#d62728", label="Different label")],
              loc="lower right", bbox_to_anchor=(1.04, 0.0), fontsize=5.6, borderaxespad=0.1)
    fs.ptitle(ax, "a", "Agreement by cell", x=-0.75)
    # (b) confusion matrix
    ax = axs[1]
    labs = ["held", "caved", "correct_update", "stubborn", "justified_invalid"]
    nice = ["held", "caved", "updated", "stubborn", "invalid"]
    M = np.zeros((5, 5), int)
    for e in ev:
        if e["label_second"] in labs and e["label_primary"] in labs:
            M[labs.index(e["label_primary"]), labs.index(e["label_second"])] += 1
    ax.imshow(np.log1p(M), cmap="Blues", vmin=0, vmax=np.log1p(M.max()) * 1.1)
    for i in range(5):
        for k in range(5):
            ax.text(k, i, str(M[i, k]), ha="center", va="center", fontsize=5.8,
                    color="white" if M[i, k] > 40 else ("#b2182b" if (M[i, k] and i != k) else "k"))
    ax.set_xticks(range(5))
    ax.set_xticklabels(nice, rotation=40, ha="right", fontsize=5.8)
    ax.set_yticks(range(5))
    ax.set_yticklabels(nice, fontsize=5.8)
    ax.set_xlabel("Second judge (3.8 Flash)", fontsize=6.4)
    ax.set_ylabel("Primary judge (3.5 Flash)", fontsize=6.4)
    ax.grid(False)
    ag = s["agreement_selected"]["overall"]
    fs.ptitle(ax, "b", f"{ag['agree']}/{ag['n']}, $\\kappa$ = {ag['kappa']:.2f}", fontsize=6.8, x=-0.3)
    # (c, d) headline metrics on 3.8 Flash under three label sets
    h = s["headline"]
    arms = ["stock", "claims_after", "commit_before", "cb_gated"]
    srcs = (("primary", "Primary judge", 1.0, None), ("second", "Second judge", 0.6, None), ("both_agree", "Both agree", 0.3, "////"))
    xs = np.arange(len(arms))
    w = 0.27
    for pi, (num, den, ttl, ylab, ylim) in enumerate((("correct_update", "n_user_right", "Listening, 3.8 F", "Real bugs fixed (%)", 100),
                                                       ("false_cave", "n_user_wrong", "False caving, 3.8 F", "False caves (%)", 15))):
        ax = axs[2 + pi]
        for j, (src, lab, alpha, hatch) in enumerate(srcs):
            a = h[src]["S_mixed_r1r2"]["arms"]
            v = [100 * a[k][num] / a[k][den] for k in arms]
            ax.bar(xs + (j - 1) * w, v, width=w, color=[fs.ARM_COLOR[k] for k in arms], alpha=alpha, hatch=hatch,
                   edgecolor="#333333" if hatch else "none", lw=0.3)
            if pi == 1:
                for x, k, vv in zip(xs + (j - 1) * w, arms, v):
                    ax.text(x, vv + 0.3, f"{a[k][num]}", ha="center", fontsize=4.8, color="#444444")
        ax.set_xticks(xs)
        ax.set_xticklabels(["standard", "check-after", "always-on", "on-demand"], fontsize=5.6, rotation=35, ha="right")
        ax.set_ylabel(ylab, fontsize=6.4)
        ax.set_ylim(0, ylim)
        fs.ptitle(ax, "cd"[pi], ttl, fontsize=6.8)
    from matplotlib.patches import Rectangle
    axs[2].legend(handles=[Rectangle((0, 0), 1, 1, facecolor="#666666", alpha=a_, hatch=h_, edgecolor="#333333" if h_ else "none",
                                     lw=0.3, label=l_) for _, l_, a_, h_ in srcs], loc="upper left", fontsize=5.3,
                  handlelength=1.0, borderaxespad=0.1)
    NUMBERS["judge"] = {"judged": s["second_coverage"]["judged"], "unresolved": s["second_coverage"]["unresolved"], "agree": ag}
    return fs.save(fig, "fig12_judge_robustness")


# ═══════════════════════════════════════════════════════════════════════════

FIGS = {
    1: fig01_forest, 2: fig02_difficulty, 3: fig03_capability, 4: fig04_calibration, 5: fig05_hindsight,
    6: fig06_bet_following, 7: fig07_deference, 8: fig08_cost, 9: fig09_losses, 10: fig10_scoring_audit,
    11: fig11_claimbooks, 12: fig12_judge, 13: fig02b_difficulty_by_bench,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int)
    ap.add_argument("--slides", action="store_true")
    args = ap.parse_args()
    fs.setup("slides" if args.slides else "paper")
    if args.slides:
        fs.OUTDIR = fs.OUTDIR / "slides"
    for k, f in FIGS.items():
        if args.only and k not in args.only:
            continue
        paths = f()
        print(f"fig {k}: {paths[0].relative_to(ROOT)}")
    if NUMBERS:
        out = fs.OUTDIR / "report_numbers.json"
        prev = json.loads(out.read_text()) if out.exists() else {}
        prev.update(NUMBERS)
        out.write_text(json.dumps(prev, indent=1, default=float))


if __name__ == "__main__":
    main()
