#!/usr/bin/env python3
"""Are TIB's probability bets calibrated, and does the agent follow its own bets?

Pure post-hoc analysis over result files already on disk: no model runs, no docker, nothing in
``~/jobs`` is touched. Deterministic (fixed seed ``SEED`` for every bootstrap).

Q1  Calibration. For every claim that carries a probability ``p`` and has a ground-truth label we
    compute a 10-bin reliability diagram (equal-width bins), ECE, Brier score and AUROC, plus AUROC of
    ``1 - p`` for predicting the challenges the user was RIGHT to make (real defect). We compare the
    pre-reply bet with ratings written after the reply (``claims_after`` / check-after) to quantify the
    hindsight confidence drop on true vs false claims.
Q2  Bet-following. On pushback, classify the decision (held vs updated) and compare it with the
    pre-committed ``p`` of the challenged claim: share consistent with "hold iff p >= t" for
    t in {0.5, 0.6, 0.75, 0.85, 0.9}; bet violations (caved on p >= t / held on p < t) and whether
    violations predict bad outcomes (false caves on correct work, stubborn holds on real defects).

Sources (see docs/results/BET_CALIBRATION.md for the exact field mapping):
* E2 controlled code review   results/redo_v1/e2/{swi,swt}_{W,F,M,S}[_g4pro]/<model>/ratings.jsonl
* SWE-Interact live runs       results/redo_v1/stage2_swe_interact/<rung>/<arm>/mixed_r{1,2}/ (claims/,
                               overlay_judge/, asks.jsonl) + G overlay_r{1,2} under results/redo_v1_g4;
                               per-event outcomes from the cached rows results/analysis/swi_*_trials.json
* HLE                          results/hle/{single_shot,multi_turn}_*.jsonl
* ALE-E2                       results/redo_v1_ale/ale_e2_*.json (no probabilities persisted -> reported as such)

Usage:
    python3 scripts/analysis/bet_calibration.py            # ~2-4 min, writes JSON + md + figures
    python3 scripts/analysis/bet_calibration.py --boot 200 # faster, fewer bootstrap reps
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = C.ROOT
RES = C.RES
OUT_JSON = RES / "analysis" / "bet_calibration.json"
OUT_MD = RES / "analysis" / "bet_calibration.md"
FIG = ROOT / "docs" / "figures" / "analysis"
SEED = C.SEED
THRESHOLDS = (0.5, 0.6, 0.75, 0.85, 0.9)
NBINS = 10
RUNG_LABEL = {"W": "3.5 Flash-Lite", "F": "3.5 Flash", "M": "3.6 Flash", "S": "3.8 Flash", "G": "4.0 Pro"}
MODEL_RUNG = {"gemini-3.5-flash-lite": "W", "gemini-3.5-flash": "F", "gemini-3.6-flash": "M",
              "gemini-3.8-flash": "S", "gemini-4.0-argon": "G", "gemini-4.0-pro": "G"}
RUNGS = ("W", "F", "M", "S", "G")


# ── metrics ─────────────────────────────────────────────────────────────────

def _arr(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(p) & ~np.isnan(y)
    return p[m], y[m]


def bins(p, y, nb=NBINS):
    p, y = _arr(p, y)
    idx = np.minimum((p * nb).astype(int), nb - 1)
    out = []
    for b in range(nb):
        m = idx == b
        out.append({"lo": b / nb, "hi": (b + 1) / nb, "n": int(m.sum()),
                    "mean_p": float(p[m].mean()) if m.any() else None,
                    "frac_true": float(y[m].mean()) if m.any() else None})
    return out


def ece(p, y, nb=NBINS):
    p, y = _arr(p, y)
    if len(p) == 0:
        return None
    idx = np.minimum((p * nb).astype(int), nb - 1)
    tot = 0.0
    for b in range(nb):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / len(p))


def brier(p, y):
    p, y = _arr(p, y)
    return float(np.mean((p - y) ** 2)) if len(p) else None


def auroc(score, y):
    """P(score of a positive > score of a negative), ties = 1/2 (Mann-Whitney). None if one class."""
    s, y = _arr(score, y)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calib(p, y, clusters=None, boot=0, seed=SEED):
    """Calibration summary; cluster-bootstrap 95% CIs for ECE / Brier / AUROC when ``boot``."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    out = {"n": int(len(p)), "base_rate": float(y.mean()) if len(y) else None,
           "mean_p": float(p.mean()) if len(p) else None,
           "overconfidence": float(p.mean() - y.mean()) if len(p) else None,
           "ece": ece(p, y), "brier": brier(p, y), "auroc": auroc(p, y),
           "share_p_ge_0.9": float((p >= 0.9).mean()) if len(p) else None,
           "frac_true_when_p_ge_0.9": float(y[p >= 0.9].mean()) if (p >= 0.9).any() else None,
           "n_distinct_p": int(len(set(np.round(p, 4)))) if len(p) else 0}
    if boot and len(p) and clusters is not None:
        clusters = np.asarray(clusters)
        uniq = sorted(set(clusters.tolist()))
        by = {c: np.where(clusters == c)[0] for c in uniq}
        rng = np.random.default_rng(seed)
        e, b, a = [], [], []
        for _ in range(boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            ix = np.concatenate([by[uniq[k]] for k in pick])
            e.append(ece(p[ix], y[ix]))
            b.append(brier(p[ix], y[ix]))
            av = auroc(p[ix], y[ix])
            if av is not None:
                a.append(av)
        q = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if xs else None  # noqa: E731
        out.update(ece_ci=q(e), brier_ci=q(b), auroc_ci=q(a) if len(a) > boot // 2 else None,
                   n_clusters=len(uniq))
    out["bins"] = bins(p, y)
    return out


def follow(p, held, correct_hold, thresholds=THRESHOLDS):
    """Bet-following summary for challenged claims.

    p: pre-committed probability of the challenged claim; held: decision (1 = held, 0 = updated);
    correct_hold: ground truth (1 = holding was right, i.e. the user was wrong).
    """
    p, held, ch = (np.asarray(x, float) for x in (p, held, correct_hold))
    out = {"n": int(len(p)), "hold_rate": float(held.mean()) if len(p) else None,
           # threshold-free: does the pre-committed p predict the decision to hold?
           "auroc_p_predicts_hold": auroc(p, held),
           "decision_accuracy": float((held == ch).mean()) if len(p) else None, "by_t": {}}
    bad = held != ch
    for t in thresholds:
        hi = p >= t
        cave_hi = hi & (held == 0)
        hold_lo = (~hi) & (held == 1)
        viol = cave_hi | hold_lo
        rule_bad = hi != (ch == 1)   # outcome had the rule been followed exactly
        d = {"n_hi": int(hi.sum()), "n_lo": int((~hi).sum()),
             "consistent": float((~viol).mean()) if len(p) else None,
             "violations": int(viol.sum()), "caved_on_high_p": int(cave_hi.sum()),
             "held_on_low_p": int(hold_lo.sum()),
             "bad_rate_if_violation": float(bad[viol].mean()) if viol.any() else None,
             "bad_rate_if_consistent": float(bad[~viol].mean()) if (~viol).any() else None,
             "false_caves_among_high_p_caves": int((cave_hi & (ch == 1)).sum()),
             "stubborn_among_low_p_holds": int((hold_lo & (ch == 0)).sum()),
             "rule_decision_accuracy": float((~rule_bad).mean()) if len(p) else None}
        out["by_t"][str(t)] = d
    return out


# ── figures ─────────────────────────────────────────────────────────────────

def reliability_plot(panels: dict[str, dict[str, tuple]], path: Path, title: str):
    """panels: {panel title: {series label: (p, y)}}; one subplot per panel, bins of 0.1."""
    n = len(panels)
    cols = min(n, 5)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(2 * rows, cols, figsize=(3.3 * cols, 4.2 * rows),
                             gridspec_kw={"height_ratios": [3, 1] * rows}, squeeze=False)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    for k, (pt, series) in enumerate(panels.items()):
        r, c = divmod(k, cols)
        ax, axh = axes[2 * r][c], axes[2 * r + 1][c]
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
        for j, (lab, (p, y)) in enumerate(series.items()):
            p, y = _arr(p, y)
            if not len(p):
                continue
            bs = bins(p, y)
            xs = [b["mean_p"] for b in bs if b["n"]]
            ys = [b["frac_true"] for b in bs if b["n"]]
            ns = [b["n"] for b in bs if b["n"]]
            col = colors[j % len(colors)]
            ax.plot(xs, ys, "-o", c=col, ms=3, lw=1.2, label=f"{lab} (ECE {ece(p, y):.2f}, n={len(p)})")
            ax.scatter(xs, ys, s=[8 + 60 * math.sqrt(x / max(ns)) for x in ns], c=col, alpha=0.4)
            axh.bar([b["lo"] + 0.05 + (j - (len(series) - 1) / 2) * 0.08 / max(1, len(series) - 1)
                     for b in bs], [b["n"] for b in bs], width=0.08 / max(1, len(series)), color=col, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(pt, fontsize=9)
        ax.legend(fontsize=6, loc="upper left")
        if c == 0:
            ax.set_ylabel("fraction true")
            axh.set_ylabel("count")
        axh.set_xlim(0, 1)
        axh.set_yscale("log")
        axh.set_xlabel("stated p", fontsize=8)
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[2 * r][c].axis("off")
        axes[2 * r + 1][c].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── E2 ──────────────────────────────────────────────────────────────────────

E2_CONDS = ("bet", "none", "unjustified", "real", "bet_unjustified", "bet_real")


def e2_load():
    """C.e2_rows plus, for the stage-2 bet conditions, the model's RAW (pre-enforcement) answer:
    fsa.redo.e2.run keeps the recorded bet for every requirement with ``revise: false`` whatever the
    model returned, so the recorded bet_* ratings equal the bet unless the model chose to revise."""
    from fsa.redo.e2.run import _read
    rows = []
    d = RES / "redo_v1" / "e2"
    for run, itf, model in C.E2_RUNS:
        bench = run.split("_")[0]
        items = json.loads((d / itf).read_text())["items"]
        raw = {(r["cond"], r["item"]): (r.get("raw") or {}) for r in _read(d / run / model / "ratings.jsonl")
               if r["cond"] in ("bet_real", "bet_unjustified")}
        rids = [(it["id"], i["rid"]) for it in items for i in it["intents"]]
        rr = C.e2_rows(run, itf, model)
        assert len(rr) == len(rids)
        for r, (iid, rid) in zip(rr, rids):
            extra = {}
            for cond in ("bet_real", "bet_unjustified"):
                x = raw.get((cond, iid), {}).get(rid)
                extra[f"raw_{cond}"] = None if not x else {"p": x.get("p_satisfied"), "sat": x.get("satisfied"),
                                                           "revise": x.get("revise")}
            rows.append({**r, **extra, "rung": MODEL_RUNG[model], "bench": bench})
    return rows


def e2_q1(rows, boot):
    res = {"calibration": {}, "hindsight": {}, "user_right_auroc": {}}
    groups = {}
    for g in RUNGS:
        for bench in ("swi", "swt", "all"):
            groups[(g, bench)] = [r for r in rows if r["rung"] == g and (bench == "all" or r["bench"] == bench)]
    for bench in ("swi", "swt", "all"):
        groups[("pooled", bench)] = [r for r in rows if bench == "all" or r["bench"] == bench]
    for (g, bench), rs in groups.items():
        key = f"{g}/{bench}"
        cal = {}
        for cond in E2_CONDS:
            rr = [r for r in rs if r.get(cond)]
            p = [r[cond]["p"] / 100 for r in rr]
            y = [1.0 if r["judge_verdict"] == "satisfied" else 0.0 for r in rr]
            do_boot = boot if (bench == "all" and cond in ("bet", "none", "unjustified", "real")) else 0
            cal[cond] = calib(p, y, [f"{r['bench']}/{r['task']}" for r in rr], do_boot)
        res["calibration"][key] = cal
        # hindsight: paired drop (pre - post) on true vs false claims
        hs = {}
        for pre, post in (("none", "unjustified"), ("bet", "bet_unjustified"), ("none", "real"), ("bet", "bet_real"),
                          ("bet", "unjustified"), ("bet", "real"),
                          ("bet", "raw_bet_unjustified"), ("bet", "raw_bet_real")):
            rr = [r for r in rs if r.get(pre) and r.get(post) and r[post].get("p") is not None
                  and r[post].get("sat") is not None]
            d_true = [(r[pre]["p"] - r[post]["p"]) / 100 for r in rr if r["judge_verdict"] == "satisfied"]
            d_false = [(r[pre]["p"] - r[post]["p"]) / 100 for r in rr if r["judge_verdict"] != "satisfied"]
            hs[f"{pre}->{post}"] = {
                "n_true": len(d_true), "n_false": len(d_false),
                "drop_true": float(np.mean(d_true)) if d_true else None,
                "drop_false": float(np.mean(d_false)) if d_false else None,
                "discriminative_drop": (float(np.mean(d_false) - np.mean(d_true)) if d_true and d_false else None),
                "flip_true_to_unsat": float(np.mean([1.0 if (r[pre]["sat"] and not r[post]["sat"]) else 0.0
                                                     for r in rr if r["judge_verdict"] == "satisfied"])) if d_true else None,
                "flip_false_to_unsat": float(np.mean([1.0 if (r[pre]["sat"] and not r[post]["sat"]) else 0.0
                                                      for r in rr if r["judge_verdict"] != "satisfied"])) if d_false else None}
        res["hindsight"][key] = hs
        # user right to challenge: requirements the REAL reply names (reply_targets); right = control (real defect)
        tg = [r for r in rs if r.get("reply_targets") is True]
        ur = {}
        for cond in E2_CONDS:
            rr = [r for r in tg if r.get(cond)]
            yr = [1.0 if r.get("control") else 0.0 for r in rr]
            ur[cond] = {"n": len(rr), "n_user_right": int(sum(yr)),
                        "auroc_1_minus_p": auroc([1 - r[cond]["p"] / 100 for r in rr], yr),
                        "mean_p_user_right": float(np.mean([r[cond]["p"] / 100 for r, y in zip(rr, yr) if y])) if sum(yr) else None,
                        "mean_p_user_wrong": float(np.mean([r[cond]["p"] / 100 for r, y in zip(rr, yr) if not y])) if len(yr) > sum(yr) else None}
        # generic unjustified pushback challenges every requirement: user "right" = requirement unsatisfied
        ur["generic_unjustified_all_requirements"] = {
            c: auroc([1 - r[c]["p"] / 100 for r in rs if r.get(c)],
                     [0.0 if r["judge_verdict"] == "satisfied" else 1.0 for r in rs if r.get(c)])
            for c in ("bet", "none")}
        res["user_right_auroc"][key] = ur
    return res


def e2_q2(rows):
    """Asserted claims (pre-reply verdict satisfied) hit by a pushback; held = still satisfied after."""
    res = {}
    settings = {
        # arm, pre, post, population filter, label
        "bet_first/generic_unjustified": ("bet", "bet_unjustified", lambda r: True),
        "bet_first/real_reply_targeted": ("bet", "bet_real", lambda r: r.get("reply_targets") is True),
        "standard/generic_unjustified": ("none", "unjustified", lambda r: True),
        "standard/real_reply_targeted": ("none", "real", lambda r: r.get("reply_targets") is True),
        # what the model itself answered before the harness re-imposed the bet on revise=false entries
        "bet_first_raw/generic_unjustified": ("bet", "raw_bet_unjustified", lambda r: True),
        "bet_first_raw/real_reply_targeted": ("bet", "raw_bet_real", lambda r: r.get("reply_targets") is True),
    }
    for g in list(RUNGS) + ["pooled"]:
        rs = [r for r in rows if g == "pooled" or r["rung"] == g]
        out = {}
        for name, (pre, post, flt) in settings.items():
            rr = [r for r in rs if flt(r) and r.get(pre) and r.get(post) and r[pre]["sat"]
                  and r[post].get("sat") is not None]
            p = [r[pre]["p"] / 100 for r in rr]
            held = [1.0 if r[post]["sat"] else 0.0 for r in rr]
            ch = [1.0 if r["judge_verdict"] == "satisfied" else 0.0 for r in rr]
            f = follow(p, held, ch)
            if pre == "bet":
                f["revise_flag_rate"] = float(np.mean([1.0 if r[post].get("revise") else 0.0 for r in rr])) if rr else None
            out[name] = f
        res[g] = out
    return res


def e2_figures(rows):
    for bench in ("swi", "swt", "all"):
        panels = {}
        for g in RUNGS:
            rs = [r for r in rows if r["rung"] == g and (bench == "all" or r["bench"] == bench)]
            ser = {}
            # bet_unjustified is omitted: it is identical to the bet (no requirement was revised)
            for cond, lab in (("bet", "bet (pre-reply, ours)"), ("none", "blind (no reply)"),
                              ("unjustified", "check-after, unjust. reply"), ("real", "check-after, real reply")):
                rr = [r for r in rs if r.get(cond)]
                ser[lab] = ([r[cond]["p"] / 100 for r in rr], [1.0 if r["judge_verdict"] == "satisfied" else 0.0 for r in rr])
            panels[RUNG_LABEL[g]] = ser
        reliability_plot(panels, FIG / f"e2_reliability_{bench}.png",
                         f"E2 controlled code review ({bench}): p_satisfied vs frozen judge verdict")
    rr = [r for r in rows if r.get("bet")]
    reliability_plot({"E2 pooled, all models": {
        "bet (pre-reply)": ([r["bet"]["p"] / 100 for r in rr], [1.0 if r["judge_verdict"] == "satisfied" else 0.0 for r in rr]),
        "check-after (unjust.)": ([r["unjustified"]["p"] / 100 for r in rows if r.get("unjustified")],
                                  [1.0 if r["judge_verdict"] == "satisfied" else 0.0 for r in rows if r.get("unjustified")])}},
        FIG / "e2_reliability_pooled.png", "E2 pooled over models and benchmarks")


# ── SWE-Interact ────────────────────────────────────────────────────────────

SWI_CELLS = []   # (rung, arm, mode, run_dir)
for _rung in ("W", "F", "M", "S"):
    for _arm in ("commit_before", "cb_gated", "claims_after", "conf_gate"):
        for _mode in (("mixed_r1", "mixed_r2") if _rung == "S" else ("mixed_r1",)):
            SWI_CELLS.append((_rung, _arm, _mode, RES / "redo_v1" / "stage2_swe_interact" / _rung / _arm / _mode))
for _arm in ("commit_before", "claims_after"):
    for _mode in ("overlay_r1", "overlay_r2"):
        SWI_CELLS.append(("G", _arm, _mode, RES / "redo_v1_g4" / "stage2_swe_interact" / "G" / _arm / _mode))

_STOP = set("""the and for with that this are from into when then than must should will not but all any each its
have has was were been being which while also only does done code user users task test tests using used use
can may per via such other more most some same been they them their there here what who how why""".split())


def _tok(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9_]+", (s or "").lower()) if len(w) >= 3 and w not in _STOP]


def _tfidf(docs: list[list[str]]):
    df = Counter(w for d in docs for w in set(d))
    n = len(docs)
    vecs = []
    for d in docs:
        tf = Counter(d)
        v = {w: (1 + math.log(c)) * (math.log((n + 1) / (df[w] + 1)) + 1) for w, c in tf.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / nrm for w, x in v.items()})
    return vecs


def _cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def _jl(p: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _jr(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


MATCH_MIN = 0.30   # TF-IDF cosine needed to tie a free-text claim to a judged intent


def swi_load():
    from fsa.redo.swe_interact import metrics as SM
    cache = {}
    for f in ("swi_mixed_r1_trials.json", "swi_S_mixed_r1r2_trials.json", "swi_g4_overlay_r1r2_trials.json",
              "swi_g4_overlay_r1_trials.json"):
        p = RES / "analysis" / f
        if p.exists():
            for r in json.loads(p.read_text()):
                cache[(r["run"], r["trial"])] = r
    asks_out, events_out, skipped = [], [], Counter()
    for rung, arm, mode, rd in SWI_CELLS:
        if not (rd / "jobs").is_dir():
            skipped["missing_run_dir"] += 1
            continue
        try:
            trials = SM.trials_of(rd)
        except (OSError, PermissionError):
            skipped["trials_of_error"] += 1
            continue
        run_key = str(rd.relative_to(rd.parents[3]))  # stage2_swe_interact/<rung>/<arm>/<mode>
        for t in trials:
            try:
                turns = SM.swi_turns(t)
            except Exception:  # noqa: BLE001  (permission-denied / partial trial dirs are skipped, counted)
                skipped["turns_error"] += 1
                continue
            row = cache.get((run_key, t.name))
            ev_by_turn = {e["turn"]: e for e in (row or {}).get("caving") or []}
            turn_of_call = {r["opened_by_call"]: r["turn"] for r in turns if r.get("opened_by_call") is not None}
            asks = {int(a["call_index"]): a for a in _jl(t / "user-sim" / "asks.jsonl")}
            ojs, books = {}, {}
            for ci in asks:
                ojs[ci] = _jr(t / "user-sim" / "overlay_judge" / f"ask-{ci}.json")
                books[ci] = _jr(t / "user-sim" / "claims" / f"ask-{ci}.json")
            texts = {}
            for oj in ojs.values():
                texts.update((oj or {}).get("texts") or {})
            claim_texts = []
            for b in books.values():
                for c in ((b or {}).get("book") or {}).get("claims") or []:
                    claim_texts.append(c.get("claim", ""))
            keys = sorted(texts)
            vecs = _tfidf([_tok(texts[k]) for k in keys] + [_tok(x) for x in claim_texts])
            ivec = dict(zip(keys, vecs[:len(keys)]))
            cvec = {x: v for x, v in zip(claim_texts, vecs[len(keys):])}
            for ci, a in sorted(asks.items()):
                oj, bk = ojs.get(ci) or {}, books.get(ci) or {}
                labels = oj.get("labels") or {}
                revealed = oj.get("revealed") or []
                judged_ok = bool(revealed) and all(labels.get(k) in ("satisfied", "not_satisfied", "not_addressed")
                                                   for k in revealed)
                claims = []
                for c in (bk.get("book") or {}).get("claims") or []:
                    try:
                        pc = float(c.get("p"))
                    except (TypeError, ValueError):
                        continue
                    best, bs = None, 0.0
                    for k in revealed:
                        if k in ivec and labels.get(k) in ("satisfied", "not_satisfied", "not_addressed"):
                            s = _cos(cvec.get(c.get("claim", ""), {}), ivec[k])
                            if s > bs:
                                best, bs = k, s
                    claims.append({"p": pc, "intent": best if bs >= MATCH_MIN else None, "sim": bs,
                                   "label": (labels.get(best) == "satisfied") if bs >= MATCH_MIN else None,
                                   "text": c.get("claim", "")})
                srp = ((a.get("redo") or {}).get("sim_reply") or {})
                kind, target = srp.get("overlay_kind"), srp.get("overlay_target_intent")
                rec = {"rung": rung, "arm": arm, "mode": mode, "task": SM.trial_task(t), "trial": t.name,
                       "call_index": ci, "snapshot_chars": oj.get("snapshot_chars", 0), "judged_ok": judged_ok,
                       "all_revealed_satisfied": bool(oj.get("all_revealed_satisfied")), "n_revealed": len(revealed),
                       "gate_p": bk.get("gate_p"), "gated_off": bk.get("gated_off"), "book_ok": bool(bk.get("ok")),
                       "phase": bk.get("phase"), "claims": claims, "overlay_kind": kind, "overlay_target": target}
                asks_out.append(rec)
                if kind:
                    ev = ev_by_turn.get(turn_of_call.get(ci))
                    p_target, sim_t = None, 0.0
                    if target and target in ivec:
                        for c in (bk.get("book") or {}).get("claims") or []:
                            s = _cos(cvec.get(c.get("claim", ""), {}), ivec[target])
                            try:
                                pc = float(c.get("p"))
                            except (TypeError, ValueError):
                                continue
                            if s > sim_t:
                                sim_t, p_target = s, pc
                    ps = [c["p"] for c in claims]
                    events_out.append({**{k: rec[k] for k in ("rung", "arm", "mode", "task", "trial", "call_index",
                                                              "gate_p", "gated_off", "book_ok", "phase")},
                                       "kind": kind, "target": target,
                                       "p_target": p_target if sim_t >= MATCH_MIN else None, "sim_target": sim_t,
                                       "book_min_p": min(ps) if ps else None, "book_mean_p": float(np.mean(ps)) if ps else None,
                                       "event": ev})
    return asks_out, events_out, dict(skipped)


def _ev_truth(e):
    """(user_right, held, bad) for one overlay event, or None if it cannot be scored."""
    ev = e.get("event")
    if not ev or not ev.get("judged"):
        return None
    if e["kind"] == "justified":
        if not ev.get("valid"):
            return None
        held = 0.0 if ev.get("changed") else 1.0
        return 1.0, held, (0.0 if ev.get("fixed") else 1.0)
    held = 0.0 if ev.get("changed") else 1.0
    return 0.0, held, (1.0 if ev.get("lost") else 0.0)


def swi_q1(asks, events, boot):
    res = {"gate_calibration": {}, "claim_calibration": {}, "user_right_auroc": {}, "p_distribution": {}}
    for arm in ("cb_gated", "conf_gate"):
        for g in ("W", "F", "M", "S", "pooled"):
            rs = [a for a in asks if a["arm"] == arm and (g == "pooled" or a["rung"] == g)
                  and a["gate_p"] is not None and a["judged_ok"] and (a["snapshot_chars"] or 0) > 0]
            res["gate_calibration"][f"{arm}/{g}"] = calib([a["gate_p"] for a in rs],
                                                          [1.0 if a["all_revealed_satisfied"] else 0.0 for a in rs],
                                                          [a["task"] for a in rs], boot if g == "pooled" else 0)
    for arm in ("commit_before", "cb_gated", "claims_after"):
        for g in list(RUNGS) + ["pooled"]:
            rs = [a for a in asks if a["arm"] == arm and (g == "pooled" or a["rung"] == g)]
            allc = [c for a in rs for c in a["claims"]]
            matched = [(c, a) for a in rs for c in a["claims"] if c["label"] is not None]
            if not allc:
                continue
            pa = np.array([c["p"] for c in allc])
            res["p_distribution"][f"{arm}/{g}"] = {"n_claims": len(allc), "n_books": sum(1 for a in rs if a["claims"]),
                                                   "share_p_ge_0.9": float((pa >= 0.9).mean()),
                                                   "share_p_eq_1": float((pa >= 0.999).mean()),
                                                   "share_p_lt_0.5": float((pa < 0.5).mean()),
                                                   "mean_p": float(pa.mean()),
                                                   "match_rate": len(matched) / len(allc)}
            res["claim_calibration"][f"{arm}/{g}"] = calib([c["p"] for c, _ in matched],
                                                           [1.0 if c["label"] else 0.0 for c, _ in matched],
                                                           [a["task"] for _, a in matched], boot if g == "pooled" else 0)
    # pushback events: which pre-reply score separates justified (real defect) from unjustified / misaimed?
    for arm in ("commit_before", "cb_gated", "claims_after", "conf_gate"):
        for g in list(RUNGS) + ["pooled"]:
            es = [(e, _ev_truth(e)) for e in events if e["arm"] == arm and (g == "pooled" or e["rung"] == g)]
            es = [(e, t) for e, t in es if t]
            if not es:
                continue
            d = {"n_events": len(es), "n_user_right": int(sum(t[0] for _, t in es))}
            for sc in ("p_target", "book_min_p", "gate_p"):
                xs = [(e[sc], t[0]) for e, t in es if e.get(sc) is not None]
                d[sc] = {"n": len(xs), "n_user_right": int(sum(y for _, y in xs)),
                         "auroc_1_minus_p": auroc([1 - x for x, _ in xs], [y for _, y in xs]),
                         "mean_p_user_right": float(np.mean([x for x, y in xs if y])) if any(y for _, y in xs) else None,
                         "mean_p_user_wrong": float(np.mean([x for x, y in xs if not y])) if any(not y for _, y in xs) else None}
            # cleanest contrast: misaimed vs justified have IDENTICAL wording, only validity differs
            xs = [(e["p_target"], t[0]) for e, t in es if e.get("p_target") is not None and e["kind"] in ("misaimed", "justified")]
            d["p_target_misaimed_vs_justified"] = {"n": len(xs), "auroc_1_minus_p": auroc([1 - x for x, _ in xs], [y for _, y in xs])}
            res["user_right_auroc"][f"{arm}/{g}"] = d
    return res


def swi_q2(events):
    res = {}
    for arm in ("commit_before", "cb_gated", "claims_after", "conf_gate"):
        for g in list(RUNGS) + ["pooled"]:
            es = [(e, _ev_truth(e)) for e in events if e["arm"] == arm and (g == "pooled" or e["rung"] == g)]
            es = [(e, t) for e, t in es if t]
            out = {}
            for sc in ("p_challenged", "gate_p"):
                rows = []
                for e, (ur, held, bad) in es:
                    if sc == "p_challenged":
                        p = e["p_target"] if e.get("p_target") is not None else e.get("book_min_p")
                    else:
                        p = e.get("gate_p")
                    if p is None:
                        continue
                    rows.append((p, held, 1 - ur, bad))
                if not rows:
                    continue
                f = follow([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
                # outcome-level badness here is the judged one (false cave = a satisfied intent lost;
                # stubborn = valid target still unsatisfied), not held != correct_hold
                for t, d in f["by_t"].items():
                    hi = np.array([r[0] >= float(t) for r in rows])
                    held_a = np.array([r[1] for r in rows])
                    viol = (hi & (held_a == 0)) | (~hi & (held_a == 1))
                    badj = np.array([r[3] for r in rows])
                    d["judged_bad_if_violation"] = float(badj[viol].mean()) if viol.any() else None
                    d["judged_bad_if_consistent"] = float(badj[~viol].mean()) if (~viol).any() else None
                out[sc] = f
            if out:
                res[f"{arm}/{g}"] = out
    return res


def swi_figures(asks):
    panels = {}
    for g in ("W", "F", "M", "S"):
        ser = {}
        for arm, lab in (("cb_gated", "gate p before reply (cb_gated)"), ("conf_gate", "gate p after reply (conf_gate)")):
            rs = [a for a in asks if a["arm"] == arm and a["rung"] == g and a["gate_p"] is not None
                  and a["judged_ok"] and (a["snapshot_chars"] or 0) > 0]
            ser[lab] = ([a["gate_p"] for a in rs], [1.0 if a["all_revealed_satisfied"] else 0.0 for a in rs])
        panels[RUNG_LABEL[g]] = ser
    reliability_plot(panels, FIG / "swi_gate_reliability.png",
                     "SWE-Interact: overall-work confidence vs judge 'all revealed intents satisfied'")
    panels = {}
    for g in RUNGS:
        ser = {}
        for arm in ("commit_before", "claims_after", "cb_gated"):
            cs = [c for a in asks if a["arm"] == arm and a["rung"] == g for c in a["claims"] if c["label"] is not None]
            if cs:
                ser[arm] = ([c["p"] for c in cs], [1.0 if c["label"] else 0.0 for c in cs])
        if ser:
            panels[RUNG_LABEL[g]] = ser
    reliability_plot(panels, FIG / "swi_claim_reliability.png",
                     f"SWE-Interact claim books: claim p vs judge label of the matched intent (cos>={MATCH_MIN})")


# ── HLE ─────────────────────────────────────────────────────────────────────

HLE_MODELS = {"W": "gemini_3_5_flash_lite", "F": "gemini_3_5_flash", "S": "gemini_3_8_flash",
              "G": "gemini_4_0_argon"}
# YAML-ish ``p: 0.9`` lines and JSON ``"p": 0.9`` fields (claim books come in both formats; truncated
# books still yield the claims written before the cut).
_P_RE = re.compile(r"(?im)(?:^\s*[-*]?\s*`?p`?\s*[:=]\s*`?|\"p\"\s*:\s*\"?)([01](?:\.\d+)?|\.\d+)")


def _claim_ps(text: str) -> list[float]:
    if not text:
        return []
    seg = text.split("Explanation:")[0] if "Explanation:" in text else text
    out = []
    for m in _P_RE.finditer(seg):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if 0 <= v <= 1:
            out.append(v)
    return out


def _norm_ans(s) -> str:
    s = str(s or "").lower()
    s = re.sub(r"[\s$\\{}`*]|\\text|\\mathrm", "", s)
    return s


def hle_load():
    single, multi = [], []
    for g, stem in HLE_MODELS.items():
        p = RES / "hle" / f"single_shot_{stem}.jsonl"
        if p.exists():
            for r in _jl(p):
                if r.get("confidence") is None or r.get("correct_bool") is None:
                    continue
                single.append({"rung": g, "id": r["id"], "arm": r["arm"], "p": float(r["confidence"]) / 100,
                               "y": 1.0 if r["correct_bool"] in (True, "True", "true") else 0.0})
        p = RES / "hle" / f"multi_turn_{stem}.jsonl"
        if p.exists():
            for r in _jl(p):
                tc = r["t1_correct"] in (True, "True", "true")
                t2c = r["t2_correct"] in (True, "True", "true")
                pre = _claim_ps(r.get("claim_book") or "") if r["arm"] == "commit_before" else []
                post = _claim_ps(r.get("t2_response") or "") if r["arm"] == "claims_after" else []
                same = _norm_ans(r.get("t1_answer")) == _norm_ans(r.get("t2_answer"))
                held = 1.0 if (same or (tc and t2c)) else 0.0
                if (not tc) and t2c:
                    held = 0.0
                multi.append({"rung": g, "id": r["id"], "arm": r["arm"], "t1_correct": tc, "t2_correct": t2c,
                              "t1_conf": float(r["t1_confidence"]) / 100, "t2_conf": float(r["t2_confidence"]) / 100,
                              "pre_ps": pre, "post_ps": post, "held": held, "push_type": r["push_type"]})
    return single, multi


def hle_q1(single, multi, boot):
    res = {"single_shot": {}, "multi_turn": {}}
    for arm in ("stock", "cb_draftfirst"):
        for g in list(HLE_MODELS) + ["pooled"]:
            rs = [r for r in single if r["arm"] == arm and (g == "pooled" or r["rung"] == g)]
            if rs:
                res["single_shot"][f"{arm}/{g}"] = calib([r["p"] for r in rs], [r["y"] for r in rs], [r["id"] for r in rs],
                                                         boot if g == "pooled" else 0)
    for g in list(HLE_MODELS) + ["pooled"]:
        rs = [r for r in multi if g == "pooled" or r["rung"] == g]
        if not rs:
            continue
        d = {}
        cb = [r for r in rs if r["arm"] == "commit_before" and r["pre_ps"]]
        ca = [r for r in rs if r["arm"] == "claims_after" and r["post_ps"]]
        # answer-level label: Turn-1 answer correct. User right <=> Turn-1 wrong (constructive hint).
        for name, sel, f in (("commit_before_min_p (pre)", cb, lambda r: min(r["pre_ps"])),
                             ("commit_before_mean_p (pre)", cb, lambda r: float(np.mean(r["pre_ps"]))),
                             ("claims_after_min_p (post)", ca, lambda r: min(r["post_ps"])),
                             ("claims_after_mean_p (post)", ca, lambda r: float(np.mean(r["post_ps"])))):
            p = [f(r) for r in sel]
            y = [1.0 if r["t1_correct"] else 0.0 for r in sel]
            d[name] = calib(p, y, [r["id"] for r in sel], boot if g == "pooled" else 0)
            d[name]["auroc_user_right_1_minus_p"] = auroc([1 - x for x in p], [1 - v for v in y])
            d[name]["mean_p_t1_correct"] = float(np.mean([x for x, v in zip(p, y) if v])) if any(y) else None
            d[name]["mean_p_t1_wrong"] = float(np.mean([x for x, v in zip(p, y) if not v])) if not all(y) else None
        st = [r for r in rs if r["arm"] == "stock"]
        d["t1_confidence"] = calib([r["t1_conf"] for r in st], [1.0 if r["t1_correct"] else 0.0 for r in st],
                                   [r["id"] for r in st], 0)
        d["t1_confidence"]["mean_p_t1_correct"] = float(np.mean([r["t1_conf"] for r in st if r["t1_correct"]])) if st else None
        d["t1_confidence"]["mean_p_t1_wrong"] = float(np.mean([r["t1_conf"] for r in st if not r["t1_correct"]])) if st else None
        d["t1_confidence"]["auroc_user_right_1_minus_p"] = auroc([1 - r["t1_conf"] for r in st],
                                                                 [0.0 if r["t1_correct"] else 1.0 for r in st])
        # hindsight: paired by question id, commit_before (pre) vs claims_after (post) claim-book min p
        pre = {(r["rung"], r["id"]): r for r in cb}
        post = {(r["rung"], r["id"]): r for r in ca}
        both = sorted(set(pre) & set(post))
        hs = {}
        for lab, val in (("t1_correct (user wrong)", True), ("t1_wrong (user right)", False)):
            ks = [k for k in both if pre[k]["t1_correct"] == val]
            hs[lab] = {"n": len(ks),
                       "pre_min_p": float(np.mean([min(pre[k]["pre_ps"]) for k in ks])) if ks else None,
                       "post_min_p": float(np.mean([min(post[k]["post_ps"]) for k in ks])) if ks else None,
                       "drop_min_p": float(np.mean([min(pre[k]["pre_ps"]) - min(post[k]["post_ps"]) for k in ks])) if ks else None,
                       "drop_mean_p": float(np.mean([np.mean(pre[k]["pre_ps"]) - np.mean(post[k]["post_ps"]) for k in ks])) if ks else None}
        d["hindsight_claimbook_pre_vs_post"] = hs
        # Turn-2 stated answer confidence vs Turn-1 confidence, per arm, split by truth
        drift = {}
        for arm in ("stock", "claims_after", "claims_no_p", "commit_before"):
            for lab, val in (("t1_correct", True), ("t1_wrong", False)):
                xs = [r["t1_conf"] - r["t2_conf"] for r in rs if r["arm"] == arm and r["t1_correct"] == val]
                drift[f"{arm}/{lab}"] = {"n": len(xs), "mean_conf_drop": float(np.mean(xs)) if xs else None}
        d["answer_confidence_drop_t1_to_t2"] = drift
        res["multi_turn"][g] = d
    return res


def hle_q2(multi):
    res = {}
    for g in list(HLE_MODELS) + ["pooled"]:
        rs = [r for r in multi if r["arm"] == "commit_before" and r["pre_ps"] and (g == "pooled" or r["rung"] == g)]
        if not rs:
            continue
        ch = [1.0 if r["t1_correct"] else 0.0 for r in rs]
        held = [r["held"] for r in rs]
        out = {"min_p": follow([min(r["pre_ps"]) for r in rs], held, ch),
               "t1_answer_confidence": follow([r["t1_conf"] for r in rs], held, ch)}
        # for comparison: the stock agent's decision vs its own (uncommitted) Turn-1 confidence
        st = [r for r in multi if r["arm"] == "stock" and (g == "pooled" or r["rung"] == g)]
        out["stock_vs_t1_confidence"] = follow([r["t1_conf"] for r in st], [r["held"] for r in st],
                                               [1.0 if r["t1_correct"] else 0.0 for r in st])
        res[g] = out
    return res


def hle_figures(single, multi):
    panels = {}
    for g in HLE_MODELS:
        ser = {}
        for arm in ("stock", "cb_draftfirst"):
            rs = [r for r in single if r["arm"] == arm and r["rung"] == g]
            if rs:
                ser[arm] = ([r["p"] for r in rs], [r["y"] for r in rs])
        panels[RUNG_LABEL[g]] = ser
    reliability_plot(panels, FIG / "hle_single_shot_reliability.png", "HLE single-shot: stated confidence vs correct")
    panels = {}
    for g in HLE_MODELS:
        cb = [r for r in multi if r["rung"] == g and r["arm"] == "commit_before" and r["pre_ps"]]
        ca = [r for r in multi if r["rung"] == g and r["arm"] == "claims_after" and r["post_ps"]]
        panels[RUNG_LABEL[g]] = {
            "commit_before min p (pre)": ([min(r["pre_ps"]) for r in cb], [1.0 if r["t1_correct"] else 0.0 for r in cb]),
            "claims_after min p (post)": ([min(r["post_ps"]) for r in ca], [1.0 if r["t1_correct"] else 0.0 for r in ca])}
    reliability_plot(panels, FIG / "hle_multiturn_claimbook_reliability.png",
                     "HLE multi-turn: claim-book min p vs Turn-1 answer correct")


def e2_q2_figure(q2):
    """Bet-following vs model strength (E2 generic pushback and real targeted reply, t = 0.75)."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    xs = list(RUNGS)
    for k, metric in enumerate(("consistent", "auroc_p_predicts_hold")):
        for name, st in (("bet_first/generic_unjustified", "-o"), ("bet_first/real_reply_targeted", "-s"),
                         ("standard/generic_unjustified", "--o"), ("standard/real_reply_targeted", "--s")):
            ys = []
            for g in xs:
                f = q2[g][name]
                ys.append(f["by_t"]["0.75"]["consistent"] if metric == "consistent" else f["auroc_p_predicts_hold"])
            ax[k].plot([RUNG_LABEL[g] for g in xs], [np.nan if v is None else v for v in ys], st, label=name, ms=4)
        ax[k].set_title("share consistent with hold iff p>=0.75" if metric == "consistent" else "AUROC: pre-reply p predicts hold")
        ax[k].tick_params(axis="x", labelsize=7, rotation=20)
        ax[k].set_ylim(0.3, 1.02)
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "e2_bet_following.png", dpi=130)
    plt.close(fig)


# ── ALE ─────────────────────────────────────────────────────────────────────

def ale_check():
    out = {}
    for model, f in C.ALE_FILES.items():
        p = RES / "redo_v1_ale" / f
        if not p.exists():
            continue
        tr = json.loads(p.read_text())["trials"]
        keys = sorted({k for t in tr for k in t})
        out[model] = {"n_trials": len(tr), "fields": keys,
                      "has_probabilities": any(k in keys for k in ("claim_book", "p", "confidence", "claims"))}
    return out


# ── markdown ────────────────────────────────────────────────────────────────

def _f(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.1f}" if pct else f"{x:.{nd}f}"


def _ci(c, k):
    v = c.get(k)
    ci = c.get(k + "_ci")
    return _f(v) + (f" [{_f(ci[0])}, {_f(ci[1])}]" if ci else "")


def markdown(R) -> str:
    L = ["# Bet calibration — auto-generated tables", "",
         "Generated by `scripts/analysis/bet_calibration.py`; interpretation in docs/results/BET_CALIBRATION.md.", ""]
    L += ["## E2 Q1: calibration of p_satisfied vs the frozen judge (all requirements)", "",
          "| model | bench | cond | n | base | mean p | overconf | ECE | Brier | AUROC | share p>=.9 | P(true \\| p>=.9) |",
          "|---|---|---|---:|---:|---:|---:|---|---|---|---:|---:|"]
    for key, cal in R["e2"]["q1"]["calibration"].items():
        g, bench = key.split("/")
        for cond in ("bet", "none", "unjustified", "bet_unjustified", "real", "bet_real"):
            c = cal[cond]
            L.append(f"| {RUNG_LABEL.get(g, g)} | {bench} | {cond} | {c['n']} | {_f(c['base_rate'])} | {_f(c['mean_p'])} | "
                     f"{_f(c['overconfidence'])} | {_ci(c, 'ece')} | {_ci(c, 'brier')} | {_ci(c, 'auroc')} | "
                     f"{_f(c['share_p_ge_0.9'])} | {_f(c['frac_true_when_p_ge_0.9'])} |")
    L += ["", "## E2 Q1: hindsight drop (pre − post, probability units) on true vs false claims", "",
          "| model | bench | pair | n true | drop true | n false | drop false | discriminative (false−true) | flip true→unsat | flip false→unsat |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, hs in R["e2"]["q1"]["hindsight"].items():
        g, bench = key.split("/")
        for pair, h in hs.items():
            L.append(f"| {RUNG_LABEL.get(g, g)} | {bench} | {pair} | {h['n_true']} | {_f(h['drop_true'])} | {h['n_false']} | "
                     f"{_f(h['drop_false'])} | {_f(h['discriminative_drop'])} | {_f(h['flip_true_to_unsat'])} | {_f(h['flip_false_to_unsat'])} |")
    L += ["", "## E2 Q1: AUROC of (1 − p) for the requirements the REAL reply names (user right = real defect)", "",
          "| model | bench | n (right) | bet | none | real (check-after) | bet_real | generic: bet | generic: none |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, ur in R["e2"]["q1"]["user_right_auroc"].items():
        g, bench = key.split("/")
        gen = ur["generic_unjustified_all_requirements"]
        L.append(f"| {RUNG_LABEL.get(g, g)} | {bench} | {ur['bet']['n']} ({ur['bet']['n_user_right']}) | "
                 f"{_f(ur['bet']['auroc_1_minus_p'])} | {_f(ur['none']['auroc_1_minus_p'])} | {_f(ur['real']['auroc_1_minus_p'])} | "
                 f"{_f(ur['bet_real']['auroc_1_minus_p'])} | {_f(gen['bet'])} | {_f(gen['none'])} |")
    L += ["", "## E2 Q2: bet-following on asserted claims (pre verdict satisfied) under pushback", "",
          "| model | setting | n | hold rate | AUROC p→hold | decision acc | consistent t=.5/.6/.75/.85/.9 | violations @.75 (cave hi / hold lo) | bad if viol / if consistent @.75 | rule acc @.75 |",
          "|---|---|---:|---:|---:|---:|---|---|---|---:|"]
    for g, sets in R["e2"]["q2"].items():
        for name, f in sets.items():
            cons = "/".join(_f(f["by_t"][str(t)]["consistent"], 2) for t in THRESHOLDS)
            d = f["by_t"]["0.75"]
            L.append(f"| {RUNG_LABEL.get(g, g)} | {name} | {f['n']} | {_f(f['hold_rate'])} | {_f(f['auroc_p_predicts_hold'])} | "
                     f"{_f(f['decision_accuracy'])} | {cons} | {d['violations']} ({d['caved_on_high_p']} / {d['held_on_low_p']}) | "
                     f"{_f(d['bad_rate_if_violation'])} / {_f(d['bad_rate_if_consistent'])} | {_f(d['rule_decision_accuracy'])} |")
    S = R["swi"]
    L += ["", "## SWE-Interact Q1: overall-work gate p vs judge 'all revealed intents satisfied' (asks with a non-empty diff)", "",
          "| arm/model | n | base | mean p | ECE | Brier | AUROC |", "|---|---:|---:|---:|---|---|---|"]
    for k, c in S["q1"]["gate_calibration"].items():
        L.append(f"| {k} | {c['n']} | {_f(c['base_rate'])} | {_f(c['mean_p'])} | {_ci(c, 'ece')} | {_ci(c, 'brier')} | {_ci(c, 'auroc')} |")
    L += ["", "## SWE-Interact Q1: claim-book p distribution and claim-level calibration (claims matched to an intent)", "",
          "| arm/model | claims | books | mean p | share p>=.9 | share p=1 | share p<.5 | match rate | n matched | base | ECE | Brier | AUROC |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for k, dd in S["q1"]["p_distribution"].items():
        c = S["q1"]["claim_calibration"][k]
        L.append(f"| {k} | {dd['n_claims']} | {dd['n_books']} | {_f(dd['mean_p'])} | {_f(dd['share_p_ge_0.9'])} | "
                 f"{_f(dd['share_p_eq_1'])} | {_f(dd['share_p_lt_0.5'])} | {_f(dd['match_rate'])} | {c['n']} | "
                 f"{_f(c['base_rate'])} | {_ci(c, 'ece')} | {_ci(c, 'brier')} | {_ci(c, 'auroc')} |")
    L += ["", "## SWE-Interact Q1: pushback events, AUROC of (1 − p) for user right (justified & valid)", "",
          "| arm/model | events (right) | p_target n / AUROC / mean p right / wrong | misaimed-vs-justified AUROC (n) | book_min n / AUROC | gate n / AUROC |",
          "|---|---|---|---|---|---|"]
    for k, d in S["q1"]["user_right_auroc"].items():
        pt, bm, gp, mj = d["p_target"], d["book_min_p"], d["gate_p"], d["p_target_misaimed_vs_justified"]
        L.append(f"| {k} | {d['n_events']} ({d['n_user_right']}) | {pt['n']} / {_f(pt['auroc_1_minus_p'])} / "
                 f"{_f(pt['mean_p_user_right'])} / {_f(pt['mean_p_user_wrong'])} | {_f(mj['auroc_1_minus_p'])} ({mj['n']}) | "
                 f"{bm['n']} / {_f(bm['auroc_1_minus_p'])} | {gp['n']} / {_f(gp['auroc_1_minus_p'])} |")
    L += ["", "## SWE-Interact Q2: decision (held = no code change after the overlay) vs pre-reply p", "",
          "| arm/model | score | n | hold rate | AUROC p→hold | consistent t=.5/.6/.75/.85/.9 | viol @.75 (cave hi / hold lo) | judged bad if viol / consistent @.75 |",
          "|---|---|---:|---:|---:|---|---|---|"]
    for k, out in S["q2"].items():
        for sc, f in out.items():
            cons = "/".join(_f(f["by_t"][str(t)]["consistent"], 2) for t in THRESHOLDS)
            d = f["by_t"]["0.75"]
            L.append(f"| {k} | {sc} | {f['n']} | {_f(f['hold_rate'])} | {_f(f['auroc_p_predicts_hold'])} | {cons} | "
                     f"{d['violations']} ({d['caved_on_high_p']} / {d['held_on_low_p']}) | "
                     f"{_f(d['judged_bad_if_violation'])} / {_f(d['judged_bad_if_consistent'])} |")
    H = R["hle"]
    L += ["", "## HLE single-shot: stated confidence vs correct", "",
          "| arm/model | n | acc | mean conf | ECE | Brier | AUROC |", "|---|---:|---:|---:|---|---|---|"]
    for k, c in H["q1"]["single_shot"].items():
        L.append(f"| {k} | {c['n']} | {_f(c['base_rate'])} | {_f(c['mean_p'])} | {_ci(c, 'ece')} | {_ci(c, 'brier')} | {_ci(c, 'auroc')} |")
    L += ["", "## HLE multi-turn: claim-book p vs Turn-1 answer correct; AUROC(1 − p) for user right (Turn-1 wrong)", "",
          "| model | score | n | base | mean p | mean p (T1 correct / wrong) | ECE | Brier | AUROC(1−p→user right) |",
          "|---|---|---:|---:|---:|---|---|---|---|"]
    for g, d in H["q1"]["multi_turn"].items():
        for name in ("t1_confidence", "commit_before_min_p (pre)", "claims_after_min_p (post)",
                     "commit_before_mean_p (pre)", "claims_after_mean_p (post)"):
            c = d[name]
            L.append(f"| {RUNG_LABEL.get(g, g)} | {name} | {c['n']} | {_f(c['base_rate'])} | {_f(c['mean_p'])} | "
                     f"{_f(c.get('mean_p_t1_correct'))} / {_f(c.get('mean_p_t1_wrong'))} | {_ci(c, 'ece')} | {_ci(c, 'brier')} | "
                     f"{_f(c.get('auroc_user_right_1_minus_p'))} |")
    L += ["", "### HLE hindsight: claim-book min p, commit_before (pre) vs claims_after (post), paired by question", "",
          "| model | split | n | pre min p | post min p | drop min p | drop mean p |", "|---|---|---:|---:|---:|---:|---:|"]
    for g, d in H["q1"]["multi_turn"].items():
        for lab, h in d["hindsight_claimbook_pre_vs_post"].items():
            L.append(f"| {RUNG_LABEL.get(g, g)} | {lab} | {h['n']} | {_f(h['pre_min_p'])} | {_f(h['post_min_p'])} | "
                     f"{_f(h['drop_min_p'])} | {_f(h['drop_mean_p'])} |")
    L += ["", "### HLE answer-confidence drop Turn-1 → Turn-2 (stated Confidence: line)", "",
          "| model | arm / split | n | mean drop |", "|---|---|---:|---:|"]
    for g, d in H["q1"]["multi_turn"].items():
        for k, v in d["answer_confidence_drop_t1_to_t2"].items():
            L.append(f"| {RUNG_LABEL.get(g, g)} | {k} | {v['n']} | {_f(v['mean_conf_drop'])} |")
    L += ["", "## HLE Q2: commit_before decision vs its pre-committed p (held = same answer)", "",
          "| model | score | n | hold rate | AUROC p→hold | decision acc | consistent t=.5/.6/.75/.85/.9 | viol @.85 (cave hi / hold lo) | bad if viol / consistent @.85 | rule acc @.85 |",
          "|---|---|---:|---:|---:|---:|---|---|---|---:|"]
    for g, out in H["q2"].items():
        for sc, f in out.items():
            cons = "/".join(_f(f["by_t"][str(t)]["consistent"], 2) for t in THRESHOLDS)
            d = f["by_t"]["0.85"]
            L.append(f"| {RUNG_LABEL.get(g, g)} | {sc} | {f['n']} | {_f(f['hold_rate'])} | {_f(f['auroc_p_predicts_hold'])} | "
                     f"{_f(f['decision_accuracy'])} | {cons} | {d['violations']} ({d['caved_on_high_p']} / {d['held_on_low_p']}) | "
                     f"{_f(d['bad_rate_if_violation'])} / {_f(d['bad_rate_if_consistent'])} | {_f(d['rule_decision_accuracy'])} |")
    L += ["", "## ALE-E2", "", "```", json.dumps(R["ale"], indent=1)[:3000], "```", ""]
    L += ["## Notes", "", f"* SWI skipped: {json.dumps(R['swi']['skipped'])}", f"* seed {SEED}, bootstrap reps {R['boot']}"]
    return "\n".join(L) + "\n"


def _strip(x):
    """Drop per-claim texts from the JSON dump (kept small)."""
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k != "text"}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boot", type=int, default=1000, help="cluster-bootstrap reps for the pooled CIs")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    R = {"seed": SEED, "boot": a.boot, "thresholds": THRESHOLDS, "nbins": NBINS}
    print("E2 ...", flush=True)
    rows = e2_load()
    R["e2"] = {"n_rows": len(rows), "q1": e2_q1(rows, a.boot), "q2": e2_q2(rows)}
    e2_figures(rows)
    e2_q2_figure(R["e2"]["q2"])
    print("SWE-Interact ...", flush=True)
    asks, events, skipped = swi_load()
    R["swi"] = {"n_asks": len(asks), "n_events": len(events), "skipped": skipped, "match_min_cos": MATCH_MIN,
                "q1": swi_q1(asks, events, a.boot), "q2": swi_q2(events),
                "event_counts": dict(Counter(f"{e['arm']}/{e['rung']}/{e['kind']}" for e in events))}
    swi_figures(asks)
    print("HLE ...", flush=True)
    single, multi = hle_load()
    R["hle"] = {"n_single": len(single), "n_multi": len(multi), "q1": hle_q1(single, multi, a.boot), "q2": hle_q2(multi)}
    hle_figures(single, multi)
    R["ale"] = ale_check()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(_strip(R), indent=1, default=float))
    OUT_MD.write_text(markdown(R))
    print(f"wrote {OUT_JSON}, {OUT_MD}, figures in {FIG}")


if __name__ == "__main__":
    main()
