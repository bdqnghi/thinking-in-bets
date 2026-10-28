#!/usr/bin/env python3
"""Multiple-comparison correction and pre-registration labels for every headline comparison in
``docs/TIB_2PAGE_SUMMARY.md``.

Pure post-hoc analysis (no model runs), fixed seeds:

    python3 scripts/analysis/multiple_comparisons.py     # ~1-3 min; results/analysis/multiple_comparisons.{json,md}

For each claim the p-value is recomputed from per-unit data (never copied from the summary):

* paired binary outcome, one replicate  -> exact McNemar (two-sided binomial test on discordant W vs L);
* paired outcome with replicates / continuous score / Brier -> paired sign-flip permutation on per-task mean
  differences (exact when <= 16 non-zero tasks, else 20,000 Monte-Carlo flips);
* event-rate metrics (caving, listening, balanced accuracy) -> within-cluster arm-swap permutation of the
  ratio-of-sums difference (clusters = tasks / scenarios), 20,000 permutations;
* E2 confidence drop -> sign-flip over tasks of the per-task mean (p_none - p_unjustified).

Holm (FWER) and Benjamini-Hochberg (FDR) are applied over the headline family (all claims in the summary that are
testable). A second, extended family adds the comparisons that were run but are not in the summary (e.g.
DeepSWE 3.8 Flash, Terminal-Bench 4.0 Pro, Finance 2-turn), to show what selective reporting does to the
correction. Labels: ``PR`` = hypothesis and metric pre-registered (Wave A / Wave B / E2 docs) for that cell;
``PP`` = pre-registered protocol but a cell, pooling or metric not named in the registration (added model,
added replicate, pooled-over-models, secondary metric); ``EX`` = exploratory (no registration covers it).
"""
from __future__ import annotations

import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import common as C  # noqa: E402
import tib_data as T  # noqa: E402

N_PERM = 20_000


def rng_for(*key):
    return np.random.default_rng(T.SEED + zlib.crc32("|".join(map(str, key)).encode()))


# ── tests ───────────────────────────────────────────────────────────────────

def signflip(x, key):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    x = x[x != 0]
    if len(x) == 0:
        return 1.0
    obs = abs(x.mean())
    if len(x) <= 16:
        signs = ((np.arange(2 ** len(x))[:, None] >> np.arange(len(x))) & 1) * 2 - 1
        return float(np.mean(np.abs((signs * x).mean(1)) >= obs - 1e-12))
    s = rng_for(key).choice([-1, 1], (N_PERM, len(x)))
    return float((np.sum(np.abs((s * x).mean(1)) >= obs - 1e-12) + 1) / (N_PERM + 1))


def mcnemar(w, l):
    if w + l == 0:
        return 1.0
    return float(stats.binomtest(w, w + l, 0.5).pvalue)


def cell_test(cell, key, outcome="y"):
    """Paired cell from tib_data.paired_cells -> dict(n, stock, arm, delta, W/T/L, p, test)."""
    ks, ka, kp = ("stock", "arm", "pairs") if outcome == "y" else ("stock_s", "arm_s", "pairs_s")
    tasks = sorted(cell)
    s = np.array([T.mean(cell[t][ks]) for t in tasks], float)
    a = np.array([T.mean(cell[t][ka]) for t in tasks], float)
    pairs = [p for t in tasks for p in cell[t][kp] if p[0] is not None and p[1] is not None]
    w = sum(1 for x, y in pairs if y > x + 1e-9); l = sum(1 for x, y in pairs if y < x - 1e-9)
    multi = any(len(cell[t][ka]) > 1 or len(cell[t][ks]) > 1 for t in tasks)
    if outcome == "y" and not multi:
        p, test = mcnemar(w, l), "exact McNemar"
    else:
        p, test = signflip(a - s, key), "paired sign-flip permutation (task)"
    return {"n": len(tasks), "n_pairs": len(pairs), "stock": float(s.mean()), "arm": float(a.mean()),
            "delta": float((a - s).mean()), "wtl": [w, len(pairs) - w - l, l], "p": p, "test": test}


def ratio_perm(ua: dict, ub: dict, stat, key):
    """Arm-swap permutation within cluster. ua/ub: cluster -> [ev1, n1, ev2, n2]. stat(totA, totB) -> float."""
    keys = sorted(set(ua) & set(ub))
    A = np.array([ua[k] for k in keys], float); B = np.array([ub[k] for k in keys], float)
    obs = stat(A.sum(0), B.sum(0))
    if obs is None:
        return None, None, len(keys)
    rng = rng_for(key)
    sw = rng.random((N_PERM, len(keys))) < 0.5
    cnt = 0; tot = 0
    for i in range(N_PERM):
        m = sw[i][:, None]
        ta = np.where(m, B, A).sum(0); tb = np.where(m, A, B).sum(0)
        v = stat(ta, tb)
        if v is None:
            continue
        tot += 1
        cnt += abs(v) >= abs(obs) - 1e-12
    return obs, (cnt + 1) / (tot + 1), len(keys)


def rate(i, j):
    return lambda a, b: (a[i] / a[j] - b[i] / b[j]) if a[j] and b[j] else None


def ba_diff(a, b):
    if not (a[1] and a[3] and b[1] and b[3]):
        return None
    ba = lambda t: 0.5 * ((1 - t[0] / t[1]) + t[2] / t[3])
    return ba(a) - ba(b)


# ── multiple comparison ─────────────────────────────────────────────────────

def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps)[::-1]
    adj = np.empty(m); run = 1.0
    for r, i in enumerate(o):
        rank = m - r
        run = min(run, ps[i] * m / rank); adj[i] = run
    return adj


# ── claims ──────────────────────────────────────────────────────────────────

def build_claims():
    recs = T.load_all()
    cells = T.paired_cells(recs, min_n=1)
    claims = []

    def add(cid, section, text, res, label, note="", family="headline", eff=None):
        claims.append({"id": cid, "section": section, "claim": text, "label": label, "note": note, "family": family,
                       **res, **({"effect": eff} if eff else {})})

    def pp(x):
        return f"{100 * x:+.1f} pp"

    # 4.1 E2 controlled code review
    e2_models = {"W": ["swi_W", "swt_W"], "F": ["swi_F", "swt_F"], "M": ["swi_M", "swt_M"], "S": ["swi_S", "swt_S"],
                 "G": ["swi_S_g4pro", "swt_S_g4pro"]}
    runs = {r[0]: r for r in C.E2_RUNS}
    pooled_std, pooled_bet = {}, {}
    for m, rr in e2_models.items():
        std, bet, drops = {}, {}, defaultdict(list)
        for run in rr:
            rows = C.e2_rows(*runs[run])
            for k, v in C.e2_units(rows, "standard").items():
                std[f"{run}/{k}"] = v
            for k, v in C.e2_units(rows, "bet_first").items():
                bet[f"{run}/{k}"] = v
            for r in rows:
                if r["judge_verdict"] == "satisfied" and r.get("none") and r.get("unjustified") and r["none"].get("p") is not None and r["unjustified"].get("p") is not None:
                    drops[f"{run}/{r['task']}"].append(r["none"]["p"] - r["unjustified"]["p"])
        pooled_std.update({f"{m}/{k}": v for k, v in std.items()}); pooled_bet.update({f"{m}/{k}": v for k, v in bet.items()})
        lab = "PR" if m in ("W", "S") else "EX"
        x = [np.mean(v) for v in drops.values()]
        allr = [d for v in drops.values() for d in v]
        add(f"E2-drop-{m}", "4.1", f"E2 {T.MODEL_LABEL[m]}: check-after confidence drops after unjustified reply",
            {"n": len(x), "delta": -float(np.mean(allr)) / 100, "p": signflip(x, f"e2drop{m}"),
             "test": "sign-flip over tasks (per-task mean p_none - p_unj)"},
            lab, "E2 primary is Cohen's d on this drop (C2a); registered for W and S only" if lab == "PR" else "model not in E2 registration (W, S only)",
            eff=f"{-np.mean(allr):+.1f} conf pts (summary table: see 4.1)")
        obs, p, n = ratio_perm(bet, std, rate(0, 1), f"e2cave{m}")
        tb, ts = C.totals(bet, sorted(set(bet) & set(std))), C.totals(std, sorted(set(bet) & set(std)))
        add(f"E2-cave-{m}", "4.1", f"E2 {T.MODEL_LABEL[m]}: false caving bet-before < check-after",
            {"n": n, "delta": obs, "p": p, "test": "within-task arm-swap permutation (ratio)"},
            "PP" if m in ("W", "S") else "EX", "flip rate is a secondary E2 metric" if m in ("W", "S") else "model not in E2 registration",
            eff=f"{100 * ts[0] / ts[1]:.1f}% ({ts[0]}/{ts[1]}) -> {100 * tb[0] / tb[1]:.1f}% ({tb[0]}/{tb[1]})")
    obs, p, n = ratio_perm(pooled_bet, pooled_std, rate(0, 1), "e2cavepool")
    tb, ts = C.totals(pooled_bet), C.totals(pooled_std)
    add("E2-cave-pooled", "4.1/5", "E2 pooled: false caving 0/6,130 with bet-before", {"n": n, "delta": obs, "p": p,
        "test": "within-task arm-swap permutation (ratio)"}, "PP", "pooled over 5 models incl. 3 unregistered",
        eff=f"{100 * ts[0] / ts[1]:.1f}% ({ts[0]}/{ts[1]}) -> {100 * tb[0] / tb[1]:.1f}% ({tb[0]}/{tb[1]})")

    # 4.2 SWE-Interact pass rates
    swi = T.swi_trials()
    mixed = json.loads((T.OUT / "swi_mixed_r1_trials.json").read_text())
    s12 = json.loads((T.OUT / "swi_S_mixed_r1r2_trials.json").read_text())
    for arm in ("commit_before", "cb_gated"):
        for m in ("W", "F", "M", "S"):
            sub = {t: v for t, v in cells[("swi_mixed", m, arm)].items()}
            # r1 only
            r1 = {t: {**v, "stock": v["stock"][:1], "arm": v["arm"][:1], "pairs": v["pairs"][:1]} for t, v in sub.items()}
            res = cell_test(r1, f"swi{arm}{m}")
            add(f"SWI-pass-{arm}-{m}", "4.2 Table A", f"SWI mixed r1 {T.MODEL_LABEL[m]}: {arm} vs stock pass", res, "PR",
                "final_pass is a Wave A/B primary metric", eff=f"{100 * res['stock']:.0f}% -> {100 * res['arm']:.0f}% ({pp(res['delta'])})")
        # pooled Flash r1
        pooled = {}
        for m in ("W", "F", "M", "S"):
            for t, v in cells[("swi_mixed", m, arm)].items():
                pooled[f"{m}/{t}"] = {**v, "stock": v["stock"][:1], "arm": v["arm"][:1], "pairs": v["pairs"][:1]}
        res = cell_test(pooled, f"swipool{arm}")
        add(f"SWI-pass-{arm}-pooledFlash", "4.2 Table A", f"SWI mixed r1 pooled Flash: {arm} vs stock pass", res, "PP",
            "pooling over models not specified", eff=f"{100 * res['stock']:.1f}% -> {100 * res['arm']:.1f}% ({pp(res['delta'])})")
        res = cell_test(cells[("swi_mixed", "S", arm)], f"swiS12{arm}")
        add(f"SWI-pass-{arm}-S-r1r2", "4.2 Table A", f"SWI mixed 3.8 Flash r1+r2: {arm} vs stock pass", res, "PP",
            "r2 replicate added after r1 was seen", eff=f"{100 * res['stock']:.1f}% -> {100 * res['arm']:.1f}% ({pp(res['delta'])})")
    res = cell_test(cells[("swi_overlay", "G", "commit_before")], "swiG")
    add("SWI-pass-commit_before-G-r1r2", "4.2 Table A / 5", "SWI 4.0 Pro overlay r1+r2: commit_before 35% vs stock 30%",
        res, "EX", "G runs documented as exploratory (G4PRO_EXPLORATORY.md)", eff=f"{100 * res['stock']:.1f}% -> {100 * res['arm']:.1f}% ({pp(res['delta'])})")

    # SWI behaviour metrics
    def units(rows, rung=None, arm=None, rep=None):
        rs = [r for r in rows if (rung is None or r["rung"] == rung) and (arm is None or r["arm"] == arm) and (rep is None or r["replicate"] in rep)]
        out = {}
        for rg in sorted({r["rung"] for r in rs}):
            for k, v in C.swi_units([r for r in rs if r["rung"] == rg]).items():
                out[f"{rg}/{k}"] = v
        return out
    for arm, lab_c in (("commit_before", "PR"), ("cb_gated", "PR")):
        ua, us = units(mixed, arm=arm), units(mixed, arm="stock")
        obs, p, n = ratio_perm(ua, us, rate(0, 1), f"cave{arm}")
        ta, ts = C.totals(ua), C.totals(us)
        add(f"SWI-cave-{arm}-pooledFlash", "4.2 Table B", f"SWI mixed r1 pooled Flash: {arm} caving vs stock", {"n": n, "delta": obs, "p": p,
            "test": "within-task arm-swap permutation (ratio)"}, lab_c, "Wave A H1 / Wave B HB1 (caving)",
            eff=f"{100 * ts[0] / ts[1]:.1f}% -> {100 * ta[0] / ta[1]:.1f}%")
        obs, p, n = ratio_perm(ua, us, rate(2, 3), f"listen{arm}")
        add(f"SWI-listen-{arm}-pooledFlash", "4.2 Table B", f"SWI mixed r1 pooled Flash: {arm} listening (correct update) vs stock",
            {"n": n, "delta": obs, "p": p, "test": "within-task arm-swap permutation (ratio)"}, lab_c,
            "Wave A H2 (stubbornness; equivalence bound +5 pp)", eff=f"{100 * ts[2] / ts[3]:.1f}% -> {100 * ta[2] / ta[3]:.1f}%")
    for arm in ("commit_before", "cb_gated"):
        ua, us = units(s12, arm=arm), units(s12, arm="stock")
        ta, ts = C.totals(ua), C.totals(us)
        obs, p, n = ratio_perm(ua, us, rate(2, 3), f"listenS{arm}")
        add(f"SWI-listen-{arm}-S-r1r2", "4.2 / 5", f"SWI 3.8 Flash r1+r2: {arm} real-bug listening vs stock",
            {"n": n, "delta": obs, "p": p, "test": "within-task arm-swap permutation (ratio)"}, "EX",
            "correct_update by model capability is listed as *exploratory* in Wave A; r2 added post hoc",
            eff=f"{100 * ts[2] / ts[3]:.1f}% -> {100 * ta[2] / ta[3]:.1f}%")
        obs, p, n = ratio_perm(ua, us, rate(0, 1), f"caveS{arm}")
        add(f"SWI-cave-{arm}-S-r1r2", "4.2", f"SWI 3.8 Flash r1+r2: {arm} caving vs stock",
            {"n": n, "delta": obs, "p": p, "test": "within-task arm-swap permutation (ratio)"}, "EX",
            "per-model subgroup of H1 with post-hoc replicate", eff=f"{100 * ts[0] / ts[1]:.1f}% -> {100 * ta[0] / ta[1]:.1f}%")
        obs, p, n = ratio_perm(ua, us, ba_diff, f"baS{arm}")
        add(f"SWI-BA-{arm}-S-r1r2", "4.2 / 5", f"SWI 3.8 Flash r1+r2: {arm} balanced accuracy vs stock",
            {"n": n, "delta": obs, "p": p, "test": "within-task arm-swap permutation (BA)"}, "EX", "BA defined post hoc",
            eff=f"ΔBA {obs:+.3f}")
    # repeat-bug (recurrence) pooled Flash r1
    for arm in ("cb_gated", "commit_before"):
        cell = {}
        for r in mixed:
            if r["arm"] in (arm, "stock"):
                k = f"{r['rung']}/{r['task']}"
                cell.setdefault(k, {"stock": [], "arm": [], "pairs": []})
                cell[k]["arm" if r["arm"] == arm else "stock"].append(int((r.get("recurrence_any_turn") or 0) > 0))
        for v in cell.values():
            v["pairs"] = list(zip(v["stock"], v["arm"]))
        cell = {k: v for k, v in cell.items() if v["stock"] and v["arm"]}
        res = cell_test(cell, f"rec{arm}")
        add(f"SWI-repeatbug-{arm}-pooledFlash", "4.2 Table B", f"SWI mixed r1 pooled Flash: {arm} repeat-bug rate vs stock",
            res, "EX", "recurrence is not a registered metric", eff=f"{100 * res['stock']:.1f}% -> {100 * res['arm']:.1f}%")
    # G overlay caving / unnecessary edits
    g = json.loads((T.OUT / "swi_g4_overlay_r1r2_trials.json").read_text())
    def g_units(arm, field):
        out = defaultdict(lambda: [0, 0, 0, 0])
        for r in g:
            if r["arm"] != arm:
                continue
            for e in r["caving"]:
                if e.get("judged") and e.get("kind") != "justified":
                    out[r["task"]][1] += 1
                    out[r["task"]][0] += 1 if (e["lost"] if field == "lost" else e.get("changed")) else 0
        return dict(out)
    for field, cid, txt in (("lost", "SWI-cave-G", "SWI 4.0 Pro overlay: caving 0/35 vs 3/38"),
                            ("changed", "SWI-edits-G", "SWI 4.0 Pro overlay: post-pushback code edits 22.9% vs 57.9%")):
        ua, us = g_units("commit_before", field), g_units("stock", field)
        obs, p, n = ratio_perm(ua, us, rate(0, 1), cid)
        ta, ts = C.totals(ua), C.totals(us)
        add(cid, "4.2 / 5", txt, {"n": n, "delta": obs, "p": p, "test": "within-task arm-swap permutation (ratio)"}, "EX",
            "G exploratory", eff=f"{ts[0]}/{ts[1]} -> {ta[0]}/{ta[1]}")
    # pure pushback S overlay_r1 (section 1 / 4.2 item 2 / 'why the number matters')
    for arm, txt in (("commit_before", "pure pushback 3.8 Flash: always-on betting 70% vs stock 55% pass"),
                     ("claims_after", "pure pushback 3.8 Flash: check-after 40% vs stock 55% pass ('worse than nothing')")):
        if ("swi_overlay", "S", arm) in cells:
            res = cell_test(cells[("swi_overlay", "S", arm)], f"ov{arm}")
            add(f"SWI-pure-pass-{arm}-S", "1 / 4.2", txt, res, "EX", "overlay_r1 S not covered by the four registration docs",
                eff=f"{100 * res['stock']:.0f}% -> {100 * res['arm']:.0f}%")
    sov = [r for r in swi["swi_overlay"] if r["rung"] == "S"]
    if sov:
        by = defaultdict(dict)
        for r in sov:
            by[r["task"]][r["arm"]] = int(bool(r["final_pass"]))
        cell = {t: {"stock": [v["claims_no_p"]], "arm": [v["commit_before"]], "pairs": [(v["claims_no_p"], v["commit_before"])]}
                for t, v in by.items() if "claims_no_p" in v and "commit_before" in v}
        res = cell_test(cell, "nump")
        add("SWI-pure-numbers-S", "2", "Why the number matters: claims+proof+p 70% vs no p 35% (pure pushback 3.8 Flash)",
            res, "EX", "recomputed claims_no_p = 25% (5/20), not 35%; reference arm here is claims_no_p",
            eff=f"{100 * res['stock']:.0f}% -> {100 * res['arm']:.0f}%")
        ua = C.swi_units([r for r in sov if r["arm"] == "commit_before"]); us = C.swi_units([r for r in sov if r["arm"] == "stock"])
        obs, p, n = ratio_perm(ua, us, rate(0, 1), "ovcave")
        ta, ts = C.totals(ua), C.totals(us)
        add("SWI-pure-cave-S", "4.2 item 2", "pure pushback 3.8 Flash: always-on caves 0% vs stock 16%",
            {"n": n, "delta": obs, "p": p, "test": "within-task arm-swap permutation (ratio)"}, "EX",
            "repo swi_units definition gives stock 1/20 = 5%, not 16%", eff=f"{ts[0]}/{ts[1]} -> {ta[0]}/{ta[1]}")

    # 4.3 ALE-E2
    pooled = {}
    for m in ("W", "F", "M", "S", "G"):
        res = cell_test(cells[("ale_e2", m, "commit_before")], f"ale{m}")
        pooled.update({f"{m}/{t}": v for t, v in cells[("ale_e2", m, "commit_before")].items()})
        add(f"ALE-E2-pass-{m}", "4.3", f"ALE-E2 {T.MODEL_LABEL[m]}: always-on vs stock pass", res, "EX", "",
            eff=f"{100 * res['stock']:.0f}% -> {100 * res['arm']:.0f}%")
    res = cell_test(pooled, "alepool")
    add("ALE-E2-pass-pooled", "4.3 / 5", "ALE-E2 pooled: 58% -> 89% (+31 pts)", res, "EX", "", eff=f"{100 * res['stock']:.0f}% -> {100 * res['arm']:.0f}%")
    ua, us = {}, {}
    for m, mod in (("W", "gemini-3.5-flash-lite"), ("F", "gemini-3.5-flash"), ("M", "gemini-3.6-flash"), ("S", "gemini-3.8-flash"), ("G", "gemini-4.0-pro")):
        tr = C.ale_trials(mod)
        ua.update({f"{m}/{k}": v for k, v in C.ale_units(tr, "commit_before").items()})
        us.update({f"{m}/{k}": v for k, v in C.ale_units(tr, "stock").items()})
    obs, p, n = ratio_perm(ua, us, rate(0, 1), "alecave")
    ta, ts = C.totals(ua), C.totals(us)
    add("ALE-E2-cave-pooled", "4.3", "ALE-E2 pooled caving 54.7% -> 12.0%", {"n": n, "delta": obs, "p": p,
        "test": "within-scenario arm-swap permutation (ratio)"}, "EX", "", eff=f"{ts[0]}/{ts[1]} -> {ta[0]}/{ta[1]}")
    pc = {}
    for m in ("W", "F", "M", "S", "G"):
        a = cells[("ale_e2", m, "commit_before")]; b = cells[("ale_e2", m, "claims_no_p")]
        for t in set(a) & set(b):
            pc[f"{m}/{t}"] = {"stock": b[t]["arm"], "arm": a[t]["arm"], "pairs": list(zip(b[t]["arm"], a[t]["arm"]))}
    res = cell_test(pc, "alenum")
    add("ALE-E2-numbers-pooled", "4.3", "ALE-E2 pooled: Bet Card with numbers 89% vs without numbers 66%", res, "EX",
        "reference arm = claims_no_p", eff=f"{100 * res['stock']:.0f}% -> {100 * res['arm']:.0f}%")
    # ALE docker (score)
    for bench, m in (("ale_t300", "G"), ("ale_t300", "S"), ("ale_t300", "F"), ("ale_t300", "W"), ("ale_s1", "F"), ("ale_s1", "W"), ("ale_s1", "S")):
        res = cell_test(cells[(bench, m, "cb_draftfirst")], f"{bench}{m}", "score")
        add(f"ALE-docker-{bench}-{m}", "4.3", f"ALE-Claw {bench} {T.MODEL_LABEL[m]}: draft-first score vs stock", res, "EX", "",
            eff=f"{100 * res['stock']:.1f} -> {100 * res['arm']:.1f} ({pp(res['delta'])})")

    # 4.4 HLE single shot accuracy + Brier
    hle = defaultdict(dict)
    for r in recs:
        if r["bench"] == "hle_ss":
            hle[(r["model"], r["task"])][r["arm"]] = r
    for m in ("G", "S", "F", "W"):
        res = cell_test(cells[("hle_ss", m, "cb_draftfirst")], f"hle{m}")
        add(f"HLE-acc-{m}", "4.4", f"HLE single-shot {T.MODEL_LABEL[m]}: draft-first accuracy vs stock", res, "EX", "",
            eff=f"{100 * res['stock']:.1f}% -> {100 * res['arm']:.1f}% ({pp(res['delta'])})")
        d, bs, ba = [], [], []
        for (mm, t), v in hle.items():
            if mm != m or "stock" not in v or "cb_draftfirst" not in v:
                continue
            if v["stock"]["conf"] is None or v["cb_draftfirst"]["conf"] is None:
                continue
            b0 = (v["stock"]["conf"] - v["stock"]["y"]) ** 2; b1 = (v["cb_draftfirst"]["conf"] - v["cb_draftfirst"]["y"]) ** 2
            d.append(b1 - b0); bs.append(b0); ba.append(b1)
        add(f"HLE-brier-{m}", "4.4", f"HLE single-shot {T.MODEL_LABEL[m]}: draft-first Brier vs stock",
            {"n": len(d), "delta": float(np.mean(d)), "p": signflip(d, f"brier{m}"), "test": "paired sign-flip permutation (question)"},
            "EX", "per-question proxy for the RMS calibration error claim", eff=f"{np.mean(bs):.3f} -> {np.mean(ba):.3f}")
    # HLE multi-turn
    mt = defaultdict(dict)
    for r in recs:
        if r["bench"] == "hle_mt":
            mt[(r["model"], r["task"])][r["arm"]] = r
    for m in ("G", "S", "F", "W"):
        res = cell_test(cells[("hle_mt", m, "commit_before")], f"hlemt{m}")
        add(f"HLE-mt-t2acc-{m}", "4.4 / 5", f"HLE 2-turn {T.MODEL_LABEL[m]}: always-on turn-2 accuracy vs stock", res, "EX", "",
            eff=f"{100 * res['stock']:.1f}% -> {100 * res['arm']:.1f}% ({pp(res['delta'])})")
        w = l = n = cs = ca = 0
        for (mm, t), v in mt.items():
            if mm != m or v.get("stock", {}).get("push_type") != "false_pushback" or "commit_before" not in v:
                continue
            n += 1; s_, a_ = v["stock"]["caved"], v["commit_before"]["caved"]; cs += s_; ca += a_
            w += (s_ and not a_); l += (a_ and not s_)
        add(f"HLE-mt-cave-{m}", "4.4", f"HLE 2-turn {T.MODEL_LABEL[m]}: always-on false-pushback caving vs stock",
            {"n": n, "delta": (ca - cs) / n, "p": mcnemar(w, l), "test": "exact McNemar", "wtl": [w, n - w - l, l]}, "EX", "",
            eff=f"{cs}/{n} -> {ca}/{n}")

    # 4.5 Finance
    pooled = {}
    for m in ("G", "S", "F", "W"):
        res = cell_test(cells[("fin_tools", m, "cb_draftfirst")], f"fin{m}", "score")
        pooled.update({f"{m}/{t}": v for t, v in cells[("fin_tools", m, "cb_draftfirst")].items()})
        add(f"FIN-tools-{m}", "4.5", f"Finance tools {T.MODEL_LABEL[m]}: draft-first rubric score vs stock", res, "EX", "",
            eff=f"{100 * res['stock']:.2f} -> {100 * res['arm']:.2f} ({pp(res['delta'])})")
    res = cell_test(pooled, "finpool", "score")
    w, l = res["wtl"][0], res["wtl"][2]
    res["p"] = mcnemar(w, l); res["test"] = "exact sign test on score W/L"
    add("FIN-tools-pooled", "4.5 / 5", "Finance tools pooled: 19 W / 180 T / 1 L", res, "EX", "", eff=f"{w}/{res['wtl'][1]}/{l}")
    for m in ("G", "S", "F", "W"):
        res = cell_test(cells[("fin_closed", m, "cb_draftfirst")], f"finc{m}", "score")
        add(f"FIN-closed-{m}", "4.5", f"Finance closed-book {T.MODEL_LABEL[m]}: draft-first score vs stock", res, "EX", "",
            eff=f"{100 * res['stock']:.1f} -> {100 * res['arm']:.1f} ({pp(res['delta'])})")

    # 4.6 Harbor
    for bench, m, txt in (("ds11m", "G", "DeepSWE mini-swe-agent 4.0 Pro: +21.4 pts (9/33/0, n=42 at writing)"),
                          ("ds11", "G", "DeepSWE opencode 4.0 Pro: 63.7% -> 66.4%"),
                          ("tb4", "S", "Terminal-Bench 4.0 3.8 Flash: 13.6% -> 18.2%")):
        res = cell_test(cells[(bench, m, "cb_draftfirst")], f"{bench}{m}")
        add(f"HARBOR-{bench}-{m}", "4.6 / 5", txt, res, "EX", "partial (still running)" if bench == "ds11m" else "",
            eff=f"{100 * res['stock']:.1f}% -> {100 * res['arm']:.1f}% ({pp(res['delta'])})")

    # ── extended family: run but not in the summary ──
    for bench, m in (("ds11", "S"), ("ds11", "F"), ("tb4", "G"), ("tb4", "W"), ("tb4", "F")):
        k = (bench, m, "cb_draftfirst")
        if k in cells and len(cells[k]) >= 10:
            res = cell_test(cells[k], f"x{bench}{m}")
            add(f"UNREP-{bench}-{m}", "not in summary", f"{T.BENCH_LABEL[bench]} {T.MODEL_LABEL[m]}: draft-first vs stock", res, "EX",
                "partial" if len(cells[k]) < 60 else "", family="unreported",
                eff=f"{100 * res['stock']:.1f}% -> {100 * res['arm']:.1f}% ({pp(res['delta'])})")
    for m in ("G", "S", "F", "W"):
        res = cell_test(cells[("fin_mt", m, "commit_before")], f"finmt{m}")
        add(f"UNREP-fin_mt-{m}", "not in summary", f"Finance 2-turn pushback {T.MODEL_LABEL[m]}: always-on t2 pass vs stock", res, "EX", "",
            family="unreported", eff=f"{100 * res['stock']:.0f}% -> {100 * res['arm']:.0f}% ({pp(res['delta'])})")
    return claims


def main():
    claims = build_claims()
    head = [c for c in claims if c["family"] == "headline" and c.get("p") is not None]
    ext = [c for c in claims if c.get("p") is not None]
    for fam, cs, tag in (("headline", head, ""), ("extended", ext, "_ext")):
        ps = [c["p"] for c in cs]
        for c, h, q in zip(cs, holm(ps), bh(ps)):
            c["holm" + tag] = float(h); c["bh" + tag] = float(q)
    pr = [c for c in head if c["label"] == "PR"]
    for c, h in zip(pr, holm([c["p"] for c in pr])):
        c["holm_prereg_only"] = float(h)
    out = {"meta": {"seed": T.SEED, "n_perm": N_PERM, "n_headline": len(head), "n_extended": len(ext),
                    "labels": {"PR": "pre-registered hypothesis+metric", "PP": "pre-registered protocol, unregistered cell/pooling/metric",
                               "EX": "exploratory"}}, "claims": claims}
    (T.OUT / "multiple_comparisons.json").write_text(json.dumps(out, indent=1))
    L = ["# Multiple-comparison correction of the 2-page summary's headline claims (auto-generated by `scripts/analysis/multiple_comparisons.py`)", "",
         f"Headline family m = {len(head)} testable claims; extended family (adds comparisons that were run but not reported) m = {len(ext)}. "
         "Survives = Holm-adjusted p < 0.05 (FWER) / BH q < 0.05 (FDR). Labels: PR pre-registered, PP pre-registered protocol but "
         "unregistered cell/pooling/metric, EX exploratory. Effect is TIB − reference unless noted.", "",
         "| id | claim | label | n | effect (recomputed) | test | raw p | Holm p | BH q | Holm (ext) | survives Holm / BH |",
         "|---|---|---|---:|---|---|---:|---:|---:|---:|---|"]
    for c in claims:
        if c.get("p") is None:
            continue
        sv = ("✔" if c.get("holm", 1) < 0.05 else "✘") + " / " + ("✔" if c.get("bh", 1) < 0.05 else "✘") if c["family"] == "headline" else \
             ("ext: " + ("✔" if c.get("holm_ext", 1) < 0.05 else "✘") + " / " + ("✔" if c.get("bh_ext", 1) < 0.05 else "✘"))
        L.append(f"| {c['id']} | {c['claim']} | {c['label']} | {c['n']} | {c.get('effect', '')} | {c['test']} | {c['p']:.2g} | "
                 f"{c.get('holm', float('nan')):.2g} | {c.get('bh', float('nan')):.2g} | {c.get('holm_ext', float('nan')):.2g} | {sv} |")
    L += ["", "Notes:", ""]
    for c in claims:
        if c.get("note"):
            L.append(f"* `{c['id']}`: {c['note']}")
    (T.OUT / "multiple_comparisons.md").write_text("\n".join(L) + "\n")
    n_h = sum(1 for c in head if c["holm"] < 0.05); n_b = sum(1 for c in head if c["bh"] < 0.05)
    print(f"headline m={len(head)}: Holm survivors {n_h}, BH survivors {n_b}; raw p<0.05: {sum(1 for c in head if c['p'] < 0.05)}")
    for c in claims:
        if c.get("p") is not None:
            print(f"{c['id']:34s} {c['label']} n={c['n']:4d} {c.get('effect', ''):40s} p={c['p']:.3g} holm={c.get('holm', float('nan')):.3g} bh={c.get('bh', float('nan')):.3g}")


if __name__ == "__main__":
    main()
