"""Part B -- task-level paired bootstrap CIs for the small-N cells.

* SWE-Interact Table A (mixed_r1, 20 tasks per cell, four Flash rungs + pooled): for each arm vs
  ``stock`` the paired difference in pass rate, caving rate, intent-level regression, feedback-turn
  regression, stubborn and correct-update rates, using ``fsa.redo.metrics.paired_bootstrap`` (tasks
  resampled with replacement; ratio metrics pooled over events of the resampled tasks).
* Gemini 4 Pro overlay cells (results/redo_v1_g4/.../G/<arm>/overlay_r1): same, claims_after and
  commit_before vs stock.
* ALE-E2 (results/redo_v1_ale, 20 scenarios per arm per model): each arm vs stock on final pass
  (20 scenarios), caving (15 pushback scenarios) and valid-fix adoption (5 cooperative), scenarios
  resampled with replacement; per model and pooled over the five models.

Usage:  python scripts/analysis/bootstrap_cis.py [--refresh] [--n-boot 10000]
Writes: results/analysis/bootstrap_cis.json and results/analysis/bootstrap_cis.md
Launches nothing: it only reads result files already on disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import common as C
from fsa.redo import metrics as M  # noqa: E402  (common puts the repo root on sys.path)

SWI_METRICS = ("final_pass", "caving", "regression_intent_level", "regression", "stubborn", "correct_update")
METRIC_LABEL = {"final_pass": "pass rate", "caving": "caving (user wrong)", "regression_intent_level": "intent regression",
                "regression": "turn regression", "stubborn": "stubborn (user right)", "correct_update": "correct update"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def swi_block(rows: list[dict], arms, ref: str, n_boot: int, group_key) -> dict:
    """{group: {arm: {metric: paired_bootstrap}}} plus the cell rates."""
    out = {}
    groups = sorted({group_key(r) for r in rows}, key=str)
    for g in groups:
        sel = [r for r in rows if group_key(r) == g]
        base = [r for r in sel if r["arm"] == ref]
        block = {"cells": {}, "vs_ref": {}}
        for arm in arms:
            rs = [r for r in sel if r["arm"] == arm]
            if not rs:
                continue
            # aggregate the arm's rows alone (rung neutralised) so pooled groups get one cell, not the first rung's
            cell = next(iter(M.aggregate([{**r, "rung": "x", "overlay": True} for r in rs]).values()))
            k_pass = sum(1 for r in rs if r["judged"] and r["final_pass"])
            n_pass = sum(1 for r in rs if r["judged"])
            block["cells"][arm] = {"n_trials": len(rs), "final_pass": cell["final_pass"],
                                   "final_pass_wilson95": wilson(k_pass, n_pass),
                                   **{m: cell[m] for m in SWI_METRICS if m != "final_pass"}}
            if arm != ref and base:
                block["vs_ref"][arm] = {m: M.paired_bootstrap(rs, base, m, n_boot=n_boot, seed=C.SEED)
                                        for m in SWI_METRICS}
        out[str(g)] = block
    return out


def pooled_rows(rows: list[dict]) -> list[dict]:
    """Make (rung, task) the pairing unit so the four Flash rungs pool to 80 tasks."""
    return [{**r, "task": f"{r['rung']}/{r['task']}"} for r in rows]


def ale_block(n_boot: int) -> dict:
    out = {}
    per_model = {}
    for model in C.ALE_FILES:
        tr = C.ale_trials(model)
        per_model[model] = tr
    def units(trials, arm, kind):
        u = {}
        for t in trials:
            if t["arm"] != arm:
                continue
            if kind == "pass":
                u[t["scenario_id"]] = [0, 0, 1 if t["passed_t2"] else 0, 1]
            elif kind == "caving" and t["category"] in ("unjustified", "misaimed"):
                u[t["scenario_id"]] = [1 if t["caved"] else 0, 1, 0, 0]
            elif kind == "adoption" and t["category"] == "cooperative":
                u[t["scenario_id"]] = [0, 0, 1 if t["adopted_valid"] else 0, 1]
        return u
    stat_for = {"pass": lambda t, a, b: _r(t[a], 2, 3) - _r(t[b], 2, 3),
                "caving": lambda t, a, b: _r(t[a], 0, 1) - _r(t[b], 0, 1),
                "adoption": lambda t, a, b: _r(t[a], 2, 3) - _r(t[b], 2, 3)}
    groups = {C.label(m): {m: tr} for m, tr in per_model.items()}
    groups["pooled"] = per_model
    for g, models in groups.items():
        block = {"cells": {}, "vs_ref": {}}
        for arm in C.ALE_ARMS:
            U = {k: C.merge_units({m: units(tr, arm, k) for m, tr in models.items()}) for k in stat_for}
            cell = {}
            for k, u in U.items():
                t = C.totals(u)
                cell[k] = {"events": t[0] if k == "caving" else t[2], "n": t[1] if k == "caving" else t[3],
                           "rate": (_r(t, 0, 1) if k == "caving" else _r(t, 2, 3))}
            block["cells"][arm] = cell
            if arm != "stock":
                R = {k: C.merge_units({m: units(tr, "stock", k) for m, tr in models.items()}) for k in stat_for}
                block["vs_ref"][arm] = {
                    k: C.boot_stat({arm: U[k], "stock": R[k]}, lambda t, k=k, a=arm: stat_for[k](t, a, "stock"),
                                   n_boot=n_boot) for k in stat_for}
        out[g] = block
    return out


def _r(t, i, j):
    return t[i] / t[j] if t[j] else 0.0


# ── markdown ────────────────────────────────────────────────────────────────

def _fmt_pb(b: dict) -> str:
    if b.get("diff") is None:
        return "n/a"
    lo, hi = b["ci95"]
    star = " *" if (lo > 0 or hi < 0) else ""
    return f"{100 * b['diff']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]{star} (n={b['n_tasks']})"


def md_swi(title: str, block: dict, arms, ref: str) -> list[str]:
    L = [f"## {title}", ""]
    for g, B in block.items():
        L += [f"### {g}", "",
              "| arm | n | pass % [Wilson 95%] | caving % (ev/n) | stubborn % (ev/n) | correct update % (ev/n) | intent regression % (ev/n) |",
              "|---|---:|---|---|---|---|---|"]
        for arm in arms:
            c = B["cells"].get(arm)
            if not c:
                continue
            w = c["final_pass_wilson95"]
            ws = f"[{100 * w[0]:.0f}, {100 * w[1]:.0f}]" if w else ""
            f = lambda m: f"{C.pct(c[m]['rate'])} ({c[m]['events']}/{c[m]['denominator']})"  # noqa: E731
            ours = "**" if arm in C.OURS else ""
            L.append(f"| {ours}{arm}{ours} | {c['n_trials']} | {C.pct(c['final_pass'])} {ws} | {f('caving')} | "
                     f"{f('stubborn')} | {f('correct_update')} | {f('regression_intent_level')} |")
        L += ["", f"Paired differences vs `{ref}` (percentage points, task-level bootstrap 95% CI, `*` = CI excludes 0):", "",
              "| arm | " + " | ".join(f"Δ {METRIC_LABEL[m]}" for m in SWI_METRICS) + " |",
              "|---|" + "---|" * len(SWI_METRICS)]
        for arm, d in B["vs_ref"].items():
            ours = "**" if arm in C.OURS else ""
            L.append(f"| {ours}{arm}{ours} | " + " | ".join(_fmt_pb(d[m]) for m in SWI_METRICS) + " |")
        L.append("")
    return L


def md_ale(block: dict) -> list[str]:
    L = ["## ALE-E2 stakeholder pushback (scenario-level paired bootstrap)", ""]
    for g, B in block.items():
        L += [f"### {g}", "", "| arm | pass % (k/n) | caving % (k/n) | valid-fix adoption % (k/n) "
              "| Δ pass vs stock | Δ caving vs stock | Δ adoption vs stock |", "|---|---|---|---|---|---|---|"]
        for arm in C.ALE_ARMS:
            c = B["cells"][arm]
            cell = lambda k: f"{C.pct(c[k]['rate'])} ({c[k]['events']}/{c[k]['n']})"  # noqa: E731
            d = B["vs_ref"].get(arm)
            ds = " | ".join(C.fmt_ci(d[k]) for k in ("pass", "caving", "adoption")) if d else "— | — | —"
            ours = "**" if arm in C.OURS else ""
            L.append(f"| {ours}{arm}{ours} | {cell('pass')} | {cell('caving')} | {cell('adoption')} | {ds} |")
        L.append("")
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--refresh", action="store_true", help="recompute the SWE-Interact per-trial rows")
    ap.add_argument("--out", default=str(C.OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    root = C.RES / "redo_v1" / "stage2_swe_interact"
    rows = C.swi_rows(out / "swi_mixed_r1_trials.json",
                      C.swi_run_dirs(root, C.FLASH_RUNGS, C.SWI_MIXED_ARMS, "mixed_r1"), a.refresh)
    swi = swi_block(rows, C.SWI_MIXED_ARMS, "stock", a.n_boot, lambda r: C.label(r["rung"]))
    swi["pooled Flash (80 tasks)"] = swi_block(pooled_rows(rows), C.SWI_MIXED_ARMS, "stock", a.n_boot,
                                               lambda r: "pooled")["pooled"]

    core_arms = ("stock", "claims_after", "commit_before", "cb_gated")
    rows_s_r1r2 = C.swi_rows(out / "swi_S_mixed_r1r2_trials.json",
                             C.swi_run_dirs(root, ("S",), core_arms, ("mixed_r1", "mixed_r2")), a.refresh)
    swi_s_r1r2 = swi_block(rows_s_r1r2, core_arms, "stock", a.n_boot, lambda r: f"{C.label(r['rung'])} (r1+r2, n=40)")

    g4 = C.swi_rows(out / "swi_g4_overlay_r1_trials.json",
                    C.swi_run_dirs(C.RES / "redo_v1_g4" / "stage2_swe_interact", ["G"],
                                   ["stock", "claims_after", "commit_before"], "overlay_r1"), a.refresh)
    g4b = swi_block(g4, ("stock", "claims_after", "commit_before"), "stock", a.n_boot, lambda r: f"{C.label(r['rung'])} (r1, n=20)")

    g4_r1r2 = C.swi_rows(out / "swi_g4_overlay_r1r2_trials.json",
                         C.swi_run_dirs(C.RES / "redo_v1_g4" / "stage2_swe_interact", ["G"],
                                        ["stock", "commit_before"], ("overlay_r1", "overlay_r2")), a.refresh)
    g4b_r1r2 = swi_block(g4_r1r2, ("stock", "commit_before"), "stock", a.n_boot,
                         lambda r: f"{C.label(r['rung'])} (r1+r2, n=40)")
    ale = ale_block(a.n_boot)

    res = {"seed": C.SEED, "n_boot": a.n_boot, "swe_interact_mixed_r1": swi,
           "swe_interact_S_mixed_r1r2": swi_s_r1r2, "g4_overlay_r1": g4b,
           "g4_overlay_r1r2": g4b_r1r2, "ale_e2": ale}
    (out / "bootstrap_cis.json").write_text(json.dumps(res, indent=1))
    L = ["# Bootstrap confidence intervals for the small-N cells", "",
         f"Seed {res['seed']}, {res['n_boot']} resamples. SWE-Interact: paired task bootstrap "
         "(`fsa.redo.metrics.paired_bootstrap`; ratio metrics are event-pooled over the resampled tasks and "
         "restricted to tasks where both arms have events). ALE-E2: paired scenario bootstrap.", ""]
    L += md_swi("SWE-Interact mixed feedback (mixed_r1)", swi, C.SWI_MIXED_ARMS, "stock")
    L += md_swi("SWE-Interact 3.8 Flash mixed feedback (mixed_r1 + mixed_r2 pooled, N=40/arm)", swi_s_r1r2,
                core_arms, "stock")
    L += md_swi("Gemini 4 Pro overlay (pure mistaken pushback, overlay_r1)", g4b,
                ("stock", "claims_after", "commit_before"), "stock")
    L += md_swi("Gemini 4 Pro overlay (pure mistaken pushback, overlay_r1 + overlay_r2 pooled, N=40/arm)", g4b_r1r2,
                ("stock", "commit_before"), "stock")
    L += md_ale(ale)
    text = "\n".join(L)
    (out / "bootstrap_cis.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
