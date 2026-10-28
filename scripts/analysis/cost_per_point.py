"""Tokens and $-equivalent cost per point of pass rate, per benchmark x model x arm (Tier-3 #5).

Pure post-hoc analysis over existing result files (read-only; no model, docker or systemd).

Sources
-------
* Harbor DeepSWE-1.1 (``ds11`` opencode, ``ds11m`` mini-swe-agent) and Terminal-Bench 4 (``tb4`` opencode):
  per-trial rows from ``scripts/analysis/harbor_trials.py`` (``~/jobs/<job>/<trial>/result.json`` +
  ATIF trajectory + opencode event stream). One trial per task, chosen with the *same* rule as
  ``scripts/harbor_collect.py`` (so pass rates match the headline tables); token totals over *all*
  attempts (incl. retried/removed ones still on disk) are reported separately as ``all_attempt_*``.
  ``ds11m_G_*`` and ``tb4_G_tib`` are still running: partial snapshot.
* SWE-Interact: ``results/analysis/swi_*_trials.json`` (``tokens.agent`` = the coding agent incl. cached
  prompt, ``tokens.claim_book`` / ``tokens.baseline_port`` = method overhead calls; user simulator and judges
  are excluded from cost). Wall time from the trial ``result.json``.
* HLE single-shot / 2-turn: ``results/hle/{single_shot,multi_turn}_<model>.jsonl`` (``usage`` incl.
  ``completion_tokens_details.reasoning_tokens``; judge calls excluded).
* Vals Finance Agent: ``results/finance_agent_tools/tools_<model>.jsonl`` (live 4-tool harness) and
  ``results/finance_agent/single_shot_<model>.jsonl`` (closed book).
* ALE-Claw docker sweep: ``~/codes/agents-last-exam/.logs/ale/tib_docker90_<arm>_<rung>_<seed>/.../run.json``
  (``usage``), same selection rule as ``scripts/ale_docker90_table.py`` (latest run per task).

Token accounting
----------------
``input`` = all prompt tokens (cached + uncached), ``cached`` = cache-read prompt tokens, ``output`` =
visible completion tokens, ``thinking`` = reasoning tokens (billed as output; *not* included in ``output``
by any of these APIs; for Gemini 4.0 Pro through opencode, which reports 0, the count attributed from the bridge
logs by ``scripts/analysis/bridge_logs.py`` is used; ``None`` when unknown, e.g. SWE-Interact and ALE-Claw).
``total = input + output + thinking``.

$-equivalent: an ASSUMED, illustrative price table (USD / 1M tokens), override with ``--prices file.json``
({"G": {"in": 2.0, "cached": 0.5, "out": 12.0}, ...}). These are NOT official prices for these models.

    python scripts/analysis/cost_per_point.py [--refresh] [--prices prices.json]

Outputs: ``results/analysis/cost_per_point.json`` / ``.md``; ``docs/figures/analysis/pareto_<bench>.png``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harbor_trials as HT  # noqa: E402

ROOT = HERE.parents[1]
RES = ROOT / "results"
OUT = RES / "analysis"
FIG = ROOT / "docs" / "figures" / "analysis"
ALE_ROOT = Path(os.path.expanduser("~/codes/agents-last-exam/.logs/ale"))

RUNG_LABEL = {"W": "3.5 Flash-Lite", "F": "3.5 Flash", "M": "3.6 Flash", "S": "3.8 Flash", "G": "4.0 Pro"}
FILE_RUNG = {"gemini_3_5_flash_lite": "W", "gemini_3_5_flash": "F", "gemini_3_6_flash": "M",
             "gemini_3_8_flash": "S", "gemini_4_0_argon": "G"}
# ASSUMED illustrative prices, USD per 1M tokens (cached = 25% of input). Parameterise with --prices.
DEFAULT_PRICES = {
    "W": {"in": 0.10, "cached": 0.025, "out": 0.40},
    "F": {"in": 0.30, "cached": 0.075, "out": 2.50},
    "M": {"in": 0.30, "cached": 0.075, "out": 2.50},
    "S": {"in": 0.50, "cached": 0.125, "out": 3.00},
    "G": {"in": 2.00, "cached": 0.50, "out": 12.00},
}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _lit(s):
    """Fields in the HLE/finance jsonl are python-repr strings."""
    if isinstance(s, dict) or s is None:
        return s or {}
    import ast
    try:
        return ast.literal_eval(s)
    except Exception:  # noqa: BLE001
        try:
            return json.loads(s)
        except Exception:  # noqa: BLE001
            return {}


def _truthy(v):
    return v is True or str(v).lower() == "true"


def unit(input_=0, cached=0, output=0, thinking=None, calls=None, steps=None, wall=None, timeout=False,
         passed=False, score=None, task=None):
    return {"input": input_ or 0, "cached": cached or 0, "output": output or 0, "thinking": thinking,
            "calls": calls, "steps": steps, "wall_sec": wall, "timeout": bool(timeout), "passed": bool(passed),
            "score": score, "task": task}


# ── loaders: each returns {(bench, rung, arm): [unit, ...]} ──────────────────

def _bridge_thoughts():
    """Per-trial thinking tokens attributed from the LLM bridge logs (scripts/analysis/bridge_logs.py)."""
    p = OUT / "bridge_harbor_trials.json"
    if not p.exists():
        return {}
    return {(r["job"], r["trial"]): r["thoughts"] for r in json.loads(p.read_text())
            if (r.get("confidence") or 0) >= 0.8 and r.get("thoughts") is not None}


def load_harbor(refresh: bool):
    rows = HT.load_trials(refresh=refresh)
    sel = HT.select_per_task(rows, "collect")
    bridge = _bridge_thoughts()
    out = defaultdict(list)
    all_att = defaultdict(lambda: {"input": 0, "output": 0, "thinking": 0, "n": 0})
    for r in rows:
        a = all_att[(r["bench"], r["rung"], r["arm"])]
        a["input"] += r["n_input_tokens"] or 0
        a["output"] += r["n_output_tokens"] or 0
        a["thinking"] += r.get("reasoning_tokens") or 0
        a["n"] += 1
    for bj, tasks in sel.items():
        for task, r in tasks.items():
            if not HT.is_valid(r):
                continue
            think = r.get("reasoning_tokens")
            if r["harness"] == "opencode" and r["rung"] == "G":
                # opencode + 4.0 Pro reports 0 reasoning tokens: take the bridge-attributed count, else unknown
                think = bridge.get((r["job"], r["trial"]))
            out[(r["bench"], r["rung"], r["arm"])].append(unit(
                r["n_input_tokens"], r["n_cache_tokens"], r["n_output_tokens"], think,
                calls=r.get("n_llm_calls"), steps=r.get("n_agent_steps"), wall=r.get("agent_sec"),
                timeout=r["timeout"], passed=r["passed"], score=r.get("partial"), task=task))
    return out, {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in all_att.items()}


def _swi_wall(run: str, trial: str):
    base = RES / ("redo_v1_g4" if "/G/" in f"/{run}/" else "redo_v1") / run / "jobs"
    for p in base.glob(f"*/{trial}/result.json"):
        try:
            r = json.loads(p.read_text())
            a = datetime.fromisoformat(r["started_at"].replace("Z", "+00:00")).timestamp()
            b = datetime.fromisoformat(r["finished_at"].replace("Z", "+00:00")).timestamp()
            return b - a
        except Exception:  # noqa: BLE001
            return None
    return None


def load_swi():
    out = defaultdict(list)
    seen = set()
    files = ["swi_mixed_r1_trials.json", "swi_S_mixed_r1r2_trials.json", "swi_g4_overlay_r1_trials.json",
             "swi_g4_overlay_r1r2_trials.json"]
    for fn in files:
        p = OUT / fn
        if not p.exists():
            continue
        for t in json.loads(p.read_text()):
            key = (t["run"], t["trial"])
            if key in seen:
                continue
            seen.add(key)
            cell = t["run"].split("/")[-1]
            bench = "swi_mixed" if cell.startswith("mixed") else "swi_overlay"
            tk = t.get("tokens") or {}
            ag, cb, bp = tk.get("agent") or {}, tk.get("claim_book") or {}, tk.get("baseline_port") or {}
            prompt = (ag.get("prompt") or 0) + (cb.get("prompt") or 0) + (bp.get("prompt") or 0)
            uncached = (ag.get("prompt_uncached") or 0) + (cb.get("prompt") or 0) + (bp.get("prompt") or 0)
            out[(bench, t["rung"], t["arm"])].append(unit(
                prompt, prompt - uncached, (ag.get("output") or 0) + (cb.get("output") or 0) + (bp.get("output") or 0),
                None, calls=(ag.get("calls") or 0) + (cb.get("calls") or 0) + (bp.get("calls") or 0),
                steps=t.get("n_turns"), wall=_swi_wall(t["run"], t["trial"]), passed=bool(t.get("final_pass")),
                score=t.get("judge_score"), task=f"{t['task']}#{cell}"))
    return out


def _usage_unit(u):
    u = _lit(u)
    ptd = u.get("prompt_tokens_details") or {}
    ctd = u.get("completion_tokens_details") or {}
    return (u.get("prompt_tokens") or 0, (ptd.get("cached_tokens") or 0) if isinstance(ptd, dict) else 0,
            u.get("completion_tokens") or 0, (ctd.get("reasoning_tokens") if isinstance(ctd, dict) else None))


def load_jsonl_benches():
    out = defaultdict(list)
    specs = [("hle_single", RES / "hle", "single_shot_", lambda r: _truthy(r.get("correct_bool")), None),
             ("hle_2turn", RES / "hle", "multi_turn_", lambda r: _truthy(r.get("t2_correct")), None),
             ("finance_tools", RES / "finance_agent_tools", "tools_", lambda r: _truthy(r.get("passed")), "score"),
             ("finance_closed", RES / "finance_agent", "single_shot_", lambda r: _truthy(r.get("passed")), "score")]
    for bench, d, prefix, ok, score_key in specs:
        for p in sorted(d.glob(f"{prefix}*.jsonl")):
            rung = FILE_RUNG.get(p.stem[len(prefix):])
            if not rung:
                continue
            for line in p.open():
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                i, c, o, th = _usage_unit(r.get("usage"))
                calls = _f(r.get("n_tool_calls"))
                out[(bench, rung, r.get("arm"))].append(unit(
                    i, c, o, th, calls=(calls + 1) if calls is not None else (2 if bench == "hle_2turn" else 1),
                    wall=_f(r.get("latency_sec")), passed=ok(r),
                    score=_f(r.get(score_key)) if score_key else None, task=r.get("id")))
    return out


def load_ale():
    out = defaultdict(list)
    if not ALE_ROOT.exists():
        return out
    for rung in "WFMSG":
        for arm in ("stock", "cb_draftfirst"):
            for seed in ("s1", "t300"):
                best = {}
                for p in glob.glob(f"{ALE_ROOT}/tib_docker90_{arm}_{rung}_{seed}/ale_claw/*/*/v0/*/run.json"):
                    task = p.split("/ale_claw/")[1].split("/")[1]
                    ts = p.split("/")[-2]
                    if task in best and best[task][0] > ts:
                        continue
                    best[task] = (ts, p)
                for task, (_, p) in best.items():
                    try:
                        r = json.loads(Path(p).read_text())
                    except Exception:  # noqa: BLE001
                        continue
                    ev = Path(p).with_name("eval_result.json")
                    sc = r.get("score")
                    if ev.exists():
                        try:
                            sc = json.loads(ev.read_text()).get("score", sc)
                        except Exception:  # noqa: BLE001
                            pass
                    if r.get("status") == "failed" and sc in (None, 0.0):
                        continue          # same exclusion as scripts/ale_docker90_table.py
                    u = r.get("usage") or {}
                    term = (r.get("termination") or {}).get("reason")
                    out[(f"ale_claw_{seed}", rung, arm)].append(unit(
                        u.get("total_input_tokens"), u.get("total_cache_read_tokens"), u.get("total_output_tokens"),
                        u.get("total_reasoning_tokens"), steps=u.get("total_steps"),
                        wall=(r.get("timings") or {}).get("duration_s"),
                        timeout=term not in (None, "completed"), passed=(sc or 0) >= 0.999, score=sc, task=task))
    return out


# ── aggregation ──────────────────────────────────────────────────────────────

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def cost_usd(u, price):
    th = u["thinking"] or 0
    unc = max(0, u["input"] - u["cached"])
    return (unc * price["in"] + u["cached"] * price["cached"] + (u["output"] + th) * price["out"]) / 1e6


def summarise(units, price, tasks=None):
    if tasks is not None:
        units = [u for u in units if u["task"] in tasks]
    n = len(units)
    if not n:
        return None
    th_known = [u["thinking"] for u in units if u["thinking"] is not None]
    tot = [u["input"] + u["output"] + (u["thinking"] or 0) for u in units]
    pass_rate = sum(u["passed"] for u in units) / n
    c = [cost_usd(u, price) for u in units]
    return {
        "n": n, "pass_rate": pass_rate, "n_pass": sum(u["passed"] for u in units),
        "mean_score": _mean([u["score"] for u in units]),
        "input_mean": _mean([u["input"] for u in units]), "cached_mean": _mean([u["cached"] for u in units]),
        "uncached_mean": _mean([u["input"] - u["cached"] for u in units]),
        "output_mean": _mean([u["output"] for u in units]),
        "thinking_mean": (sum(th_known) / len(th_known)) if len(th_known) >= 0.9 * n else None,
        "thinking_reported_frac": len(th_known) / n,
        "total_tokens_mean": sum(tot) / n, "total_tokens_median": st.median(tot),
        "calls_mean": _mean([u["calls"] for u in units]), "steps_mean": _mean([u["steps"] for u in units]),
        "wall_sec_mean": _mean([u["wall_sec"] for u in units]), "timeouts": sum(u["timeout"] for u in units),
        "usd_per_task": sum(c) / n, "usd_total": sum(c),
        "usd_per_pass": (sum(c) / sum(u["passed"] for u in units)) if any(u["passed"] for u in units) else None,
        "pass_per_Mtok": pass_rate / (sum(tot) / n / 1e6) if sum(tot) else None,
    }


def build(prices, refresh=False):
    data = {}
    harbor, all_att = load_harbor(refresh)
    for src in (harbor, load_swi(), load_jsonl_benches(), load_ale()):
        for k, v in src.items():
            data[k] = v
    benches = sorted({k[0] for k in data})
    res = {"price_table_usd_per_Mtok": prices,
           "price_note": "ASSUMED illustrative prices (not official); cached prompt = 25% of input price; "
                         "thinking billed at output price.",
           "harbor_all_attempt_tokens": all_att, "cells": [], "vs_stock": []}
    for b in benches:
        for rung in "WFMSG":
            arms = sorted({k[2] for k in data if k[0] == b and k[1] == rung})
            if not arms:
                continue
            ref = "stock" if "stock" in arms else None
            for arm in arms:
                s = summarise(data[(b, rung, arm)], prices[rung])
                if s:
                    res["cells"].append({"bench": b, "rung": rung, "model": RUNG_LABEL[rung], "arm": arm, **s})
                if ref and arm != ref:
                    # paired: tasks present in both arms
                    common = {u["task"] for u in data[(b, rung, arm)]} & {u["task"] for u in data[(b, rung, ref)]}
                    a = summarise(data[(b, rung, arm)], prices[rung], common)
                    r0 = summarise(data[(b, rung, ref)], prices[rung], common)
                    if not a or not r0:
                        continue
                    dp = 100 * (a["pass_rate"] - r0["pass_rate"])
                    dtok = a["total_tokens_mean"] - r0["total_tokens_mean"]
                    dusd = a["usd_per_task"] - r0["usd_per_task"]
                    res["vs_stock"].append({
                        "bench": b, "rung": rung, "model": RUNG_LABEL[rung], "arm": arm, "n_paired": len(common),
                        "stock_pass": r0["pass_rate"], "arm_pass": a["pass_rate"], "delta_pass_pts": dp,
                        "stock_tokens": r0["total_tokens_mean"], "arm_tokens": a["total_tokens_mean"],
                        "delta_tokens_per_task": dtok, "token_ratio": a["total_tokens_mean"] / r0["total_tokens_mean"]
                        if r0["total_tokens_mean"] else None,
                        "delta_output_plus_thinking": (a["output_mean"] + (a["thinking_mean"] or 0))
                        - (r0["output_mean"] + (r0["thinking_mean"] or 0)),
                        "delta_usd_per_task": dusd,
                        # incremental cost-effectiveness: extra tokens / $ per +1 pass-rate point (None if dp<=0)
                        # only defined when the arm buys points with MORE spend; cheaper-and-better = dominates
                        "extra_tokens_per_pass_point": dtok / dp if (dp > 0 and dtok > 0) else None,
                        "extra_usd_per_pass_point": dusd / dp if (dp > 0 and dusd > 0) else None,
                        "dominates_stock": dp >= 0 and dtok <= 0, "dominated_by_stock": dp <= 0 and dtok >= 0,
                        "stock_timeouts": r0["timeouts"], "arm_timeouts": a["timeouts"],
                    })
    return res


def plot(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("matplotlib unavailable: skipping figures")
        return []
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"W": "#9e9e9e", "F": "#1f77b4", "M": "#2ca02c", "S": "#ff7f0e", "G": "#d62728"}
    markers = {"stock": "o", "cb_draftfirst": "*", "tib": "*", "commit_before": "*", "cb_gated": "P",
               "claims_after": "X", "conf_gate": "v", "sc_vote3": "s", "verify_then_respond": "D"}
    files = []
    for b in sorted({c["bench"] for c in res["cells"]}):
        cells = [c for c in res["cells"] if c["bench"] == b]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for c in cells:
            ax.scatter(c["total_tokens_mean"], 100 * c["pass_rate"], c=colors[c["rung"]],
                       marker=markers.get(c["arm"], "^"), s=70 if c["arm"] != "stock" else 45,
                       edgecolors="k", linewidths=0.4, zorder=3)
            ax.annotate(c["arm"].replace("cb_draftfirst", "tib"), (c["total_tokens_mean"], 100 * c["pass_rate"]),
                        fontsize=6, xytext=(3, 2), textcoords="offset points")
        for rung in "WFMSG":
            rc = sorted([c for c in cells if c["rung"] == rung], key=lambda c: c["total_tokens_mean"])
            if len(rc) < 2:
                continue
            # Pareto frontier within the model (max pass for <= tokens)
            front, best = [], -1
            for c in rc:
                if c["pass_rate"] > best:
                    front.append(c)
                    best = c["pass_rate"]
            ax.plot([c["total_tokens_mean"] for c in front], [100 * c["pass_rate"] for c in front],
                    c=colors[rung], lw=1, alpha=0.6, label=f"{RUNG_LABEL[rung]} frontier")
        ax.set_xscale("log")
        ax.set_xlabel("mean total tokens per task (input+output+thinking, log)")
        ax.set_ylabel("pass rate (%)")
        ax.set_title(f"{b}: pass vs tokens (per model x arm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="best")
        fig.tight_layout()
        fp = FIG / f"pareto_{b}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        files.append(str(fp.relative_to(ROOT)))
    return files


def _icer(v, kind):
    if v["delta_pass_pts"] > 0 and (v["delta_tokens_per_task"] if kind == "tok" else v["delta_usd_per_task"]) <= 0:
        return "dominates (cheaper & better)"
    if v["delta_pass_pts"] <= 0:
        return "n/a (no gain)"
    x = v["extra_tokens_per_pass_point"] if kind == "tok" else v["extra_usd_per_pass_point"]
    return "—" if x is None else (f"{x / 1e3:,.1f}k" if kind == "tok" else f"{x:.4f}")


def to_md(res):
    L = ["# Cost per point of pass rate (auto-generated by scripts/analysis/cost_per_point.py)", "",
         "Prices are ASSUMED (USD / 1M tokens): " + ", ".join(
             f"{RUNG_LABEL[k]} in {v['in']}/cached {v['cached']}/out {v['out']}"
             for k, v in res["price_table_usd_per_Mtok"].items()), "",
         "## Per cell", "",
         "| bench | model | arm | n | pass | input (k) | cached (k) | output (k) | thinking (k) | calls | steps | wall (min) | timeouts | $/task | $/pass | pass per Mtok |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    k = lambda x: "—" if x is None else f"{x / 1e3:,.1f}"
    g = lambda x, f="{:.1f}": "—" if x is None else f.format(x)
    for c in res["cells"]:
        L.append(f"| {c['bench']} | {c['model']} | {c['arm']} | {c['n']} | {100 * c['pass_rate']:.1f}% | {k(c['input_mean'])} | "
                 f"{k(c['cached_mean'])} | {k(c['output_mean'])} | {k(c['thinking_mean'])} | {g(c['calls_mean'])} | "
                 f"{g(c['steps_mean'])} | {g(c['wall_sec_mean'] and c['wall_sec_mean'] / 60)} | {c['timeouts']} | "
                 f"{g(c['usd_per_task'], '{:.3f}')} | {g(c['usd_per_pass'], '{:.3f}')} | {g(c['pass_per_Mtok'], '{:.3f}')} |")
    L += ["", "## Arm vs stock (paired tasks)", "",
          "| bench | model | arm | n | stock → arm pass | Δ pts | tokens ratio | Δ tokens/task (k) | Δ $/task | extra tokens per +1 pt (k) | extra $ per +1 pt |",
          "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for v in res["vs_stock"]:
        L.append(f"| {v['bench']} | {v['model']} | {v['arm']} | {v['n_paired']} | {100 * v['stock_pass']:.1f}% → {100 * v['arm_pass']:.1f}% | "
                 f"{v['delta_pass_pts']:+.1f} | {g(v['token_ratio'], '{:.2f}')}× | {v['delta_tokens_per_task'] / 1e3:+,.1f} | "
                 f"{v['delta_usd_per_task']:+.4f} | {_icer(v, 'tok')} | {_icer(v, 'usd')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-scan ~/jobs (Harbor) instead of the cached trial table")
    ap.add_argument("--prices", help="JSON file overriding the assumed price table")
    a = ap.parse_args()
    prices = dict(DEFAULT_PRICES)
    if a.prices:
        prices.update(json.loads(Path(a.prices).read_text()))
    res = build(prices, refresh=a.refresh)
    res["figures"] = plot(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost_per_point.json").write_text(json.dumps(res, indent=1))
    (OUT / "cost_per_point.md").write_text(to_md(res))
    print(f"{len(res['cells'])} cells, {len(res['vs_stock'])} arm-vs-stock rows -> results/analysis/cost_per_point.json")
    for v in res["vs_stock"]:
        if v["arm"] in ("tib", "cb_draftfirst", "commit_before", "cb_gated"):
            print(f"  {v['bench']:15s} {v['model']:15s} {v['arm']:14s} n={v['n_paired']:3d} "
                  f"{100 * v['stock_pass']:5.1f}->{100 * v['arm_pass']:5.1f} ({v['delta_pass_pts']:+5.1f}) "
                  f"tok x{v['token_ratio'] or 0:.2f}  d$={v['delta_usd_per_task']:+.4f}")


if __name__ == "__main__":
    main()
