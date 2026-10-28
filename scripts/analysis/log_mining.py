"""Log mining: harness discrepancy, loss taxonomy, timeouts/exceptions, and biases in the Harbor / ALE / SWE-Interact runs.

Pure post-hoc analysis (read-only; no model, docker or systemd). ``~/jobs`` is read-only and ``ds11m_G_*`` /
``tb4_G_tib`` may still be running -> partial snapshot. Re-run any time:

    python scripts/analysis/harbor_trials.py --refresh     # re-scan ~/jobs (optional; log_mining does it with --refresh)
    python scripts/analysis/bridge_logs.py                 # per-trial bridge attribution (optional input)
    python scripts/analysis/log_mining.py [--refresh]

Sections (keys of ``results/analysis/log_mining.json``):

* ``retry_audit``      every attempt ever collected (``results/harbor/<job>/*/result.json`` archive + live ``~/jobs``)
                       vs the trial that is counted. Headline pass rates use ``scripts/harbor_collect.py``'s rule, which
                       prefers a *passing* attempt when a task has several; trials with exceptions (incl.
                       ``AgentTimeoutError`` / ``NonZeroAgentExitCodeError`` that *were* scored) were deleted and re-run.
                       We recompute an intent-to-treat (ITT) score = the first attempt that produced a verifier reward.
* ``harness``          DeepSWE G: opencode (``ds11_G``) vs mini-swe-agent (``ds11m_G``) per-task join, compliance, budgets.
* ``losses``           failure-mode taxonomy for stock-pass / TIB-fail tasks (ds11 S, tb4 G, ds11 G, ALE-Claw t300) with quotes.
* ``timeouts``         AgentTimeoutError / exception rates per arm, and whether TIB's milestone commits rescue timed-out work.
* ``other``            run-window confounds, empty / blocked / zero-token responses, MAX_TOKENS, SWE-Interact provider errors
                       and quota-degraded trials, unequal trial counts, paired-subset difficulty.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harbor_trials as HT  # noqa: E402

ROOT = HERE.parents[1]
RES = ROOT / "results"
OUT = RES / "analysis"
FIG = ROOT / "docs" / "figures" / "analysis"
ARCHIVE = RES / "harbor"
ALE_ROOT = Path(os.path.expanduser("~/codes/agents-last-exam/.logs/ale"))
LABEL = {"W": "3.5 Flash-Lite", "F": "3.5 Flash", "M": "3.6 Flash", "S": "3.8 Flash", "G": "4.0 Pro"}
# exceptions that indicate an infrastructure failure (agent crash / API / verifier), as opposed to the agent's own
# outcome; AgentTimeoutError is NOT excused (running out of wall-clock is an outcome of the agent + its protocol)
INFRA_EXC = {"NonZeroAgentExitCodeError", "ApiRateLimitError", "NetworkConnectionError", "CancelledError",
             "VerifierTimeoutError", "PermissionError", "RewardFileNotFoundError", "AgentSetupTimeoutError"}


def _iso(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001
        return None


def _day(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%m-%d %H:%M") if ts else None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _clip(s, n=280):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _passed(rew):
    return rew is not None and float(rew) >= 0.999


def wtl(a: dict, b: dict, keys):
    """a = stock, b = arm; returns pass rates and W/T/L on pass (b better = W)."""
    keys = list(keys)
    w = sum(1 for k in keys if b[k] and not a[k])
    l_ = sum(1 for k in keys if a[k] and not b[k])
    n = len(keys)
    return {"n": n, "stock": sum(a[k] for k in keys) / n if n else None, "arm": sum(b[k] for k in keys) / n if n else None,
            "delta_pts": 100 * (sum(b[k] for k in keys) - sum(a[k] for k in keys)) / n if n else None,
            "W": w, "T": n - w - l_, "L": l_}


# ── 1. retry / selection audit ───────────────────────────────────────────────

def all_attempts(rows):
    """Union of archived result.json copies and live trials: {(base_job, task): [attempt,...]}."""
    att = {}
    for f in glob.glob(str(ARCHIVE / "*" / "*__*" / "result.json")):
        job = f.split("/")[-3]
        m = HT.JOB_RE.match(job)
        if not m:
            continue
        try:
            r = json.loads(Path(f).read_text())
        except Exception:  # noqa: BLE001
            continue
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        e = r.get("exception_info") or {}
        att[(f"{m.group(1)}_{m.group(2)}_{m.group(3)}", f.split("/")[-2])] = {
            "task": (r.get("task_name") or "").split("/")[-1], "reward": rew, "exc": e.get("exception_type"),
            "started": _iso(r.get("started_at")), "live": False}
    for r in rows:
        att[(r["base_job"], r["trial"])] = {"task": r["task"], "reward": r["reward"], "exc": r["exc_type"],
                                           "started": r["started_at"], "live": True}
    live = {(r["base_job"], r["trial"]) for r in rows}
    out = defaultdict(list)
    for (bj, trial), a in att.items():
        a["live"] = (bj, trial) in live
        a["trial"] = trial
        out[(bj, a["task"])].append(a)
    for v in out.values():
        v.sort(key=lambda a: a["started"] or 0)
    return out


def retry_audit(rows):
    att = all_attempts(rows)
    sel = HT.select_per_task(rows, "collect")
    per_job = defaultdict(Counter)
    itt = defaultdict(dict)     # first attempt that produced a reward
    itt_fair = defaultdict(dict)  # first scored attempt that did not crash for infra reasons (timeouts count)
    strict = defaultdict(dict)  # first attempt, no reward -> fail
    counted = defaultdict(dict)
    for (bj, task), v in att.items():
        c = per_job[bj]
        c["tasks"] += 1
        c["attempts"] += len(v)
        c["tasks_multi_attempt"] += len(v) > 1
        removed = [a for a in v if not a["live"]]
        for a in removed:
            key = f"removed_{a['exc'] or 'clean'}_{'pass' if _passed(a['reward']) else ('fail' if a['reward'] is not None else 'noreward')}"
            c[key] += 1
        first = next((a for a in v if a["reward"] is not None), None)
        if first is not None:
            itt[bj][task] = _passed(first["reward"])
        fair = next((a for a in v if a["reward"] is not None and a["exc"] not in INFRA_EXC), None)
        if fair is not None:
            itt_fair[bj][task] = _passed(fair["reward"])
        strict[bj][task] = _passed(v[0]["reward"])
        cur = sel.get(bj, {}).get(task)
        if cur is not None and HT.is_valid(cur):
            counted[bj][task] = cur["passed"]
            if first is not None and _passed(first["reward"]) != cur["passed"]:
                c["counted_differs_from_first_scored"] += 1
                c["first_fail_counted_pass" if cur["passed"] else "first_pass_counted_fail"] += 1
    pairs = []
    for bench in ("ds11", "ds11m", "tb4"):
        for rung in "WFMSG":
            s, t = f"{bench}_{rung}_stock", f"{bench}_{rung}_tib"
            if s not in counted or t not in counted:
                continue
            row = {"bench": bench, "rung": rung, "model": LABEL[rung]}
            for name, src in (("counted", counted), ("itt_first_scored", itt), ("itt_excuse_infra", itt_fair),
                              ("strict_first_attempt", strict)):
                keys = sorted(set(src[s]) & set(src[t]))
                row[name] = wtl(src[s], src[t], keys) if keys else None
            pairs.append(row)
    return {"per_job": {k: dict(v) for k, v in sorted(per_job.items())}, "paired": pairs,
            "note": "counted = scripts/harbor_collect.py rule (prefers a passing attempt); itt_first_scored = earliest "
                    "attempt with a verifier reward (incl. scored timeouts / non-zero exits); strict_first_attempt = "
                    "earliest attempt, unscored -> fail."}


# ── 2. harness discrepancy ───────────────────────────────────────────────────

def _job_stats(rs):
    n = len(rs)
    if not n:
        return None
    return {"n": n, "pass": sum(r["passed"] for r in rs) / n,
            "claim_book_written": sum(bool(r.get("claim_book")) for r in rs) / n,
            "claim_book_with_p": sum((r.get("claim_book_n_p") or 0) > 0 for r in rs) / n,
            "claim_book_median_n_p": _med([r.get("claim_book_n_p") for r in rs if r.get("claim_book")]),
            "claim_book_frac_all_p_ge_0.95": _mean([float((r.get("claim_book_min_p") or 0) >= 0.95) for r in rs
                                                    if r.get("claim_book_n_p")]),
            "claim_book_position_median": _med([r.get("claim_book_first_frac") for r in rs]),
            "edits_after_claim_book_frac": _mean([float((r.get("edits_after_claim_book") or 0) > 0) for r in rs
                                                  if r.get("claim_book")]),
            "steps_median": _med([r.get("n_agent_steps") for r in rs]),
            "steps_p90": sorted([r.get("n_agent_steps") or 0 for r in rs])[int(0.9 * (n - 1))],
            "agent_min_median": _med([(r.get("agent_sec") or 0) / 60 for r in rs]),
            "sec_per_step_median": _med([r.get("sec_per_step") for r in rs]),
            "timeouts": sum(r["timeout"] for r in rs), "empty_patch": sum(r.get("patch_bytes") == 0 for r in rs),
            "commits_median": _med([r.get("n_commits") for r in rs]),
            "any_commit_frac": _mean([float((r.get("n_commits") or 0) > 0) for r in rs]),
            "first_commit_frac_median": _med([r.get("first_commit_frac") for r in rs]),
            "test_runs_median": _med([r.get("n_test_runs") for r in rs]),
            "output_tokens_median": _med([r.get("n_output_tokens") for r in rs]),
            "input_tokens_median": _med([r.get("n_input_tokens") for r in rs])}


def harness(rows, sel):
    g = {k: {t: r for t, r in sel.get(k, {}).items() if HT.is_valid(r)}
         for k in ("ds11_G_stock", "ds11_G_tib", "ds11m_G_stock", "ds11m_G_tib")}
    four = sorted(set.intersection(*[set(v) for v in g.values()]))
    oc = sorted(set(g["ds11_G_stock"]) & set(g["ds11_G_tib"]))
    ms = sorted(set(g["ds11m_G_stock"]) & set(g["ds11m_G_tib"]))
    P = {k: {t: r["passed"] for t, r in v.items()} for k, v in g.items()}
    res = {
        "opencode_all": wtl(P["ds11_G_stock"], P["ds11_G_tib"], oc),
        "mswea_paired": wtl(P["ds11m_G_stock"], P["ds11m_G_tib"], ms),
        "opencode_on_mswea_paired_tasks": wtl(P["ds11_G_stock"], P["ds11_G_tib"], [t for t in ms if t in P["ds11_G_tib"]]),
        "four_way_tasks": len(four),
        "stock_harness_agreement": None,
    }
    if four:
        res["four_way"] = {
            "opencode": wtl(P["ds11_G_stock"], P["ds11_G_tib"], four),
            "mswea": wtl(P["ds11m_G_stock"], P["ds11m_G_tib"], four),
            "stock_opencode_vs_stock_mswea": wtl(P["ds11_G_stock"], P["ds11m_G_stock"], four),
            "tib_opencode_vs_tib_mswea": wtl(P["ds11_G_tib"], P["ds11m_G_tib"], four),
        }
        # per-task effect agreement
        agree = Counter()
        for t in four:
            d_oc = int(P["ds11_G_tib"][t]) - int(P["ds11_G_stock"][t])
            d_ms = int(P["ds11m_G_tib"][t]) - int(P["ds11m_G_stock"][t])
            agree[f"oc{d_oc:+d}_ms{d_ms:+d}"] += 1
        res["per_task_effect_pattern"] = dict(agree)
    res["profile"] = {k: _job_stats(list(v.values())) for k, v in g.items()}
    # the ds11m wins: what did stock do on those tasks?
    wins = [t for t in ms if P["ds11m_G_tib"][t] and not P["ds11m_G_stock"][t]]
    res["mswea_wins_stock_side"] = [{
        "task": t, "stock_timeout": g["ds11m_G_stock"][t]["timeout"], "stock_patch_bytes": g["ds11m_G_stock"][t]["patch_bytes"],
        "stock_partial": g["ds11m_G_stock"][t]["partial"], "tib_partial": g["ds11m_G_tib"][t]["partial"],
        "stock_steps": g["ds11m_G_stock"][t].get("n_agent_steps"), "tib_steps": g["ds11m_G_tib"][t].get("n_agent_steps"),
        "stock_commits": g["ds11m_G_stock"][t].get("n_commits"), "tib_commits": g["ds11m_G_tib"][t].get("n_commits"),
        "tib_claim_book": g["ds11m_G_tib"][t].get("claim_book"),
        "opencode_stock": P["ds11_G_stock"].get(t), "opencode_tib": P["ds11_G_tib"].get(t)} for t in wins]
    stock_wins_counter = Counter()
    for w in res["mswea_wins_stock_side"]:
        if w["stock_timeout"]:
            stock_wins_counter["stock_timed_out"] += 1
        elif w["stock_patch_bytes"] == 0:
            stock_wins_counter["stock_empty_patch"] += 1
        elif (w["stock_partial"] or 0) >= 0.95:
            stock_wins_counter["stock_near_miss_partial>=0.95"] += 1
        else:
            stock_wins_counter["stock_wrong"] += 1
    res["mswea_wins_stock_failure_kind"] = dict(stock_wins_counter)
    # paired-subset difficulty: is the (partial) ds11m paired set easier than the full 113?
    rest = [t for t in P["ds11_G_stock"] if t not in set(ms)]
    res["paired_subset_difficulty"] = {
        "opencode_stock_pass_on_mswea_paired": _mean([float(P["ds11_G_stock"][t]) for t in ms if t in P["ds11_G_stock"]]),
        "opencode_stock_pass_on_rest": _mean([float(P["ds11_G_stock"][t]) for t in rest]),
        "n_paired": len(ms), "n_rest": len(rest)}
    res["prompt_differences"] = [
        "mini-swe-agent TIB instance template (configs/harbor/mswea_tib_draftfirst.yaml) tells the agent to branch, "
        "commit the Phase-1 draft by step 15, and `git commit --amend` after EVERY edit, stating that uncommitted "
        "work scores 0 (the verifier collects `git diff <base> HEAD`). The stock mini.yaml template never mentions git; "
        "only the task text ends with 'commit everything when you are done'.",
        "The opencode TIB prompt (configs/harbor/tib_draftfirst.j2) has no commit instruction at all.",
        "Both harnesses: no step limit (mini-swe-agent step_limit=0, cost_limit=0); the binding budget is wall-clock "
        "(agent timeout 5400 s x agent_timeout_multiplier: 1.0 for ds11/opencode, 2.5 for ds11m).",
    ]
    return res


# ── 3. losses ────────────────────────────────────────────────────────────────

def _bridge():
    p = OUT / "bridge_harbor_trials.json"
    if not p.exists():
        return {}
    return {(r["job"], r["trial"]): r for r in json.loads(p.read_text())}


def classify_loss(t, s, br_t, job_lat_med):
    """t = TIB trial row, s = stock trial row (stock passed, TIB failed). Priority-ordered single label."""
    lat = (br_t or {}).get("latency_p50")
    if t["timeout"]:
        if lat is not None and job_lat_med and (lat >= 3 * job_lat_med or (br_t or {}).get("retries", 0) >= 20):
            return "timeout_api_starved"
        return "timeout_budget_exhausted"
    if t.get("patch_bytes") == 0 and t["bench"] != "tb4":
        return "no_committed_patch"
    if (br_t or {}).get("blocked", 0) + (br_t or {}).get("empty", 0) >= 5:
        return "empty_or_safety_blocked_responses"
    if not t.get("claim_book"):
        return "claim_book_skipped"
    if (t.get("edits_after_claim_book") or 0) > 0:
        return "post_audit_edits_still_wrong"
    if t.get("claim_book_n_p") and (t.get("claim_book_min_p") or 0) >= 0.95:
        return "overconfident_audit_all_p_ge_0.95"
    if (t.get("n_agent_steps") or 0) < 0.5 * (s.get("n_agent_steps") or 0):
        return "early_stop_fewer_than_half_stock_steps"
    return "other_wrong_answer"


def _quote(t, cat):
    if cat.startswith("timeout") or cat == "no_committed_patch":
        q = t.get("final_msg_tail") or t.get("exc_msg") or ""
    elif cat in ("post_audit_edits_still_wrong", "overconfident_audit_all_p_ge_0.95", "other_wrong_answer",
                 "early_stop_fewer_than_half_stock_steps"):
        q = t.get("claim_book_excerpt") or t.get("final_msg_tail") or ""
    else:
        q = t.get("final_msg_tail") or ""
    return _clip(q, 300)


def loss_taxonomy(sel, bridge, bench, rung):
    s_job, t_job = f"{bench}_{rung}_stock", f"{bench}_{rung}_tib"
    S = {k: r for k, r in sel.get(s_job, {}).items() if HT.is_valid(r)}
    T = {k: r for k, r in sel.get(t_job, {}).items() if HT.is_valid(r)}
    common = sorted(set(S) & set(T))
    lat_ok = [bridge.get((r["job"], r["trial"]), {}).get("latency_p50") for r in T.values() if not r["timeout"]]
    job_lat_med = _med(lat_ok)
    losses = [k for k in common if S[k]["passed"] and not T[k]["passed"]]
    wins = [k for k in common if T[k]["passed"] and not S[k]["passed"]]
    cats = defaultdict(list)
    for k in losses:
        t = T[k]
        c = classify_loss(t, S[k], bridge.get((t["job"], t["trial"])), job_lat_med)
        br = bridge.get((t["job"], t["trial"])) or {}
        cats[c].append({"task": k, "tib_steps": t.get("n_agent_steps"), "stock_steps": S[k].get("n_agent_steps"),
                        "tib_min": round((t.get("agent_sec") or 0) / 60, 1), "stock_min": round((S[k].get("agent_sec") or 0) / 60, 1),
                        "tib_partial": t.get("partial"), "tib_latency_p50": br.get("latency_p50"),
                        "tib_retries": br.get("retries"), "claim_book": t.get("claim_book"),
                        "min_p": t.get("claim_book_min_p"), "quote": _quote(t, c)})
    # tokens / steps of losing tasks vs stock
    def agg(rs):
        return {"steps_median": _med([r.get("n_agent_steps") for r in rs]),
                "agent_min_median": _med([(r.get("agent_sec") or 0) / 60 for r in rs]),
                "output_tokens_median": _med([r.get("n_output_tokens") for r in rs]),
                "input_tokens_median": _med([r.get("n_input_tokens") for r in rs]),
                "timeouts": sum(r["timeout"] for r in rs)}
    # overall TIB failure modes (not just losses): all TIB failures on common tasks
    allfail = Counter()
    for k in common:
        if not T[k]["passed"]:
            allfail[classify_loss(T[k], S[k], bridge.get((T[k]["job"], T[k]["trial"])), job_lat_med)] += 1
    return {"bench": bench, "rung": rung, "model": LABEL[rung], "n_common": len(common),
            "stock_pass": sum(S[k]["passed"] for k in common), "tib_pass": sum(T[k]["passed"] for k in common),
            "W": len(wins), "L": len(losses), "tib_job_latency_p50_non_timeout": job_lat_med,
            "loss_categories": {c: len(v) for c, v in sorted(cats.items(), key=lambda kv: -len(kv[1]))},
            "examples": {c: v[:3] for c, v in cats.items()}, "all_loss_rows": {c: [x["task"] for x in v] for c, v in cats.items()},
            "losing_tasks_tib": agg([T[k] for k in losses]), "losing_tasks_stock": agg([S[k] for k in losses]),
            "all_tib_failures_by_mode": dict(allfail),
            "counterfactual_excluding_api_starved_timeouts": _cf(S, T, common, bridge, job_lat_med)}


def _cf(S, T, common, bridge, job_lat_med):
    keep = [k for k in common if classify_loss(T[k], S[k], bridge.get((T[k]["job"], T[k]["trial"])), job_lat_med)
            != "timeout_api_starved" or T[k]["passed"]]
    A = {k: S[k]["passed"] for k in keep}
    B = {k: T[k]["passed"] for k in keep}
    return wtl(A, B, keep)


# ALE-Claw t300 -----------------------------------------------------------------

def _ale_runs(arm, rung, seed):
    best = {}
    for p in glob.glob(f"{ALE_ROOT}/tib_docker90_{arm}_{rung}_{seed}/ale_claw/*/*/v0/*/run.json"):
        task = p.split("/ale_claw/")[1].split("/")[1]
        ts = p.split("/")[-2]
        if task in best and best[task][0] > ts:
            continue
        best[task] = (ts, p)
    out = {}
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
            continue
        u = r.get("usage") or {}
        tj = Path(p).with_name("trajectory.json")
        claim, cb_txt, final = False, "", ""
        n_agent = 0
        if tj.exists():
            try:
                t = json.loads(tj.read_text())
                ag = [s for s in t.get("steps", []) if s.get("source") == "agent"]
                n_agent = len(ag)
                for s in ag:
                    txt = (s.get("message") or "") + "\n" + (s.get("reasoning") or "")
                    m = HT.CLAIM_BLOCK_RE.search(txt)
                    if m and len(HT.CLAIM_ID_RE.findall(txt[m.start():m.start() + 6000])) >= 2:
                        claim, cb_txt = True, txt[m.start():]
                final = (ag[-1].get("message") or "") if ag else ""
            except Exception:  # noqa: BLE001
                pass
        pv = [float(v) for v in HT.P_RE.findall(cb_txt[:20000])] if cb_txt else []
        out[task] = {"score": sc or 0.0, "steps": u.get("total_steps") or n_agent, "output": u.get("total_output_tokens"),
                     "input": u.get("total_input_tokens"), "sec": (r.get("timings") or {}).get("duration_s"),
                     "termination": (r.get("termination") or {}).get("reason"), "status": r.get("status"),
                     "claim_book": claim, "min_p": min(pv) if pv else None, "cb": cb_txt[:1200], "final": final[-400:]}
    return out


def ale_losses():
    res = []
    for rung in ("G", "S", "F", "W"):
        S, T = _ale_runs("stock", rung, "t300"), _ale_runs("cb_draftfirst", rung, "t300")
        common = sorted(set(S) & set(T))
        if not common:
            continue
        losses = [k for k in common if T[k]["score"] < S[k]["score"] - 1e-9]
        wins = [k for k in common if T[k]["score"] > S[k]["score"] + 1e-9]
        cats = defaultdict(list)
        for k in losses:
            t, s = T[k], S[k]
            if t["termination"] not in (None, "completed"):
                c = f"terminated_{t['termination']}"
            elif (t["steps"] or 0) >= 295:
                c = "turn_budget_exhausted"
            elif not t["claim_book"]:
                c = "claim_book_skipped"
            elif t["min_p"] is not None and t["min_p"] >= 0.95:
                c = "overconfident_audit_all_p_ge_0.95"
            elif (t["steps"] or 0) < 0.5 * (s["steps"] or 0):
                c = "early_stop_fewer_than_half_stock_steps"
            else:
                c = "other_lower_score"
            cats[c].append({"task": k, "stock_score": s["score"], "tib_score": t["score"], "tib_steps": t["steps"],
                            "stock_steps": s["steps"], "quote": _clip(t["cb"] or t["final"], 300)})
        delta = [T[k]["score"] - S[k]["score"] for k in common]
        res.append({"rung": rung, "model": LABEL[rung], "n_common": len(common), "W": len(wins), "L": len(losses),
                    "mean_delta_score": sum(delta) / len(delta),
                    "claim_book_rate_tib": _mean([float(T[k]["claim_book"]) for k in common]),
                    "steps_median_stock": _med([S[k]["steps"] for k in common]),
                    "steps_median_tib": _med([T[k]["steps"] for k in common]),
                    "turn_budget_hits_stock": sum((S[k]["steps"] or 0) >= 295 for k in common),
                    "turn_budget_hits_tib": sum((T[k]["steps"] or 0) >= 295 for k in common),
                    "terminations_stock": dict(Counter(S[k]["termination"] for k in common)),
                    "terminations_tib": dict(Counter(T[k]["termination"] for k in common)),
                    "abs_delta_mean": _mean([abs(d) for d in delta]),
                    "loss_categories": {c: len(v) for c, v in sorted(cats.items(), key=lambda kv: -len(kv[1]))},
                    "examples": {c: v[:3] for c, v in cats.items()}})
    return res


# ── 4. timeouts / exceptions ─────────────────────────────────────────────────

def timeouts(rows, sel):
    per = {}
    by = defaultdict(list)
    for r in rows:
        by[r["base_job"]].append(r)
    for bj, rs in sorted(by.items()):
        ex = Counter(r["exc_type"] or "none" for r in rs)
        to = [r for r in rs if r["timeout"]]
        per[bj] = {"trials_live": len(rs), "exceptions_live": dict(ex),
                   "timeout_rate_live": len(to) / len(rs),
                   "timeouts_with_nonempty_patch": sum((r.get("patch_bytes") or 0) > 0 for r in to),
                   "timeouts_passed": sum(r["passed"] for r in to),
                   "timeouts_partial_mean": _mean([r.get("partial") for r in to]),
                   "timeout_commits_median": _med([r.get("n_commits") for r in to])}
    # the counted (selected) trials
    for bj, tasks in sel.items():
        v = [r for r in tasks.values() if HT.is_valid(r)]
        if bj in per and v:
            per[bj]["counted_timeouts"] = sum(r["timeout"] for r in v)
            per[bj]["counted_n"] = len(v)
    return per


# ── 4b. are the agentic bets informative? ───────────────────────────────────

def _auroc(pos, neg):
    if not pos or not neg:
        return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def bet_calibration(rows):
    """Final claim book p-values vs the verifier outcome, per TIB job (all live trials with parsed p values)."""
    out = {}
    by = defaultdict(list)
    for r in rows:
        if r["arm"] == "tib" and r.get("claim_book_n_p") and r["scored"]:
            by[r["base_job"]].append(r)
    for bj, rs in sorted(by.items()):
        meanp = [sum(r["claim_book_p"]) / len(r["claim_book_p"]) for r in rs if r.get("claim_book_p")]
        minp = [r["claim_book_min_p"] for r in rs]
        out[bj] = {"n": len(rs), "pass_rate": _mean([float(r["passed"]) for r in rs]),
                   "mean_of_mean_p": _mean(meanp), "frac_min_p_ge_0.95": _mean([float(p >= 0.95) for p in minp]),
                   "frac_any_p_lt_0.9": _mean([float(p < 0.9) for p in minp]),
                   "overconfidence_gap_pts": 100 * ((_mean(meanp) or 0) - (_mean([float(r["passed"]) for r in rs]) or 0)),
                   "auroc_min_p_predicts_pass": _auroc([r["claim_book_min_p"] for r in rs if r["passed"]],
                                                        [r["claim_book_min_p"] for r in rs if not r["passed"]]),
                   "median_claims": _med([r["claim_book_n_p"] for r in rs])}
    return out


# ── 5. other biases ──────────────────────────────────────────────────────────

def other(rows, sel):
    res = {}
    by = defaultdict(list)
    for r in rows:
        by[r["base_job"]].append(r)
    res["run_windows"] = {bj: {"first_start": _day(min(r["started_at"] for r in rs)),
                               "last_start": _day(max(r["started_at"] for r in rs)), "n": len(rs),
                               "sec_per_step_median": _med([r.get("sec_per_step") for r in rs])}
                          for bj, rs in sorted(by.items())}
    res["opencode_stream"] = {bj: {"zero_token_steps": sum(r.get("oc_zero_token_steps") or 0 for r in rs),
                                   "length_finishes": sum(r.get("oc_length_finishes") or 0 for r in rs),
                                   "steps": sum(r.get("n_agent_steps") or 0 for r in rs),
                                   "trials_with_zero_token_step": sum((r.get("oc_zero_token_steps") or 0) > 0 for r in rs)}
                              for bj, rs in sorted(by.items()) if rs[0]["harness"] == "opencode"}
    res["mswea_exit_status"] = {bj: dict(Counter(r.get("mswea_exit_status") or "none" for r in rs))
                                for bj, rs in sorted(by.items()) if rs[0]["harness"] == "mini-swe-agent"}
    res["empty_agent_messages"] = {bj: sum(r.get("empty_agent_msgs") or 0 for r in rs) for bj, rs in sorted(by.items())}
    # bridge summary (if present)
    bp = OUT / "bridge_logs.json"
    if bp.exists():
        b = json.loads(bp.read_text())
        bj = (b.get("harbor") or {}).get("by_job") or {}
        keep = ("trials", "mean_requests_per_trial", "mean_thoughts_per_trial", "failure_rate", "retries_per_request",
                "max_tokens_rate", "empty_rate", "blocked_rate", "mean_latency", "latency_mean", "mean_latency_per_request")
        res["bridge_by_job"] = {k: {kk: vv for kk, vv in v.items() if kk in keep or "latency" in kk or "timeout" in kk}
                                for k, v in bj.items()}
        res["bridge_overall"] = b.get("overall")
        res["bridge_bad_windows_with_harbor"] = [w for w in (b.get("bad_windows_all_models") or [])
                                                 if w.get("overlapping_harbor_jobs")]
    # SWE-Interact provider errors and degraded trials
    pe = defaultdict(Counter)
    seen = set()
    for fn in ("swi_mixed_r1_trials.json", "swi_S_mixed_r1r2_trials.json", "swi_g4_overlay_r1r2_trials.json",
               "swi_g4_overlay_r1_trials.json"):
        p = OUT / fn
        if not p.exists():
            continue
        for t in json.loads(p.read_text()):
            if (t["run"], t["trial"]) in seen:
                continue
            seen.add((t["run"], t["trial"]))
            k = f"{t['rung']}|{t['arm']}|{t['run'].split('/')[-1]}"
            pe[k]["trials"] += 1
            for kk, vv in (t.get("provider_errors") or {}).items():
                pe[k][kk] += vv or 0
            pe[k]["trials_with_any_provider_error"] += any((t.get("provider_errors") or {}).values())
    res["swi_provider_errors"] = {k: dict(v) for k, v in sorted(pe.items())}
    dq = RES / "redo_v1_g4" / "stage2_swe_interact" / "_degraded_quota"
    res["swi_g4_degraded_quota_trials"] = {p.name: len(list((p / "overlay_r1" / "jobs").glob("*"))) for p in dq.iterdir()
                                           if p.is_dir()} if dq.exists() else {}
    res["unequal_trial_counts"] = {bj: len([r for r in v.values() if HT.is_valid(r)]) for bj, v in sorted(sel.items())}
    return res


# ── report helpers ───────────────────────────────────────────────────────────

def plot(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    FIG.mkdir(parents=True, exist_ok=True)
    files = []
    # loss taxonomy stacked bars
    L = [x for x in res["losses"]]
    cats = sorted({c for x in L for c in x["loss_categories"]})
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bottom = [0] * len(L)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cats):
        vals = [x["loss_categories"].get(c, 0) for x in L]
        ax.bar(range(len(L)), vals, bottom=bottom, label=c, color=cmap(i % 10))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(L)))
    ax.set_xticklabels([f"{x['bench']} {x['model']}\n(W{x['W']}/L{x['L']})" for x in L], fontsize=7)
    ax.set_ylabel("# stock-pass / TIB-fail tasks")
    ax.set_title("Why TIB lost: failure modes of losing tasks")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fp = FIG / "loss_taxonomy.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    files.append(str(fp.relative_to(ROOT)))
    # counted vs ITT deltas
    P = [p for p in res["retry_audit"]["paired"] if p.get("itt_first_scored")]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    xs = range(len(P))
    ax.bar([x - 0.27 for x in xs], [p["counted"]["delta_pts"] for p in P], width=0.27, label="counted (best-of-N rule)")
    ax.bar([x for x in xs], [(p["itt_excuse_infra"] or {}).get("delta_pts") or 0 for p in P], width=0.27,
           label="ITT, infra crashes excused")
    ax.bar([x + 0.27 for x in xs], [p["itt_first_scored"]["delta_pts"] for p in P], width=0.27, label="ITT first scored attempt")
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{p['bench']} {p['model']}" for p in P], fontsize=7, rotation=20)
    ax.set_ylabel("TIB − stock (pts)")
    ax.legend(fontsize=7)
    ax.set_title("Harbor: headline Δ vs intent-to-treat Δ")
    fig.tight_layout()
    fp = FIG / "harbor_itt_vs_counted.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    files.append(str(fp.relative_to(ROOT)))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-scan ~/jobs")
    a = ap.parse_args()
    rows = HT.load_trials(refresh=a.refresh)
    sel = HT.select_per_task(rows, "collect")
    bridge = _bridge()
    res = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_trials_live": len(rows), "bridge_attribution_available": bool(bridge)}
    res["retry_audit"] = retry_audit(rows)
    res["harness"] = harness(rows, sel)
    res["losses"] = [loss_taxonomy(sel, bridge, b, r) for b, r in (("ds11", "S"), ("tb4", "G"), ("ds11", "G"),
                                                                   ("ds11m", "G"), ("tb4", "S"))]
    res["ale_t300"] = ale_losses()
    res["timeouts"] = timeouts(rows, sel)
    res["bet_calibration"] = bet_calibration(rows)
    res["other"] = other(rows, sel)
    res["figures"] = plot(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "log_mining.json").write_text(json.dumps(res, indent=1, default=str))
    print("-> results/analysis/log_mining.json")
    for p in res["retry_audit"]["paired"]:
        c, i, f = p["counted"], p["itt_first_scored"], p["itt_excuse_infra"] or {}
        print(f"  {p['bench']:5s} {p['model']:14s} counted {c['n']:3d} {100*c['stock']:5.1f}->{100*c['arm']:5.1f} "
              f"({c['delta_pts']:+5.1f}, {c['W']}W/{c['L']}L) | ITT {i['n']:3d} {100*i['stock']:5.1f}->{100*i['arm']:5.1f} "
              f"({i['delta_pts']:+5.1f}, {i['W']}W/{i['L']}L) | ITT-infra-excused {f.get('n')} "
              f"{100*(f.get('stock') or 0):5.1f}->{100*(f.get('arm') or 0):5.1f} ({(f.get('delta_pts') or 0):+5.1f}, "
              f"{f.get('W')}W/{f.get('L')}L)")
    for x in res["losses"]:
        print(f"  losses {x['bench']} {x['model']}: W{x['W']}/L{x['L']} {x['loss_categories']}  cf-excl-starved: "
              f"{x['counterfactual_excluding_api_starved_timeouts']['delta_pts']:+.1f}")
    for x in res["ale_t300"]:
        print(f"  ALE t300 {x['model']}: W{x['W']}/L{x['L']} {x['loss_categories']}")


if __name__ == "__main__":
    main()
