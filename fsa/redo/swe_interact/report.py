"""Per-trial / per-step report for SWE-Interact trials (Stage 0 sizing) and the turn index.

    python -m fsa.redo.swe_interact.report <run_dir> [...] [--out report.json]

Sources in a Harbor trial dir (``<run_dir>/jobs/<job>/<trial>/``):

* ``result.json``: per step ``agent_execution`` / ``verifier`` timing, reward, exception.
* ``steps/<step>/agent/opencode.attempt-*.txt`` (RedoOpenCode): every opencode attempt,
  including discarded throttled ones. One ``step_finish`` event = one agent model call; for
  the native Gemini provider ``tokens.input`` is the uncached prompt and ``tokens.cache.read``
  the implicit cache hits, so prompt = input + cache.read, output = output + reasoning.
  Provider errors are counted by kind (429 / 503 / mid-stream "model turn" 400 / other).
* ``steps/<step>/agent/redo/step-K.{json}``, ``step-K-end.{patch,json}``: attempts,
  throttle retries, HEAD before/after, end-of-step snapshot.
* ``user-sim/`` (sidecar, local patch): ``asks.jsonl`` (one line per ``ask_user`` call),
  ``sim_calls.jsonl`` (one line per simulator LLM attempt with usage), ``snapshots/ask-N.*``.
* ``steps/05_test_handoff/verifier/``: ``reward.json``; rf_*: ``rubrics_results.json``
  (``llm_calls`` with grader usage, local patch).

Turn index (``turns()``): agent turn *t* is opened by a step instruction (``opened_by="step"``)
or by the simulator's reply to an ``ask_user`` call (``opened_by="sim"``), and ends with the
next ``ask_user`` call (snapshot ``user-sim/snapshots/ask-N.patch``, taken before the simulator
is consulted) or with the end of its step (snapshot ``steps/<step>/agent/redo/step-K-end.patch``).
An ``ask_user`` call is attributed to the step whose agent phase contains its timestamp.

Simulator interventions: every ``ask_user`` reply. ``revision_request`` is a heuristic label
(reply is not an approval by keyword match), reported for sizing only.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

APPROVAL = re.compile(r"\b(lgtm|looks good|ship it|approved?|go ahead|sounds good|good to go|that works|"
                      r"perfect|all good|fine by me|yep|yeah,? go)\b", re.I)


def _ts(s: str | None) -> float | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def classify_error(text: str) -> str:
    t = text.lower()
    if "requests ending with a model turn" in t:
        return "midstream_503_as_400"
    if "429" in t or "resource_exhausted" in t or "resource has been exhausted" in t or "rate limit" in t:
        return "429"
    if "503" in t or "unavailable" in t or "high demand" in t or "overloaded" in t:
        return "503"
    return "other"


def agent_calls(agent_dir: Path) -> tuple[list[dict], dict]:
    calls, errs = [], {}
    files = sorted(glob.glob(str(agent_dir / "opencode.attempt-*.txt"))) or \
        ([str(agent_dir / "opencode.txt")] if (agent_dir / "opencode.txt").exists() else [])
    for f in files:
        for line in open(f, errors="replace"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "step_finish":
                t = (e.get("part") or {}).get("tokens") or {}
                c = t.get("cache") or {}
                calls.append({"ts": (e.get("timestamp") or 0) / 1000.0, "input_uncached": t.get("input") or 0,
                              "cache_read": c.get("read") or 0,
                              "prompt_total": (t.get("input") or 0) + (c.get("read") or 0),
                              "output": t.get("output") or 0, "reasoning": t.get("reasoning") or 0})
            elif e.get("type") == "error":
                k = classify_error(json.dumps(e.get("error")))
                errs[k] = errs.get(k, 0) + 1
    return sorted(calls, key=lambda c: c["ts"]), errs


def peak_60s(calls: list[dict], key: str = "prompt_total") -> int:
    best, j, run = 0, 0, 0
    for c in calls:
        run += c[key]
        while c["ts"] - calls[j]["ts"] > 60:
            run -= calls[j][key]
            j += 1
        best = max(best, run)
    return best


def _sum(xs, k):
    return sum((x.get(k) or 0) for x in xs)


def step_rows(trial: Path) -> list[dict]:
    res = json.loads((trial / "result.json").read_text()) if (trial / "result.json").exists() else {}
    asks = _jsonl(trial / "user-sim" / "asks.jsonl")
    sims = _jsonl(trial / "user-sim" / "sim_calls.jsonl")
    rows = []
    for k, s in enumerate(res.get("step_results") or [], start=1):
        name = s["step_name"]
        ad = trial / "steps" / name / "agent"
        ae = s.get("agent_execution") or {}
        t0, t1 = _ts(ae.get("started_at")), _ts(ae.get("finished_at"))
        vr = s.get("verifier") or {}
        step_asks = [a for a in asks if t0 and t1 and t0 <= a["t_start"] <= t1 + 1]
        idx = {a["call_index"] for a in step_asks}
        step_sims = [c for c in sims if c.get("call_index") in idx]
        calls, errs = agent_calls(ad)
        rec = json.loads((ad / "redo" / f"step-{k}.json").read_text()) if (ad / "redo" / f"step-{k}.json").exists() else {}
        head_before, head_after = rec.get("head_before"), rec.get("head_after")
        n_commits = None
        endj = ad / "redo" / f"step-{k}-end.json"
        if endj.exists():
            n_commits = len([l for l in (json.loads(endj.read_text()).get("commits_since_anchor") or "").splitlines() if l])
        rows.append({
            "step": name,
            "reward": ((s.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
            "exception": ((s.get("exception_info") or {}).get("exception_message") or "")[:200] or None,
            "agent_wall_sec": round(t1 - t0, 1) if t0 and t1 else None,
            "verifier_wall_sec": round(_ts(vr.get("finished_at")) - _ts(vr.get("started_at")), 1)
            if vr.get("started_at") and vr.get("finished_at") else None,
            "ask_user_calls": len(step_asks),
            "revision_requests_heuristic": sum(1 for a in step_asks if not APPROVAL.search(a.get("reply") or "")),
            "last_resort_replies": sum(1 for a in step_asks if a.get("last_resort")),
            "agent_turns": len(step_asks) + 1,
            "agent_llm_calls": len(calls),
            "agent_prompt_total": _sum(calls, "prompt_total"), "agent_cache_read": _sum(calls, "cache_read"),
            "agent_output": _sum(calls, "output") + _sum(calls, "reasoning"),
            "agent_cached_share": round(_sum(calls, "cache_read") / max(_sum(calls, "prompt_total"), 1), 3),
            "agent_peak_prompt_60s": peak_60s(calls) if calls else 0,
            "agent_peak_uncached_60s": peak_60s(calls, "input_uncached") if calls else 0,
            "agent_provider_errors": errs,
            "agent_attempts": len(rec.get("attempts") or []), "agent_throttle_retries": rec.get("throttle_retries"),
            "sim_llm_attempts": len(step_sims), "sim_failed_attempts": sum(1 for c in step_sims if not c.get("ok")),
            "sim_prompt": _sum(step_sims, "prompt_tokens"), "sim_cached": _sum(step_sims, "cached_tokens"),
            "sim_completion": _sum(step_sims, "completion_tokens"), "sim_reasoning": _sum(step_sims, "reasoning_tokens"),
            "sim_latency_sec_total": round(_sum(step_sims, "latency_sec"), 1),
            "commits_since_anchor_at_step_end": n_commits,
            "head_changed": bool(head_before and head_after and head_before != head_after),
        })
    return rows


def rubric(trial: Path) -> dict | None:
    p = trial / "steps" / "05_test_handoff" / "verifier" / "rubrics_results.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    lc = d.get("llm_calls") or []
    ok = [c for c in lc if c.get("ok")]
    return {"must_have_pass": d.get("must_have_pass"), "agg_score": d.get("agg_score"),
            "rubric_count": d.get("rubric_count"), "evaluable_count": d.get("evaluable_count"),
            "llm_attempts": len(lc), "llm_failed_attempts": len(lc) - len(ok),
            "prompt_tokens": _sum(ok, "prompt_tokens"), "completion_tokens": _sum(ok, "completion_tokens"),
            "reasoning_tokens": _sum(ok, "reasoning_tokens"), "cached_tokens": _sum(ok, "cached_tokens"),
            "finish_reasons": sorted({c.get("finish_reason") for c in ok if c.get("finish_reason")}),
            "settings": d.get("llm_settings"), "error": d.get("error")}


def turns(trial: Path) -> list[dict]:
    """Turn index (module docstring). Each row names the snapshot taken when the turn ended."""
    res = json.loads((trial / "result.json").read_text()) if (trial / "result.json").exists() else {}
    asks = _jsonl(trial / "user-sim" / "asks.jsonl")
    out, t = [], 0
    for k, s in enumerate(res.get("step_results") or [], start=1):
        ae = s.get("agent_execution") or {}
        t0, t1 = _ts(ae.get("started_at")), _ts(ae.get("finished_at"))
        opened_by, msg = "step", s["step_name"]
        for a in [a for a in asks if t0 and t1 and t0 <= a["t_start"] <= t1 + 1]:
            out.append({"turn": t, "step": s["step_name"], "opened_by": opened_by, "opening_message": msg,
                        "ended_by": "ask_user", "question": a.get("question"),
                        "snapshot": f"user-sim/snapshots/ask-{a['call_index']}.patch",
                        "snapshot_head": a.get("snapshot_head")})
            t += 1
            opened_by, msg = "sim", a.get("reply")
        out.append({"turn": t, "step": s["step_name"], "opened_by": opened_by, "opening_message": msg,
                    "ended_by": "step_end", "snapshot": f"steps/{s['step_name']}/agent/redo/step-{k}-end.patch"})
        t += 1
    return out


def trial_report(trial: Path) -> dict:
    res = json.loads((trial / "result.json").read_text()) if (trial / "result.json").exists() else {}
    steps = step_rows(trial)
    fin = trial / "steps" / "05_test_handoff" / "verifier" / "reward.json"
    agent_calls_all = []
    for s in steps:
        agent_calls_all += agent_calls(trial / "steps" / s["step"] / "agent")[0]
    agent_calls_all.sort(key=lambda c: c["ts"])
    wall = (_ts(res.get("finished_at")) - _ts(res.get("started_at"))) if res.get("finished_at") else None
    active = max(agent_calls_all[-1]["ts"] - agent_calls_all[0]["ts"], 1) if agent_calls_all else None
    sims = _jsonl(trial / "user-sim" / "sim_calls.jsonl")
    tot = lambda k: sum(s[k] or 0 for s in steps)  # noqa: E731
    return {
        "trial": trial.name, "task": res.get("task_name"),
        "reward": ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
        # reward.txt of the final step: the benchmark's score even when Harbor 0.23.0 without the
        # redo patch rejected reward.json ({"reward": 0.0, "error": ...}) and raised instead
        "final_reward_txt": (lambda p: float(p.read_text().strip()) if p.exists() and p.read_text().strip() else None)(
            trial / "steps" / "05_test_handoff" / "verifier" / "reward.txt"),
        "final_verifier": json.loads(fin.read_text()) if fin.exists() else None,
        "exception": (res.get("exception_info") or {}).get("exception_type"),
        "trial_wall_sec": round(wall, 1) if wall else None,
        "steps": steps,
        "rubric": rubric(trial),
        "totals": {
            "ask_user_calls": tot("ask_user_calls"), "agent_turns": tot("agent_turns"),
            "agent_llm_calls": tot("agent_llm_calls"), "agent_prompt_total": tot("agent_prompt_total"),
            "agent_cache_read": tot("agent_cache_read"), "agent_output": tot("agent_output"),
            "agent_cached_share": round(tot("agent_cache_read") / max(tot("agent_prompt_total"), 1), 3),
            "agent_prompt_tokens_per_min": int(tot("agent_prompt_total") / (active / 60)) if active else None,
            "agent_uncached_tokens_per_min": int((tot("agent_prompt_total") - tot("agent_cache_read")) / (active / 60))
            if active else None,
            "agent_peak_prompt_60s": peak_60s(agent_calls_all) if agent_calls_all else 0,
            "agent_peak_uncached_60s": peak_60s(agent_calls_all, "input_uncached") if agent_calls_all else 0,
            "agent_throttle_retries": tot("agent_throttle_retries"),
            "sim_attempts": len(sims), "sim_failed_attempts": sum(1 for c in sims if not c.get("ok")),
            "sim_prompt": _sum(sims, "prompt_tokens"), "sim_cached": _sum(sims, "cached_tokens"),
            "sim_completion": _sum(sims, "completion_tokens"), "sim_reasoning": _sum(sims, "reasoning_tokens"),
            "sim_latency_median": statistics.median([c["latency_sec"] for c in sims if c.get("ok")])
            if any(c.get("ok") for c in sims) else None,
        },
        "provider_errors": {k: sum(s["agent_provider_errors"].get(k, 0) for s in steps)
                            for k in ("429", "503", "midstream_503_as_400", "other")},
        "n_turns": len(turns(trial)),
    }


def find_trials(run_dir: Path) -> list[Path]:
    return sorted(p.parent for p in run_dir.glob("jobs/*/*/result.json"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = []
    for rd in a.run_dirs:
        for t in find_trials(Path(rd)):
            rows.append({"run": Path(rd).name, "job": t.parent.name, **trial_report(t)})
    text = json.dumps(rows, indent=2)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
