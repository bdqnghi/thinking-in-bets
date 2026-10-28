"""Token / call / wall-time accounting for SWE-Together trials (Stage 0 sizing).

    python -m fsa.redo.swe_together.tokens results/redo_v1/stage0/W_bringup [...more run dirs]

Per trial, per seat:

* agent (opencode, inside the sandbox): every ``step_finish`` event in the
  per-turn archives ``agent/opencode.txt.turn-*`` (throttled/discarded attempts
  included) is one model call. For the native Gemini provider opencode reports
  ``tokens.input`` = uncached prompt tokens and ``tokens.cache.read`` = implicit
  cache hits (verified: input + cache.read == Gemini promptTokenCount), so
  prompt_total = input + cache.read; output_total = output + reasoning.
* user simulator (host LiteLLM): ``agent/user_sim_calls.jsonl`` (local patch):
  prompt_tokens (includes cached), cache_tokens, completion_tokens, latency, attempts.
* claim book (Stage 1 claim-book arms, host LiteLLM on the agent's model):
  ``agent/claim_calls.jsonl``, one line per attempt (failed attempts included).
* in-trial intent judge (overlay trials): ``agent/overlay_judge/turn-*.json``
  ``judge_stats_delta``.
* judge (codex exec): the "tokens used" line the CLI prints (judge_verdict.json
  ``judge_stdout_tail``, local patch).

Rates: calls/min and prompt tokens/min over the agent's wall time (first to last
step), plus the peak prompt-token volume in any 60 s window, which is what a
per-minute input-token quota sees.
"""
from __future__ import annotations

import glob
import json
import re
import statistics
import sys
from pathlib import Path


def agent_calls(trial: Path) -> list[dict]:
    calls = []
    files = sorted(glob.glob(str(trial / "agent" / "opencode.txt.turn-*")))
    for f in files:
        name = Path(f).name.removeprefix("opencode.txt.turn-")
        for line in open(f, errors="replace"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "step_finish":
                continue
            t = (e.get("part") or {}).get("tokens") or {}
            c = t.get("cache") or {}
            calls.append({
                "file": name, "ts": (e.get("timestamp") or 0) / 1000.0,
                "input_uncached": t.get("input") or 0, "cache_read": c.get("read") or 0,
                "cache_write": c.get("write") or 0,
                "prompt_total": (t.get("input") or 0) + (c.get("read") or 0),
                "output": t.get("output") or 0, "reasoning": t.get("reasoning") or 0,
                "cost_reported": (e.get("part") or {}).get("cost"),
            })
    # the same stdout can be archived twice (turn-N and a recovered copy); de-dup on (ts, tokens)
    seen, out = set(), []
    for c in calls:
        k = (c["ts"], c["prompt_total"], c["output"], c["reasoning"])
        if k not in seen:
            seen.add(k)
            out.append(c)
    return sorted(out, key=lambda c: c["ts"])


def agent_errors(trial: Path) -> dict:
    n, kinds = 0, {}
    for f in glob.glob(str(trial / "agent" / "opencode.txt.turn-*")):
        for line in open(f, errors="replace"):
            if '"type": "error"' in line or '"type":"error"' in line:
                n += 1
                m = re.search(r'"message":\s*"([^"]{0,80})', line)
                k = m.group(1) if m else "?"
                kinds[k] = kinds.get(k, 0) + 1
    return {"n": n, "kinds": kinds}


def peak_60s(calls: list[dict], key: str = "prompt_total") -> int:
    best, j, run = 0, 0, 0
    for i, c in enumerate(calls):
        run += c[key]
        while c["ts"] - calls[j]["ts"] > 60:
            run -= calls[j][key]
            j += 1
        best = max(best, run)
    return best


def sim_calls(trial: Path) -> list[dict]:
    p = trial / "agent" / "user_sim_calls.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def claim_calls(trial: Path) -> list[dict]:
    p = trial / "agent" / "claim_calls.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def port_calls(trial: Path) -> list[dict]:
    """tracker / mediator port calls (``agent/port_calls.jsonl``, every attempt)."""
    p = trial / "agent" / "port_calls.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def overlay_judge_stats(trial: Path) -> dict:
    tot = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    for f in sorted(glob.glob(str(trial / "agent" / "overlay_judge" / "turn-*.json"))):
        d = (json.load(open(f)).get("judge_stats_delta") or {})
        for k in tot:
            tot[k] += d.get(k) or 0
    return tot


def judge_tokens(trial: Path) -> int | None:
    v = trial / "judge_verdict.json"
    if not v.exists():
        return None
    d = json.loads(v.read_text())
    tail = (d.get("judge_stdout_tail") or "") + "\n" + (d.get("judge_stderr_tail") or "")
    m = re.search(r"tokens used\s*\n?\s*([\d,]+)", tail)
    return int(m.group(1).replace(",", "")) if m else None


def trial_report(trial: Path) -> dict:
    ac = agent_calls(trial)
    sc = sim_calls(trial)
    cc = claim_calls(trial)
    pc = port_calls(trial)
    res = json.loads((trial / "result.json").read_text()) if (trial / "result.json").exists() else {}
    ae = res.get("agent_execution") or {}
    wall = None
    if ac:
        wall = max(ac[-1]["ts"] - ac[0]["ts"], 1.0)
    tot = lambda k, xs: sum(x.get(k) or 0 for x in xs)  # noqa: E731
    r = {
        "trial": trial.name,
        "agent_calls": len(ac),
        "agent_prompt_total": tot("prompt_total", ac),
        "agent_prompt_uncached": tot("input_uncached", ac),
        "agent_cache_read": tot("cache_read", ac),
        "agent_output": tot("output", ac), "agent_reasoning": tot("reasoning", ac),
        "agent_cached_fraction": round(tot("cache_read", ac) / max(tot("prompt_total", ac), 1), 3),
        "agent_prompt_per_call_median": int(statistics.median([c["prompt_total"] for c in ac])) if ac else None,
        "agent_prompt_per_call_max": max((c["prompt_total"] for c in ac), default=None),
        "agent_active_wall_sec": round(wall, 1) if wall else None,
        "agent_calls_per_min": round(len(ac) / (wall / 60), 2) if wall else None,
        "agent_prompt_tokens_per_min": int(tot("prompt_total", ac) / (wall / 60)) if wall else None,
        "agent_uncached_prompt_tokens_per_min": int(tot("input_uncached", ac) / (wall / 60)) if wall else None,
        "agent_peak_prompt_tokens_60s": peak_60s(ac) if ac else None,
        "agent_peak_uncached_tokens_60s": peak_60s(ac, "input_uncached") if ac else None,
        "agent_errors": agent_errors(trial),
        "agent_throttle_retries": int((trial / "agent" / "throttle_retries.txt").read_text().strip())
        if (trial / "agent" / "throttle_retries.txt").exists() else 0,
        "sim_calls": len(sc),
        "sim_prompt_total": tot("prompt_tokens", sc), "sim_cache": tot("cache_tokens", sc),
        "sim_completion": tot("completion_tokens", sc),
        "sim_retry_attempts": sum(max((c.get("attempts") or 1) - 1, 0) for c in sc),
        "sim_failed_calls": sum(1 for c in sc if not c.get("ok")),
        "sim_latency_median": statistics.median([c["latency_sec"] for c in sc]) if sc else None,
        "claim_calls_ok": sum(1 for c in cc if c.get("ok")), "claim_attempts": len(cc),
        "claim_prompt_total": tot("prompt_tokens", cc), "claim_cached": tot("cached_tokens", cc),
        "claim_completion": tot("completion_tokens", cc), "claim_reasoning": tot("reasoning_tokens", cc),
        "claim_latency_median": statistics.median([c["latency_sec"] for c in cc if c.get("ok")])
        if any(c.get("ok") for c in cc) else None,
        "port_calls_ok": sum(1 for c in pc if c.get("ok")), "port_attempts": len(pc),
        "port_prompt_total": tot("prompt_tokens", pc), "port_completion": tot("completion_tokens", pc),
        "port_reasoning": tot("reasoning_tokens", pc),
        "intent_judge_in_trial": overlay_judge_stats(trial),
        "judge_tokens_used": judge_tokens(trial),
        "agent_setup_to_finish_sec": None,
        "exception": (res.get("exception_info") or {}).get("exception_type"),
    }
    if ae.get("started_at") and ae.get("finished_at"):
        from datetime import datetime
        f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
        r["agent_setup_to_finish_sec"] = round((f(ae["finished_at"]) - f(ae["started_at"])).total_seconds(), 1)
    return r


def main(argv: list[str]) -> int:
    rows = []
    for rd in argv:
        for t in sorted(Path(rd, "trials").iterdir()):
            if t.is_dir() and "__" in t.name:
                rows.append({"run": Path(rd).name, **trial_report(t)})
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
