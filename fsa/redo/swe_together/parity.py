"""Sandbox parity check: the same rung / arm / tasks on local docker (arm64 images) vs E2B.

    python -m fsa.redo.swe_together.parity <docker_root> <e2b_root> [--out summary.json]

Per trial: benchmark judge score (``eval/per_trial.json``), tests reward
(``result.json`` verifier reward), executed turns, simulator messages, agent wall time
(Harbor ``agent_execution``), agent model calls / tokens, tool-call wall time by command
kind (from opencode ``tool_use`` start/end times), provider errors. Per task and backend:
means over replicates; overall: mean differences docker - e2b.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ..metrics import find_run_dirs
from .tokens import agent_calls


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def tool_time(trial: Path) -> dict:
    """Seconds spent in tool calls, split into test/build runs and everything else."""
    out = defaultdict(float)
    n = defaultdict(int)
    for f in sorted((trial / "agent").glob("opencode.txt.turn-*")):
        if "throttled" in f.name:
            continue
        for line in f.read_text(errors="replace").splitlines():
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "tool_use":
                continue
            st = (e.get("part") or {}).get("state") or {}
            tm = st.get("time") or {}
            if not (tm.get("start") and tm.get("end")):
                continue
            dur = (tm["end"] - tm["start"]) / 1000
            cmd = json.dumps(st.get("input") or {})
            kind = "test_or_build" if re.search(r"go (test|build|vet)|pytest|npm (run )?test|cargo (test|build)|make|mise run", cmd) \
                else (e["part"].get("tool") or "?")
            out[kind] += dur
            n[kind] += 1
    return {"seconds": {k: round(v, 1) for k, v in out.items()}, "calls": dict(n)}


def trial_row(trial: Path, eval_rows: dict) -> dict:
    res = json.loads((trial / "result.json").read_text()) if (trial / "result.json").exists() else {}
    ae = res.get("agent_execution") or {}
    wall = (_ts(ae.get("finished_at")) - _ts(ae.get("started_at"))).total_seconds() \
        if ae.get("started_at") and ae.get("finished_at") else None
    rw = (res.get("verifier_result") or {}).get("rewards") or {}
    turns = [json.loads(l) for l in (trial / "agent" / "turns.jsonl").read_text().splitlines() if l.strip()] \
        if (trial / "agent" / "turns.jsonl").exists() else []
    ex = [t for t in turns if t.get("executed", True) and t.get("opened_by") != "session_end"]
    ac = agent_calls(trial)
    er = eval_rows.get(trial.name) or {}
    return {"trial": trial.name, "task": trial.name.rsplit("__", 1)[0],
            "judge_score": er.get("judge_score"), "tests_reward": rw.get("reward") if isinstance(rw, dict) else rw,
            "user_correction": er.get("user_correction"),
            "turns": len(ex), "sim_messages": sum(1 for t in ex if t.get("opened_by") == "sim"),
            "agent_wall_sec": round(wall, 1) if wall else None,
            "turn_sec": [t.get("turn_sec") for t in ex],
            "agent_calls": len(ac), "agent_prompt_tokens": sum(c["prompt_total"] for c in ac),
            "agent_output_tokens": sum(c["output"] + c["reasoning"] for c in ac),
            "tool_time": tool_time(trial),
            "exception": (res.get("exception_info") or {}).get("exception_type"),
            "final_patch_bytes": (trial / "agent" / "final.patch").stat().st_size
            if (trial / "agent" / "final.patch").exists() else 0}


def rows_for(root: str) -> list[dict]:
    out = []
    for rd in find_run_dirs([root]):
        ev = {}
        if (rd / "eval" / "per_trial.json").exists():
            ev = {r["trial_id"]: r for r in json.loads((rd / "eval" / "per_trial.json").read_text())}
        for t in sorted((rd / "trials").iterdir()):
            if t.is_dir() and "__" in t.name and (t / "result.json").exists():
                out.append({**trial_row(t, ev), "run": str(rd)})
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def summarize(docker: list[dict], e2b: list[dict]) -> dict:
    keys = ("judge_score", "tests_reward", "turns", "sim_messages", "agent_wall_sec", "agent_calls",
            "agent_prompt_tokens", "user_correction")
    per_task = {}
    for task in sorted({r["task"] for r in docker + e2b}):
        d = [r for r in docker if r["task"] == task]
        e = [r for r in e2b if r["task"] == task]
        per_task[task] = {"docker": {k: _mean([r[k] for r in d]) for k in keys} | {"n": len(d)},
                          "e2b": {k: _mean([r[k] for r in e]) for k in keys} | {"n": len(e)},
                          "docker_test_or_build_sec": _mean([r["tool_time"]["seconds"].get("test_or_build", 0) for r in d]),
                          "e2b_test_or_build_sec": _mean([r["tool_time"]["seconds"].get("test_or_build", 0) for r in e])}
    overall = {k: {"docker": _mean([r[k] for r in docker]), "e2b": _mean([r[k] for r in e2b])} for k in keys}
    return {"per_task": per_task, "overall": overall, "trials": {"docker": docker, "e2b": e2b}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("docker_root")
    ap.add_argument("e2b_root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    s = summarize(rows_for(a.docker_root), rows_for(a.e2b_root))
    txt = json.dumps(s, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    print(json.dumps({"per_task": s["per_task"], "overall": s["overall"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
