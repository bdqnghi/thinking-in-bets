"""Per-trial feature extraction for every Harbor job (DeepSWE 1.1 ``ds11`` / ``ds11m``, Terminal-Bench ``tb4``).

Pure post-hoc analysis: reads ``~/jobs/<job>/<trial>/{result.json, agent/trajectory.json,
artifacts/logs/artifacts/model.patch}`` READ-ONLY. Never launches a model, docker or a systemd unit.
Jobs ``ds11m_G_*`` and ``tb4_G_tib`` may still be running when this is executed: unfinished trials
(no ``result.json`` / no ``finished_at``) are skipped and the output is a partial snapshot.

One row per trial (not per task): ``load_trials()`` returns the list and caches it to
``results/analysis/harbor_trials.json``. Use ``select_per_task()`` to pick one trial per task, either
with the *same* "best-of-N" rule as ``scripts/harbor_collect.py`` (``rule="collect"``) or the first
finished trial (``rule="first"``, unbiased).

    python scripts/analysis/harbor_trials.py [--refresh]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
CACHE = OUT / "harbor_trials.json"
JOBS = Path(os.path.expanduser("~/jobs"))
JOB_RE = re.compile(r"^(tb4|ds11m|ds11)_([WFMSG])_(stock|tib)(?:_p(\d+))?$")
HARNESS = {"tb4": "opencode", "ds11": "opencode", "ds11m": "mini-swe-agent"}

TEST_RE = re.compile(r"\b(pytest|go test|npm (?:run )?test|yarn test|pnpm test|cargo test|jest|vitest|mocha|"
                     r"tox|unittest|make test|bun test|deno test|rspec|phpunit|ctest|mvn test|gradle test|"
                     r"dotnet test|mix test|zig build test)\b")
CLAIM_RE = re.compile(r"claim[_ ]?book", re.I)            # any mention (incl. plans to write one)
CLAIM_BLOCK_RE = re.compile(r"\[CLAIM_BOOK\]|#+\s*`?\[?CLAIM[_ ]BOOK", re.I)  # an actual written block header
CLAIM_ID_RE = re.compile(r"\b(?:C\d+|claim_\w+|claim_id)\b", re.I)
# p values: "p: 0.95", "**p:** 1.00", "p=0.9", "\"p\": 0.9", or the tuple form "(C1, req, evidence, 0.99)"
P_RE = re.compile(r"(?:\bp[\s`*]*[=:][\s`*]*|\"p\"\s*:\s*|,\s*`?|\|\s*`?)(0?\.\d{1,3}|1\.0{1,3}|1(?=`?\s*[\)|]))`?(?=\s*[\)\s,;|*`]|$)")
EDIT_TOOLS = {"edit", "write", "apply_patch", "patch", "multiedit", "str_replace_editor"}
EDIT_CMD_RE = re.compile(r"(cat\s*<<|cat\s*>|sed\s+-i|\btee\b|python3?\s+-\s*<<|>\s*[\w./-]+\.(py|go|ts|js|rs|java|rb|c|h|cc|cpp|md|json|yaml|toml)\b|apply_patch|git apply|patch\s+-p)")


def _iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001
        return None


def _dur(block):
    if not block:
        return None
    a, b = _iso(block.get("started_at")), _iso(block.get("finished_at"))
    return (b - a) if (a and b) else None


def _text(step) -> str:
    parts = [step.get("message") or "", step.get("reasoning_content") or ""]
    return "\n".join(p if isinstance(p, str) else json.dumps(p) for p in parts)


def _cmds(step):
    out = []
    for tc in step.get("tool_calls") or []:
        name = tc.get("function_name") or ""
        args = tc.get("arguments") or {}
        cmd = args.get("command") if isinstance(args, dict) else None
        out.append((name, cmd if isinstance(cmd, str) else "", args))
    return out


def _obs_text(step) -> str:
    o = step.get("observation")
    if not o:
        return ""
    try:
        return "\n".join(str(r.get("content", "")) for r in o.get("results", []))
    except Exception:  # noqa: BLE001
        return str(o)


def trajectory_features(traj_path: Path) -> dict:
    f: dict = {"traj": False}
    if not traj_path.exists():
        return f
    try:
        t = json.loads(traj_path.read_text())
    except Exception:  # noqa: BLE001
        return f
    steps = t.get("steps") or []
    agent = [s for s in steps if s.get("source") == "agent"]
    fm = t.get("final_metrics") or {}
    f.update(traj=True, n_agent_steps=len(agent),
             n_llm_calls=sum(int(s.get("llm_call_count") or 1) for s in agent),
             reasoning_tokens=(fm.get("extra") or {}).get("total_reasoning_tokens"))
    n_tools = n_tests = n_commits = n_edits = 0
    first_edit = first_commit = first_claim = last_claim = mention = None
    claim_text = ""
    empty_msgs = 0
    tool_hist: dict = {}
    max_prompt = 0
    for i, s in enumerate(agent):
        txt = _text(s)
        m = s.get("metrics") or {}
        max_prompt = max(max_prompt, int(m.get("prompt_tokens") or 0))
        if not txt.strip() and not s.get("tool_calls"):
            empty_msgs += 1
        if CLAIM_RE.search(txt):
            mention = mention if mention is not None else i
        blk = CLAIM_BLOCK_RE.search(txt)
        if blk and len(CLAIM_ID_RE.findall(txt[blk.start():blk.start() + 6000])) >= 2:
            if first_claim is None:
                first_claim = i
            last_claim = i
            claim_text = txt[blk.start():]     # keep the LAST written claim book (the final audit)
        for name, cmd, args in _cmds(s):
            n_tools += 1
            tool_hist[name] = tool_hist.get(name, 0) + 1
            if name in EDIT_TOOLS or (cmd and EDIT_CMD_RE.search(cmd)):
                n_edits += 1
                first_edit = i if first_edit is None else first_edit
            if cmd and TEST_RE.search(cmd):
                n_tests += 1
            if cmd and "git commit" in cmd:
                n_commits += 1
                first_commit = i if first_commit is None else first_commit
    n = max(1, len(agent))
    final_txt = _text(agent[-1]) if agent else ""
    ts = [_iso(s.get("timestamp")) for s in agent if s.get("timestamp")]
    ts = [x for x in ts if x]
    pv = [float(v) for v in P_RE.findall(claim_text[:20000])] if claim_text else []
    pv = [v for v in pv if 0.0 <= v <= 1.0]
    # edits made after the (first) written claim book = the audit changed code
    edits_after_claim = 0
    if first_claim is not None:
        for s in agent[first_claim + 1:]:
            for name, cmd, _ in _cmds(s):
                if name in EDIT_TOOLS or (cmd and EDIT_CMD_RE.search(cmd)):
                    edits_after_claim += 1
    f.update(n_tool_calls=n_tools, n_test_runs=n_tests, n_commits=n_commits, n_edit_actions=n_edits,
             first_edit_frac=None if first_edit is None else first_edit / n,
             first_commit_frac=None if first_commit is None else first_commit / n,
             claim_mention=mention is not None,
             claim_book=first_claim is not None,
             claim_book_first_frac=None if first_claim is None else first_claim / n,
             claim_book_last_frac=None if last_claim is None else last_claim / n,
             claim_book_n_p=len(pv), claim_book_p=pv[:60],
             claim_book_min_p=min(pv) if pv else None,
             claim_book_chars=len(claim_text),
             claim_book_low_p=sum(1 for v in pv if v < 0.95),
             claim_book_excerpt=claim_text[:1500],
             steps_after_claim_book=None if first_claim is None else len(agent) - 1 - first_claim,
             edits_after_claim_book=edits_after_claim,
             empty_agent_msgs=empty_msgs, max_prompt_tokens=max_prompt,
             tool_hist=tool_hist, final_msg_tail=final_txt[-400:],
             submitted_echo="COMPLETE_TASK_AND_SUBMIT" in json.dumps(agent[-1].get("tool_calls") or []) if agent else False,
             traj_span_sec=(max(ts) - min(ts)) if len(ts) > 1 else None)
    return f


def opencode_log_features(p: Path) -> dict:
    """Token / finish stats from opencode's JSON event stream (agent/opencode.txt): thinking tokens are only here."""
    if not p.exists():
        return {}
    inp = out = rsn = cread = cwrite = 0
    reasons: dict = {}
    zero = 0
    errors = 0
    for line in p.open(errors="replace"):
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        typ = e.get("type")
        if typ == "error":
            errors += 1
        if typ != "step_finish":
            continue
        part = e.get("part") or {}
        tk = part.get("tokens") or {}
        rs = part.get("reason") or "none"
        reasons[rs] = reasons.get(rs, 0) + 1
        c = tk.get("cache") or {}
        inp += int(tk.get("input") or 0)
        out += int(tk.get("output") or 0)
        rsn += int(tk.get("reasoning") or 0)
        cread += int(c.get("read") or 0)
        cwrite += int(c.get("write") or 0)
        if not (tk.get("input") or tk.get("output") or tk.get("reasoning") or c.get("read")):
            zero += 1
    return {"oc_reasoning_tokens": rsn, "oc_output_tokens": out, "oc_input_uncached": inp, "oc_cache_read": cread,
            "oc_finish_reasons": reasons, "oc_zero_token_steps": zero, "oc_error_events": errors,
            "oc_length_finishes": reasons.get("length", 0)}


def trial_row(d: Path, job: str) -> dict | None:
    rp = d / "result.json"
    if not rp.exists():
        return None
    try:
        r = json.loads(rp.read_text())
    except Exception:  # noqa: BLE001
        return None
    if not r.get("finished_at"):
        return None
    m = JOB_RE.match(job)
    bench, rung, arm, part = m.group(1), m.group(2), m.group(3), m.group(4)
    rew = (r.get("verifier_result") or {}).get("rewards") or {}
    exc = r.get("exception_info") or None
    ar = r.get("agent_result") or {}
    patch = d / "artifacts" / "logs" / "artifacts" / "model.patch"
    row = {
        "job": job, "base_job": f"{bench}_{rung}_{arm}", "bench": bench, "rung": rung, "arm": arm,
        "part": int(part or 1), "harness": HARNESS[bench], "trial": d.name,
        "task": (r.get("task_name") or "").split("/")[-1],
        "reward": rew.get("reward"), "partial": rew.get("partial"),
        "f2p": rew.get("f2p"), "p2p": rew.get("p2p"),
        "exc_type": exc.get("exception_type") if exc else None,
        "exc_msg": (exc.get("exception_message") or "")[:300] if exc else None,
        "started_at": _iso(r.get("started_at")), "finished_at": _iso(r.get("finished_at")),
        "agent_sec": _dur(r.get("agent_execution")), "verifier_sec": _dur(r.get("verifier")),
        "total_sec": (_iso(r.get("finished_at")) or 0) - (_iso(r.get("started_at")) or 0),
        "agent_timeout_multiplier": (r.get("config") or {}).get("agent_timeout_multiplier"),
        "n_input_tokens": ar.get("n_input_tokens"), "n_cache_tokens": ar.get("n_cache_tokens"),
        "n_output_tokens": ar.get("n_output_tokens"),
        "patch_bytes": patch.stat().st_size if patch.exists() else None,
        "model": ((r.get("agent_info") or {}).get("model_info") or {}).get("name"),
    }
    row["passed"] = bool(row["reward"] is not None and float(row["reward"]) >= 0.999)
    row["scored"] = row["reward"] is not None
    row["timeout"] = row["exc_type"] == "AgentTimeoutError"
    row.update(trajectory_features(d / "agent" / "trajectory.json"))
    if HARNESS[bench] == "opencode":
        row.update(opencode_log_features(d / "agent" / "opencode.txt"))
        if row.get("reasoning_tokens") is None and "oc_reasoning_tokens" in row:
            row["reasoning_tokens"] = row["oc_reasoning_tokens"]
    if row.get("n_agent_steps") and row.get("agent_sec"):
        row["sec_per_step"] = row["agent_sec"] / row["n_agent_steps"]
    if bench == "ds11m":
        mt = d / "agent" / "mini-swe-agent.trajectory.json"
        if mt.exists():
            try:
                info = json.loads(mt.read_text()).get("info") or {}
                row["mswea_exit_status"] = info.get("exit_status")
                row["mswea_api_calls"] = (info.get("model_stats") or {}).get("api_calls")
            except Exception:  # noqa: BLE001
                pass
    return row


def load_trials(refresh: bool = False) -> list[dict]:
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    rows = []
    for jd in sorted(JOBS.iterdir()):
        if not jd.is_dir() or not JOB_RE.match(jd.name):
            continue
        for d in sorted(jd.glob("*__*")):
            if d.is_dir():
                row = trial_row(d, jd.name)
                if row:
                    rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(rows, indent=0))
    return rows


def _collect_key(r):
    """harbor_collect.py's preference: passing > non-errored > scored timeout > errored."""
    errored = (not r["passed"]) and r["exc_type"] and not (r["timeout"] and r["scored"])
    return (r["passed"], not errored, not r["timeout"])


def select_per_task(rows: list[dict], rule: str = "collect") -> dict:
    """{base_job: {task: row}}. rule='collect' mirrors scripts/harbor_collect.py (best-of-N over retries);
    rule='first' keeps the earliest-started *scored or timed-out* trial (no outcome-dependent choice)."""
    out: dict = {}
    for r in sorted(rows, key=lambda r: r["started_at"] or 0):
        bj = out.setdefault(r["base_job"], {})
        cur = bj.get(r["task"])
        if rule == "first":
            if cur is None or (not (cur["scored"]) and r["scored"]):
                bj[r["task"]] = r
        else:
            if cur is None or _collect_key(r) > _collect_key(cur):
                bj[r["task"]] = r
    return out


def is_valid(r) -> bool:
    """Mirrors harbor_collect: a trial counts if it was scored (incl. scored timeouts) or passed."""
    errored = (not r["passed"]) and r["exc_type"] and not (r["timeout"] and r["scored"])
    return r["scored"] and not errored


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    rows = load_trials(refresh=a.refresh or not CACHE.exists())
    from collections import Counter
    c = Counter(r["job"] for r in rows)
    for k in sorted(c):
        print(k, c[k])
    print(len(rows), "trials ->", CACHE)
