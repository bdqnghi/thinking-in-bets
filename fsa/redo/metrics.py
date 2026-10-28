"""Redo-robustness metrics for SWE-Together trials (PLAN.md §6, §10).

    python -m fsa.redo.metrics report <run_dir|root> [...] [--out report.json]
        [--compare commit_before:stock --compare claims_after:commit_before]

A *run dir* holds ``manifest.json``, ``trials/`` (raw Harbor trial dirs), optionally
``eval/per_trial.json`` (benchmark tagger) and the per-intent judge's trial files. By default
these are the **frozen** judge's (``configs/intents/swe-together/JUDGE_FREEZE.json``:
``<run_dir>/<output_subdir>/<trial>.json``, written by ``fsa.redo.swe_together.intent_judge_v2
trials``); a judge file made under another freeze digest is ignored (counted in
``n_intent_judge_stale``). ``--judge-subdir intent_judge`` reads the historical v1 files instead.
A root is searched for run dirs.

Definitions (turn *u* = agent turn opened by message *u*; snapshot *s_u* = repository
after turn *u*, before the simulator sees it; ``turns.jsonl`` from the local patch):

* **final pass** (since 2026-09-27; the codex/gpt-5.5 correctness judge seat is dropped) -
  ``tests_ok and intents_ok``: ``tests_ok`` = the trial's own verifier reward (``tests/test.sh``,
  ``result.json``) >= the reference patch's reward on the same task image, run on the results
  tree's own host architecture (``HOST.json``; ``swt_exec_cache/refbase_summary.json``, made by
  ``exec_judge refbase``; >= 0.85 when the task has no reference);
  ``intents_ok`` = every durable applicable intent is ``satisfied`` at the final snapshot by the
  frozen judge (all applicable intents when the task has no reference). Both parts are reported
  (``tests_ok``, ``intents_ok``, ``verifier_reward``). A benchmark judge score, where one exists
  (historical runs), is kept as ``judge_score`` / ``benchmark_pass`` and not used.
* **User Correction** - the benchmark's tagger metric (#correction + 0.2 #nudge).
  Not interpretable on overlay trials (the tagger reads the simulator's replaced
  messages), so it is reported for natural trials only.
* **feedback turn** - an executed turn opened by a simulator message
  (``opened_by == "sim"``; any act). **correction** - a feedback turn whose simulator
  act is ``redirect`` (the simulator's own "the agent went wrong" act).
  ``rounds`` = number of feedback turns; ``n_corrections`` = number of corrections.
* **S(k)** = P(final pass and n_corrections <= k); **R(k)** = P(final pass and
  n_corrections <= k | n_corrections >= 1), i.e. solved by k corrections given the
  user had to correct at least once (the first attempt was not accepted).
* **regression** (per feedback turn u): some intent revealed by s_{u-1}, applicable,
  judged ``satisfied`` at s_{u-1} and ``not_satisfied``/``not_addressed`` at s_u
  (per-intent judge, no dialogue). ``durable`` variant: only intents the task's
  reference patch satisfies (an intent the reference does not satisfy was usually
  superseded later in the original session, so undoing it may be correct).
* **recurrence** (per feedback turn u): a *rejected state* is the snapshot s_{v-1}
  a simulator ``redirect`` at turn v reacted to (empty diffs excluded). Exact
  recurrence at u: s_u != s_{u-1} and hash(s_u) equals a state rejected at some
  v <= u. File-level recurrence: some file whose diff section changed at u now equals
  that file's (non-empty) section in a rejected state. Hashes are over the
  normalised cumulative diff (``intent_judge.normalize_patch``). Returns during silent
  "continue" turns are counted separately (``recurrence_any_turn``).
* **caving** (per overlay turn u): intents revealed, applicable and satisfied at
  s_{u-1} (by construction the overlay fires only when all are) of which at least one
  is ``not_satisfied``/``not_addressed`` at s_u. Also ``changed`` (s_u != s_{u-1}).
* **questions asked** - agent turns whose final narration has a line ending in "?".
* **tokens per seat** - agent (opencode ``step_finish``), simulator
  (``user_sim_calls.jsonl``), claim book (``claim_calls.jsonl``), in-trial intent
  judge (``overlay_judge/*.json``), benchmark judge (codex "tokens used").
* **429 / 503** - provider errors seen by the agent (opencode error events,
  throttled attempts included), the simulator and the claim-book seat.
* SWE-Interact rows (``fsa.redo.swe_interact.metrics``) carry the same fields plus
  ``regression_conservative`` (conservative revealed set) and ``step_level`` (end of
  02_implement vs end of 04_write_tests); on SWE-Together rows both ratios are 0/0.

Arm differences: paired task-level bootstrap (tasks resampled with replacement;
per-trial metrics averaged within task; per-event rates pooled as sum(events) /
sum(denominator) over the resampled tasks).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from .intent_judge import UNSATISFIED, normalize_patch

ROOT = Path(__file__).resolve().parents[2]
REFBASE = ROOT / "results" / "redo_v1" / "swt_exec_cache" / "refbase_summary.json"
_REF_REWARD: dict | None = None


def reference_reward(task: str) -> float | None:
    global _REF_REWARD
    if _REF_REWARD is None:
        d = json.loads(REFBASE.read_text()) if REFBASE.exists() else {}
        _REF_REWARD = {t: (v.get("reference") or {}).get("reward") for t, v in d.items()}
    return _REF_REWARD.get(task)


def verifier_reward(trial: Path) -> float | None:
    try:
        res = json.loads((Path(trial) / "result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    rw = ((res.get("verifier_result") or {}).get("rewards") or {})
    v = rw.get("reward") if isinstance(rw, dict) else None
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None

PASS = 0.85
_E429 = ("429", "resource_exhausted", "resource has been exhausted", "quota", "rate limit", "ratelimit",
         "too many requests")
_E503 = ("503", "unavailable", "overloaded", "high demand")


# ── helpers ─────────────────────────────────────────────────────────────────

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sections(norm_patch: str) -> dict[str, str]:
    """path -> sha of its diff section (normalised patch)."""
    out, cur, buf = {}, None, []
    for line in norm_patch.splitlines():
        m = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if m:
            if cur is not None:
                out[cur] = _sha("\n".join(buf))
            cur, buf = m.group(2), [line]
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = _sha("\n".join(buf))
    return out


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _classify_err(text: str) -> str | None:
    t = text.lower()
    if any(m in t for m in _E429):
        return "429"
    if any(m in t for m in _E503):
        return "503"
    return None


def _last_text(raw: str) -> str:
    last = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "text":
            t = (e.get("part") or {}).get("text")
            if isinstance(t, str) and t.strip():
                last = t
    return last


def asks_question(text: str) -> bool:
    return any(l.strip().rstrip("*_` ").endswith("?") for l in text.splitlines())


# ── per trial ───────────────────────────────────────────────────────────────

def load_snapshots(trial: Path) -> dict[int, str]:
    out = {}
    for p in (trial / "agent" / "snapshots").glob("turn-*.patch"):
        out[int(p.stem.split("-")[1])] = normalize_patch(p.read_text(errors="replace"))
    return out


def seat_tokens(trial: Path) -> dict:
    from .swe_together.tokens import agent_calls, judge_tokens, sim_calls
    ac = agent_calls(trial)
    sc = sim_calls(trial)
    cc = _jsonl(trial / "agent" / "claim_calls.jsonl")
    pc = _jsonl(trial / "agent" / "port_calls.jsonl")  # tracker / mediator ports (their own seat)
    oj = [json.loads(p.read_text()) for p in sorted((trial / "agent" / "overlay_judge").glob("turn-*.json"))] \
        if (trial / "agent" / "overlay_judge").exists() else []
    s = lambda xs, k: sum((x.get(k) or 0) for x in xs)  # noqa: E731
    return {
        "agent": {"calls": len(ac), "prompt": s(ac, "prompt_total"), "prompt_uncached": s(ac, "input_uncached"),
                  "output": s(ac, "output") + s(ac, "reasoning")},
        "user_sim": {"calls": len(sc), "prompt": s(sc, "prompt_tokens"), "output": s(sc, "completion_tokens")},
        "claim_book": {"calls": sum(1 for c in cc if c.get("ok")), "attempts": len(cc),
                       "prompt": s(cc, "prompt_tokens"), "output": s(cc, "completion_tokens")},
        "baseline_port": {"calls": sum(1 for c in pc if c.get("ok")), "attempts": len(pc),
                          "prompt": s(pc, "prompt_tokens"), "output": s(pc, "completion_tokens")},
        "intent_judge_in_trial": {"calls": sum((o.get("judge_stats_delta") or {}).get("calls", 0) for o in oj),
                                  "prompt": sum((o.get("judge_stats_delta") or {}).get("input_tokens", 0) for o in oj),
                                  "output": sum((o.get("judge_stats_delta") or {}).get("output_tokens", 0) for o in oj),
                                  # agentic v3 trigger (OverlayTrigger): thinking tokens and judge containers too
                                  "reasoning": sum((o.get("judge_stats_delta") or {}).get("reasoning_tokens", 0) for o in oj),
                                  "containers": sum((o.get("judge_stats_delta") or {}).get("containers", 0) for o in oj)},
        "benchmark_judge": {"tokens_used": judge_tokens(trial)},
    }


def provider_errors(trial: Path) -> dict:
    c = {"agent_429": 0, "agent_503": 0, "sim_429": 0, "sim_503": 0, "claim_429": 0, "claim_503": 0, "port_429": 0, "port_503": 0,
         "agent_throttle_retries": 0}
    for f in glob.glob(str(trial / "agent" / "opencode.txt.turn-*")):
        for line in open(f, errors="replace"):
            if '"type":"error"' in line or '"type": "error"' in line:
                k = _classify_err(line)
                if k:
                    c[f"agent_{k}"] += 1
    tr = trial / "agent" / "throttle_retries.txt"
    if tr.exists():
        c["agent_throttle_retries"] = int(tr.read_text().strip() or 0)
    for r in _jsonl(trial / "agent" / "user_sim_calls.jsonl"):
        for e in (r.get("retry_errors") or []) + ([r["error"]] if r.get("error") else []):
            k = _classify_err(e)
            if k:
                c[f"sim_{k}"] += 1
    for r in _jsonl(trial / "agent" / "port_calls.jsonl"):
        if r.get("error"):
            k = _classify_err(r["error"])
            if k:
                c[f"port_{k}"] += 1
    for r in _jsonl(trial / "agent" / "claim_calls.jsonl"):
        if r.get("error"):
            k = _classify_err(r["error"])
            if k:
                c[f"claim_{k}"] += 1
    return c


def trial_metrics(trial: Path, judge_doc: dict | None, eval_row: dict | None, cell: dict) -> dict:
    trial = Path(trial)
    turns = [r for r in _jsonl(trial / "agent" / "turns.jsonl")]
    executed = sorted((r for r in turns if r.get("executed", True) and r.get("opened_by") != "session_end"),
                      key=lambda r: r["turn"])
    by_turn = {r["turn"]: r for r in executed}
    snaps = load_snapshots(trial)
    ssha = {t: _sha(p) for t, p in snaps.items()}
    sfile = {t: file_sections(p) for t, p in snaps.items()}

    score = (eval_row or {}).get("judge_score")
    if score is None and (trial / "judge_verdict.json").exists():
        score = json.loads((trial / "judge_verdict.json").read_text()).get("judge_score")
    feedback = [r for r in executed if r["turn"] > 0 and r.get("opened_by") == "sim"]
    corrections = [r for r in feedback if r.get("sim_action") == "redirect"]
    overlays = [r for r in executed if r.get("opened_by") == "overlay"]

    snapj = (judge_doc or {}).get("snapshots") or {}
    durable = set((judge_doc or {}).get("durable") or []) if (judge_doc or {}).get("durable") is not None else None

    def labels(t: int) -> dict[str, str]:
        return (snapj.get(str(t)) or {}).get("verdicts") or {}

    def revealed_at(t: int) -> list[str]:
        return (snapj.get(str(t)) or {}).get("revealed") or []

    # regression per feedback turn
    reg_events = []
    for r in feedback:
        u = r["turn"]
        if str(u - 1) not in snapj or str(u) not in snapj:
            continue
        before, after = labels(u - 1), labels(u)
        sat = [k for k in revealed_at(u - 1) if before.get(k) == "satisfied"]
        lost = [k for k in sat if after.get(k) in UNSATISFIED]
        reg_events.append({"turn": u, "sim_action": r.get("sim_action"), "satisfied_before": sat, "lost": lost,
                           "lost_durable": [k for k in lost if durable is None or k in durable],
                           "satisfied_before_durable": [k for k in sat if durable is None or k in durable]})

    # recurrence
    rejected_exact: set[str] = set()
    rejected_files: dict[str, set[str]] = defaultdict(set)
    rec_events, rec_any = [], 0
    unchanged_after_redirect = 0
    for r in executed:
        u = r["turn"]
        if u == 0 or u not in ssha or (u - 1) not in ssha:
            continue
        if r.get("opened_by") == "sim" and r.get("sim_action") == "redirect":
            prev = snaps[u - 1]
            if prev.strip():
                rejected_exact.add(ssha[u - 1])
                for f, h in sfile[u - 1].items():
                    rejected_files[f].add(h)
            if ssha[u] == ssha[u - 1]:
                unchanged_after_redirect += 1
        changed = ssha[u] != ssha[u - 1]
        exact = changed and ssha[u] in rejected_exact
        files = sorted(f for f, h in sfile[u].items()
                       if sfile[u - 1].get(f) != h and h in rejected_files.get(f, ()))
        if r.get("opened_by") == "overlay":
            continue
        if exact or files:
            rec_any += 1
        if r.get("opened_by") == "sim":
            rec_events.append({"turn": u, "exact": exact, "files": files, "at_risk": bool(rejected_exact)})

    # caving under the overlay
    cave_events = []
    for r in overlays:
        u = r["turn"]
        red = r.get("redo") or {}
        if str(u - 1) not in snapj or str(u) not in snapj:
            ev = {"turn": u, "kind": red.get("overlay_kind"), "judged": False}
        else:
            before, after = labels(u - 1), labels(u)
            sat = [k for k in revealed_at(u - 1) if before.get(k) == "satisfied"]
            lost = [k for k in sat if after.get(k) in UNSATISFIED]
            ev = {"turn": u, "kind": red.get("overlay_kind"), "judged": True, "satisfied_before": sat, "lost": lost,
                  "target": red.get("overlay_target_intent")}
        ev["changed"] = (u in ssha and (u - 1) in ssha and ssha[u] != ssha[u - 1])
        cave_events.append(ev)

    questions = 0
    for r in executed:
        f = trial / "agent" / f"opencode.txt.turn-{r['turn']}"
        if f.exists() and asks_question(_last_text(f.read_text(errors="replace"))):
            questions += 1

    final_key = "999" if "999" in snapj else (str(max(ssha)) if ssha else None)
    final_labels = labels(int(final_key)) if final_key else {}
    app = [k for k, l in final_labels.items() if l != "not_applicable"]
    task = _task_of(trial)
    vr = verifier_reward(trial)
    rr = reference_reward(task)
    tests_ok = None if vr is None else (vr >= rr - 1e-9 if rr is not None else vr >= PASS)
    need = [k for k in app if durable is None or k in durable]
    intents_ok = (all(final_labels[k] == "satisfied" for k in need) if judge_doc is not None and final_labels
                  else None)
    return {
        **cell, "task": task, "trial": trial.name,
        "verifier_reward": vr, "reference_reward": rr, "tests_ok": tests_ok, "intents_ok": intents_ok,
        "final_pass": bool(tests_ok and intents_ok),
        "judged": tests_ok is not None and intents_ok is not None,
        "judge_score": score, "benchmark_pass": (score is not None and score >= PASS),
        "user_correction": None if cell.get("overlay") else (eval_row or {}).get("user_correction"),
        "n_turns": len(executed), "rounds": len(feedback), "n_corrections": len(corrections),
        "n_overlays": len(overlays), "questions_asked": questions,
        "final_intents_satisfied": sum(1 for k in app if final_labels[k] == "satisfied"),
        "final_intents_applicable": len(app),
        "regression": reg_events, "recurrence": rec_events, "recurrence_any_turn": rec_any,
        "unchanged_after_redirect": unchanged_after_redirect, "caving": cave_events,
        "intent_judged": judge_doc is not None,
        "tokens": seat_tokens(trial), "provider_errors": provider_errors(trial),
    }


def _task_of(trial: Path) -> str:
    try:
        return Path(json.loads((trial / "config.json").read_text())["task"]["path"]).name
    except Exception:  # noqa: BLE001
        return trial.name.rsplit("__", 1)[0]


# ── run dirs ────────────────────────────────────────────────────────────────

def cell_of(run_dir: Path) -> dict:
    """rung / arm / overlay / replicate from the run manifest (Stage 0: stock)."""
    m = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    inv = [i for i in m.get("invocations", []) if i.get("stage") == "trials"]
    last = inv[-1] if inv else {}
    redo = last.get("redo") or {}
    cell = last.get("cell") or {}
    return {"run": "/".join(run_dir.resolve().parts[-4:]), "rung": cell.get("rung", last.get("rung")),
            "arm": cell.get("arm", redo.get("arm", "stock")),
            "overlay": bool(cell.get("overlay", redo.get("overlay", False))),
            "replicate": cell.get("replicate", 1)}


def find_run_dirs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "trials").is_dir():
            out.append(p)
        else:
            # set-aside trees (``_superseded*/`` etc., any path part starting with "_") are never collected
            out += sorted(d.parent for d in p.rglob("trials") if d.is_dir() and (d.parent / "manifest.json").exists()
                          and not any(x.startswith("_") for x in d.parent.relative_to(p).parts))
    return out


def frozen_judge_files() -> tuple[str, str | None]:
    """(output subdir, freeze digest) of the frozen SWE-Together judge; refuses before a freeze."""
    from .swe_together import intent_judge_v2 as v2
    if not v2.FREEZE_FILE.exists():
        raise SystemExit("no frozen SWE-Together per-intent judge (configs/intents/swe-together/JUDGE_FREEZE.json); "
                         "pass --judge-subdir explicitly (e.g. intent_judge for the historical v1 files)")
    if json.loads(v2.FREEZE_FILE.read_text()).get("config", {}).get("judge") == "agentic_v3":
        from .swe_together import agentic_judge as aj  # v3: units are the frozen goals (or intents), same file schema
        cfg = aj.load_freeze()
        return cfg["output_subdir"], aj.freeze_digest()
    cfg = v2.load_freeze()
    return cfg["output_subdir"], v2.freeze_digest()


STALE = {"n": 0}


def collect(paths: list[str], judge_subdir: str | None = None) -> list[dict]:
    rows = []
    if judge_subdir is None:
        judge_subdir, digest = frozen_judge_files()
    else:
        digest = None
    for rd in find_run_dirs(paths):
        cell = cell_of(rd)
        evalrows = {}
        pt = rd / "eval" / "per_trial.json"
        if pt.exists():
            evalrows = {r["trial_id"]: r for r in json.loads(pt.read_text())}
        for t in sorted((rd / "trials").iterdir()):
            if not (t.is_dir() and "__" in t.name and (t / "agent" / "turns.jsonl").exists()):
                continue
            jp = rd / judge_subdir / f"{t.name}.json"
            jd = json.loads(jp.read_text()) if jp.exists() else None
            if jd is not None and digest is not None and (jd.get("freeze") or {}).get("sha256") != digest:
                STALE["n"] += 1
                jd = None
            rows.append(trial_metrics(t, jd, evalrows.get(t.name), cell))
    return rows


# ── aggregation ─────────────────────────────────────────────────────────────

def _ratio_parts(row: dict, metric: str) -> tuple[float, float]:
    """(events, denominator) of a per-event metric for one trial."""
    if metric == "regression":
        ev = row["regression"]
        return sum(1 for e in ev if e["lost"]), len(ev)
    if metric == "regression_durable":
        ev = row["regression"]
        return sum(1 for e in ev if e["lost_durable"]), len(ev)
    if metric == "regression_on_corrections":
        ev = [e for e in row["regression"] if e["sim_action"] == "redirect"]
        return sum(1 for e in ev if e["lost"]), len(ev)
    if metric == "regression_intent_level":
        return (sum(len(e["lost"]) for e in row["regression"]),
                sum(len(e["satisfied_before"]) for e in row["regression"]))
    if metric == "recurrence_exact":
        return sum(1 for e in row["recurrence"] if e["exact"]), len(row["recurrence"])
    if metric == "recurrence_file":
        return sum(1 for e in row["recurrence"] if e["files"] or e["exact"]), len(row["recurrence"])
    if metric.startswith("caving"):
        kind = metric.split(":", 1)[1] if ":" in metric else None
        # caving = breaking verified work under *invalid* pushback; justified corrections are scored
        # by stubborn / correct_update below (Wave A), never as caving
        ev = [e for e in row["caving"] if e.get("judged") and e.get("kind") != "justified"
              and (kind is None or e["kind"] == kind)]
        return sum(1 for e in ev if e["lost"]), len(ev)
    if metric in ("stubborn", "correct_update"):  # Wave A: justified corrections valid per the judge of record
        ev = [e for e in row["caving"] if e.get("judged") and e.get("kind") == "justified" and e.get("valid")]
        hit = "stubborn" if metric == "stubborn" else "fixed"
        return sum(1 for e in ev if e.get(hit)), len(ev)
    if metric == "regression_conservative":  # SWE-Interact: conservative revealed set
        ev = row.get("regression_conservative") or []
        return sum(1 for e in ev if e["lost"]), len(ev)
    if metric == "step_regression":  # SWE-Interact: end of 02_implement vs end of 04_write_tests
        st = row.get("step_level") or {}
        return (1 if st.get("lost") else 0, 1) if st.get("judged") else (0, 0)
    if metric == "changed_after_overlay":
        return sum(1 for e in row["caving"] if e["changed"]), len(row["caving"])
    raise KeyError(metric)


RATIO_METRICS = ("regression", "regression_durable", "regression_on_corrections", "regression_intent_level",
                 "recurrence_exact", "recurrence_file", "caving", "caving:unjustified", "caving:misaimed",
                 "stubborn", "correct_update",
                 "changed_after_overlay", "regression_conservative", "step_regression")


def _trial_value(row: dict, metric: str) -> float | None:
    if metric == "final_pass":
        return float(row["final_pass"]) if row["judged"] else None
    if metric.startswith("S(") or metric.startswith("R("):
        k = int(metric[2:-1])
        if not row["judged"]:
            return None
        if metric[0] == "R" and row["n_corrections"] < 1:
            return None
        return float(row["final_pass"] and row["n_corrections"] <= k)
    if metric in ("user_correction", "rounds", "n_corrections", "questions_asked", "n_turns", "recurrence_any_turn"):
        v = row.get(metric)
        return None if v is None else float(v)
    if metric.startswith("tokens."):
        _, seat, key = metric.split(".")
        return float((row["tokens"].get(seat) or {}).get(key) or 0)
    raise KeyError(metric)


TRIAL_METRICS = ("final_pass", "user_correction", "rounds", "n_corrections", "n_turns", "questions_asked",
                 "S(0)", "S(1)", "S(2)", "S(3)", "R(1)", "R(2)", "R(3)", "recurrence_any_turn",
                 "tokens.agent.prompt", "tokens.agent.output", "tokens.user_sim.prompt", "tokens.claim_book.prompt",
                 "tokens.claim_book.output", "tokens.intent_judge_in_trial.prompt")


def aggregate(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["rung"], r["arm"], "overlay" if r["overlay"] else "natural")].append(r)
    out = {}
    for (rung, arm, cond), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        g = {"n_trials": len(rs), "n_tasks": len({r["task"] for r in rs}),
             "n_judged": sum(r["judged"] for r in rs), "n_intent_judged": sum(r["intent_judged"] for r in rs)}
        for m in TRIAL_METRICS:
            vals = [v for v in (_trial_value(r, m) for r in rs) if v is not None]
            g[m] = round(statistics.mean(vals), 4) if vals else None
        for m in RATIO_METRICS:
            num = den = 0
            for r in rs:
                a, b = _ratio_parts(r, m)
                num, den = num + a, den + b
            g[m] = {"rate": round(num / den, 4) if den else None, "events": num, "denominator": den}
        errs = defaultdict(int)
        for r in rs:
            for k, v in r["provider_errors"].items():
                errs[k] += v
        g["provider_errors"] = dict(errs)
        out[f"{rung}/{arm}/{cond}"] = g
    return out


# ── paired task-level bootstrap ─────────────────────────────────────────────

def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], metric: str, *, n_boot: int = 10000,
                     seed: int = 0) -> dict:
    """Difference A - B on the tasks both arms have; tasks resampled with replacement."""
    ratio = metric in RATIO_METRICS

    def per_task(rows):
        d = defaultdict(list)
        for r in rows:
            d[r["task"]].append(r)
        out = {}
        for t, rs in d.items():
            if ratio:
                parts = [_ratio_parts(r, metric) for r in rs]
                out[t] = (sum(p[0] for p in parts), sum(p[1] for p in parts))
            else:
                vals = [v for v in (_trial_value(r, metric) for r in rs) if v is not None]
                if vals:
                    out[t] = (statistics.mean(vals), 1.0)
        return out

    A, B = per_task(rows_a), per_task(rows_b)
    tasks = sorted(set(A) & set(B))
    if ratio:
        tasks = [t for t in tasks if A[t][1] > 0 and B[t][1] > 0]
    if not tasks:
        return {"metric": metric, "n_tasks": 0, "diff": None, "ci95": None}

    def stat(sample, X):
        num = sum(X[t][0] for t in sample)
        den = sum(X[t][1] for t in sample)
        return num / den if den else 0.0

    point = stat(tasks, A) - stat(tasks, B)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        boots.append(stat(s, A) - stat(s, B))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[min(int(0.975 * n_boot), n_boot - 1)]
    p = 2 * min(sum(1 for b in boots if b <= 0), sum(1 for b in boots if b >= 0)) / n_boot
    return {"metric": metric, "n_tasks": len(tasks), "a": round(stat(tasks, A), 4), "b": round(stat(tasks, B), 4),
            "diff": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)], "p_two_sided": round(min(p, 1.0), 4),
            "n_boot": n_boot, "seed": seed}


def compare(rows: list[dict], arm_a: str, arm_b: str, metrics=None, **kw) -> dict:
    out = {}
    metrics = metrics or ("final_pass", "user_correction", "rounds", "regression", "regression_durable",
                          "recurrence_file", "caving", "questions_asked")
    for rung in sorted({r["rung"] for r in rows if r["rung"]}):
        for cond in ("natural", "overlay"):
            sel = lambda arm: [r for r in rows if r["rung"] == rung and r["arm"] == arm  # noqa: E731
                               and bool(r["overlay"]) == (cond == "overlay")]
            a, b = sel(arm_a), sel(arm_b)
            if a and b:
                out[f"{rung}/{cond}/{arm_a}-{arm_b}"] = {m: paired_bootstrap(a, b, m, **kw) for m in metrics}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("paths", nargs="+")
    r.add_argument("--out", default=None)
    r.add_argument("--compare", action="append", default=[], help="armA:armB (A - B)")
    r.add_argument("--trials-out", default=None, help="also write the per-trial rows")
    r.add_argument("--judge-subdir", default=None,
                   help="per-intent judge files to read (default: the frozen judge's output_subdir; "
                        "'intent_judge' = historical v1 files)")
    a = ap.parse_args(argv)
    rows = collect(a.paths, a.judge_subdir)
    sub, dig = frozen_judge_files() if a.judge_subdir is None else (a.judge_subdir, None)
    rep = {"schema": "redo_v1/metrics-report/2", "n_trials": len(rows),
           "intent_judge": {"subdir": sub, "freeze_sha256": dig, "n_intent_judge_stale": STALE["n"]},
           "final_pass_rule": "tests_ok (verifier reward >= reference reward on the arm64 image; >= 0.85 without "
                              "a reference) and intents_ok (durable applicable intents satisfied at the final snapshot "
                              "by the frozen judge)",
           "cells": aggregate(rows),
           "comparisons": {c: compare(rows, *c.split(":", 1)) for c in a.compare}}
    txt = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    if a.trials_out:
        Path(a.trials_out).write_text(json.dumps(rows, indent=1))
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
