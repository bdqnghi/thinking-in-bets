"""SWE-Interact adapter for the redo metrics (PLAN.md §6): same definitions and aggregation as
``fsa.redo.metrics`` (SWE-Together), fed from SWE-Interact trial dirs.

    python -u -m fsa.redo.swe_interact.metrics judge <run_dir|root> [...] [--rf-exec all|steps|off]
    python -u -m fsa.redo.swe_interact.metrics report <run_dir|root> [...] [--out r.json] [--compare a:b]

``judge`` writes ``<run_dir>/intent_judge/<job>.json`` per trial: the turn index, the simulator
messages with their tags (revelation + act), the revealed sets per turn (``tagger`` and
``conservative``, ``tagger.py``) and the per-intent verdicts of every snapshot (``judge.py``:
executable hidden tests where mapped, the no-dialogue LLM / rubric judge otherwise). Snapshot
s_t (the repository after agent turn t) is judged on the intents revealed by t (both sets); the
end-of-step snapshots of 02_implement and 04_write_tests and the final snapshot on all intents.

``report`` builds rows with exactly the fields ``fsa.redo.metrics`` aggregates, so the cells,
ratio metrics and the paired task bootstrap are the SWE-Together code:

* turn index: agent turn t is opened by a step instruction (``step``), a simulator reply
  (``sim``) or an overlay message (``overlay``), and ends with an ``ask_user`` call or the step's
  end (``report.turns``); ``sim_action`` = the tagger's act, with ``revision_request`` mapped to
  ``redirect`` (SWE-Together's correction act), so **correction** = a revision request;
* **regression**, **recurrence**, **caving**, **S(k)/R(k)**, questions, tokens per seat and
  429/503 counts as in ``fsa.redo.metrics``; ``final pass`` = the final step's reward (1.0);
  ``durable`` intents = those the reference solution satisfies (``validation`` of the set);
* extras: ``regression_conservative`` (conservative revealed set) and ``step_regression``
  (per trial: some intent revealed by the end of 02_implement, satisfied there and unsatisfied
  at the end of 04_write_tests, which must not change the implementation);
* ``User Correction`` does not exist in SWE-Interact (None); ``mcp_ask_timeouts`` counts
  ``ask_user`` calls that failed on opencode's MCP timeout (a reply the agent never got).
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

from .. import metrics as M
from ..intent_judge import normalize_patch
from . import report as R
from . import tagger
from .intents import load as load_intents

UNSAT = frozenset({"not_satisfied", "not_addressed"})
STEP_IMPL, STEP_TESTS = "02_implement", "04_write_tests"


def _jsonl(p: Path) -> list[dict]:
    return R._jsonl(p)


def trial_task(trial: Path) -> str:
    try:
        return Path(json.loads((trial / "config.json").read_text())["task"]["path"]).name
    except Exception:  # noqa: BLE001
        return json.loads((trial / "result.json").read_text()).get("task_name", trial.name)


def swi_turns(trial: Path) -> list[dict]:
    """report.turns() + who opened each turn (sim / overlay / step), the ask that ended it, and
    the message the benchmark's user said (the overlay message on overlay turns)."""
    asks = {a["call_index"]: a for a in _jsonl(trial / "user-sim" / "asks.jsonl")}
    rows = R.turns(trial)
    prev_ask = None
    for r in rows:
        r["opened_by_call"] = None
        if r["opened_by"] == "sim" and prev_ask is not None:
            a = asks.get(prev_ask) or {}
            r["opened_by_call"] = prev_ask
            if (a.get("redo") or {}).get("opened_by_next") == "overlay":
                r["opened_by"] = "overlay"
                r["opening_message"] = a.get("history_reply")
                r["sim_message_replaced"] = a.get("reply")
            else:
                r["opening_message"] = a.get("history_reply", a.get("reply"))
            r["last_resort"] = bool(a.get("last_resort"))
        if r["ended_by"] == "ask_user":
            prev_ask = int(r["snapshot"].rsplit("ask-", 1)[1].split(".")[0])
            r["ask_call_index"] = prev_ask
        else:
            prev_ask = None
    return rows


def _snap_text(trial: Path, rel: str) -> str | None:
    p = trial / rel
    return p.read_text(errors="surrogateescape") if p.exists() else None


def _plan_text(trial: Path) -> str | None:
    for p in sorted((trial / "steps").glob("*/agent/redo/plan-*.md")):
        return p.read_text(errors="replace")
    return None


# ── judging ─────────────────────────────────────────────────────────────────

def judge_trial(trial: Path, judge, *, rf_exec: str = "all") -> dict:
    trial = Path(trial)
    task = trial_task(trial)
    doc = load_intents(task)
    its = {i["key"]: i for i in doc["intents"]}
    turns = swi_turns(trial)
    # tags of the delivered simulator messages (overlay messages and replaced replies reveal nothing)
    tags, revealed, rev_cons = [], {}, {}
    plan = _plan_text(trial)
    plan_tag = tagger.tag_message(plan, doc) if plan else None
    for r in turns:
        if r["opened_by"] == "sim" and not r.get("last_resort") and r.get("opening_message"):
            tg = tagger.tag_message(r["opening_message"], doc)
            r["act"] = tg["act"]
            r["tag"] = tg["levels"]
            tags.append(tg)
        elif r["opened_by"] == "overlay":
            r["act"] = "revision_request"
        sets = tagger.revealed_sets(tags, doc, plan_tag if r["step"] > "01_plan" else None)
        revealed[r["turn"]], rev_cons[r["turn"]] = sets["tagger"], sets["conservative"]
    final_turns = {r["turn"] for r in turns if r["ended_by"] == "step_end" and r["step"] in (STEP_IMPL, STEP_TESTS)}
    if turns:
        final_turns.add(turns[-1]["turn"])
    snaps = {}
    for r in turns:
        text = _snap_text(trial, r["snapshot"])
        if text is None:
            continue
        keys = list(its) if r["turn"] in final_turns else sorted(set(revealed[r["turn"]]) | set(rev_cons[r["turn"]]))
        sel = [its[k] for k in keys]
        if doc["family"] == "rf" and rf_exec != "all":
            if rf_exec == "off" or r["turn"] not in final_turns:
                sel = [i for i in sel if i["judge"] != "exec_rf"]
        v = judge.verdicts(task, text, sel, label=f"{trial.name} turn {r['turn']}") if sel else {"_meta": {}}
        meta = v.pop("_meta", {})
        snaps[str(r["turn"])] = {"snapshot": r["snapshot"], "patch_sha256": M._sha(normalize_patch(text)),
                                 "revealed": revealed[r["turn"]], "revealed_conservative": rev_cons[r["turn"]],
                                 "verdicts": {k: x["verdict"] for k, x in v.items()},
                                 "judges": {k: x.get("judge") for k, x in v.items()},
                                 "evidence": {k: (x.get("evidence") or "")[:300] for k, x in v.items()},
                                 "exec": meta.get("exec")}
    ref = ((doc.get("validation") or {}).get("reference") or {}).get("labels")
    return {"schema": "redo_v1/swi-intent-judge-trial/1", "trial": trial.name, "task": task,
            "intent_digest": doc["digest"], "judge": judge_config_record(), "tagger": tagger.config_record(),
            "turns": [{k: r.get(k) for k in ("turn", "step", "opened_by", "opened_by_call", "ask_call_index", "act",
                                                "tag", "ended_by", "snapshot")} for r in turns],
            "plan_tag": plan_tag and {"levels": plan_tag["levels"]},
            "durable": sorted(k for k, lab in ref.items() if lab == "satisfied") if ref else None,
            "snapshots": snaps}


def judge_config_record() -> dict:
    from .judge import config_record
    return config_record()


# ── rows for fsa.redo.metrics ──────────────────────────────────────────────

def _final_reward(trial: Path) -> float | None:
    p = trial / "steps" / "05_test_handoff" / "verifier" / "reward.txt"
    if p.exists() and p.read_text().strip():
        try:
            return float(p.read_text().strip())
        except ValueError:
            pass
    try:
        return ((json.loads((trial / "result.json").read_text()).get("verifier_result") or {}).get("rewards") or {}).get("reward")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def seat_tokens(trial: Path) -> dict:
    ac = []
    for sd in sorted((trial / "steps").glob("*/agent")):
        ac += R.agent_calls(sd)[0]
    sc = [c for c in _jsonl(trial / "user-sim" / "sim_calls.jsonl") if c.get("ok")]
    cc = _jsonl(trial / "user-sim" / "claim_calls.jsonl")
    pc = _jsonl(trial / "user-sim" / "port_calls.jsonl")  # tracker / mediator ports (their own seat)
    oj = [json.loads(p.read_text()) for p in sorted((trial / "user-sim" / "overlay_judge").glob("ask-*[0-9].json"))] \
        if (trial / "user-sim" / "overlay_judge").exists() else []
    rub = R.rubric(trial) or {}
    s = lambda xs, k: sum((x.get(k) or 0) for x in xs)  # noqa: E731
    return {
        "agent": {"calls": len(ac), "prompt": s(ac, "prompt_total"), "prompt_uncached": s(ac, "input_uncached"),
                  "output": s(ac, "output") + s(ac, "reasoning")},
        "user_sim": {"calls": len(sc), "prompt": s(sc, "prompt_tokens"), "output": s(sc, "completion_tokens")},
        "claim_book": {"calls": sum(1 for c in cc if c.get("ok")), "attempts": len(cc),
                       "prompt": s(cc, "prompt_tokens"), "output": s(cc, "completion_tokens")},
        "baseline_port": {"calls": sum(1 for c in pc if c.get("ok")), "attempts": len(pc),
                          "prompt": s(pc, "prompt_tokens"), "output": s(pc, "completion_tokens")},
        "intent_judge_in_trial": {"calls": sum((o.get("stats_delta") or {}).get("calls", 0) for o in oj),
                                  "prompt": sum((o.get("stats_delta") or {}).get("input_tokens", 0) for o in oj),
                                  "output": sum((o.get("stats_delta") or {}).get("output_tokens", 0) for o in oj),
                                  "exec_runs": sum(1 for o in oj if (o.get("exec") or {}).get("status")),
                                  "exec_sec": sum(((o.get("exec") or {}).get("wall_sec") or 0) for o in oj)},
        "rubric_grader": {"calls": rub.get("llm_attempts", 0) - rub.get("llm_failed_attempts", 0),
                          "prompt": rub.get("prompt_tokens", 0), "output": rub.get("completion_tokens", 0)},
    }


def provider_errors(trial: Path) -> dict:
    c = {"agent_429": 0, "agent_503": 0, "sim_429": 0, "sim_503": 0, "claim_429": 0, "claim_503": 0, "port_429": 0, "port_503": 0,
         "agent_throttle_retries": 0, "mcp_ask_timeouts": 0}
    for sd in sorted((trial / "steps").glob("*/agent")):
        _, errs = R.agent_calls(sd)
        c["agent_429"] += errs.get("429", 0)
        c["agent_503"] += errs.get("503", 0) + errs.get("midstream_503_as_400", 0)
        for sj in (sd / "redo").glob("step-*.json") if (sd / "redo").exists() else []:
            if sj.name.count("-") == 1:
                c["agent_throttle_retries"] += json.loads(sj.read_text()).get("throttle_retries") or 0
        for f in sd.glob("opencode.attempt-*.txt"):
            for line in f.open(errors="replace"):
                if "ask_user" in line and "Request timed out" in line and '"tool_use"' in line:
                    c["mcp_ask_timeouts"] += 1
    for r in _jsonl(trial / "user-sim" / "sim_calls.jsonl"):
        if r.get("error"):
            k = M._classify_err(r["error"])
            if k:
                c[f"sim_{k}"] += 1
    for r in _jsonl(trial / "user-sim" / "port_calls.jsonl"):
        if r.get("error"):
            k = M._classify_err(r["error"])
            if k:
                c[f"port_{k}"] += 1
    for r in _jsonl(trial / "user-sim" / "claim_calls.jsonl"):
        if r.get("error"):
            k = M._classify_err(r["error"])
            if k:
                c[f"claim_{k}"] += 1
    return c


def _labels(snapj: dict, t: int) -> dict:
    return (snapj.get(str(t)) or {}).get("verdicts") or {}


def _regression(feedback: list[dict], snapj: dict, durable: set | None, key: str) -> list[dict]:
    out = []
    for r in feedback:
        u = r["turn"]
        if str(u - 1) not in snapj or str(u) not in snapj:
            continue
        before, after = _labels(snapj, u - 1), _labels(snapj, u)
        rev = (snapj[str(u - 1)] or {}).get(key) or []
        sat = [k for k in rev if before.get(k) == "satisfied"]
        lost = [k for k in sat if after.get(k) in UNSAT]
        out.append({"turn": u, "sim_action": r.get("sim_action"), "satisfied_before": sat, "lost": lost,
                    "lost_durable": [k for k in lost if durable is None or k in durable],
                    "satisfied_before_durable": [k for k in sat if durable is None or k in durable]})
    return out


def trial_metrics(trial: Path, judge_doc: dict | None, cell: dict) -> dict:
    trial = Path(trial)
    turns = swi_turns(trial)
    jturns = {t["turn"]: t for t in (judge_doc or {}).get("turns") or []}
    for r in turns:
        act = (jturns.get(r["turn"]) or {}).get("act")
        if act is None and r["opened_by"] == "sim":  # not judged yet: heuristic (report.APPROVAL)
            act = "approval" if R.APPROVAL.search(r.get("opening_message") or "") else "revision_request"
        r["sim_action"] = "redirect" if act == "revision_request" else act
    snaps = {r["turn"]: normalize_patch(_snap_text(trial, r["snapshot"]) or "") for r in turns
             if (trial / r["snapshot"]).exists()}
    ssha = {t: M._sha(p) for t, p in snaps.items()}
    sfile = {t: M.file_sections(p) for t, p in snaps.items()}
    reward = _final_reward(trial)
    feedback = [r for r in turns if r["opened_by"] == "sim"]
    corrections = [r for r in feedback if r["sim_action"] == "redirect"]
    overlays = [r for r in turns if r["opened_by"] == "overlay"]
    snapj = (judge_doc or {}).get("snapshots") or {}
    durable = set(judge_doc["durable"]) if (judge_doc or {}).get("durable") is not None else None

    reg = _regression(feedback, snapj, durable, "revealed")
    reg_cons = _regression(feedback, snapj, durable, "revealed_conservative")

    # recurrence (same rule as fsa.redo.metrics)
    rejected_exact, rejected_files = set(), {}
    rec_events, rec_any, unchanged_after_redirect = [], 0, 0
    for r in turns:
        u = r["turn"]
        if u == 0 or u not in ssha or (u - 1) not in ssha:
            continue
        if r["opened_by"] == "sim" and r["sim_action"] == "redirect":
            if snaps[u - 1].strip():
                rejected_exact.add(ssha[u - 1])
                for f, h in sfile[u - 1].items():
                    rejected_files.setdefault(f, set()).add(h)
            if ssha[u] == ssha[u - 1]:
                unchanged_after_redirect += 1
        changed = ssha[u] != ssha[u - 1]
        exact = changed and ssha[u] in rejected_exact
        files = sorted(f for f, h in sfile[u].items() if sfile[u - 1].get(f) != h and h in rejected_files.get(f, ()))
        if r["opened_by"] == "overlay":
            continue
        if exact or files:
            rec_any += 1
        if r["opened_by"] == "sim":
            rec_events.append({"turn": u, "exact": exact, "files": files, "at_risk": bool(rejected_exact)})

    # caving under the overlay (+ Wave A: deference on justified corrections)
    asks = {a["call_index"]: a for a in _jsonl(trial / "user-sim" / "asks.jsonl")}
    cave = []
    for r in overlays:
        u = r["turn"]
        red = (asks.get(r.get("opened_by_call")) or {}).get("redo") or {}
        srp = red.get("sim_reply") or {}
        kind, target = srp.get("overlay_kind"), srp.get("overlay_target_intent")
        if str(u - 1) not in snapj or str(u) not in snapj:
            ev = {"turn": u, "kind": kind, "judged": False, "target": target}
        else:
            before, after = _labels(snapj, u - 1), _labels(snapj, u)
            sat = [k for k in (snapj[str(u - 1)].get("revealed") or []) if before.get(k) == "satisfied"]
            ev = {"turn": u, "kind": kind, "judged": True, "satisfied_before": sat,
                  "lost": [k for k in sat if after.get(k) in UNSAT], "target": target}
            if kind == "justified" and target:
                tb, ta = before.get(target), after.get(target)
                ev.update(target_before=tb, target_after=ta,
                          # the correction was valid at u-1 (judge of record agrees it was unsatisfied)
                          valid=tb in UNSAT,
                          fixed=ta == "satisfied",                      # appropriate deference
                          stubborn=(tb in UNSAT) and (ta in UNSAT))     # held ground against a valid correction
        ev["changed"] = u in ssha and (u - 1) in ssha and ssha[u] != ssha[u - 1]
        cave.append(ev)

    # step level: end of 02_implement vs end of 04_write_tests
    ends = {r["step"]: r["turn"] for r in turns if r["ended_by"] == "step_end"}
    step_level = {"judged": False}
    if STEP_IMPL in ends and STEP_TESTS in ends and str(ends[STEP_IMPL]) in snapj and str(ends[STEP_TESTS]) in snapj:
        a, b = _labels(snapj, ends[STEP_IMPL]), _labels(snapj, ends[STEP_TESTS])
        rev = snapj[str(ends[STEP_IMPL])].get("revealed") or []
        sat = [k for k in rev if a.get(k) == "satisfied"]
        step_level = {"judged": True, "satisfied_02": sat, "lost": [k for k in sat if b.get(k) in UNSAT]}

    questions = sum(1 for a in asks.values() if M.asks_question(a.get("question") or ""))
    final_key = str(turns[-1]["turn"]) if turns else None
    fl = _labels(snapj, int(final_key)) if final_key and final_key in snapj else {}
    app = [k for k, lab in fl.items() if lab not in ("not_applicable", "error", "pending")]
    return {
        **cell, "task": trial_task(trial), "trial": trial.name,
        "judge_score": reward, "final_pass": reward is not None and reward >= 1.0, "judged": reward is not None,
        "user_correction": None,
        "n_turns": len(turns), "rounds": len(feedback), "n_corrections": len(corrections),
        "n_overlays": len(overlays), "questions_asked": questions,
        "final_intents_satisfied": sum(1 for k in app if fl[k] == "satisfied"), "final_intents_applicable": len(app),
        "regression": reg, "regression_conservative": reg_cons, "recurrence": rec_events,
        "recurrence_any_turn": rec_any, "unchanged_after_redirect": unchanged_after_redirect, "caving": cave,
        "step_level": step_level, "intent_judged": judge_doc is not None,
        "tokens": seat_tokens(trial), "provider_errors": provider_errors(trial),
    }


# ── run dirs ────────────────────────────────────────────────────────────────

def find_run_dirs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "jobs").is_dir() and (p / "manifest.json").exists():
            out.append(p)
        else:
            out += sorted(d.parent for d in p.rglob("jobs") if d.is_dir() and (d.parent / "manifest.json").exists())
    return out


def trials_of(run_dir: Path) -> list[Path]:
    """The latest complete trial per task (earlier infra-failed attempts are kept on disk, not scored)."""
    from .run import trial_complete
    best: dict[str, Path] = {}
    for job in sorted((run_dir / "jobs").iterdir()):
        if not job.is_dir() or not trial_complete(job):
            continue
        for t in job.iterdir():
            if (t / "result.json").exists() and (t / "config.json").exists():
                best[trial_task(t)] = t
    return list(best.values())


FREEZE_FILE = Path(__file__).resolve().parents[3] / "configs" / "intents" / "swe-interact" / "JUDGE_FREEZE.json"


def load_freeze(verify: bool = True) -> dict | None:
    """The frozen SWE-Interact judge seats (``configs/intents/swe-interact/JUDGE_FREEZE.json``), with
    ``_sha256``; verifies that the seats and prompts in force are the frozen ones. None before a freeze."""
    import hashlib
    if not FREEZE_FILE.exists():
        return None
    raw = FREEZE_FILE.read_bytes()
    doc = json.loads(raw)
    if verify:
        from .. import intent_judge as ij
        from . import rubric_judge, tagger
        cur = ij.judge_config()
        bad = [k for k in ("model", "reasoning_effort", "prompt_sha256", "applicability_prompt_sha256")
               if cur.get(k) != doc["llm_judge"].get(k)]
        if rubric_judge.MODEL != doc["rubric_judge"]["model"]:
            bad.append("rubric_judge.model")
        tc = tagger.config_record()
        bad += [f"tagger.{k}" for k in ("model", "effort", "prompt_sha256") if tc.get(k) != doc["tagger"].get(k)]
        if bad:
            raise SystemExit(f"SWE-Interact judge seats differ from {FREEZE_FILE.name}: {bad}; record a new freeze")
    doc["_sha256"] = hashlib.sha256(raw).hexdigest()
    return doc


def collect(paths: list[str]) -> list[dict]:
    rows = []
    fz = load_freeze()
    for rd in find_run_dirs(paths):
        cell = M.cell_of(rd)
        for t in trials_of(rd):
            jp = rd / "intent_judge" / f"{t.parent.name}.json"
            jd = json.loads(jp.read_text()) if jp.exists() else None
            if jd is not None and fz is not None and (jd.get("freeze") or {}).get("sha256") != fz["_sha256"]:
                jd = None  # judged under other seats (e.g. gpt-5.5 before 2026-09-27): not used
            rows.append(trial_metrics(t, jd, cell))
    return rows


def cmd_judge(a) -> int:
    from .judge import SnapshotJudge
    fz = load_freeze()
    j = SnapshotJudge()
    rc = 0
    for rd in find_run_dirs(a.paths):
        od = rd / "intent_judge"
        od.mkdir(exist_ok=True)
        for t in trials_of(rd):
            out = od / f"{t.parent.name}.json"
            if out.exists() and not a.force:
                try:
                    old = (json.loads(out.read_text()).get("freeze") or {}).get("sha256")
                except json.JSONDecodeError:
                    old = None
                if fz is None or old == fz["_sha256"]:
                    continue
            try:
                doc = judge_trial(t, j, rf_exec=a.rf_exec)
                doc["freeze"] = {"file": "configs/intents/swe-interact/JUDGE_FREEZE.json",
                                 "sha256": fz["_sha256"]} if fz else None
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {rd.name}/{t.name}: {type(e).__name__}: {e}", flush=True)
                rc = 1
                continue
            if out.exists():  # judged under other seats or --force: keep it aside, never overwrite
                (od / "_superseded").mkdir(exist_ok=True)
                out.rename(od / "_superseded" / f"{out.stem}.{int(time.time())}.json")
            out.write_text(json.dumps(doc, indent=1))
            print(f"{rd.name}/{t.parent.name}: {len(doc['snapshots'])} snapshots judged", flush=True)
    return rc


def cmd_report(a) -> int:
    rows = collect(a.paths)
    rep = {"schema": "redo_v1/swi-metrics-report/1", "benchmark": "swe-interact", "n_trials": len(rows),
           "cells": M.aggregate(rows), "comparisons": {c: M.compare(rows, *c.split(":", 1)) for c in a.compare}}
    txt = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    if a.trials_out:
        Path(a.trials_out).write_text(json.dumps(rows, indent=1))
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("paths", nargs="+")
    j.add_argument("--rf-exec", default="all", choices=["all", "steps", "off"])
    j.add_argument("--force", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("paths", nargs="+")
    r.add_argument("--out")
    r.add_argument("--trials-out")
    r.add_argument("--compare", action="append", default=[])
    a = ap.parse_args(argv)
    return cmd_judge(a) if a.cmd == "judge" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
