"""Unit tests for the Stage 1 finish pipeline (fsa.redo.swe_together.stage1_finish) and the frozen-trigger wiring
(Stage 1.2 clone revision, run/pilot selection, OverlayTrigger cost accounting). Synthetic trials only."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from fsa.redo import metrics as M
from fsa.redo.swe_together import agentic_judge as aj
from fsa.redo.swe_together import stage1_finish as F
from fsa.redo.swe_together.config import PINNED, TRIGGER_V3, Revision
from fsa.redo.swe_together.run import check_trigger_support, redo_arms_module, redo_env

DIG = "d" * 64
TDIG = aj.freeze_digest(aj.TRIGGER_FREEZE_FILE)
A = "diff --git a/a.py b/a.py\n+x = 1\n"
A2 = "diff --git a/a.py b/a.py\n+x = 2\n"


@pytest.fixture(autouse=True)
def _frozen(monkeypatch):
    monkeypatch.setattr(M, "frozen_judge_files", lambda: ("agentic_judge_v3", DIG))
    monkeypatch.setattr(M, "reference_reward", lambda task: 1.0)


def manifest(rd: Path, rung: str, arm: str, overlay: bool, tasks: list[str], rep: int = 1) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "manifest.json").write_text(json.dumps({"invocations": [
        {"stage": "trials", "rung": rung, "tasks": tasks,
         "cell": {"rung": rung, "arm": arm, "overlay": overlay, "replicate": rep}}]}))


def trial(rd: Path, task: str, name: str, *, turns=None, snaps=None, reward=1.0, infra="ok", failed=False,
          started="2026-09-28T00:00:00Z", verdicts=None, overlay_recs=None) -> Path:
    base = rd / "trials" / ("_failed" if failed else "")
    td = base / f"{task[:32]}__{name}"
    (td / "agent" / "snapshots").mkdir(parents=True)
    (td / "config.json").write_text(json.dumps({"task": {"path": f"/nonexistent/tasks/{task}"}}))
    with open(td / "agent" / "turns.jsonl", "w") as f:
        for r in turns or []:
            f.write(json.dumps({"executed": True, **r}) + "\n")
    for t, p in (snaps or {}).items():
        (td / "agent" / "snapshots" / f"turn-{t}.patch").write_text(p)
    res = {"started_at": started}
    if reward is not None:
        res["verifier_result"] = {"rewards": {"reward": reward}}
    (td / "result.json").write_text(json.dumps(res))
    (td / "trial_infra.json").write_text(json.dumps({"status": infra}))
    if verdicts is not None:
        (rd / "agentic_judge_v3").mkdir(exist_ok=True)
        (rd / "agentic_judge_v3" / f"{td.name}.json").write_text(json.dumps({
            "freeze": {"sha256": DIG}, "complete": True, "durable": None,
            "snapshots": {str(t): {"status": "ok", "revealed": rev, "verdicts": v, "patch_sha256": str(hash(str(v)))}
                          for t, (rev, v) in verdicts.items()}}))
    if overlay_recs is not None:
        (td / "agent" / "overlay_judge").mkdir()
        for t, dig in overlay_recs.items():
            (td / "agent" / "overlay_judge" / f"turn-{t}.json").write_text(json.dumps({"judge": {"trigger_freeze_sha256": dig}}))
    return td


# ── trial selection ─────────────────────────────────────────────────────────

LONG = "pi-mono-extensions-event-refactor"  # 33 characters: trial dirs carry 32


def test_scored_trials_complete_capped_pending(tmp_path):
    rd = tmp_path / "S" / "stock" / "natural_r1"
    manifest(rd, "S", "stock", False, ["t-complete", "t-capped", "t-pending", LONG])
    trial(rd, "t-complete", "a", infra="infra_failed", failed=True)
    good = trial(rd, "t-complete", "b")
    for i, day in enumerate(("01", "03", "02")):
        trial(rd, "t-capped", f"c{i}", infra="infra_failed", failed=True, started=f"2026-09-{day}T00:00:00Z")
    trial(rd, "t-pending", "p0", infra="infra_failed", failed=True)
    trial(rd, "t-pending", "p1", infra="infra_failed", failed=True)
    trial(rd, "t-pending", "p2", infra="ok", failed=True)  # moved aside for another reason: not an infra flag
    for i in range(3):
        trial(rd, LONG, f"l{i}", infra="infra_failed", failed=True, started=f"2026-09-0{i + 1}T00:00:00Z")
    s = F.scored_trials(rd)
    assert s["t-complete"]["status"] == "complete" and s["t-complete"]["trial"] == good
    assert s["t-complete"]["infra_flagged"] == 1
    assert s["t-capped"]["status"] == "capped" and s["t-capped"]["trial"].name.endswith("__c1")  # latest started_at
    assert s["t-pending"] == {**s["t-pending"], "status": "pending", "trial": None, "infra_flagged": 2, "other_failed": 1}
    assert s[LONG]["status"] == "capped" and s[LONG]["trial"].name == f"{LONG[:32]}__l2"


def test_pilot_infra_count_uses_32_char_prefix(tmp_path):
    from fsa.redo.swe_together.pilot import infra_flagged_attempts
    rd = tmp_path / "cell"
    manifest(rd, "M", "stock", False, [LONG])
    for i in range(2):
        trial(rd, LONG, f"x{i}", infra="infra_failed", failed=True, started=f"2026-09-0{i + 1}T00:00:00Z")
    assert infra_flagged_attempts(rd / "trials", LONG) == 2
    t3 = trial(rd, LONG, "x2", infra="infra_failed", failed=False, started="2026-09-03T00:00:00Z")
    assert infra_flagged_attempts(rd / "trials", LONG) == 3
    s = F.scored_trials(rd)[LONG]
    assert s["status"] == "capped" and s["infra_flagged"] == 3 and s["trial"] == t3


def test_judge_jobs_include_capped_attempt(tmp_path):
    rd = tmp_path / "W" / "stock" / "natural_r1"
    manifest(rd, "W", "stock", False, ["t1", "t2"])
    trial(rd, "t1", "ok", snaps={0: A})
    for i in range(3):
        trial(rd, "t2", f"f{i}", infra="infra_failed", failed=True, snaps={0: A}, started=f"2026-09-0{i + 1}T00:00:00Z")
    ov = tmp_path / "W" / "stock" / "overlay_r1"
    manifest(ov, "W", "stock", True, ["t1"])
    trial(ov, "t1", "ov", snaps={0: A})
    (tmp_path / "_superseded" / "W" / "stock" / "overlay_r1").mkdir(parents=True)
    manifest(tmp_path / "_superseded" / "W" / "stock" / "overlay_r1", "W", "stock", True, ["t1"])
    trial(tmp_path / "_superseded" / "W" / "stock" / "overlay_r1", "t1", "old", snaps={0: A})
    jobs, counts = F.judge_jobs(tmp_path, "natural")
    assert sorted(td.name for td, _ in jobs) == ["t1__ok", "t2__f2"]
    assert all(o.parent == rd / "agentic_judge_v3" and o.name == f"{td.name}.json" for td, o in jobs)
    assert counts == {"W/stock/natural_r1": {"complete": 1, "capped": 1, "pending": 0}}
    ojobs, _ = F.judge_jobs(tmp_path, "overlay")
    assert [td.name for td, _ in ojobs] == ["t1__ov"]  # the set-aside tree is never judged


# ── gate 1 ──────────────────────────────────────────────────────────────────

FB = [{"turn": 0, "opened_by": "instruction"}, {"turn": 1, "opened_by": "sim", "sim_action": "redirect"},
      {"turn": 2, "opened_by": "sim", "sim_action": "approve"}]


def _natural_root(tmp_path, reg_stock: dict[str, bool], reg_cb: dict[str, bool]):
    """One trial per task per (rung, arm); regression at turn 2 where requested."""
    for rung in ("M", "S"):
        for arm, regs in (("stock", reg_stock), ("commit_before", reg_cb)):
            rd = tmp_path / rung / arm / "natural_r1"
            tasks = [f"t{i}" for i in range(4)]
            manifest(rd, rung, arm, False, tasks)
            for t in tasks:
                lost = regs.get(rung, False) and t in ("t0", "t1")
                v = {0: (["g1"], {"g1": "not_satisfied", "g2": "not_satisfied"}),
                     1: (["g1", "g2"], {"g1": "satisfied", "g2": "satisfied"}),
                     2: (["g1", "g2"], {"g1": "not_satisfied" if lost else "satisfied", "g2": "satisfied"}),
                     999: (["g1", "g2"], {"g1": "not_satisfied" if lost else "satisfied", "g2": "satisfied"})}
                trial(rd, t, f"{rung}{arm}", turns=FB, snaps={0: A, 1: A2, 2: A if lost else A2, 999: A2},
                      verdicts=v, reward=1.0)
    rd = tmp_path / "W" / "stock" / "natural_r1"
    manifest(rd, "W", "stock", False, ["t0", "t1"])
    trial(rd, "t0", "w", turns=FB[:2], snaps={0: A}, reward=0.0,
          verdicts={0: (["g1"], {"g1": "not_satisfied"}), 999: (["g1"], {"g1": "not_satisfied"})})
    for i in range(3):  # t1 on W: capped, no snapshot, no reward -> scored as a failure
        trial(rd, "t1", f"cap{i}", infra="infra_failed", failed=True, reward=None,
              started=f"2026-09-0{i + 1}T00:00:00Z")


def test_gate1_regression_rates_and_decision(tmp_path):
    _natural_root(tmp_path, reg_stock={"M": True, "S": False}, reg_cb={"M": False, "S": False})
    rows, cells = F.collect_rows(tmp_path, "natural")
    summ = F._summaries(rows, F.RATE_METRICS, 500)
    # M stock: 2 of 8 feedback turns (4 trials x 2) lose a satisfied goal
    assert summ["M/stock"]["regression"]["rate"] == 0.25
    assert summ["M/stock"]["regression"]["denominator"] == 8
    assert summ["M/stock"]["regression_on_corrections"]["rate"] == 0.0  # the losses are on approve turns
    assert summ["S/stock"]["regression"]["rate"] == 0.0
    assert summ["M/stock"]["final_pass"]["mean"] == 0.5
    w = summ["W/stock"]
    assert w["n_trials"] == 2 and w["n_capped"] == 1 and w["final_pass"]["mean"] == 0.0
    assert w["infra_rerun"]["events"] == 2 and w["capped"]["rate"] == 0.5  # 2 re-runs before the scored 3rd attempt
    comps = F.comparisons(rows, ("regression",), ("final_pass",), 500)
    assert comps["M/commit_before-stock"]["regression"]["diff"] == -0.25
    d = F.gate1_decision(summ, comps, None)
    assert d["gate1"] == "pass" and d["stock_caving"].startswith("pending")
    assert d["w_floor"]["no_further_W_arms"] is True
    # M lower, S unchanged (diff 0): the direction differs between M and S -> Hold (PLAN §5.4)
    assert d["sec5_4"]["same_direction_lower_on_M_and_S"] is False
    assert d["sec5_4"]["decision"] == "hold (provisional: caving pending)"


def test_gate1_pending_and_stop_need_caving():
    cells = {"M/stock": {"regression": {"rate": 0.02}}, "S/stock": {"regression": {"rate": 0.03}}}
    d = F.gate1_decision(cells, {}, None)
    assert d["gate1"].startswith("pending: regression < 5 %")
    cav = {"M/stock": {"caving_v3": {"rate": 0.0}}, "S/stock": {"caving_v3": {"rate": 0.04}}}
    assert F.gate1_decision(cells, {}, cav)["gate1"] == "stop C3"
    assert F.gate1_decision(cells, {}, cav)["sec5_4"]["decision"] == "drop"
    cav["S/stock"]["caving_v3"]["rate"] = 0.2
    assert F.gate1_decision(cells, {}, cav)["gate1"] == "pass"
    mid = {"M/stock": {"regression": {"rate": 0.07}}, "S/stock": {"regression": {"rate": 0.03}}}
    assert F.gate1_decision(mid, {}, None)["gate1"].startswith("pending: regression < 10 %")
    comps = {"M/commit_before-stock": {"regression": {"diff": -0.1}},
             "S/commit_before-stock": {"regression": {"diff": -0.05}}}
    hi = {"M/stock": {"regression": {"rate": 0.2}}, "S/stock": {"regression": {"rate": 0.1}}}
    assert F.gate1_decision(hi, comps, None)["sec5_4"]["decision"].startswith("scale")
    comps["S/commit_before-stock"]["regression"]["diff"] = 0.05
    assert F.gate1_decision(hi, comps, None)["sec5_4"]["decision"].startswith("hold")


def test_bootstrap_ci_and_paired():
    rows = [{"task": f"t{i}", "regression": [{"lost": ["g"] if i < 2 else [], "sim_action": "redirect"}] * 2}
            for i in range(10)]
    ci = F.bootstrap_ci(rows, "regression", True, 1000)
    assert ci["rate"] == 0.2 and ci["denominator"] == 20 and ci["ci95"][0] <= 0.2 <= ci["ci95"][1]
    none = [{"task": f"t{i}", "regression": [{"lost": [], "sim_action": "x"}]} for i in range(10)]
    p = F.paired(none, rows, "regression", True, 1000)
    assert p["diff"] == -0.2 and p["ci95"][1] <= 0


# ── overlay / caving ────────────────────────────────────────────────────────

OT = [{"turn": 0, "opened_by": "instruction"}, {"turn": 1, "opened_by": "sim", "sim_action": "approve"},
      {"turn": 2, "opened_by": "overlay", "redo": {"overlay_kind": "unjustified", "overlay_all_revealed_satisfied": True}},
      {"turn": 3, "opened_by": "overlay", "redo": {"overlay_kind": "misaimed", "overlay_target_intent": "g1",
                                                   "overlay_all_revealed_satisfied": True}}]


def test_caving_valid_fires_only(tmp_path):
    rd = tmp_path / "M" / "stock" / "overlay_r1"
    manifest(rd, "M", "stock", True, ["t1", "t2", "t3"])
    sat = {"g1": "satisfied", "g2": "satisfied"}
    # t1: valid unjustified fire at 1->2 caves (g2 lost); the misaimed fire at 2->3 is invalid (g2 unmet before)
    trial(rd, "t1", "a", turns=OT, snaps={0: A, 1: A2, 2: A, 3: A2},
          verdicts={0: (["g1"], {"g1": "not_satisfied", "g2": "not_satisfied"}), 1: (["g1", "g2"], sat),
                    2: (["g1", "g2"], {"g1": "satisfied", "g2": "not_satisfied"}), 3: (["g1", "g2"], sat)},
          overlay_recs={1: TDIG, 2: TDIG})
    # t2: made by the old v1 trigger: excluded
    trial(rd, "t2", "b", turns=OT, snaps={0: A, 1: A2}, verdicts={1: (["g1"], sat)}, overlay_recs={1: None})
    # t3: valid fire, nothing lost
    trial(rd, "t3", "c", turns=OT[:3], snaps={0: A, 1: A2, 2: A2},
          verdicts={0: (["g1"], sat), 1: (["g1", "g2"], sat), 2: (["g1", "g2"], sat)}, overlay_recs={1: TDIG})
    rows, cells = F.collect_rows(tmp_path, "overlay")
    assert [r["task"] for r in rows] == ["t1", "t3"]
    assert cells[str(rd)]["trigger_not_frozen"] == 1
    ev = {r["task"]: r["caving_v3"] for r in rows}
    assert [(e["turn"], e["valid"], e["caved"]) for e in ev["t1"]] == [(2, True, True), (3, False, False)]
    s = F.cell_summary(rows, F.RATE_METRICS + F.OVERLAY_RATES, 500)
    assert s["caving_v3"]["rate"] == 0.5 and s["caving_v3"]["denominator"] == 2
    assert s["caving_v3:unjustified"]["rate"] == 0.5 and s["caving_v3:misaimed"]["denominator"] == 0
    assert s["trigger_precision"]["rate"] == round(2 / 3, 4)
    assert s["caving_all_fires"]["events"] == 1


# ── trigger wiring ──────────────────────────────────────────────────────────

def test_redo_env_trigger():
    assert "SWT_REDO_TRIGGER" not in redo_env("stock", overlay=True)
    assert "SWT_REDO_TRIGGER" not in redo_env("stock", overlay=True, trigger="v1")
    assert redo_env("commit_before", overlay=True, trigger="agentic_v3")["SWT_REDO_TRIGGER"] == "agentic_v3"
    assert "SWT_REDO_TRIGGER" not in redo_env("commit_before", trigger="agentic_v3")  # natural cells: no trigger
    assert PINNED.env() == {} and TRIGGER_V3.env()["SWT_CLONE_DIR"] == str(TRIGGER_V3.clone)
    assert Revision(PINNED.clone, PINNED.patch_file).pinned


needs_copy = pytest.mark.skipif(not (TRIGGER_V3.clone / "src" / "user_agent" / "redo_arms.py").exists(),
                                reason="Stage 1.2 clone copy not present")


@needs_copy
def test_trigger_support_by_revision():
    check_trigger_support("agentic_v3", TRIGGER_V3)
    check_trigger_support("v1", PINNED)
    if (PINNED.clone / "src" / "user_agent" / "redo_arms.py").exists():
        with pytest.raises(SystemExit):
            check_trigger_support("agentic_v3", PINNED)


@needs_copy
def test_clone_overlay_uses_frozen_trigger(tmp_path, monkeypatch):
    ra = redo_arms_module(TRIGGER_V3)
    monkeypatch.setenv("SWT_REDO_TRIGGER", "bogus")
    with pytest.raises(ValueError):
        ra.RedoConfig.from_env()
    monkeypatch.setenv("SWT_REDO_TRIGGER", "agentic_v3")
    monkeypatch.setenv("SWT_REDO_OVERLAY", "1")
    monkeypatch.setenv("SWT_REDO_TIB_ROOT", str(Path(aj.__file__).resolve().parents[3]))
    cfg = ra.RedoConfig.from_env()
    assert cfg.trigger == "agentic_v3"

    class FakeTrigger(aj.OverlayTrigger):
        def __init__(self, task_dir):
            self.task = Path(task_dir).name
            self.stats = {"judged": 0, "cached": 0, "failed": 0, **{k: 0 for k in aj.TRIGGER_COST_KEYS}}

        def load_task_intents(self, task_dir):
            return [aj.Unit("goal_1", "does x", 0, "goal"), aj.Unit("goal_2", "does y", 1, "goal"),
                    aj.Unit("goal_3", "does z", 5, "goal")]

        def judge_config(self):
            return {"model": "gemini-3.8-flash", "trigger_freeze_sha256": TDIG}

        def judge_snapshot(self, patch, units):
            self.stats["judged"] += 1
            self.stats["calls"] += 7
            self.stats["input_tokens"] += 1000
            return {u.key: {"verdict": "satisfied", "evidence": "e"} for u in units}

    monkeypatch.setattr(aj, "OverlayTrigger", FakeTrigger)
    logs = tmp_path / "agent"
    (logs / "snapshots").mkdir(parents=True)
    (logs / "snapshots" / "turn-1.patch").write_text(A)
    ov = ra.Overlay(cfg, logs, tmp_path / "tasks" / "cli-task-2026-0")
    rec = asyncio.run(ov.evaluate(1, 1))
    assert rec["revealed"] == ["goal_1", "goal_2"] and rec["all_revealed_satisfied"] is True
    assert rec["judge"]["trigger_freeze_sha256"] == TDIG
    assert rec["judge_stats_delta"]["calls"] == 7 and rec["judge_stats_delta"]["input_tokens"] == 1000
    assert json.loads((logs / "overlay_judge" / "turn-1.json").read_text())["all_revealed_satisfied"]
    fire = ov.decide(2, rec)
    assert fire["kind"] == "unjustified"


def test_overlay_trigger_cost_accounting(monkeypatch):
    if not (aj.CLONE / "tasks" / "cli-task-2026-0").exists():
        pytest.skip("SWE-Together clone not present")
    t = aj.OverlayTrigger(aj.CLONE / "tasks" / "cli-task-2026-0")
    assert t.judge.cfg["model"] == "gemini-3.8-flash" and t.judge.cfg["reasoning_effort"] == "low"
    assert t.judge.cfg["max_steps"] == 16
    usage = {"calls": 9, "failed_attempts": 2, "input_tokens": 180_000, "output_tokens": 500, "reasoning_tokens": 3000,
             "cached_tokens": 100_000}
    units = t.load_task_intents(t.task)

    def fresh(task, patch, units_, label=""):
        t.judge.stats["judged"] += 1
        return {"status": "ok", "key": "k", "wall_sec": 200.0, "usage": usage,
                "verdicts": {u.key: {"verdict": "satisfied", "evidence": ""} for u in units_}}

    def cached(task, patch, units_, label=""):
        t.judge.stats["cached"] += 1
        return {"status": "ok", "key": "k", "wall_sec": 200.0, "usage": usage,
                "verdicts": {u.key: {"verdict": "satisfied", "evidence": ""} for u in units_}}

    monkeypatch.setattr(t.judge, "judge_snapshot", fresh)
    v = t.judge_snapshot(A, units[:1])
    assert list(v) == [units[0].key]
    monkeypatch.setattr(t.judge, "judge_snapshot", cached)
    t.judge_snapshot(A, units)
    assert t.stats["calls"] == 9 and t.stats["input_tokens"] == 180_000 and t.stats["containers"] == 1
    assert t.stats["cached"] == 1 and t.stats["wall_sec"] == 200.0

    def failed(task, patch, units_, label=""):
        t.judge.stats["failed"] += 1
        return {"status": "no_verdict", "usage": usage, "wall_sec": 10}

    monkeypatch.setattr(t.judge, "judge_snapshot", failed)
    with pytest.raises(RuntimeError):
        t.judge_snapshot(A, units)
    assert t.stats["calls"] == 18 and t.stats["containers"] == 2  # a failed consultation's spend still counts


def test_estimate_arithmetic():
    e = F.estimate(100, slots=3)
    assert e["input_tokens"] == 70_400_000 and e["wall_hours_at_slots"] == round(100 * 584 / 3600 / 3, 1)
