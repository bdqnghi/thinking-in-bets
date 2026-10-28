"""Unit tests for the agentic snapshot judge v3 (fsa.redo.swe_together.agentic_judge) and its validation
helpers, with a fake container and a fake model (no docker, no network)."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from fsa.redo.swe_together import agentic_judge as aj
from fsa.redo.swe_together import judge_v3_validate as jv


@pytest.mark.skipif(not (aj.CLONE / "tasks").exists(), reason="SWE-Together clone not present")
def test_goal_units_exclude_floor_zero_weight_and_chat_goals():
    units = aj.goal_units("sd-scripts-fp8-lumina")
    keys = {u.key for u in units}
    assert "goal_0" not in keys                     # anti-gaming floor
    assert not keys & {"goal_7", "goal_8"}          # chat-only
    assert all(u.weight and u.weight > 0 for u in units)
    assert all(u.kind == "goal" for u in units)
    # every PRESERVE / CHAT goal id exists in its task's rubric
    for table in (aj.PRESERVE_GOALS, aj.CHAT_ONLY_GOALS):
        for task, ids in table.items():
            have = {g["id"] for g in aj.all_goals(task)}
            assert set(ids) <= have, (task, ids)


@pytest.mark.skipif(not (aj.CLONE / "tasks").exists(), reason="SWE-Together clone not present")
def test_units_digest_changes_with_text():
    u = aj.goal_units("cli-task-2026-0")
    d = aj.units_digest(u)
    u2 = [aj.Unit(**{**x.__dict__, "text": x.text + " "}) if i == 0 else x for i, x in enumerate(u)]
    assert aj.units_digest(u2) != d
    assert aj.units_digest(aj.goal_units("cli-task-2026-0")) == d


def test_validate_submit():
    ids = ["g1", "g2"]
    err, _ = aj.AgenticJudge._validate({"verdicts": [{"id": "g1", "label": "satisfied", "evidence": "x"}]}, ids, aj.LABELS)
    assert "missing" in err
    err, _ = aj.AgenticJudge._validate({"verdicts": [{"id": "g1", "label": "not_applicable", "evidence": ""},
                                                     {"id": "g2", "label": "satisfied", "evidence": ""}]}, ids, aj.LABELS)
    assert "label" in err
    err, _ = aj.AgenticJudge._validate({"verdicts": [{"id": "g1", "label": "satisfied", "evidence": ""},
                                                     {"id": "g1", "label": "satisfied", "evidence": ""}]}, ids, aj.LABELS)
    assert "twice" in err
    err, out = aj.AgenticJudge._validate({"verdicts": [{"id": "g1", "label": "satisfied", "evidence": "a:1"},
                                                       {"id": "g2", "label": "not_addressed", "evidence": ""}]}, ids,
                                         aj.LABELS)
    assert err is None and out["verdicts"]["g1"] == {"verdict": "satisfied", "evidence": "a:1"}


def test_clip():
    assert aj.clip("abc", 10) == "abc"
    c = aj.clip("x" * 100 + "END", 30)
    assert c.endswith("END") and "clipped" in c


def test_kappa_and_bootstrap():
    k = jv.kappa([(True, True)] * 8 + [(False, False)] * 8 + [(True, False)] * 2 + [(False, True)] * 2)
    assert k["agreement"] == 0.8 and abs(k["cohen_kappa"] - 0.6) < 1e-9
    rows = [{"snapshot": f"s{i // 4}", "judge": "satisfied" if i % 3 else "not_satisfied",
             "ref": "satisfied" if i % 3 else "not_addressed"} for i in range(40)]
    b = jv.cluster_bootstrap(rows, "snapshot", n_boot=50)
    assert b["clusters"] == 10 and b["agreement_ci95"] == [1.0, 1.0]


# ── the tool loop with a fake container and a fake model ───────────────────

class FakeSandbox:
    ran: list = []

    def __init__(self, task):
        self.task, self.repo, self.name = task, "/repo", "fake"

    def start(self):
        pass

    def apply(self, patch):
        return True, "applied"

    def run(self, cmd, timeout):
        FakeSandbox.ran.append(cmd)
        return 0, "ok output", 0.01

    def close(self):
        pass


def _tc(i, name, args):
    return types.SimpleNamespace(id=f"c{i}", function=types.SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeMsg:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls, self.content = tool_calls, content

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": t.id, "type": "function", "function": {"name": t.function.name,
                                                                             "arguments": t.function.arguments}}
                               for t in self.tool_calls]
        return {k: v for k, v in d.items() if v is not None}


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.seen.append(kw)
        msg = self.script.pop(0)
        usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=130,
                                      completion_tokens_details=None, prompt_tokens_details=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")], usage=usage,
                                     model="fake", id="r")


@pytest.fixture()
def fake_env(monkeypatch, tmp_path):
    monkeypatch.setattr(aj, "Sandbox", FakeSandbox)
    monkeypatch.setattr(aj.exec_judge, "image", lambda task: f"fake/{task}")
    monkeypatch.setattr(aj.exec_judge, "image_id", lambda task: ("sha256:img", "arm64"))
    monkeypatch.setattr(aj, "RESULTS", tmp_path)
    return tmp_path


def test_session_bash_then_bad_then_good_submit(fake_env):
    units = [aj.Unit("g1", "do a", 0, "goal", "core", 0.5), aj.Unit("g2", "do b", 0, "goal", "core", 0.5)]
    client = FakeClient([
        FakeMsg([_tc(1, "bash", {"command": "git diff HEAD"})]),
        FakeMsg(content="thinking out loud"),                                   # nudged
        FakeMsg([_tc(2, "submit_verdict", {"verdicts": [{"id": "g1", "label": "satisfied", "evidence": "a"}]})]),
        FakeMsg([_tc(3, "submit_verdict", {"verdicts": [{"id": "g1", "label": "satisfied", "evidence": "a"},
                                                       {"id": "g2", "label": "not_addressed", "evidence": "b"}]})]),
    ])
    j = aj.AgenticJudge({"slots": 1}, cache_dir=fake_env / "cache", client=client)
    rec = j.judge_snapshot("cli-task-2026-0", "diff --git a/x b/x\n+1\n", units)
    assert rec["status"] == "ok"
    assert rec["verdicts"]["g2"]["verdict"] == "not_addressed"
    assert rec["steps"] == 4 and rec["nudges"] == 1 and rec["bad_submits"] == 1
    assert "git diff HEAD" in FakeSandbox.ran
    assert rec["usage"]["calls"] == 4 and rec["usage"]["reasoning_tokens"] == 4 * 20
    # cached: a second call makes no model call
    rec2 = j.judge_snapshot("cli-task-2026-0", "diff --git a/x b/x\n+1\n", units)
    assert rec2["key"] == rec["key"] and j.stats["cached"] == 1
    # a different config is a different cache key
    j2 = aj.AgenticJudge({"slots": 1, "reasoning_effort": "low"}, cache_dir=fake_env / "cache", client=client)
    assert j2.key("cli-task-2026-0", "p", units) != j.key("cli-task-2026-0", "p", units)


def test_session_forced_submit_at_budget(fake_env):
    units = [aj.Unit("g1", "do a", 0, "goal", "core", 1.0)]
    script = [FakeMsg([_tc(i, "bash", {"command": f"echo {i}"})]) for i in range(3)]
    script.append(FakeMsg([_tc(9, "submit_verdict", {"verdicts": [{"id": "g1", "label": "not_satisfied", "evidence": "x"}]})]))
    client = FakeClient(script)
    j = aj.AgenticJudge({"slots": 1, "max_steps": 5}, cache_dir=fake_env / "cache", client=client)
    rec = j.judge_snapshot("cli-task-2026-0", "", units)
    assert rec["status"] == "ok" and rec["forced_submit"]
    last_tools = [t["function"]["name"] for t in client.seen[-1]["tools"]]
    assert last_tools == ["submit_verdict"]


def test_judge_trial_schema(fake_env, monkeypatch):
    td = fake_env / "cell" / "trials" / "cli-task-2026-0__x"
    (td / "agent" / "snapshots").mkdir(parents=True)
    (td / "config.json").write_text(json.dumps({"task": {"path": "/nonexistent/tasks/cli-task-2026-0"}}))
    (td / "agent" / "turns.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"turn": 0, "opened_by": "instruction", "executed": True},
        {"turn": 1, "opened_by": "sim", "executed": True}]) + "\n")
    for t in (0, 1, 999):
        (td / "agent" / "snapshots" / f"turn-{t}.patch").write_text(f"diff --git a/x b/x\n+{t}\n")

    class J:
        cfg = {"unit": "goals"}
        units = staticmethod(lambda task: [aj.Unit("g1", "a", 0, "goal"), aj.Unit("g2", "b", 1, "goal"),
                                           aj.Unit("g3", "c", 5, "goal")])

        def judge_snapshot(self, task, patch, units, label=""):
            sat = "satisfied" if "+0" in patch or "ref" in patch else "not_satisfied"
            return {"status": "ok", "key": "k", "verdicts": {u.key: {"verdict": sat, "evidence": ""} for u in units}}

        def config_record(self):
            return {"v": 3}
    monkeypatch.setattr(aj.exec_judge, "reference_patch", lambda task: "diff --git a/r b/r\n+ref\n")
    doc = aj.judge_trial(td, J())
    assert doc["schema"] == aj.TRIAL_SCHEMA and doc["complete"]
    assert doc["snapshots"]["0"]["revealed"] == ["g1"]
    assert doc["snapshots"]["1"]["revealed"] == ["g1", "g2"]
    assert doc["snapshots"]["999"]["revealed"] == ["g1", "g2", "g3"]
    assert doc["durable"] == ["g1", "g2", "g3"]
    assert doc["snapshots"]["0"]["verdicts"]["g1"] == "satisfied"
    assert doc["snapshots"]["1"]["verdicts"]["g1"] == "not_satisfied"


def test_session_refute_pass(fake_env):
    units = [aj.Unit("g1", "do a", 0, "goal", "core", 0.5), aj.Unit("g2", "do b", 0, "goal", "core", 0.5)]
    client = FakeClient([
        FakeMsg([_tc(1, "submit_verdict", {"verdicts": [{"id": "g1", "label": "satisfied", "evidence": "a"},
                                                       {"id": "g2", "label": "satisfied", "evidence": "b"}]})]),
        FakeMsg([_tc(2, "bash", {"command": "python /tmp/probe.py"})]),
        FakeMsg([_tc(3, "submit_verdict", {"verdicts": [{"id": "g1", "label": "satisfied", "evidence": "probe ok"},
                                                       {"id": "g2", "label": "not_satisfied", "evidence": "probe fails"}]})]),
    ])
    j = aj.AgenticJudge({"slots": 1, "refute": True}, cache_dir=fake_env / "cache", client=client)
    rec = j.judge_snapshot("cli-task-2026-0", "diff --git a/x b/x\n+2\n", units)
    assert rec["status"] == "ok" and rec["refuted"] == ["g2"]
    assert rec["draft_verdicts"]["g2"]["verdict"] == "satisfied" and rec["verdicts"]["g2"]["verdict"] == "not_satisfied"
    assert any("REFUTE" in (m.get("content") or "") for m in client.seen[-1]["messages"] if m["role"] == "user")


def test_refcheck_labels_agreement_and_tiebreak(tmp_path, monkeypatch):
    d = tmp_path / "refcheck"
    d.mkdir()
    monkeypatch.setattr(jv, "REFCHECK", d)
    (d / "ref_units.json").write_text(json.dumps({"tasks": {"t": [{"goal_id": g} for g in ("g1", "g2", "g3")]}}))
    lab = lambda xs: json.dumps({"labels": {f"t/{g}": {"label": l} for g, l in xs.items()}})  # noqa: E731
    (d / "labels_G1a.json").write_text(lab({"g1": "satisfied", "g2": "satisfied", "g3": "not_satisfied"}))
    (d / "labels_G1b.json").write_text(lab({"g1": "satisfied", "g2": "not_satisfied", "g3": "not_satisfied"}))
    rc = jv.refcheck_labels()
    assert rc["unresolved"] == ["t/g2"] and rc["agreement"]["agree"] == 2
    (d / "labels_tiebreak.json").write_text(lab({"g2": "satisfied"}))
    rc = jv.refcheck_labels()
    assert not rc["unresolved"] and rc["final"] == {"t/g1": "satisfied", "t/g2": "satisfied", "t/g3": "not_satisfied"}
