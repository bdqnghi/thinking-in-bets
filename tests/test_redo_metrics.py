"""Unit tests for fsa.redo.metrics / fsa.redo.intent_judge on synthetic trial dirs."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from fsa.redo import intent_judge as ij
from fsa.redo import metrics as M


@pytest.fixture(autouse=True)
def _v1_judge_files(monkeypatch):
    # synthetic run dirs carry v1-style judge files under intent_judge/ (no freeze digest)
    monkeypatch.setattr(M, "frozen_judge_files", lambda: ("intent_judge", None))

A = "diff --git a/a.py b/a.py\n+x = 1\n"
A2 = "diff --git a/a.py b/a.py\n+x = 2\n"
B = "diff --git a/b.py b/b.py\n+y = 1\n"


def make_trial(root: Path, name: str, turns: list[dict], snaps: dict[int, str], task: str = "t1",
               judge: dict | None = None, score: float | None = None, run: str = "cell") -> Path:
    rd = root / run
    td = rd / "trials" / f"{task}__{name}"
    (td / "agent" / "snapshots").mkdir(parents=True)
    (td / "config.json").write_text(json.dumps({"task": {"path": f"/nonexistent/tasks/{task}"}}))
    with open(td / "agent" / "turns.jsonl", "w") as f:
        for r in turns:
            f.write(json.dumps({"executed": True, **r}) + "\n")
    for t, p in snaps.items():
        (td / "agent" / "snapshots" / f"turn-{t}.patch").write_text("=== /repo (cumulative vs harbor-base) ===\n" + p)
    if score is not None:  # the trial's own verifier reward (final pass since 2026-09-27)
        (td / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": score}}}))
    if judge is not None:
        (rd / "intent_judge").mkdir(parents=True, exist_ok=True)
        (rd / "intent_judge" / f"{td.name}.json").write_text(json.dumps(judge))
    if not (rd / "manifest.json").exists():
        (rd / "manifest.json").write_text(json.dumps({"invocations": [
            {"stage": "trials", "rung": "W", "cell": {"rung": "W", "arm": run.split("-")[0],
                                                      "overlay": "overlay" in run, "replicate": 1}}]}))
    return td


def jdoc(per_turn: dict[int, tuple[list[str], dict[str, str]]], durable=None) -> dict:
    return {"durable": durable, "snapshots": {str(t): {"revealed": rev, "verdicts": v}
                                              for t, (rev, v) in per_turn.items()}}


def test_normalize_and_sections():
    assert ij.normalize_patch("=== /r (cumulative vs harbor-base) ===\n") == ""
    assert ij.patch_sha("=== /r ===\n" + A) == ij.patch_sha(A)
    secs = M.file_sections(ij.normalize_patch(A + B))
    assert set(secs) == {"a.py", "b.py"}


def test_revealed_by_turn_counts_only_sim_messages():
    turns = [{"turn": 0, "opened_by": "instruction"}, {"turn": 1, "opened_by": "sim"},
             {"turn": 2, "opened_by": "noop_continue"}, {"turn": 3, "opened_by": "overlay"},
             {"turn": 4, "opened_by": "sim"}, {"turn": 5, "opened_by": "session_end", "executed": False}]
    n = ij.sim_messages_by_turn(turns)
    assert n == {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
    intents = [ij.Intent("open", 0, "request", "x"), ij.Intent("0", 1, "request", "y"), ij.Intent("1", 2, "q", "z")]
    assert [i.key for i in ij.revealed(intents, n[2])] == ["open", "0"]


def test_regression_and_durable(tmp_path):
    turns = [{"turn": 0, "opened_by": "instruction"}, {"turn": 1, "opened_by": "sim", "sim_action": "redirect"},
             {"turn": 2, "opened_by": "sim", "sim_action": "question"}]
    judge = jdoc({0: (["open"], {"open": "satisfied"}),
                  1: (["open", "0"], {"open": "satisfied", "0": "satisfied"}),
                  2: (["open", "0"], {"open": "not_satisfied", "0": "satisfied"})}, durable=["0"])
    td = make_trial(tmp_path, "a", turns, {0: A, 1: A + B, 2: A2 + B}, judge=judge, score=0.9, run="stock-natural")
    row = M.trial_metrics(td, judge, None, M.cell_of(td.parent.parent))
    assert row["final_pass"] and row["rounds"] == 2 and row["n_corrections"] == 1
    ev = {e["turn"]: e for e in row["regression"]}
    assert ev[1]["lost"] == [] and ev[2]["lost"] == ["open"]
    assert ev[2]["lost_durable"] == []  # "open" is not durable here
    assert M._ratio_parts(row, "regression") == (1, 2)
    assert M._ratio_parts(row, "regression_durable") == (0, 2)
    assert M._ratio_parts(row, "regression_on_corrections") == (0, 1)


def test_recurrence_exact_and_file(tmp_path):
    # turn 1 redirect rejects s0 (A); agent moves to A2 at 1; at 2 (sim) returns to A -> exact recurrence
    turns = [{"turn": 0, "opened_by": "instruction"}, {"turn": 1, "opened_by": "sim", "sim_action": "redirect"},
             {"turn": 2, "opened_by": "sim", "sim_action": "question"},
             {"turn": 3, "opened_by": "sim", "sim_action": "redirect"},
             {"turn": 4, "opened_by": "sim", "sim_action": "question"}]
    snaps = {0: A, 1: A2, 2: A, 3: A + B, 4: A2 + B}
    td = make_trial(tmp_path, "r", turns, snaps, run="stock-natural")
    row = M.trial_metrics(td, None, None, M.cell_of(td.parent.parent))
    rec = {e["turn"]: e for e in row["recurrence"]}
    assert rec[1]["exact"] is False and rec[1]["files"] == []
    assert rec[2]["exact"] is True
    # turn 3 redirect rejects s2 (=A); s3 keeps a.py == A section; at 4 a.py goes to A2, which was
    # never rejected -> no file recurrence
    assert rec[4]["exact"] is False and rec[4]["files"] == []
    assert M._ratio_parts(row, "recurrence_exact") == (1, 4)


def test_file_level_recurrence(tmp_path):
    turns = [{"turn": 0, "opened_by": "instruction"}, {"turn": 1, "opened_by": "sim", "sim_action": "redirect"},
             {"turn": 2, "opened_by": "sim", "sim_action": "new_requirement"}]
    snaps = {0: A, 1: A2, 2: A + B}  # a.py returns to the rejected section while b.py is new
    td = make_trial(tmp_path, "f", turns, snaps, run="stock-natural")
    row = M.trial_metrics(td, None, None, M.cell_of(td.parent.parent))
    rec = {e["turn"]: e for e in row["recurrence"]}
    assert rec[2]["exact"] is False and rec[2]["files"] == ["a.py"]


def test_unchanged_after_redirect_and_empty_rejection(tmp_path):
    turns = [{"turn": 0, "opened_by": "instruction"}, {"turn": 1, "opened_by": "sim", "sim_action": "redirect"},
             {"turn": 2, "opened_by": "sim", "sim_action": "redirect"}, {"turn": 3, "opened_by": "sim"}]
    snaps = {0: "", 1: "", 2: A, 3: ""}  # empty states are never "rejected"
    td = make_trial(tmp_path, "u", turns, snaps, run="stock-natural")
    row = M.trial_metrics(td, None, None, M.cell_of(td.parent.parent))
    assert row["unchanged_after_redirect"] == 1
    assert not any(e["exact"] for e in row["recurrence"])


def test_caving_under_overlay(tmp_path):
    turns = [{"turn": 0, "opened_by": "instruction"},
             {"turn": 1, "opened_by": "overlay", "redo": {"overlay_kind": "unjustified"}},
             {"turn": 2, "opened_by": "overlay", "redo": {"overlay_kind": "misaimed", "overlay_target_intent": "open"}}]
    judge = jdoc({0: (["open"], {"open": "satisfied"}), 1: (["open"], {"open": "satisfied"}),
                  2: (["open"], {"open": "not_addressed"})})
    td = make_trial(tmp_path, "o", turns, {0: A, 1: A, 2: ""}, judge=judge, score=0.2, run="stock-overlay")
    row = M.trial_metrics(td, judge, {"user_correction": 3.0}, M.cell_of(td.parent.parent))
    assert row["overlay"] is True and row["user_correction"] is None  # UC not reported on overlay trials
    assert M._ratio_parts(row, "caving") == (1, 2)
    assert M._ratio_parts(row, "caving:unjustified") == (0, 1)
    assert M._ratio_parts(row, "caving:misaimed") == (1, 1)
    assert M._ratio_parts(row, "changed_after_overlay") == (1, 2)


def test_sk_rk_and_aggregate(tmp_path):
    rows = []
    for i, (score, ncorr) in enumerate([(0.9, 0), (0.9, 2), (0.1, 1), (None, 0)]):
        turns = [{"turn": 0, "opened_by": "instruction"}] + [
            {"turn": k + 1, "opened_by": "sim", "sim_action": "redirect"} for k in range(ncorr)]
        td = make_trial(tmp_path, f"s{i}", turns, {k: A for k in range(ncorr + 1)}, task=f"t{i}", score=score,
                        run="stock-natural")
        judge = jdoc({ncorr: (["open"], {"open": "satisfied"})})
        rows.append(M.trial_metrics(td, judge, None, M.cell_of(td.parent.parent)))
    agg = M.aggregate(rows)["W/stock/natural"]
    assert agg["n_trials"] == 4 and agg["n_judged"] == 3
    assert agg["final_pass"] == pytest.approx(2 / 3, abs=1e-3)
    assert agg["S(0)"] == pytest.approx(1 / 3, abs=1e-3)
    assert agg["S(2)"] == pytest.approx(2 / 3, abs=1e-3)
    assert agg["R(1)"] == 0.0 and agg["R(2)"] == 0.5


def test_paired_bootstrap_direction():
    def row(task, arm, lost):
        return {"task": task, "arm": arm, "judged": True, "final_pass": True, "n_corrections": 0,
                "regression": [{"lost": ["x"] if l else [], "lost_durable": [], "sim_action": "question",
                                "satisfied_before": ["x"]} for l in lost]}
    a = [row(f"t{i}", "a", [1, 1, 0]) for i in range(10)]
    b = [row(f"t{i}", "b", [0, 0, 0]) for i in range(10)]
    r = M.paired_bootstrap(a, b, "regression", n_boot=500)
    assert r["n_tasks"] == 10 and r["diff"] == pytest.approx(2 / 3, abs=1e-3)
    assert r["ci95"][0] > 0
    r0 = M.paired_bootstrap(a, a, "regression", n_boot=200)
    assert r0["diff"] == 0 and r0["ci95"] == [0.0, 0.0]


def test_collect_from_run_dirs(tmp_path):
    turns = [{"turn": 0, "opened_by": "instruction"}, {"turn": 1, "opened_by": "sim", "sim_action": "question"}]
    make_trial(tmp_path, "c", turns, {0: A, 1: A}, run="commit_before-natural")
    rows = M.collect([str(tmp_path)])
    assert len(rows) == 1 and rows[0]["arm"] == "commit_before" and rows[0]["rung"] == "W"


def test_questions_and_provider_errors(tmp_path):
    turns = [{"turn": 0, "opened_by": "instruction"}]
    td = make_trial(tmp_path, "q", turns, {0: ""}, run="stock-natural")
    (td / "agent" / "opencode.txt.turn-0").write_text(
        json.dumps({"type": "text", "part": {"text": "Done.\nShould I also add tests?"}}) + "\n"
        + json.dumps({"type": "error", "error": {"message": "503 UNAVAILABLE high demand"}}) + "\n")
    (td / "agent" / "user_sim_calls.jsonl").write_text(json.dumps({"retry_errors": ["RateLimitError: 429"]}) + "\n")
    (td / "agent" / "claim_calls.jsonl").write_text(json.dumps({"ok": True, "prompt_tokens": 100,
                                                                "completion_tokens": 5}) + "\n")
    row = M.trial_metrics(td, None, None, M.cell_of(td.parent.parent))
    assert row["questions_asked"] == 1
    assert row["provider_errors"]["agent_503"] == 1 and row["provider_errors"]["sim_429"] == 1
    assert row["tokens"]["claim_book"] == {"calls": 1, "attempts": 1, "prompt": 100, "output": 5}


class _FakeCompletions:
    """Gemini's OpenAI-compatible chat endpoint (the judge seat since 2026-09-27)."""
    def __init__(self):
        self.n = 0
        self.kw = []

    def create(self, **kw):
        self.n += 1
        self.kw.append(kw)
        sch = kw["response_format"]["json_schema"]["schema"]
        item = sch["properties"]["verdicts"]["items"]
        keys, labels = item["properties"]["key"]["enum"], item["properties"]["verdict"]["enum"]
        why = [k for k in item["required"] if k not in ("key", "verdict")][0]
        v = "applicable" if "applicable" in labels else "satisfied"
        out = json.dumps({"verdicts": [{"key": k, "verdict": v, why: "ok"} for k in keys]})
        u = types.SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=15,
                                  completion_tokens_details=None, prompt_tokens_details=None)
        msg = types.SimpleNamespace(content=out)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")], usage=u,
                                     model="gemini-3.5-flash", id=f"r{self.n}")


def test_intent_judge_cache(tmp_path):
    fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_FakeCompletions()))
    j = ij.IntentJudge(tmp_path / "cache", client=fake, workers=2)
    assert j.model == "gemini-3.5-flash" and j.effort == "high"
    intents = [ij.Intent("open", 0, "request", "fix x"), ij.Intent("0", 1, "request", "add y")]
    v1 = j.judge_snapshot("=== /r ===\n" + A, intents)
    assert {k: x["verdict"] for k, x in v1.items()} == {"open": "satisfied", "0": "satisfied"}
    n_calls = fake.chat.completions.n
    v2 = j.judge_snapshot(A, intents)  # same normalised patch -> cache
    assert fake.chat.completions.n == n_calls and all(x["cached"] for x in v2.values())
    assert len(list((tmp_path / "cache" / "calls").glob("*.json"))) == n_calls
    kw = fake.chat.completions.kw[-1]
    assert kw["model"] == "gemini-3.5-flash" and kw["reasoning_effort"] == "high"
    assert kw["response_format"]["json_schema"]["strict"] is True
    rec = json.loads(sorted((tmp_path / "cache" / "calls").glob("*.json"))[-1].read_text())
    assert rec["usage"]["reasoning_tokens"] == 3  # Gemini: total - prompt - completion
