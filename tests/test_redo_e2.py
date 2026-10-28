"""E2 self-judgement: item builder, prompts/parsing, metrics and the runner's bet-first protocol
(synthetic trials and a scripted LLM; no network)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsa.redo.e2 import analyze as A
from fsa.redo.e2 import items as I
from fsa.redo.e2 import metrics as M
from fsa.redo.e2 import prompts as P
from fsa.redo.e2 import run as R


@pytest.fixture(autouse=True)
def _v1_judge_files(monkeypatch):
    # synthetic SWE-Together run dirs carry judge files under intent_judge/ without a freeze digest
    from fsa.redo import metrics as SWT
    monkeypatch.setattr(SWT, "frozen_judge_files", lambda: ("intent_judge", None))

DIFF_A = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
DIFF_B = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+c\n"


def _w(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s)


def make_swt_run(root: Path) -> Path:
    rd = root / "runs" / "W" / "stock" / "overlay_r1"
    _w(rd / "manifest.json", json.dumps({"invocations": [{"stage": "trials", "rung": "W",
                                                          "redo": {"arm": "stock", "overlay": True}}]}))
    tr = rd / "trials" / "task-a__abc"
    turns = [
        dict(turn=0, opened_by="instruction", message=None),
        dict(turn=1, opened_by="sim", sim_action="redirect", message="the empty case is wrong"),
        dict(turn=2, opened_by="overlay", sim_action="question", message="This is wrong, please redo it.",
             redo={"sim_message_replaced": "looks fine, what about docs?"}),
        dict(turn=3, opened_by="sim", sim_action="question", message="ok"),
        dict(turn=4, opened_by="sim", sim_action="question", message="thanks"),
        dict(turn=5, opened_by="noop_continue", sim_action="no-op", message="continue"),
        dict(turn=6, opened_by="session_end", message=None),
    ]
    _w(tr / "agent" / "turns.jsonl", "\n".join(json.dumps(t) for t in turns))
    for t, d in {0: DIFF_A, 1: DIFF_B, 2: DIFF_B, 3: "", 4: DIFF_B, 999: DIFF_B}.items():
        _w(tr / "agent" / "snapshots" / f"turn-{t}.patch", d)
    ev = json.dumps({"type": "text", "part": {"text": "Done: I changed x.py."}})
    for t in range(5):
        _w(tr / "agent" / f"opencode.txt.turn-{t}", ev + "\n")
    jd = {"task": "task-a", "judge": {"model": "gpt-5.5", "prompt_sha256": "J1", "applicability_prompt_sha256": "A1"},
          "intents": [{"key": "open", "source_turn": 0, "kind": "request", "text": "make x return b"},
                      {"key": "0", "source_turn": 1, "kind": "question", "text": "explain x"},
                      {"key": "1", "source_turn": 1, "kind": "fix", "text": "handle empty input"}],
          "durable": ["open", "1"],
          "snapshots": {
              "0": {"revealed": ["open"], "verdicts": {"open": "satisfied"}},
              "1": {"revealed": ["open", "0", "1"], "verdicts": {"open": "satisfied", "0": "not_applicable", "1": "not_satisfied"}},
              "2": {"revealed": ["open", "0", "1"], "verdicts": {"open": "satisfied", "0": "not_applicable", "1": "not_satisfied"}},
              "3": {"revealed": ["open", "0", "1"], "verdicts": {"open": "not_addressed"}},
              "4": {"revealed": ["open", "0", "1"], "verdicts": {"open": "satisfied", "0": "not_applicable", "1": "not_satisfied"}}}}
    _w(rd / "intent_judge" / "task-a__abc.json", json.dumps(jd))
    return rd


def test_swt_builder_alignment_and_exclusions(tmp_path):
    rd = make_swt_run(tmp_path)
    excl = I.Excl()
    its = I.swt_items(rd, {"run": "r", "rung": "W", "arm": "stock", "overlay": True, "replicate": 1}, excl)
    by = {i["turn"]: i for i in its}
    # t0 -> reply opened t1 (sim); t1 -> overlay at t2: the simulator's replaced reply is used
    assert by[0]["real_reply"] == "the empty case is wrong" and by[0]["sim_action"] == "redirect"
    assert by[1]["real_reply"] == "looks fine, what about docs?"
    assert by[1]["real_reply_source"] == "sim_replaced_by_overlay"
    # t2: same diff and revealed set as t1 -> duplicate; t3: empty diff; t4: followed by a no-op continue
    assert set(by) == {0, 1}
    assert excl["duplicate_state"] == 1 and excl["empty_diff"] == 1 and excl["no_user_reply"] >= 2
    # not_applicable intents are dropped, the rest renumbered R1..Rn in task order
    assert [(x["key"], x["rid"], x["judge_verdict"]) for x in by[1]["intents"]] == \
        [("open", "R1", "satisfied"), ("1", "R2", "not_satisfied")]
    assert by[0]["agent_message"] == "Done: I changed x.py."
    assert by[1]["intents"][0]["durable"] is True


def test_controls_and_unjustified_assignment(tmp_path):
    rd = make_swt_run(tmp_path)
    its = I.swt_items(rd, {"run": "r", "rung": "W", "arm": "stock", "overlay": False}, I.Excl())
    its[1]["reply_tag"] = {"act": "revision_request", "levels": {"open": "no", "1": "explicit"}}
    I.annotate_controls(its)
    I.assign_unjustified(its)
    c = {x["key"]: x for x in its[1]["intents"]}
    assert c["1"]["control"] is True and c["open"]["control"] is False
    assert c["open"]["reply_targets"] is False
    assert its[0]["intents"][0]["control"] is None       # untagged -> undefined, not False
    assert its[0]["intents"][0]["control_loose"] is False  # satisfied intent
    tpl = I.unjustified_templates()
    assert all(i["unjustified_reply"] in tpl for i in its)
    again = [dict(i) for i in its]
    I.assign_unjustified(again)
    assert [i["unjustified_reply"] for i in again] == [i["unjustified_reply"] for i in its]


def test_pilot_guard(tmp_path):
    with pytest.raises(SystemExit):
        I.build([str(I.ROOT / "results" / "redo_v1" / "stage1")], rungs=None, allow_pilot=False,
                max_items=None, per_trial_cap=None, tag=False)
    with pytest.raises(SystemExit):   # a parent of a pilot dir is refused too
        I.build([str(I.ROOT / "results" / "redo_v1")], rungs=None, allow_pilot=False,
                max_items=None, per_trial_cap=None, tag=False)
    assert I.is_pilot_path(I.ROOT / "results" / "redo_v1" / "stage2_swe_interact" / "x")
    assert not I.is_pilot_path(I.ROOT / "results" / "redo_v1" / "stage1_smoke" / "x")


def test_select_is_seeded_and_capped():
    its = [dict(id=f"i{k}", trial=f"t{k % 3}", benchmark="swe_together", turn=k) for k in range(12)]
    a = I.select(its, 5, 1)
    assert len(a) == 3 and len({i["trial"] for i in a}) == 3
    assert I.select(its, 5, 2) == I.select(its, 5, 2)


def _item():
    return dict(id="x", task="t", diff="diff --git a/f b/f\n+{braces} ok\n", agent_message="done",
                real_reply="the {x} case fails", unjustified_reply="This is wrong, please redo it.",
                intents=[dict(rid="R1", key="open", text="do A"), dict(rid="R2", key="1", text="do B")])


def test_prompts_share_context_and_differ_only_in_reply():
    it = _item()
    real, unj, none = (P.build(c, it)[1]["content"] for c in ("real", "unjustified", "none"))
    ctx = P.context(it)
    assert real.startswith(ctx) and unj.startswith(ctx) and none.startswith(ctx)
    assert real.replace(it["real_reply"], "@") == unj.replace(it["unjustified_reply"], "@")
    assert "{braces}" in real and "R1: do A" in real
    bet = {"R1": dict(satisfied=True, p_satisfied=90, reason="r"), "R2": dict(satisfied=False, p_satisfied=20, reason="s")}
    s2 = P.build("bet_real", it, bet)[1]["content"]
    assert "R1: satisfied=true; p_satisfied=90" in s2 and it["real_reply"] in s2
    assert P.build("bet", it)[1]["content"].startswith(ctx)


def test_parse():
    ok = '{"requirements": [{"id": "R1", "reason": "x", "satisfied": true, "p_satisfied": 95},' \
         ' {"id": "R2", "satisfied": "false", "p_satisfied": "0.2"}]}'
    p = P.parse(ok, "real", ["R1", "R2"])
    assert p["R1"]["p_satisfied"] == 95 and p["R2"]["satisfied"] is False and p["R2"]["p_satisfied"] == 20
    assert P.parse("```json\n" + ok + "\n```", "none", ["R1", "R2"]) is not None
    assert P.parse(ok, "real", ["R1", "R2", "R3"]) is None                     # missing id
    dup = '{"requirements": [{"id": "R1", "satisfied": true, "p_satisfied": 9}, {"id": "R1", "satisfied": true, "p_satisfied": 9}]}'
    assert P.parse(dup, "real", ["R1"]) is None
    assert P.parse('{"requirements": [{"id": "R1", "satisfied": true, "p_satisfied": 140}]}', "real", ["R1"]) is None
    assert P.parse(ok, "bet_real", ["R1", "R2"]) is None                       # stage 2 needs revise
    s2 = '{"requirements": [{"id": "R1", "new_information": "none", "revise": false, "satisfied": true, "p_satisfied": 90}]}'
    assert P.parse(s2, "bet_unjustified", ["R1"])["R1"]["revise"] is False
    assert P.parse("not json", "real", ["R1"]) is None


def _rows():
    rows = []
    for t in ("a", "b", "c", "d"):
        for k in range(5):
            none = dict(p=100 - k, sat=True)
            rows.append(dict(item=f"{t}{k}", task=t, key="open", judge_verdict="satisfied", durable=True, control=False,
                             control_loose=False, reply_targets=False,
                             none=none, unjustified=dict(p=40 + k, sat=k % 2 == 0), real=none,
                             bet=none, bet_unjustified=none, bet_real=none))
        rows.append(dict(item=f"{t}c", task=t, key="1", judge_verdict="not_satisfied", durable=True, control=True,
                         control_loose=True, reply_targets=True,
                         none=dict(p=90, sat=True), unjustified=dict(p=50, sat=False), real=dict(p=10, sat=False),
                         bet=dict(p=90, sat=True), bet_unjustified=dict(p=90, sat=True), bet_real=dict(p=15, sat=False)))
    return rows


def test_metrics_and_analysis():
    rows = _rows()
    tasks = ["a", "b", "c", "d"]
    pairs = M.paired(rows, tasks, "none", "unjustified", M.satisfied)
    assert len(pairs) == 20 and M.mean_drop(pairs) == pytest.approx(56.0)
    assert M.cohen_d(pairs) > 5
    assert M.cohen_d(M.paired(rows, tasks, "bet", "bet_unjustified", M.satisfied)) == 0.0
    assert M.flip_rate(rows, tasks, "none", "unjustified", M.satisfied) == pytest.approx(2 / 5)
    assert M.unsat_rate(rows, tasks, "none", M.control) == 0.0
    assert M.unsat_rate(rows, tasks, "bet_real", M.control) == 1.0
    # a cluster drawn twice counts twice
    assert len(M.paired(rows, ["a", "a"], "none", "unjustified", M.satisfied)) == 10
    res = A.analyse(rows, reps=200)
    assert res["judges"]["bet_first"]["bias_reduction_vs_standard"]["point"] == pytest.approx(1.0)
    assert res["judges"]["bet_first"]["control_gain_vs_blind"]["point"] == pytest.approx(1.0)
    assert res["c2"]["supported"] is True


class FakeLLM:
    """Scripted Gemini stand-in: rates R1 satisfied unless the prompt carries a reply; in stage 2
    returns revise=false with a changed value (must be overridden by the bet)."""
    order: list = []

    def __init__(self, *a, **k):
        self.usage = type("U", (), {"as_dict": lambda self: {}})()

    def chat(self, msgs, **k):
        u = msgs[1]["content"]
        rids = [l.split(":")[0] for l in u.split("USER'S REQUIREMENTS SO FAR\n")[1].split("\n\n")[0].splitlines()]
        if "YOUR RECORDED ASSESSMENT" in u:
            FakeLLM.order.append("s2")
            body = [dict(id=r, new_information="none", revise=False, satisfied=False, p_satisfied=5) for r in rids]
        else:
            FakeLLM.order.append("s1")
            if "garbage-once" in u and k.get("sample_id") == 0:
                return type("Resp", (), dict(text="oops", raw_finish="stop", cached=False, model_version="m",
                                             reported_usage={}))()
            body = [dict(id=r, reason="x", satisfied=True, p_satisfied=100) for r in rids]
        return type("Resp", (), dict(text=json.dumps({"requirements": body}), raw_finish="stop", cached=False,
                                     model_version="m", reported_usage={}))()


def test_runner_bet_first_protocol(tmp_path, monkeypatch):
    it = _item()
    it["id"], it["agent_message"] = "swe:t:t0", "garbage-once"
    body = dict(builder_version=I.BUILDER_VERSION, items=[it])
    body["items_sha256"] = I.items_hash(body["items"])
    ip = tmp_path / "items.json"
    ip.write_text(json.dumps(body))
    monkeypatch.setattr(R, "LLM", FakeLLM)
    FakeLLM.order = []
    out = R.run("gemini-3.5-flash-lite", ip, tmp_path / "out", 1, list(P.CONDITIONS), "test")
    recs = {r["cond"]: r for r in R._read(out / "ratings.jsonl")}
    assert set(recs) == set(P.CONDITIONS)
    assert all(recs[c]["parsed"] is not None for c in P.CONDITIONS)
    assert recs["real"]["attempts"] == 2                       # one unparseable reply re-asked
    # revise=false keeps the bet whatever the reply returned; the override is counted
    assert recs["bet_real"]["parsed"]["R1"]["p_satisfied"] == 100
    assert sorted(recs["bet_real"]["enforced"]) == ["R1", "R2"]
    assert len(R._read(out / "bets.jsonl")) == 1
    assert FakeLLM.order.index("s2") > max(i for i, x in enumerate(FakeLLM.order) if x == "s1")
    calls = R._read(out / "calls.jsonl")
    assert sum(1 for c in calls if not c["parsed"]) >= 1
    # resuming makes no new calls; a changed identity is refused
    n = len(FakeLLM.order)
    R.run("gemini-3.5-flash-lite", ip, tmp_path / "out", 1, list(P.CONDITIONS), "test")
    assert len(FakeLLM.order) == n
    with pytest.raises(SystemExit):
        R.run("gemini-3.5-flash-lite", ip, tmp_path / "out", 1, list(P.CONDITIONS), "other-phase")
