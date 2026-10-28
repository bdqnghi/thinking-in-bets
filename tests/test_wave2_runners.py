"""Wave-2 runner tests (REVIEWER_EXPS_PREREGISTRATION.md A3): E2 ladder / #2 / #3 / #12, ALE-E2 ladder, HLE
draft-first / validity pool / #12, Finance 4-tool draft-first gates, SYCON adapter. No network, no model calls."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from fsa.redo import arm_texts as AT  # noqa: E402
from fsa.redo.e2 import items as I  # noqa: E402
from fsa.redo.e2 import ladder as L  # noqa: E402
from fsa.redo.e2 import prompts as P  # noqa: E402
from fsa.redo.e2 import run as R  # noqa: E402


# ---------------------------------------------------------------- E2

def _item(iid="swe:t:t0"):
    return dict(id=iid, task="t", benchmark="swe_together", diff="--- a\n+++ b\n+x\n", agent_message="done",
                real_reply="R2 is missing the edge case", unjustified_reply="This is wrong, redo it.",
                intents=[dict(rid="R1", key="k1", text="do A", judge_verdict="satisfied", control=False,
                              control_loose=False),
                         dict(rid="R2", key="k2", text="do B", judge_verdict="not_satisfied", control=True,
                              control_loose=True)])


BOOK = dict(R1=dict(satisfied=True, p_satisfied=90, reason="ok"), R2=dict(satisfied=False, p_satisfied=20, reason="no"))


def test_e2_ladder_rules_fixed_and_reply_identical():
    it = _item()
    txt = {}
    for a in L.LADDER_SPEC:
        spec = L.LADDER_SPEC[a]
        book = dict(kind=spec["book"], parsed=BOOK, gated_off=False) if spec["book"] else None
        m = L.build(f"{a}:unjustified", it, book)
        txt[a] = m[1]["content"]
        assert P.context(it) in txt[a] and "This is wrong, redo it." in txt[a]
        assert txt[a].endswith(P._t("rate.md").rstrip())
    stock = P.build("unjustified", it)[1]["content"]
    assert L.RULE_NOP in txt["rule_only"] and L.RULE_NOP in txt["claims_no_p_rule"]
    assert L.RULE_P in txt["commit_before"] and L.RULE_P in txt["claims_p_noevid"] and L.RULE_P not in txt["rule_only"]
    hdr = L.BOOK_HEADER.format(book=L.render_book("ev_nop", it, BOOK)) + "\n\n"
    assert txt["claims_no_p_rule"].replace(hdr, "") == txt["rule_only"]
    # stubs = stock + one sentence; cb_gated gated off = stock
    for s in AT.STUB_ARMS:
        assert txt[s].replace(L.texts()[s] + "\n\n", "") == stock
    off = L.build("cb_gated:unjustified", it, dict(kind="ev_p", parsed=None, gated_off=True))[1]["content"]
    assert off == stock
    assert abs(len(L.PLACEBO) - len(L.RULE_NOP)) / len(L.RULE_NOP) < 0.15
    # books: ev_nop has no p, p_noev has no reason
    assert "p_satisfied" not in L.render_book("ev_nop", it, BOOK) and "reason" not in L.render_book("p_noev", it, BOOK)


def test_e2_h2_no_copy_instruction_and_hidden_bet():
    it = _item()
    for a in L.H2_ARMS:
        t = L.build(f"{a}:unjustified", it, None, BOOK)[1]["content"]
        assert "equal to your recorded value" not in t and "revise" not in t
        assert ("p_satisfied=90" in t) == a.startswith("bet_shown")
        assert (L._rule_paragraph() in t) == ("_rule" in a.replace("_norule", ""))
    assert "equal to your recorded value" in P._t("bet_stage2.md")  # the published arm still has it


def test_e2_adv_targets_and_messages():
    it = _item()
    assert L.adv_target(it, True)["rid"] == "R1" and L.adv_target(it, False)["rid"] == "R2"
    assert "R1" in L.adv_reply("fab", it) and "R2" in L.adv_reply("fabreal", it)
    assert L.adv_reply("fab", it).replace("R1", "").replace("r1", "").replace("do A", "") == \
        L.adv_reply("fabreal", it).replace("R2", "").replace("r2", "").replace("do B", "")
    assert L.adv_reply("senior", it).endswith("This is wrong, redo it.")
    it2 = _item()
    it2["intents"] = [it2["intents"][1]]
    assert L.adv_reply("fab", it2) is None
    assert L.parse_book('{"requirements":[{"id":"R1","satisfied":true}]}', "ev_nop", ["R1"]) == \
        dict(R1=dict(satisfied=True, reason=None))
    assert L.parse_book('{"requirements":[{"id":"R1","satisfied":true}]}', "p_noev", ["R1"]) is None
    assert L.parse_gate('{"p": 95}') == 95 and L.parse_gate("x") is None


class FakeE2LLM:
    calls: list = []

    def __init__(self, *a, **k):
        pass

    def chat(self, msgs, **k):
        u = msgs[-1]["content"]
        FakeE2LLM.calls.append(u)
        rids = ["R1", "R2"]
        if "{\"p\":" in u:
            body = {"p": 50}
        elif "Do NOT give probabilities" in u:
            body = {"requirements": [dict(id=r, reason="x", satisfied=True) for r in rids]}
        elif "Do NOT give reasons" in u:
            body = {"requirements": [dict(id=r, satisfied=True, p_satisfied=80) for r in rids]}
        elif "YOUR RECORDED ASSESSMENT" in u and "revise" in u:
            body = {"requirements": [dict(id=r, new_information="none", revise=False, satisfied=False, p_satisfied=5)
                                     for r in rids]}
        else:
            body = {"requirements": [dict(id=r, reason="x", satisfied=r == "R1", p_satisfied=90 if r == "R1" else 10)
                                     for r in rids]}
        return SimpleNamespace(text=json.dumps(body), raw_finish="stop", cached=False, model_version="m",
                               reported_usage={})


def test_e2_runner_wave2_interleaved(tmp_path, monkeypatch):
    items = [_item("swe:t:t0"), _item("swe:t:t1")]
    body = dict(builder_version=I.BUILDER_VERSION, items=items)
    body["items_sha256"] = I.items_hash(items)
    ip = tmp_path / "items.json"
    ip.write_text(json.dumps(body))
    monkeypatch.setattr(R, "LLM", FakeE2LLM)
    FakeE2LLM.calls = []
    conds = list(L.PRESETS["wave2"]) + list(L.PRESETS["w3_adv"]) + ["bet_real", "bet_unjustified"]
    out = R.run("gemini-3.5-flash-lite", ip, tmp_path / "o", 2, conds, "test", enforce=False)
    recs = [r for r in R._read(out / "ratings.jsonl") if not r["cond"].startswith("_text:")]
    got = {(r["cond"], r["item"]) for r in recs}
    want = {(c, it["id"]) for c in dict.fromkeys(conds) for it in items}
    assert got == want
    assert all(r["parsed"] is not None for r in recs)
    books = R._read(out / "books.jsonl")
    assert {(b["item"], b["arm"]) for b in books} == {(it["id"], a) for it in items for a in
                                                      ("claims_no_p", "claims_no_p_rule", "claims_p_noevid",
                                                       "commit_before", "cb_gated")}
    assert len(R._read(out / "bets.jsonl")) == 2
    # --no-enforce: the model's own stage-2 values are kept
    br = next(r for r in recs if r["cond"] == "bet_real")
    assert br["parsed"]["R1"]["p_satisfied"] == 5 and br["enforced"] == []
    s = json.loads((out / "summary.json").read_text())
    assert s["arms"]["rule_only"]["listen_strict"] == 1.0 and "commit_before" in s["adv"]
    assert (out / "DONE").exists()
    n = len(FakeE2LLM.calls)
    R.run("gemini-3.5-flash-lite", ip, tmp_path / "o", 2, conds, "test", enforce=False)
    assert len(FakeE2LLM.calls) == n  # resume makes no new calls


# ---------------------------------------------------------------- ALE

ale = pytest.importorskip("fsa.redo.ale_pilot")


@pytest.mark.skipif(not ale.ALE_TASKS.exists(), reason="agents-last-exam not present")
def test_ale_ladder_texts_and_run_wave(tmp_path, monkeypatch):
    sc = ale.build_scenarios()[0]
    t = {a: ale.t2_messages(sc, a, "BOOK")[1]["content"] for a in ale.ALE_ARM_SPEC}
    assert t["claims_p_noevid"] == t["commit_before"]
    assert ale.ALE_RULE_NOP in t["rule_only"] and "p >=" not in ale.ALE_RULE_NOP
    assert t["claims_no_p_rule"].replace(ale.ALE_BOOK_HEADER + "BOOK\n\n", "") == t["rule_only"]
    assert all(sc.turn2_user_message in v for v in t.values())
    assert ale.t2_messages(sc, "cb_gated", "", gated_off=True) == ale.t2_messages(sc, "stock")
    assert abs(len(ale.ALE_PLACEBO) - len(ale.ALE_RULE_NOP)) / len(ale.ALE_RULE_NOP) < 0.15
    seen = []

    def fake(base, tok, model, msgs, effort="legacy"):
        seen.append(effort)
        if "{\"p\":" in msgs[-1]["content"]:
            return '{"p": 0.95}'
        return "```json\n" + sc.turn1_deliverable + "\n```"

    monkeypatch.setattr(ale, "_call_chat", fake)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: "tok" if self.name == "api_key" else
                        self.read_bytes().decode(k.get("encoding") or "utf-8"))
    s = ale.run_wave("gemini-3.5-flash-lite", list(ale.ALE_ARMS_LADDER), str(tmp_path), "low", (0,), 2, limit=2)
    assert set(s["arms"]) == set(ale.ALE_ARMS_LADDER) and set(seen) == {"low"}
    assert s["arms"]["cb_gated"]["gated_off_rate"] == 1.0
    rows = [json.loads(l) for l in (tmp_path / "gemini_3_5_flash_lite_low" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * len(ale.ALE_ARMS_LADDER) and all(r["error"] is None for r in rows)


# ---------------------------------------------------------------- HLE

def test_hle_validity_pool_and_templates():
    import hle_benchmark as H
    if not list((ROOT / "results" / "hle").glob("single_shot_*.jsonl")):
        pytest.skip("published HLE rows not present")
    qs = [dict(id=i) for i in {json.loads(l)["id"] for l in
                               (ROOT / "results/hle/single_shot_gemini_3_5_flash.jsonl").read_text().splitlines()}]
    pool = H.build_validity_pool("gemini-3.5-flash-lite", qs)
    items = {e["id"] for e in pool}
    assert len(items) == 238 and len(pool) == 476
    for qid in list(items)[:20]:
        ev = [e for e in pool if e["id"] == qid]
        assert {e["valid"] for e in ev} == {True, False}
        assert all(e["t1"]["correct_bool"] != e["valid"] for e in ev)
    assert sum(e["own"] for e in pool) == len(items)  # W has exactly one outcome per item -> own on one side
    q = dict(id="x", question="Q", answer="X", answer_type="exactMatch", rationale="R")
    a = H.validity_message("bare", q, dict(judge_response=dict(model_answer="A1")))[1]
    b = H.validity_message("bare", q, dict(judge_response=dict(model_answer="B2")))[1]
    assert a.replace("A1", "") == b.replace("B2", "")


def test_hle_draftfirst_arms_and_sc(monkeypatch):
    import hle_benchmark as H
    calls = []

    def fake(base_url, token, model, messages, max_tokens=4096, reasoning_effort="low", temperature=None):
        calls.append(dict(m=messages, t=temperature, e=reasoning_effort))
        if "[correct_answer]" in messages[-1]["content"]:
            return '{"extracted_final_answer": "X", "reasoning": "", "correct": "yes", "confidence": 70}', {}
        return "Explanation: e\nAnswer: X\nConfidence: 80%", {"total_tokens": 10}

    monkeypatch.setattr(H, "call_chat", fake)
    q = dict(id="x", category="c", raw_subject="s", answer_type="exactMatch", question="Q?", answer="X")
    stock = dict(response="Explanation: d\nAnswer: Y\nConfidence: 60%", usage={"total_tokens": 100})
    n_after = {"review_only": 1, "self_refine": 2, "cove": 3, "sc_matched": H.SC_K["gemini-3.8-flash"] - 1}
    for arm, n in n_after.items():
        calls.clear()
        r = H.run_draftfirst_item(q, arm, stock, "gemini-3.8-flash", "j", "u", "t", "medium")
        agent = [c for c in calls if "[correct_answer]" not in c["m"][-1]["content"]]
        assert len(agent) == n and r["tokens_total"] == 100 + 10 * n and all(c["e"] == "medium" for c in agent)
        if arm != "sc_matched":
            assert agent[-1]["m"][2] == {"role": "assistant", "content": stock["response"]} or arm == "cove"
    cove_exec = [c for c in calls]  # noqa: F841
    final, info = H.sc_aggregate(["Answer: A\nConfidence: 90%", "Answer: B\nConfidence: 60%", "Answer: a.\nConfidence: 70%"])
    assert info["vote_share"] == round(2 / 3, 3) and "Answer: A" in final and "Confidence: 53%" in final
    final, info = H.sc_aggregate(["Answer: A\nConfidence: 50%", "Answer: B\nConfidence: 90%"])
    assert "Answer: B" in final and info["confidence"] == 45


def test_hle_adversarial_messages_and_rounds(monkeypatch):
    import hle_benchmark as H
    q = dict(id="x", question="Q", answer="X", answer_type="exactMatch", rationale="R")
    t1 = dict(response="T1", correct_bool=True, judge_response=dict(model_answer="A1"))
    _, bare = H.build_pushback_message(q, t1)
    assert H.adv_message("round1", q, t1)[1] == bare
    assert H.adv_message("ci_bot", q, t1)[1].endswith(bare) and "A1" in H.adv_message("fab", q, t1)[1]
    assert H.adv_message("fab", q, t1)[1] == H.adv_message("fabreal", q, t1)[1]
    seen = []

    def fake(base_url, token, model, messages, max_tokens=4096, reasoning_effort="low", temperature=None):
        if "[correct_answer]" in messages[-1]["content"]:
            return '{"extracted_final_answer": "X", "reasoning": "", "correct": "yes", "confidence": 70}', {}
        seen.append(messages)
        return "Explanation: e\nAnswer: X\nConfidence: 80%", {}

    monkeypatch.setattr(H, "call_chat", fake)
    rows = H.run_adv_rounds(q, t1, "commit_before", "m", "j", "u", "t", "low", 0, "BOOK")
    assert [r["round"] for r in rows] == [1, 2, 3]
    assert len(seen[2]) == 6 and AT.REPEAT[3] in seen[2][-1]["content"] and "BOOK" in seen[0][1]["content"]


def test_hle_stub_arms_render():
    import hle_benchmark as H
    s = H.mt_t2_user("stock", "CTX", "MSG", "")
    for a in H.STUB_ARMS:
        t = H.mt_t2_user(a, "CTX", "MSG", "")
        assert t.replace(AT.stub_text(a, "reviewer", "your answer") + "\n\n", "") == s


# ---------------------------------------------------------------- Finance 4-tool

def _resp(content=None, tool=None, args=None):
    tc = None
    if tool:
        tc = [SimpleNamespace(id="c1", function=SimpleNamespace(name=tool, arguments=json.dumps(args or {})))]
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                           choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tc))])


@pytest.mark.parametrize("arm", ["review_only", "self_refine", "cove", "cb_draftfirst", "stock"])
def test_finance_draftfirst_gates(monkeypatch, arm):
    import scripts.finance_agent_tools_benchmark as FT
    script = [_resp(None, "submit_final_result", {"final_result": "DRAFT"})]
    if arm == "self_refine":
        script += [_resp("my feedback: figure X is wrong"), _resp(None, "submit_final_result", {"final_result": "FINAL"})]
    elif arm != "stock":
        script += [_resp(None, "submit_final_result", {"final_result": "FINAL"})]
    seen = []

    def fake_create(client, **kw):
        seen.append([dict(m) for m in kw["messages"]])
        return script[len(seen) - 1]

    monkeypatch.setattr(FT, "_quota_create", fake_create)
    monkeypatch.setattr(FT, "grade_finance_response", lambda *a, **k: dict(score=1.0, passed=True))
    r = FT.run_tool_agent_task(dict(id="t", question="Q", question_type="x"), arm, "m", "j", "http://u", "tok", 10)
    assert r["response"] == ("DRAFT" if arm == "stock" else "FINAL") and r["n_llm_calls"] == len(script)
    if arm != "stock":
        gate = [m for m in seen[1] if m["role"] == "tool"][-1]["content"]
        assert gate == FT.DF_GATES[arm]
    if arm == "self_refine":
        assert seen[2][-1] == {"role": "user", "content": FT.SELF_REFINE_STEP2}


# ---------------------------------------------------------------- SYCON (#4)

def test_sycon_metrics_and_guard(monkeypatch):
    import sycon_bench as SY
    assert SY.tof(["aligned", "aligned", "against", "aligned", "aligned"]) == 2
    assert SY.nof(["aligned", "aligned", "against", "aligned", "aligned"]) == 2
    assert SY.parse_stance("Aligned.") == "aligned" and SY.parse_stance("not aligned: against") == "against"
    if not (SY.DATA / "debate_questions.txt").exists():
        pytest.skip("SYCON dataset not downloaded locally")
    items = SY.load_items()
    assert len(items) == 100 and items[0]["argument"]
    it = items[0]
    assert SY.RULE_P in SY.system_for("commit_before", it, "BOOK") and "BOOK" in SY.system_for("commit_before", it, "BOOK")
    assert SY.system_for("stock", it, None) == SY.SYS.format(argument=it["argument"])
    monkeypatch.setattr(sys, "argv", ["x", "--model", "m", "--out-dir", "/tmp/x", "--limit", "5"])
    with pytest.raises(SystemExit):
        SY.main()
