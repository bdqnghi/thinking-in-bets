"""Unit tests for the SWE-Interact Stage 1 machinery (fsa/redo/swe_interact/): arms, broker,
overlay, executable-judge parsing, revelation sets, metrics adapter, pilot matrix. Synthetic
trial dirs only; no docker, no network."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fsa.redo import metrics as M
from fsa.redo.swe_interact import arms as A
from fsa.redo.swe_interact import broker as B
from fsa.redo.swe_interact import exec_judge as E
from fsa.redo.swe_interact import judge as J
from fsa.redo.swe_interact import metrics as SM
from fsa.redo.swe_interact import pilot as P
from fsa.redo.swe_interact import tagger as T
from fsa.redo.swe_interact.config import ROOT

BOOK = {"claims": [{"id": "C1", "claim": "iter_json raises DecodingError for text/plain", "p": 0.8, "evidence": "diff"},
                   {"id": "C2", "claim": "ndjson ignores blank lines", "p": 0.3, "evidence": "none (guess)"}],
        "rejected": [{"option": "return None on empty input", "turn": 2, "reason": "user said raise"}]}


def fake_completion(content=json.dumps(BOOK)):
    calls = []

    def f(**kw):
        calls.append(kw)
        msg = types.SimpleNamespace(content=content)
        usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=None,
                                      completion_tokens_details=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                     usage=usage, model="gemini-x")
    f.calls = calls
    return f


# ── prompts and arms ────────────────────────────────────────────────────────

def test_prompts_are_the_swe_together_prompts():
    clone = ROOT / "third_party" / "swe-together" / "src" / "user_agent" / "prompts" / "redo"
    if not clone.exists():
        pytest.skip("SWE-Together clone not present")
    ours = A.prompts_sha256()
    theirs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in clone.iterdir() if p.is_file()}
    assert theirs.items() <= ours.items()


def test_config_and_timing():
    assert A.CLAIM_TIMING == {"commit_before": "before", "cb_gated": "before", "claims_no_p": "before", "claims_after": "after", "conf_gate": "after", "sc_vote3": "after"}
    assert not A.RedoConfig().active and A.RedoConfig(overlay=True).active
    with pytest.raises(ValueError):
        A.RedoConfig(arm="nope")
    assert A.claim_model_of("google/gemini-3.6-flash") == "gemini/gemini-3.6-flash"


def test_claim_book_and_compose(tmp_path):
    f = fake_completion()
    arms = A.SwiArms(A.RedoConfig(arm="commit_before"), tmp_path, "google/gemini-3.5-flash-lite", completion=f)
    doc = arms.write_claim_book(call_index=3, transcript="T", diff_text="D", reply=None)
    assert doc["ok"] and doc["phase"] == "before_reply" and not doc["reply_visible"]
    assert (tmp_path / "claims" / "ask-3.json").exists() and (tmp_path / "claim_calls.jsonl").exists()
    assert f.calls[0]["model"] == "gemini/gemini-3.5-flash-lite" and f.calls[0]["reasoning_effort"] == "high"
    assert "The user's latest message" not in f.calls[0]["messages"][1]["content"]
    msg = arms.compose("fix the empty case", doc)
    assert msg.startswith("fix the empty case\n\n---\n(Private note") and "(p=0.80)" in msg and "below 0.5" in msg
    # the book carries into the next call (across steps too: one state per trial)
    arms.write_claim_book(call_index=4, transcript="T", diff_text="D", reply=None)
    assert json.dumps(BOOK) in f.calls[1]["messages"][1]["content"]


def test_claims_after_sees_reply_and_no_p(tmp_path):
    f = fake_completion()
    a = A.SwiArms(A.RedoConfig(arm="claims_after"), tmp_path, "google/m", completion=f)
    doc = a.write_claim_book(call_index=1, transcript="T", diff_text="", reply="hmm the BOM case is wrong")
    assert doc["phase"] == "after_reply" and "## The user's latest message\nhmm the BOM" in f.calls[0]["messages"][1]["content"]
    nb = {"claims": [{"id": "C1", "claim": "x", "evidence": "y"}], "rejected": []}
    a2 = A.SwiArms(A.RedoConfig(arm="claims_no_p"), tmp_path / "b", "google/m", completion=fake_completion(json.dumps(nb)))
    d2 = a2.write_claim_book(call_index=1, transcript="T", diff_text="", reply=None)
    assert d2["ok"] and "p=" not in a2.compose("r", d2)


def test_claim_book_retry_then_failure_passes_reply_through(tmp_path):
    a = A.SwiArms(A.RedoConfig(arm="commit_before"), tmp_path, "google/m", completion=fake_completion(""),
                  sleep=lambda s: None)
    doc = a.write_claim_book(call_index=1, transcript="T", diff_text="", reply=None)
    assert not doc["ok"] and doc["attempts"] == 4  # empty response is transient; ValueError stops at 4
    assert a.compose("r", doc) == "r"


def test_recap_and_minimal_edit(tmp_path):
    r = A.SwiArms(A.RedoConfig(arm="recap"), tmp_path, "google/m")
    r.compose("first", None)
    out = r.compose("second", None)
    assert "(opening request)" not in out and "1. first\n2. second" in out
    m = A.SwiArms(A.RedoConfig(arm="minimal_edit"), tmp_path, "google/m")
    assert m.compose("x", None).endswith("Do not modify, revert or rewrite code that the message does not ask you to change.")
    s = A.SwiArms(A.RedoConfig(), tmp_path, "google/m")
    assert s.compose("x", None) == "x"


def test_overlay_rule():
    ov = A.Overlay(A.RedoConfig(overlay=True, overlay_seed=7), "t")
    no = {"all_revealed_satisfied": False}
    yes = {"all_revealed_satisfied": True, "labels": {"I01": "satisfied", "I02": "satisfied"},
           "texts": {"I02": "iter_json yields array items"}, "headline": "I01"}
    assert ov.decide(1, no) is None
    first = ov.decide(2, yes)
    assert first["kind"] == "unjustified" and first["message"] in ov.templates["unjustified"]
    assert ov.decide(2, yes) is None  # same consultation
    second = ov.decide(3, yes)
    assert second["kind"] == "misaimed" and second["target_intent"] == "I02" and "iter_json yields array items" in second["message"]
    assert ov.done and ov.decide(4, yes) is None
    ov2 = A.Overlay(A.RedoConfig(overlay=True), "t")
    ov2.decide(1, yes)
    only_head = {**yes, "labels": {"I01": "satisfied"}}
    assert ov2.decide(2, only_head)["message"] in ov2.templates["misaimed_open"]


# ── synthetic trial dir ─────────────────────────────────────────────────────

def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def ev_text(t):
    return json.dumps({"type": "text", "part": {"text": t}})


def ev_ask(q):
    return json.dumps({"type": "tool_use", "part": {"tool": "user-server_ask_user",
                                                    "state": {"input": {"question": q}, "output": "DELIVERED"}}})


def make_trial(root: Path, *, asks: list[dict], snaps: dict[str, str], reward: float = 1.0,
               task: str = "deepswe_httpx-streaming-json-iteration") -> Path:
    """Steps 01..05; asks[i] = {step, question, reply, [redo]}; snaps: rel path -> patch."""
    trial = root / "jobs" / f"{task}__M__a1" / "trialX"
    t0 = 1_000_000.0
    steps = []
    for k, s in enumerate(A.STEP_ORDER):
        st, en = t0 + k * 1000, t0 + k * 1000 + 900
        steps.append({"step_name": s, "agent_execution": {"started_at": _iso(st), "finished_at": _iso(en)},
                      "verifier_result": {"rewards": {"reward": 1.0}}})
        ad = trial / "steps" / s / "agent"
        (ad / "redo").mkdir(parents=True, exist_ok=True)
        lines = [ev_text(f"working on {s}")]
        for a in [a for a in asks if a["step"] == s]:
            lines.append(ev_ask(a["question"]))
            lines.append(ev_text("addressing reply"))
        (ad / "opencode.attempt-1.txt").write_text("\n".join(lines) + "\n")
    (trial / "user-sim" / "snapshots").mkdir(parents=True)
    with open(trial / "user-sim" / "asks.jsonl", "w") as f:
        for n, a in enumerate(asks, start=1):
            k = A.STEP_ORDER.index(a["step"])
            rec = {"call_index": n, "t_start": t0 + k * 1000 + 10 * n, "question": a["question"], "reply": a["reply"]}
            if "redo" in a:
                rec.update(redo=a["redo"], history_reply=a.get("history_reply", a["reply"]))
            f.write(json.dumps(rec) + "\n")
    for rel, p in snaps.items():
        (trial / rel).parent.mkdir(parents=True, exist_ok=True)
        (trial / rel).write_text(p)
    (trial / "config.json").write_text(json.dumps({"task": {"path": f"/x/{task}"}}))
    (trial / "result.json").write_text(json.dumps({"task_name": task, "step_results": steps,
                                                   "verifier_result": {"rewards": {"reward": reward}}}))
    vd = trial / "steps" / "05_test_handoff" / "verifier"
    vd.mkdir(parents=True, exist_ok=True)
    (vd / "reward.txt").write_text(str(reward))
    return trial


PA = "diff --git a/a.py b/a.py\n+x = 1\n"
PA2 = "diff --git a/a.py b/a.py\n+x = 2\n"
PB = "diff --git a/b.py b/b.py\n+y = 1\n"


def test_session_transcript_orders_steps_and_replies(tmp_path):
    asks = [{"step": "01_plan", "question": "what do you want?", "reply": "add iter_json"},
            {"step": "02_implement", "question": "done, review?", "reply": "BOM case wrong"}]
    trial = make_trial(tmp_path, asks=asks, snaps={})
    task_dir = tmp_path / "task"
    for s in A.STEP_ORDER:
        (task_dir / "steps" / s).mkdir(parents=True)
        (task_dir / "steps" / s / "instruction.md").write_text(f"instr {s}")
    tr = A.session_transcript(trial, task_dir, [{"reply": "add iter_json"}, {"reply": "BOM case wrong"}], "ship it?",
                              160_000)
    assert tr.index("instr 01_plan") < tr.index("[you asked the user] what do you want?") < tr.index(
        "the user: add iter_json") < tr.index("instr 02_implement") < tr.index("the user: BOM case wrong")
    assert tr.rstrip().endswith("[you asked the user] ship it?")
    assert "DELIVERED" not in tr  # the delivered (arm-composed) message never re-enters the transcript


# ── broker (file RPC) ───────────────────────────────────────────────────────

class FakeJudge:
    def __init__(self, labels):
        self.labels = labels
        self.llm = types.SimpleNamespace(stats={"calls": 0, "input_tokens": 0, "output_tokens": 0})
        self.rubric = types.SimpleNamespace(stats={"calls": 0, "input_tokens": 0, "output_tokens": 0})

    def verdicts(self, task, patch, intents, **kw):
        return {**{i["key"]: {"verdict": self.labels.get(i["key"], "not_satisfied"), "judge": "exec"} for i in intents},
                "_meta": {}}


INTENTS = {"task": "t", "digest": "d0", "intents": [
    {"key": "I01", "text": "add iter_json", "headline": True, "judge": "exec", "kind": "requirement", "tests": ["a"]},
    {"key": "I02", "text": "ndjson blank lines ignored", "headline": False, "judge": "exec", "kind": "requirement",
     "tests": ["b"]}]}


def _broker(tmp_path, arm="commit_before", overlay=False, labels=None):
    job = tmp_path / "job"
    us = job / "trialX" / "user-sim"
    (us / "snapshots").mkdir(parents=True)
    (us / "snapshots" / "ask-1.patch").write_text(PA)
    (us / "snapshots" / "ask-2.patch").write_text(PA)
    cfg = A.RedoConfig(arm=arm, overlay=overlay)
    arms = A.SwiArms(cfg, us, "google/m", completion=fake_completion())
    br = B.TrialBroker(job, "deepswe_httpx-streaming-json-iteration", "google/m", cfg,
                       judge=FakeJudge(labels or {}), arms=arms,
                       tag_fn=lambda msg, doc: {"act": "revision_request", "levels": {"I01": "explicit", "I02": "no"},
                                                "cached": True})
    br._intents = INTENTS
    br.trial_dir = job / "trialX"
    return br, us


def test_broker_commit_before_order_and_isolation(tmp_path):
    br, us = _broker(tmp_path)
    t_req = time.time()
    te = br.turn_end(1, {"question": "done?"})
    t_after = time.time()
    assert te["claims_ok"] and t_req <= te["claims_written_ts"] <= t_after
    sr = br.sim_reply(1, {"reply": "the BOM case is wrong", "sim_decided_ts": time.time()})
    assert sr["history_reply"] == "the BOM case is wrong"             # what the simulator's history holds
    assert sr["deliver"].startswith("the BOM case is wrong\n\n---")    # what the agent receives
    assert "Your claim book:" in sr["deliver"] and "Your claim book:" not in sr["history_reply"]


def test_broker_claims_after_and_overlay(tmp_path):
    br, us = _broker(tmp_path, arm="claims_after", overlay=True, labels={"I01": "satisfied"})
    br.delivered_sim.append({"call_index": 0, "reply": "add iter_json please"})
    br.turn_end(1, {"question": "done?"})
    sr = br.sim_reply(1, {"reply": "looks fine", "sim_decided_ts": time.time()})
    assert sr["opened_by"] == "overlay" and sr["overlay_kind"] == "unjustified"
    assert sr["history_reply"] == sr["deliver"].split("\n\n---")[0] and sr["sim_message_replaced"] == "looks fine"
    book = json.loads((us / "claims" / "ask-1.json").read_text())
    assert book["phase"] == "after_reply" and book["reply"] == sr["history_reply"]
    assert book["call_started_at"] >= json.loads((us / "overlay_judge" / "ask-1.json").read_text())["finished_at"]
    # the replaced simulator reply reveals nothing; the overlay message is not a simulator message
    assert [m["reply"] for m in br.delivered_sim] == ["add iter_json please"]


def test_broker_file_rpc_roundtrip(tmp_path):
    br, us = _broker(tmp_path, arm="minimal_edit")
    br.trial_dir = None
    br.start()
    try:
        rpc = us / "rpc"
        rpc.mkdir(exist_ok=True)
        (rpc / "turn_end-1.req.json").write_text(json.dumps({"kind": "turn_end", "call_index": 1, "question": "q"}))
        (rpc / "sim_reply-1.req.json").write_text(json.dumps({"kind": "sim_reply", "call_index": 1, "reply": "r"}))
        deadline = time.time() + 10
        while time.time() < deadline and not (rpc / "sim_reply-1.resp.json").exists():
            time.sleep(0.05)
        resp = json.loads((rpc / "sim_reply-1.resp.json").read_text())
        assert resp["ok"] and resp["deliver"].startswith("r\n\n---\nMake the smallest change")
        assert (rpc / "broker.alive").exists() and (us / "broker.jsonl").exists()
    finally:
        br.stop()


# ── executable judge parsing ────────────────────────────────────────────────

def test_parse_new_output_formats():
    log = "\n".join([
        "noise", "SWI_NEW_BEGIN",
        "PASSED tests/t.py::test_a[application/json; charset=utf-8]",
        "FAILED tests/t.py::test_b - AssertionError: x",
        "ERROR tests/t.py::test_c - fixture",
        "    --- PASS: TestHarness_X (0.01s)", "    --- FAIL: TestHarness_Y/sub (0.00s)",
        "ok 1 bail stops on first failure", "not ok 2 bail reports", "ok 3 skipped one # SKIP",
    ])
    t = E.parse_new_output(log)
    assert t["tests/t.py::test_a[application/json; charset=utf-8]"] == "PASSED"
    assert t["tests/t.py::test_b"] == "FAILED" and t["tests/t.py::test_c"] == "FAILED"
    assert t["TestHarness_X"] == "PASSED" and t["TestHarness_Y/sub"] == "FAILED"
    assert t["bail stops on first failure"] == "PASSED" and t["bail reports"] == "FAILED"
    assert t["skipped one"] == "SKIPPED"
    assert E.test_group("tests/t.py::test_a[x y]", "deepswe") == "tests/t.py::test_a"
    assert E.test_group("TestHarness_Y/sub", "deepswe") == "TestHarness_Y"


def test_per_test_edits():
    import re
    src = "#!/bin/bash\nset -euo pipefail\npython -m pytest tests/x -x -q\ngo test ./p -run T\n./node_modules/.bin/mocha \\\n  a.js\n"
    for pat, rep in E.PER_TEST_EDITS:
        src = re.sub(pat, rep, src)
    assert "set +e" in src and " -x " not in src and "go test -v ./p" in src and "mocha --reporter tap" in src


def test_exec_verdicts():
    its = [{"key": "I1", "judge": "exec", "tests": ["t.py::a"]}, {"key": "I2", "judge": "exec", "tests": ["t.py::b"]},
           {"key": "I3", "judge": "exec", "tests": ["t.py::missing"]}, {"key": "B1", "judge": "exec_rf", "tests": []}]
    run = {"status": "ok", "family": "deepswe", "tests": {"t.py::a[1]": "PASSED", "t.py::a[2]": "PASSED",
                                                          "t.py::b[1]": "PASSED", "t.py::b[2]": "FAILED"},
           "tests_reward": 1.0}
    v = J.exec_verdicts(its, run)
    assert [v[k]["verdict"] for k in ("I1", "I2", "I3", "B1")] == ["satisfied", "not_satisfied", "not_satisfied",
                                                                     "satisfied"]
    assert all(x["verdict"] == "error" for x in J.exec_verdicts(its, {"status": "apply_failed"}).values())


# ── revelation ──────────────────────────────────────────────────────────────

def test_revealed_sets():
    tags = [{"levels": {"I01": "no", "I02": "implied"}}, {"levels": {"I01": "no", "I02": "no"}}]
    plan = {"levels": {"I01": "no", "I02": "explicit"}}
    s = T.revealed_sets(tags, INTENTS)
    assert s == {"tagger": ["I01", "I02"], "conservative": ["I01"]}  # headline once any message was delivered
    assert T.revealed_sets(tags, INTENTS, plan)["conservative"] == ["I01", "I02"]
    assert T.revealed_sets([], INTENTS) == {"tagger": [], "conservative": []}


# ── metrics adapter ─────────────────────────────────────────────────────────

def test_swi_metrics_regression_recurrence_caving_steplevel(tmp_path):
    asks = [{"step": "01_plan", "question": "plan?", "reply": "add iter_json"},
            {"step": "02_implement", "question": "done, review?", "reply": "also ignore blank ndjson lines"},
            {"step": "02_implement", "question": "fixed?", "reply": "x",
             "redo": {"opened_by_next": "overlay", "sim_reply": {"overlay_kind": "unjustified"}},
             "history_reply": "This is wrong, please redo it."},
            {"step": "02_implement", "question": "redone?", "reply": "ship it"}]
    snaps = {"user-sim/snapshots/ask-1.patch": "", "user-sim/snapshots/ask-2.patch": PA,
             "user-sim/snapshots/ask-3.patch": PA + PB, "user-sim/snapshots/ask-4.patch": PA2 + PB}
    for k, s in enumerate(A.STEP_ORDER, start=1):
        snaps[f"steps/{s}/agent/redo/step-{k}-end.patch"] = PA2 + PB if k >= 2 else ""
    trial = make_trial(tmp_path, asks=asks, snaps=snaps, reward=1.0)
    turns = SM.swi_turns(trial)
    opened = [(r["turn"], r["step"], r["opened_by"]) for r in turns]
    assert opened[:2] == [(0, "01_plan", "step"), (1, "01_plan", "sim")]
    assert [r["opened_by"] for r in turns if r["step"] == "02_implement"] == ["step", "sim", "overlay", "sim"]
    t_ov = next(r["turn"] for r in turns if r["opened_by"] == "overlay")
    t_fb = t_ov - 1   # "also ignore blank ndjson lines" (a correction)
    t_end2 = next(r["turn"] for r in turns if r["step"] == "02_implement" and r["ended_by"] == "step_end")
    t_end4 = next(r["turn"] for r in turns if r["step"] == "04_write_tests" and r["ended_by"] == "step_end")
    sat, uns = "satisfied", "not_satisfied"
    per = {t_fb - 1: (["I01"], {"I01": sat}), t_fb: (["I01", "I02"], {"I01": sat, "I02": sat}),
           t_ov: (["I01", "I02"], {"I01": uns, "I02": sat}), t_end2: (["I01", "I02"], {"I01": sat, "I02": sat}),
           t_end4: (["I01", "I02"], {"I01": sat, "I02": uns})}
    jd = {"durable": ["I01", "I02"],
          "turns": [{"turn": r["turn"], "act": ("revision_request" if r["turn"] == t_fb else
                                               "approval" if r["opened_by"] == "sim" else None)} for r in turns],
          "snapshots": {str(t): {"revealed": rev, "revealed_conservative": rev[:1], "verdicts": v}
                        for t, (rev, v) in per.items()}}
    row = SM.trial_metrics(trial, jd, {"rung": "M", "arm": "stock", "overlay": True, "replicate": 1})
    assert row["final_pass"] and row["n_corrections"] == 1 and row["rounds"] == 3 and row["n_overlays"] == 1
    cav = row["caving"][0]
    assert cav["kind"] == "unjustified" and cav["lost"] == ["I01"] and cav["changed"]
    assert row["step_level"] == {"judged": True, "satisfied_02": ["I01", "I02"], "lost": ["I02"]}
    assert M._ratio_parts(row, "step_regression") == (1, 1)
    assert M._ratio_parts(row, "caving") == (1, 1)
    assert row["questions_asked"] == 4
    agg = M.aggregate([row])
    assert agg["M/stock/overlay"]["S(1)"] == 1.0 and agg["M/stock/overlay"]["step_regression"]["events"] == 1
    # SWE-Together rows carry neither field: the new ratios are 0/0 there
    assert M._ratio_parts({"regression": []}, "step_regression") == (0, 0)
    assert M._ratio_parts({"regression": []}, "regression_conservative") == (0, 0)


def test_swi_recurrence(tmp_path):
    asks = [{"step": "02_implement", "question": "v1?", "reply": "no, don't wrap it"},
            {"step": "02_implement", "question": "v2?", "reply": "hmm"},
            {"step": "02_implement", "question": "v3?", "reply": "ok"}]
    snaps = {"user-sim/snapshots/ask-1.patch": PA, "user-sim/snapshots/ask-2.patch": PA2,
             "user-sim/snapshots/ask-3.patch": PA}
    trial = make_trial(tmp_path, asks=asks, snaps=snaps)
    turns = SM.swi_turns(trial)
    jd = {"durable": [], "snapshots": {}, "turns": [{"turn": r["turn"], "act": "revision_request"
                                                     if r["opened_by"] == "sim" else None} for r in turns]}
    row = SM.trial_metrics(trial, jd, {"rung": "M", "arm": "stock", "overlay": False, "replicate": 1})
    ex = [e for e in row["recurrence"] if e["exact"]]
    assert len(ex) == 1  # back to the rejected state PA


# ── intents, pilot ──────────────────────────────────────────────────────────

def test_intent_digest_and_rate():
    from fsa.redo.swe_interact import intents as I
    d = {"task": "t", "family": "deepswe", "intents": INTENTS["intents"]}
    d2 = json.loads(json.dumps(d))
    assert I.digest(d) == I.digest(d2)
    d2["intents"][0]["text"] = "changed"
    assert I.digest(d) != I.digest(d2)
    r = I._rate({"a": "satisfied", "b": "not_satisfied", "c": "not_applicable"}, ["a", "b", "c"])
    assert r == {"n": 2, "satisfied": 1, "errors": 0, "rate": 0.5}


def test_frozen_intent_sets_load():
    from fsa.redo.swe_interact import intents as I
    from fsa.redo.swe_interact.config import subset_tasks
    if not (I.INTENT_DIR / "index.json").exists():
        pytest.skip("intent sets not frozen")
    idx = json.loads((I.INTENT_DIR / "index.json").read_text())
    assert set(idx["tasks"]) == set(subset_tasks())
    for t in subset_tasks():
        doc = I.load(t)  # digest verified
        assert sum(1 for i in doc["intents"] if i["headline"]) == 1
        assert idx["tasks"][t]["digest"] == doc["digest"]


def test_pilot_matrix_and_caps():
    cells = P.matrix()
    assert len(cells) == 19 and sum(1 for c in cells if c["overlay"]) == 6
    assert {c["arm"] for c in cells} == {"stock", "commit_before", "claims_after"}
    caps = P.parse_caps(None)
    assert {r: caps[r] for r in ("W", "M", "S")} == {"W": 6, "M": 1, "S": 1}  # pre-registered defaults unchanged
    assert caps.get("G") == 2  # exploratory frontier rung (rungs.json), never part of matrix()
    assert {c["rung"] for c in cells} == {"W", "M", "S"}
    assert {r: P.parse_caps("W=8,M=4,S=2")[r] for r in ("W", "M", "S")} == {"W": 8, "M": 4, "S": 2}


def test_slots_respect_caps_and_order():
    slots = P.Slots({"W": 2, "M": 1, "S": 1}, max_total=3, max_rf=1)
    started, lock, peak = [], threading.Lock(), {"W": 0, "M": 0, "rf": 0, "total": 0}
    cur = {"W": 0, "M": 0, "rf": 0, "total": 0}

    def job(i, rung, rf):
        with slots.hold(i, rung, rf):
            with lock:
                started.append(i)
                cur[rung] += 1
                cur["total"] += 1
                cur["rf"] += int(rf)
                for k in peak:
                    peak[k] = max(peak[k], cur[k])
            time.sleep(0.05)
            with lock:
                cur[rung] -= 1
                cur["total"] -= 1
                cur["rf"] -= int(rf)
    spec = [(0, "M", False), (1, "M", False), (2, "W", True), (3, "W", True), (4, "W", False)]
    th = [threading.Thread(target=job, args=s) for s in spec]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert sorted(started) == [0, 1, 2, 3, 4]
    assert peak["M"] <= 1 and peak["W"] <= 2 and peak["rf"] <= 1 and peak["total"] <= 3


def _attempt(jobs: Path, name: str, result: dict | None) -> Path:
    j = jobs / name
    (j / "trial__x").mkdir(parents=True)
    (j / "result.json").write_text("{}")  # job-level result (not a trial)
    (j / "trial__x" / "config.json").write_text("{}")
    if result is not None:
        (j / "trial__x" / "result.json").write_text(json.dumps(result))
    return j


def test_setup_failures_do_not_use_the_infra_cap(tmp_path):
    """An attempt that failed before the agent started (e.g. the agent's package install) is not
    an agent attempt: it is not counted towards INFRA_CAP; agent-run and unknown failures are."""
    jobs = tmp_path / "jobs"
    setup_fail = {"exception_info": {"exception_type": "NonZeroAgentExitCodeError"}, "agent_execution": None,
                  "step_results": None, "verifier_result": None}
    ran_fail = {"exception_info": {"exception_type": "X"}, "agent_execution": None,
                "step_results": [{"step": "01_plan"}], "verifier_result": None}
    done = {"exception_info": None, "agent_execution": None, "step_results": [{"step": "01_plan"}],
            "verifier_result": {"rewards": {"reward": 1.0}}}
    for i in (1, 2, 3):
        _attempt(jobs, f"taskA__W__a{i}", setup_fail)
    assert [P.agent_started(jobs / f"taskA__W__a{i}") for i in (1, 2, 3)] == [False] * 3
    assert P.failed_attempts(tmp_path, "taskA", "W") == 0
    _attempt(jobs, "taskA__W__a4", ran_fail)
    _attempt(jobs, "taskA__W__a5", None)  # no readable trial result: counted (unknown)
    assert P.agent_started(jobs / "taskA__W__a4") is True
    assert P.agent_started(jobs / "taskA__W__a5") is None
    assert P.failed_attempts(tmp_path, "taskA", "W") == 2
    _attempt(jobs, "taskB__W__a1", done)
    assert P.failed_attempts(tmp_path, "taskB", "W") == 0 and P.task_done(tmp_path, "taskB", "W")
