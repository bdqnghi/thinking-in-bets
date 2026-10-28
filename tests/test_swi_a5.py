"""Unit tests for A5.11 / A5.14 (SWE-Pushback Claim Book on SWE-Interact): sidecar gate texts and state machine,
artifact parsing, notes, broker hooks, the sidecar + broker round trip, test-file globs / source-only patch comparison,
after-the-fact tracks and outcomes."""
from __future__ import annotations

import json
import time

import pytest

from fsa.redo.swe_interact import a5
from fsa.redo.swe_interact import arms as A
from fsa.redo.swe_interact import broker as B
from fsa.redo.swe_pushback import texts as T

BOOK = {"claims": [{"id": "C1", "requirement": "Source.Get returns the snapshot", "evidence": "ok", "p": 0.9},
                   {"id": "C2", "requirement": "Subscribe sends updates on the channel", "evidence": "not tested",
                    "p": 0.4}], "p_all": 0.5}


def ev(t):
    return json.dumps({"type": "text", "part": {"text": t}})


# ---- texts ---------------------------------------------------------------------------------------------------
def test_arms_and_gate_prompts_reuse_dspush_texts():
    assert a5.ARMS == ("stock", "rule_only", "review_only", "commit_before")
    assert set(a5.REDO_ARMS) <= set(A.ARMS)
    assert a5.base_arm("a5_commit_before") == "commit_before" and a5.base_arm("commit_before") is None
    assert a5.base_arm("a5_nope") is None and a5.base_arm(None) is None
    for arm in ("stock", "rule_only", "stub_3"):
        assert a5.gate_prompt(arm) is None
    for arm, kind in (("review_only", "review"), ("commit_before", "claims_ev_p"), ("claims_p_noevid", "claims_p")):
        g = a5.gate_prompt(arm)
        body = T.ARTIFACT_PROMPT[kind][:-len(T._ART_TAIL)]
        assert g == a5.DEMAND_HEAD + body + a5.DEMAND_TAIL          # A5.14: sidecar demand frame, A5 body verbatim
        assert "bash tool" not in g and "submit again" not in g     # dspush tail (bash tool / submit) dropped
        assert "ask_user" in g and "held" in g
    assert len(a5.texts_sha256()) == 64


def test_stream_text_and_parse_artifact():
    out = "\n".join(["noise", json.dumps({"type": "tool_use", "part": {"tool": "bash"}}), ev("checking"),
                     ev("[CLAIM_BOOK]\n" + json.dumps(BOOK) + "\n[/CLAIM_BOOK]"), "{bad json"])
    text = a5.stream_text(out)
    assert text.startswith("checking\n[CLAIM_BOOK]")
    art, why = a5.parse_artifact("commit_before", text)
    assert why is None and art == BOOK
    assert a5.artifact_summary("commit_before", art) == {"n_claims": 2, "p_all": 0.5, "p_min": 0.4,
                                                         "frac_p_ge_095": 0.0}
    art, why = a5.parse_artifact("commit_before", "no block here")
    assert art is None and "no [CLAIM_BOOK]" in why
    nevid = {"claims": [{"id": "C1", "requirement": "r", "p": 0.9}], "p_all": 0.9}
    assert a5.parse_artifact("commit_before", "[CLAIM_BOOK]" + json.dumps(nevid) + "[/CLAIM_BOOK]")[1] == \
        "claims[0].evidence must be a non-empty string"
    assert a5.parse_artifact("claims_p_noevid", "[CLAIM_BOOK]" + json.dumps(nevid) + "[/CLAIM_BOOK]")[1] is None
    rev = {"requirements": [{"id": "R1", "requirement": "r", "evidence": "not tested"}]}
    assert a5.parse_artifact("review_only", "[REVIEW]" + json.dumps(rev) + "[/REVIEW]") == (rev, None)
    assert a5.gate_invalid("review_only", "x") == T.ARTIFACT_INVALID.format(tag="REVIEW", why="x") + a5.RESUBMIT_TAIL


# ---- routing + notes -----------------------------------------------------------------------------------------
def test_route_router_fallback_and_rule():
    r = a5.route(BOOK, "the subscription never sends anything", chat=lambda m, msgs: {"content": '{"claim_id": "C2"}'})
    assert r["claim_id"] == "C2" and r["via"] == "router" and r["decision"] == "CHECK" and r["p"] == 0.4
    r = a5.route(BOOK, "Get returns the wrong snapshot", chat=lambda m, msgs: {"content": '{"claim_id": "C9"}'})
    assert r["via"] == "fallback" and r["claim_id"] == "C1" and r["decision"] == "HOLD"

    def boom(m, msgs):
        raise RuntimeError("bridge down")
    r = a5.route(BOOK, "Subscribe updates channel", chat=boom)
    assert r["via"] == "fallback" and r["claim_id"] == "C2" and "bridge down" in r["router_error"]


def test_notes_per_arm():
    hold = {"claim_id": "C1", "p": 0.9, "requirement": "Get", "decision": "HOLD"}
    check = {"claim_id": "C2", "p": 0.4, "requirement": "Sub", "decision": "CHECK"}
    assert a5.note("stock", None, False) == ("", False)
    assert a5.note("stub_3", None, False) == (T.STUB3_NOTE, False)
    assert a5.note("rule_only", None, False) == (T.RULE_NOP + "\n" + a5.OVERRIDE_FORMAT_SWI, True)
    assert a5.note("review_only", None, True) == (T.RULE_NOP + "\n" + a5.OVERRIDE_FORMAT_SWI, True)
    t, enf = a5.note("commit_before", hold, True)
    assert enf and t.startswith(T.RULE_P_HOLD.format(cid="C1", req="Get", p=0.9)) and "[OVERRIDE]" in t
    t, enf = a5.note("commit_before", check, True)
    assert not enf and t == T.RULE_P_CHECK.format(cid="C2", req="Sub", p=0.4)
    # p-arm without a valid book: p-free rule
    assert a5.note("commit_before", None, False) == (T.RULE_NOP + "\n" + a5.OVERRIDE_FORMAT_SWI, True)
    assert "[step N]" not in a5.OVERRIDE_FORMAT_SWI
    assert a5.compose("r", "") == "r" and a5.compose("r", "n") == "r\n\nn"


def test_check_override_swi():
    ok = '[OVERRIDE] {"quote": "FAIL: Test_SourceSubscribe (0.01s)"} [/OVERRIDE]'
    assert a5.check_override_swi(ok, ["--- FAIL: Test_SourceSubscribe (0.01s)\n"]) is None
    assert "verbatim" in a5.check_override_swi(ok, ["nothing"])
    assert "20 characters" in a5.check_override_swi('[OVERRIDE] {"quote": "short"} [/OVERRIDE]', ["short"])
    assert "no [OVERRIDE]" in a5.check_override_swi("none", ["x"])


# ---- tracks and outcomes ---------------------------------------------------------------------------------------
DOC = {"intents": [{"key": "I01", "judge": "exec", "tests": ["Test_SourceGet"]},
                   {"key": "I02", "judge": "exec", "tests": ["Test_SourceSubscribe", "Test_Other"]},
                   {"key": "I03", "judge": "llm", "tests": []}]}


def tag(act, **levels):
    return {"act": act, "levels": levels}


def test_concerned_groups_and_tracks():
    tags = [tag("revision_request", I01="no", I02="implied", I03="explicit"),
            tag("approval", I01="explicit")]                     # an approval is not pushback
    conc = a5.concerned_groups(tags, DOC)
    assert conc == ["Test_Other", "Test_SourceSubscribe"]
    snap = {"Test_SourceGet": True, "Test_SourceSubscribe": True, "Test_Other": False}
    assert a5.assign_track(tags, conc, snap) == "valid"
    assert a5.assign_track(tags, conc, {"Test_SourceSubscribe": True, "Test_Other": True}) == "invalid"
    assert a5.assign_track(tags, conc, {"Test_SourceSubscribe": True}) == "valid"        # absent = failed
    only_llm = [tag("revision_request", I03="explicit")]
    assert a5.concerned_groups(only_llm, DOC) == [] and a5.assign_track(only_llm, [], snap) == "ambiguous"
    assert a5.assign_track([tag("answer", I01="explicit")], [], snap) == "no_pushback"
    assert a5.assign_track([], [], snap) == "no_pushback"


def test_group_status_and_outcomes():
    g = a5.group_status({"pkg::T1::a": "PASSED", "pkg::T1::b": "FAILED", "pkg::T2::a": "passed"},
                        lambda t: t.rsplit("::", 1)[0])
    assert g == {"pkg::T1": False, "pkg::T2": True}
    pre = {"A": True, "B": False, "C": True}
    fin = {"A": False, "B": True, "C": True}
    o = a5.outcomes("valid", ["B"], pre, fin, ["A", "B", "C"], True)
    assert o["regressions"] == ["A"] and o["false_cave"] and o["gained"] == ["B"] and o["correct_update"]
    assert not o["stubborn"]
    o = a5.outcomes("invalid", ["C"], pre, {"A": True, "C": False}, ["A", "B", "C"], True)
    assert o["target_broken"] and o["regressions"] == ["C"]
    o = a5.outcomes("valid", ["B"], pre, pre, ["A", "B", "C"], False)
    assert o["stubborn"] and not o["false_cave"]
    assert a5.brier([(1.0, 1), (0.5, 0)]) == pytest.approx(0.125) and a5.brier([]) is None


# ---- A5.14 sidecar gate (pure state machine) -------------------------------------------------------------------
BOOK_TXT = "[CLAIM_BOOK]\n" + json.dumps(BOOK) + "\n[/CLAIM_BOOK]"


def test_gate_artifact_arm_withholds_then_releases_on_valid_block():
    g = a5.Gate("commit_before")
    assert g.on_ask(1, "plan ok?", "01_plan")["phase"] == "pre_gate"          # planning Q&A: no gate
    assert g.on_ask(2, "is the plan fine?", None)["phase"] == "pre_gate"      # no step dir yet: not a gate step
    d = g.on_ask(3, "done, please review", "02_implement", now=10.0)
    assert d == {"phase": "withheld", "action": "withhold", "message": a5.gate_prompt("commit_before"), "gate_call": 3}
    assert g.doc["state"] == "pending" and g.doc["status"] == "unresolved" and not g.resolved
    d = g.on_ask(4, "here: " + BOOK_TXT, "02_implement", now=12.0)
    assert d == {"phase": "release", "action": "release", "question": "done, please review", "withheld_call": 3}
    assert g.doc["status"] == "ok" and g.doc["artifact"] == BOOK and g.doc["p_all"] == 0.5 and g.resolved
    assert g.doc["resolved_call"] == 4 and g.doc["resolved_ts"] == 12.0 and g.doc["call_index"] == 3
    assert g.on_ask(5, "anything else?", "04_write_tests")["phase"] == "post_gate"   # once per trial
    assert [c["phase"] for c in g.doc["calls"]] == ["pre_gate", "pre_gate", "withheld", "release", "post_gate"]


def test_gate_three_invalid_payloads_noncompliant_and_release():
    g = a5.Gate("review_only")
    g.on_ask(1, "q", "02_implement")
    d = g.on_ask(2, "no block", "02_implement")
    assert d["action"] == "withhold" and d["message"] == a5.gate_invalid("review_only", a5.parse_artifact(
        "review_only", "no block")[1]) and "no [REVIEW]" in d["message"]
    assert g.on_ask(3, "[REVIEW]{bad json[/REVIEW]", "03_handoff")["action"] == "withhold"   # tries persist across steps
    d = g.on_ask(4, "still nothing", "03_handoff")
    assert d["action"] == "release" and d["question"] == "q" and g.doc["status"] == "noncompliant"
    assert len(g.doc["tries"]) == a5.GATE_TRIES and g.doc["artifact"] is None


def test_gate_never_withholds_in_no_artifact_arms():
    for arm in ("stock", "rule_only", "stub_3"):
        g = a5.Gate(arm)
        assert g.on_ask(1, "q", "01_plan") == {"phase": "pre_gate"}
        assert g.on_ask(2, "q", "02_implement") == {"phase": "gate_open"}
        assert g.on_ask(3, "q", "05_test_handoff") == {"phase": "post_gate"}
        assert g.doc["status"] == "not_required" and g.doc["call_index"] == 2 and g.resolved
    g = a5.Gate("stock")
    assert g.on_ask(1, "q", "01_plan")["phase"] == "pre_gate" and g.doc["status"] == "never_opened"


def test_gate_fail_never_withholds_again():
    g = a5.Gate("commit_before")
    g.on_ask(1, "q", "02_implement")
    g.fail(2, "RuntimeError: boom")
    assert g.resolved and g.doc["status"] == "broker_error" and g.doc["errors"][0]["call_index"] == 2
    assert g.on_ask(3, "q2", "02_implement") == {"phase": "post_gate"}


def test_current_step(tmp_path):
    assert a5.current_step(tmp_path) is None and a5.current_step(tmp_path / "nope") is None
    (tmp_path / "steps" / "01_plan" / "agent").mkdir(parents=True)
    assert a5.current_step(tmp_path) == "01_plan" and not a5.in_gate_steps("01_plan")
    (tmp_path / "steps" / "02_implement" / "agent").mkdir(parents=True)   # created at the START of 02_implement
    (tmp_path / "steps" / "notastep").mkdir()
    assert a5.current_step(tmp_path) == "02_implement" and a5.in_gate_steps("02_implement")
    (tmp_path / "steps" / "05_test_handoff").mkdir()
    assert a5.current_step(tmp_path) == "05_test_handoff" and a5.in_gate_steps("05_test_handoff")


# ---- A5.14 (5) test globs and source-only comparison ------------------------------------------------------------
def _fdiff(path, body, new=False):
    hdr = f"diff --git a/{path} b/{path}\n" + ("new file mode 100644\nindex 0000000..1111111\n--- /dev/null\n"
                                               if new else "index 2222222..3333333 100644\n--- a/" + path + "\n")
    return hdr + f"+++ b/{path}\n@@ -0,0 +1 @@\n+{body}\n"


def test_test_globs_and_matching():
    tp = _fdiff("internal/oci/file_test.go", "x") + _fdiff(".azure-pipelines/templates/test.yml", "y")
    g = a5.test_globs("go", tp)
    assert g[:4] == list(a5.TEST_GLOBS_BY_LANG["go"]) and ".azure-pipelines/templates/test.yml" in g
    assert a5.is_test_path("internal/oci/ecr/new_test.go", g) and a5.is_test_path("pkg/testdata/a.json", g)
    assert a5.is_test_path(".azure-pipelines/templates/test.yml", g)
    assert not a5.is_test_path("internal/oci/file.go", g) and not a5.is_test_path("internal/testing.go", g)
    py = a5.test_globs("python", "")
    assert a5.is_test_path("openlibrary/catalog/marc/tests/test_parse.py", py) and a5.is_test_path("tests/unit/x.py", py)
    assert a5.is_test_path("conftest.py", py) and a5.is_test_path("lib/foo_test.py", py)
    assert not a5.is_test_path("openlibrary/catalog/marc/parse.py", py) and not a5.is_test_path("contest.py", py)
    js = a5.test_globs("js", "")
    assert a5.is_test_path("test/user.js", js) and a5.is_test_path("src/a/MPollBody-test.tsx", js)
    assert a5.is_test_path("applications/mail/src/x/tests/M.test.tsx", js) and a5.is_test_path("src/__tests__/a.ts", js)
    assert not a5.is_test_path("src/components/Latest.tsx", js) and not a5.is_test_path("src/attest.ts", js)
    assert a5.test_globs("cobol", "") == list(a5.TEST_GLOBS_FALLBACK)


def test_source_only_patch_comparison():
    g = a5.test_globs("go", "")
    pre = _fdiff("lib/a.go", "fix") + _fdiff("lib/new.go", "n", new=True)
    # step 04 adds a test file only -> the source comparison does not move, the all-files one does
    fin = _fdiff("lib/a.go", "fix") + _fdiff("lib/new.go", "n", new=True) + _fdiff("lib/a_test.go", "t", new=True)
    assert a5.patch_changes(pre, fin, g) == {"source": False, "all": True}
    # the untracked file got committed: index/mode lines and the order differ, the change itself does not
    fin2 = _fdiff("lib/new.go", "n", new=True).replace("index 0000000..1111111\n", "index 0000000..9999999\n") \
        + _fdiff("lib/a.go", "fix")
    assert a5.patch_changes(pre, fin2, g)["source"] is False
    fin3 = _fdiff("lib/a.go", "fix2") + _fdiff("lib/new.go", "n", new=True)
    assert a5.patch_changes(pre, fin3, g) == {"source": True, "all": True}
    assert a5.patch_changes("", "", g) == {"source": False, "all": False}
    assert a5.patch_changes("", _fdiff("lib/b_test.go", "t", new=True), g) == {"source": False, "all": True}
    assert a5.patch_changes("", _fdiff("lib/b.go", "t", new=True), g) == {"source": True, "all": True}
    q = 'diff --git "a/dir with space/x.go" "b/dir with space/x.go"\n+++ b/x\n+1\n'
    assert list(a5.patch_file_map(q)) == ["dir with space/x.go"]
    assert a5.patch_paths("diff --git a/a b/b c\n") == ["b c"]


def test_outcomes_source_vs_all_columns():
    pre = {"A": True, "B": False}
    o = a5.outcomes("valid", ["B"], pre, pre, ["A", "B"], False, True)
    assert o["stubborn"] and not o["stubborn_all"] and o["patch_changed"] is False and o["patch_changed_all"] is True


def test_task_test_globs_recorded_for_every_a5_task():
    from fsa.redo.swe_interact import a5_lane as L
    from fsa.redo.swe_interact.config import TASKS
    if not (TASKS.is_dir() and (TASKS.parents[1] / ".git").exists()):
        pytest.skip("SWE-Interact clone not present")
    for t in L.a5_tasks():
        g = L.task_test_globs(t)
        c = L.task_config(t)
        exact = set(a5.patch_paths(c["test_patch"]))
        assert exact and all(a5.is_test_path(p, g) for p in exact)
        assert not any(a5.is_test_path(p, g) for p in a5.patch_paths(c["patch"]) if p not in exact), t
        ns = L.task_nonsource_globs(t)
        assert ns[:len(g)] == g and "plan.md" in ns
        assert not any(a5.is_test_path(p, ns) for p in a5.patch_paths(c["patch"]) if p not in exact), t


# ---- broker hooks ------------------------------------------------------------------------------------------------
def _broker(tmp_path, arm, step="01_plan"):
    job = tmp_path / "job"
    trial = job / "trialX"
    us = trial / "user-sim"
    (us / "snapshots").mkdir(parents=True)
    for s in ("01_plan", "02_implement", "03_handoff", "04_write_tests", "05_test_handoff"):
        if s <= step:
            (trial / "steps" / s / "agent").mkdir(parents=True)
    cfg = A.RedoConfig(arm=arm)
    br = B.TrialBroker(job, "swebenchpro_x", "google/m", cfg, arms=A.SwiArms(cfg, us, "google/m"))
    br.trial_dir = trial
    br._a5_chat = lambda model, msgs: {"content": '{"claim_id": "C2"}'}
    return br, us, trial


def _step(trial, s):
    (trial / "steps" / s / "agent").mkdir(parents=True, exist_ok=True)


def test_broker_plan_replies_no_gate_no_note(tmp_path):
    for arm in ("a5_stock", "a5_rule_only", "a5_commit_before"):
        br, us, _ = _broker(tmp_path / arm, arm)
        te = br.turn_end(1, {"question": "plan?"})
        assert te["a5_gate"]["phase"] == "pre_gate" and "action" not in te["a5_gate"]
        sr = br.sim_reply(1, {"reply": "plan looks fine", "sim_decided_ts": time.time()})
        assert sr["deliver"] == "plan looks fine" and sr["a5_phase"] == "pre_gate"
        assert not (us / "a5" / "ask-1.json").exists()


def test_broker_gate_withhold_validate_release_routed_note(tmp_path):
    br, us, trial = _broker(tmp_path, "a5_commit_before")
    br.turn_end(1, {"question": "plan?"})
    br.sim_reply(1, {"reply": "ok"})
    _step(trial, "02_implement")
    te = br.turn_end(2, {"question": "implemented, review please", "snapshot_patch_sha256": "s2"})
    g = te["a5_gate"]
    assert g["action"] == "withhold" and g["message"] == a5.gate_prompt("commit_before") and g["step"] == "02_implement"
    assert g["snapshot_patch_sha256"] == "s2" and g["gate_call"] == 2
    assert json.loads((us / "a5" / "gate.json").read_text())["state"] == "pending"
    te = br.turn_end(3, {"question": "oops"})
    assert te["a5_gate"]["action"] == "withhold" and "could not accept" in te["a5_gate"]["message"]
    te = br.turn_end(4, {"question": BOOK_TXT})
    assert te["a5_gate"]["action"] == "release" and te["a5_gate"]["question"] == "implemented, review please"
    sr = br.sim_reply(4, {"reply": "updates never arrive on the channel", "sim_decided_ts": time.time()})
    assert sr["history_reply"] == "updates never arrive on the channel"         # the simulator never sees the note
    assert sr["deliver"] == "updates never arrive on the channel\n\n" + T.RULE_P_CHECK.format(
        cid="C2", req=BOOK["claims"][1]["requirement"], p=0.4)
    assert br.asks[-1]["question"] == "implemented, review please"
    rec = json.loads((us / "a5" / "ask-4.json").read_text())
    assert rec["route"]["claim_id"] == "C2" and rec["gate_call"] == 2 and rec["resolved_call"] == 4
    gd = json.loads((us / "a5" / "gate.json").read_text())
    assert gd["status"] == "ok" and gd["artifact"] == BOOK and len(gd["tries"]) == 2


def test_broker_no_artifact_arms_and_noncompliant(tmp_path):
    br, _, _ = _broker(tmp_path / "s", "a5_stock", step="02_implement")
    assert "action" not in br.turn_end(1, {"question": "q"})["a5_gate"]
    assert br.sim_reply(1, {"reply": "r"})["deliver"] == "r"
    br, _, _ = _broker(tmp_path / "r", "a5_rule_only", step="03_handoff")
    assert br.turn_end(1, {"question": "q"})["a5_gate"]["phase"] == "gate_open"
    assert br.sim_reply(1, {"reply": "r"})["deliver"] == "r\n\n" + T.RULE_NOP + "\n" + a5.OVERRIDE_FORMAT_SWI
    br, _, _ = _broker(tmp_path / "t", "a5_stub_3", step="02_implement")
    br.turn_end(1, {"question": "q"})
    assert br.sim_reply(1, {"reply": "r"})["deliver"] == "r\n\n" + T.STUB3_NOTE
    # p-arm whose payloads were all invalid: p-free rule, no router call
    br, _, _ = _broker(tmp_path / "n", "a5_commit_before", step="02_implement")
    br._a5_chat = None
    for k in range(1, 5):
        te = br.turn_end(k, {"question": "nope"})
    assert te["a5_gate"]["action"] == "release" and br.gate.doc["status"] == "noncompliant"
    sr = br.sim_reply(4, {"reply": "r"})
    assert sr["deliver"].endswith(T.RULE_NOP + "\n" + a5.OVERRIDE_FORMAT_SWI) and sr["a5_route"] is None
    assert sr["a5_artifact_status"] == "noncompliant"


def test_broker_gate_error_never_withholds(tmp_path, monkeypatch):
    br, us, _ = _broker(tmp_path, "a5_commit_before", step="02_implement")
    monkeypatch.setattr(a5, "current_step", lambda d: (_ for _ in ()).throw(OSError("disk")))
    te = br.turn_end(1, {"question": "q"})
    assert te["a5_gate"]["phase"] == "broker_error" and "action" not in te["a5_gate"]
    assert json.loads((us / "a5" / "gate.json").read_text())["status"] == "broker_error"
    monkeypatch.undo()
    assert "action" not in br.turn_end(2, {"question": BOOK_TXT})["a5_gate"]


# ---- the sidecar (overlay/server.py) + the real broker over the file RPC ------------------------------------------
def _load_server(monkeypatch, tmp_path, redo_dir, arm):
    """Import overlay/server.py with stub fastmcp / litellm modules (not installed on the host) and no network."""
    import importlib.util
    import sys
    import types
    fm = types.ModuleType("fastmcp")

    class FastMCP:
        def __init__(self, *a, **k):
            pass

        def tool(self):
            return lambda f: f
    fm.FastMCP = FastMCP
    monkeypatch.setitem(sys.modules, "fastmcp", fm)
    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
    for k, v in {"LITELLM_API_KEY": "x", "LITELLM_BASE_URL": "http://127.0.0.1:9", "SWI_REDO_DIR": str(redo_dir),
                 "SWI_REDO_ARM": arm, "SIM_USER_PRIVATE_LOG_DIR": str(tmp_path / "private")}.items():
        monkeypatch.setenv(k, v)
    src = (B.Path(B.__file__).parent / "overlay" / "server.py").read_text()
    assert src.count('Path("/app/persona.md").read_text()') == 1
    src = src.replace('Path("/app/persona.md").read_text()', '"persona"')
    spec = importlib.util.spec_from_loader(f"swi_server_{arm}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "server.py", "exec"), mod.__dict__)
    state = {"patch": "", "sim_inputs": []}

    def repo_exec(command, max_chars=0, timeout=0):
        if command.startswith("sed -n 1p") or command == "git rev-parse HEAD":
            return {"ok": True, "stdout": "a" * 40 + "\n"}
        if command.startswith("git diff --binary"):
            return {"ok": True, "stdout": state["patch"]}
        return {"ok": True, "stdout": ""}
    mod._repo_exec_raw = repo_exec
    mod.repo_tool = lambda *a, **k: "{}"

    def sim(messages):
        state["sim_inputs"].append(messages)
        return f"sim reply #{len(state['sim_inputs'])}"
    mod.complete_with_tools = sim
    return mod, state


@pytest.mark.parametrize("arm", ["a5_commit_before", "a5_stock"])
def test_sidecar_and_broker_round_trip(tmp_path, monkeypatch, arm):
    job = tmp_path / "job"
    trial = job / "trialA"
    us = trial / "user-sim"
    us.mkdir(parents=True)
    _step(trial, "01_plan")
    srv, st = _load_server(monkeypatch, tmp_path, us, arm)
    cfg = A.RedoConfig(arm=arm)
    br = B.TrialBroker(job, "swebenchpro_x", "google/m", cfg, arms=A.SwiArms(cfg, us, "google/m"))
    br._a5_chat = lambda model, msgs: {"content": '{"claim_id": "C1"}'}
    br.start()
    try:
        assert srv.ask_user("Here is my plan. OK?") == "sim reply #1"                      # 01_plan: no gate
        _step(trial, "02_implement")
        st["patch"] = "diff --git a/lib/a.go b/lib/a.go\n+fix\n"
        out2 = srv.ask_user("I implemented it, please review")
        if arm == "a5_stock":   # never withheld, no note, byte-identical delivery
            assert out2 == "sim reply #2" and len(st["sim_inputs"]) == 2
            assert br.gate.doc["status"] == "not_required"
            return
        assert out2 == a5.gate_prompt("commit_before") and len(st["sim_inputs"]) == 1     # simulator not consulted
        out3 = srv.ask_user("not a block")
        assert out3.startswith("The harness could not accept your [CLAIM_BOOK] block") and len(st["sim_inputs"]) == 1
        out4 = srv.ask_user(BOOK_TXT)
        assert out4.startswith("sim reply #2\n\n[harness] ") and "C1" in out4               # HOLD at p = 0.9
        sim_in = json.dumps(st["sim_inputs"][-1])
        assert "I implemented it, please review" in sim_in and "CLAIM_BOOK" not in sim_in and "not a block" not in sim_in
        assert [m["content"] for m in st["sim_inputs"][-1] if m["role"] == "user"] == [
            "Here is my plan. OK?", "I implemented it, please review"]
        out5 = srv.ask_user("Anything else?")
        assert out5.startswith("sim reply #3\n\n[harness] ")
        asks = [json.loads(x) for x in (us / "asks.jsonl").read_text().splitlines()]
        assert [bool(a["redo"].get("a5_withheld")) for a in asks] == [False, True, True, False, False]
        assert asks[3]["sim_question"] == "I implemented it, please review" and asks[3]["redo"]["a5_released_call"] == 2
        assert (us / "snapshots" / "ask-2.patch").read_text() == st["patch"]              # pre-review snapshot
        gd = json.loads((us / "a5" / "gate.json").read_text())
        assert gd["call_index"] == 2 and gd["resolved_call"] == 4 and gd["status"] == "ok"
    finally:
        br.stop()


def test_sidecar_releases_held_question_when_broker_fails(tmp_path, monkeypatch):
    us = tmp_path / "us"
    us.mkdir()
    srv, st = _load_server(monkeypatch, tmp_path, us, "a5_review_only")
    replies = iter([{"ok": True, "a5_gate": {"action": "withhold", "message": "DEMAND"}},
                    {"ok": False, "error": "broker not alive"}, {"ok": False}])
    srv._redo_rpc = lambda kind, n, payload: next(replies) if kind == "turn_end" else {"ok": False}
    assert srv.ask_user("review my change") == "DEMAND" and not st["sim_inputs"]
    assert srv.ask_user("[REVIEW]{}[/REVIEW]") == "sim reply #1"
    users = [m["content"] for m in st["sim_inputs"][0] if m["role"] == "user"]
    assert users == ["review my change"]                       # the payload never reaches the simulator
    rec = [json.loads(x) for x in (us / "asks.jsonl").read_text().splitlines()][-1]
    assert rec["redo"]["a5_release_without_broker"] and rec["sim_question"] == "review my change"
