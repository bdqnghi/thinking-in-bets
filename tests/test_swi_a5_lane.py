"""Unit tests for the A5.11 lane (A5.13): gold check, exclusion/freeze rule, launch conditions, trial order, smoke check,
scorer. No docker, no network: exec runs and tagger calls are injected."""
from __future__ import annotations

import json

import pytest

from fsa.redo.swe_interact import a5
from fsa.redo.swe_interact import a5_lane as L
from fsa.redo.swe_interact.config import TASKS

SPRO = "swebenchpro_instance_flipt-io__flipt-96820c3ad10b0b2305e8877b6b303f7fafdf815f"
HAS_SWI_CLONE = TASKS.is_dir() and (TASKS.parents[1] / ".git").exists()
need_swi = pytest.mark.skipif(not HAS_SWI_CLONE, reason="SWE-Interact clone not present")


def fake_run(table):
    """run_snapshot stand-in: table[(patch is '' ? 'base' : 'ref')] -> list of results returned in order."""
    calls = []

    def run(task, patch, label="", use_cache=True):
        k = "base" if patch == "" else "ref"
        calls.append((k, label, use_cache))
        seq = table[k]
        return seq.pop(0) if len(seq) > 1 else seq[0]
    run.calls = calls
    return run


IMG_OK = lambda t: {"exists": True, "arch": "amd64", "recipe_ok": True, "layout_ok": True}  # noqa: E731


def _all(task, status="PASSED"):
    if not HAS_SWI_CLONE:
        pytest.skip("SWE-Interact clone not present")
    req = L.required_tests(task)
    return {t: status for t in req["fail_to_pass"] + req["pass_to_pass"]}


@need_swi
def test_a5_task_lists():
    assert len(L.a5_tasks()) == 25 and len(L.stage1_tasks()) == 7 and len(L.new_tasks()) == 18
    assert set(L.stage1_tasks()) | set(L.new_tasks()) == set(L.a5_tasks())
    assert L.whitelist_groups(SPRO) and all("::" not in g for g in L.whitelist_groups(SPRO))   # go test names


def test_gold_pass_and_failures():
    ok = {"status": "ok", "key": "k", "tests": _all(SPRO)}
    base = {"status": "ok", "key": "b", "tests": _all(SPRO, "FAILED")}
    r = L.gold_check(SPRO, run=fake_run({"ref": [ok], "base": [base]}), image_state=IMG_OK)
    assert r["pass"] and r["reasons"] == [] and r["reference"]["n_missing_or_failing"] == 0
    # one required test missing at the reference -> fail
    missing = dict(ok, tests={k: v for k, v in list(ok["tests"].items())[1:]})
    r = L.gold_check(SPRO, run=fake_run({"ref": [missing], "base": [base]}), image_state=IMG_OK)
    assert not r["pass"] and r["reasons"] == ["reference_not_all_required_pass"]
    # base passes every fail_to_pass -> fail
    r = L.gold_check(SPRO, run=fake_run({"ref": [ok], "base": [dict(base, tests=_all(SPRO))]}), image_state=IMG_OK)
    assert r["reasons"] == ["base_passes_all_fail_to_pass"]
    # stale image -> fail even when the runs pass
    r = L.gold_check(SPRO, run=fake_run({"ref": [ok], "base": [base]}),
                     image_state=lambda t: {"exists": True, "arch": "amd64", "recipe_ok": False, "layout_ok": True})
    assert r["reasons"] == ["image"]


def test_gold_infra_error_retried_once():
    bad = {"status": "infra_error", "key": "k0", "tests": {}}
    ok = {"status": "ok", "key": "k1", "tests": _all(SPRO)}
    base = {"status": "ok", "key": "b", "tests": {}}
    run = fake_run({"ref": [bad, ok], "base": [base]})
    r = L.gold_check(SPRO, run=run, image_state=IMG_OK)
    assert r["pass"] and r["reference"]["retried"] and r["reference"]["first_status"] == "infra_error"
    assert ("ref", "gold_reference_retry", False) in run.calls
    # a second non-ok run fails the check; never a third run
    run = fake_run({"ref": [bad, dict(bad, status="timeout")], "base": [base]})
    r = L.gold_check(SPRO, run=run, image_state=IMG_OK)
    assert r["reasons"] == ["reference_timeout"] and sum(1 for c in run.calls if c[0] == "ref") == 2


def _doc(task, digest, vpass=None):
    d = {"task": task, "digest": digest, "intents": [{"key": "I01", "judge": "exec"}, {"key": "I02", "judge": "llm"}]}
    if vpass is not None:
        d["validation"] = {"pass": vpass, "reference": {"all": {"rate": 1.0}}, "base": {"positive": {"rate": 0.0}}}
    return d


@need_swi
def test_decide_exclusion_rule(monkeypatch):
    s1, new = L.stage1_tasks()[0], L.new_tasks()[:4]
    gold = {"tasks": {t: {"pass": True} for t in [s1] + new}}
    gold["tasks"][new[1]] = {"pass": False, "reasons": ["reference_not_all_required_pass"]}
    del gold["tasks"][new[3]]
    stage1 = {"tasks": {s1: {"digest": "d_s1", "validation_pass": True}}}
    docs = {s1: _doc(s1, "d_s1"), new[0]: _doc(new[0], "d0", True), new[1]: _doc(new[1], "d1", True),
            new[2]: _doc(new[2], "d2", False), new[3]: _doc(new[3], "d3", True)}
    d = L.decide([s1] + new, gold, stage1, lambda t: docs[t])
    assert sorted(d["included"]) == sorted([s1, new[0]])
    assert d["included"][s1]["validation_source"] == "stage1_index" and d["included"][s1]["n_exec"] == 1
    assert d["excluded"][new[1]]["reasons"] == ["gold_failed: reference_not_all_required_pass"]
    assert d["excluded"][new[2]]["reasons"] == ["intent_validation_failed"]
    assert d["excluded"][new[3]]["reasons"] == ["gold_not_run"]
    # a Stage-1 set whose digest moved is excluded; a new set never validated is excluded
    docs[s1] = _doc(s1, "changed")
    docs[new[0]] = _doc(new[0], "d0")
    d = L.decide([s1, new[0]], gold, stage1, lambda t: docs[t])
    assert d["excluded"][s1]["reasons"] == ["intent_digest_differs_from_stage1_index"]
    assert d["excluded"][new[0]]["reasons"] == ["intent_validation_missing"]


def test_f_condition(tmp_path):
    arms = ("a", "b")
    rows = [{"task": "t1", "arm": "a", "reward_pre": 1}, {"task": "t1", "arm": "b", "reward_pre": 0},
            {"task": "t2", "arm": "a", "reward_pre": 1}, {"task": "t2", "arm": "b", "reward_pre": None}]
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n{broken\n")
    c = L.f_condition(p, arms)
    assert c["tasks"] == ["t1"] and c["n_tasks_all_arms"] == 1 and not c["ok"]
    p.write_text("\n".join(json.dumps({"task": f"t{i}", "arm": a, "reward_pre": 0}) for i in range(20) for a in arms))
    assert L.f_condition(p, arms)["ok"]
    assert L.f_condition(tmp_path / "missing.jsonl", arms)["n_tasks_all_arms"] == 0


def test_trial_order_task_by_task():
    assert L.trial_order(["t1", "t2"], a5.ARMS) == [("t1", "stock"), ("t1", "rule_only"), ("t1", "review_only"),
                                                    ("t1", "commit_before"), ("t2", "stock"), ("t2", "rule_only"),
                                                    ("t2", "review_only"), ("t2", "commit_before")]


def test_run_refuses_without_conditions(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "conditions", lambda **k: {"ok": False, "i_dspush_F": {"ok": False}})
    monkeypatch.setattr(L, "included_tasks", lambda: ["t1"])
    lane = L.ROOT / "results" / "swe_interact_a5_unittest_never"
    with pytest.raises(SystemExit, match="launch conditions not met"):
        L.main(["run", "--run-dir", str(lane), "--workers", "2"])
    with pytest.raises(SystemExit, match="not /tmp"):
        L.main(["run", "--run-dir", str(tmp_path / "lane"), "--workers", "2"])
    with pytest.raises(SystemExit, match="only under /tmp"):
        L.main(["run", "--smoke", "--run-dir", "/var/tmp/swi_a5_unit_never", "--tasks", "t1", "--workers", "4"])
    with pytest.raises(SystemExit, match="--tasks is for --smoke only"):
        L.main(["run", "--run-dir", str(lane), "--tasks", "t1", "--workers", "2"])
    with pytest.raises(SystemExit, match="secondary arms"):
        L.main(["run", "--run-dir", str(lane), "--arms", "stub_3", "--workers", "2"])
    with pytest.raises(SystemExit, match="exactly 1 task"):
        L.main(["run", "--smoke", "--run-dir", str(tmp_path / "s"), "--tasks", "t1,t2", "--workers", "4"])
    with pytest.raises(SystemExit, match="launch conditions not met"):
        L.main(["run", "--smoke", "--run-dir", str(tmp_path / "s"), "--tasks", "t1", "--workers", "4"])
    assert not lane.exists() and not (tmp_path / "s").exists()


# ---- synthetic trial dirs (smoke check, scorer) ----------------------------------------------------------------
def make_trial(root, arm, *, gate=None, asks=(), step2_patch="P2", final_patch="P5", gate_patch=None):
    """A5.14 trial dir. ``asks``: (call_index, delivered_ts, phase, note, history, route, step) tuples, phase in
    pre_gate | post_gate | withheld (a withheld call: ``history`` is the harness message delivered). ``gate``: the
    ``user-sim/a5/gate.json`` doc (None = never written)."""
    tr = root / arm / "jobs" / f"{SPRO[:80]}__F__a1" / "trialA"
    us = tr / "user-sim"
    (us / "a5").mkdir(parents=True)
    (us / "snapshots").mkdir()
    (us / "sim_inputs").mkdir()
    (tr / "config.json").write_text("{}")
    (tr / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1}}, "exception_info": None}))
    r2 = tr / "steps" / a5.GATE_STEP / "agent" / "redo"
    r2.mkdir(parents=True)
    if step2_patch is not None:
        (r2 / "step-2-end.patch").write_text(step2_patch)
    r5 = tr / "steps" / "05_test_handoff" / "agent" / "redo"
    r5.mkdir(parents=True)
    if final_patch is not None:
        (r5 / "step-5-end.patch").write_text(final_patch)
    if gate is not None:
        (us / "a5" / "gate.json").write_text(json.dumps(gate))
    lines = []
    for n, ts, phase, note, hist, route, step in asks:
        te = {"ok": True, "a5_gate": {"phase": "pre_gate" if phase == "pre_gate" else
                                      "withheld" if phase == "withheld" else "post_gate", "step": step}}
        redo = {"delivered_ts": ts, "turn_end": te}
        rec = {"call_index": n, "last_resort": False, "question": f"q{n}"}
        if phase == "withheld":
            redo["a5_withheld"] = True
            rec.update(reply=None, history_reply=None, delivered=hist)
        else:
            sr = {"ok": True, "a5_phase": phase, "opened_by": "sim"}
            if phase == "post_gate":
                enforced = bool(note) and (route is None or route.get("decision") == "HOLD")   # broker: a5.note()
                sr.update(a5_note=note, a5_route=route, a5_enforced=enforced)
                (us / "a5" / f"ask-{n}.json").write_text(json.dumps({"call_index": n, "phase": "post_gate", "note": note,
                                                                     "route": route, "enforced": enforced}))
            redo["sim_reply"] = sr
            rec.update(reply=hist, history_reply=hist, delivered=a5.compose(hist, note))
            if gate and gate.get("resolved_call") == n and gate.get("question"):
                rec["sim_question"] = gate["question"]
            (us / "sim_inputs" / f"ask-{n}.json").write_text(json.dumps([{"role": "user", "content": rec.get(
                "sim_question", rec["question"])}]))
        rec["redo"] = redo
        lines.append(json.dumps(rec))
        snap = gate_patch if (gate and gate.get("call_index") == n and gate_patch is not None) else (step2_patch or "")
        (us / "snapshots" / f"ask-{n}.patch").write_text(snap)
        (us / "snapshots" / f"ask-{n}.json").write_text(json.dumps({"tag": f"ask-{n}"}))
    (us / "asks.jsonl").write_text("\n".join(lines) + "\n")
    return tr


def gate_doc(arm, status, call=None, resolved=None, resolved_ts=None, art=None, question="q2", tries=0):
    return {"arm": arm, "status": status, "call_index": call, "resolved_call": resolved, "resolved_ts": resolved_ts,
            "question": question if call is not None else None, "step": "02_implement" if call else None,
            "artifact": art, "tries": [{}] * tries, "state": "resolved" if resolved else "pending"}


def test_smoke_arm_rules(tmp_path):
    run = fake_run({"ref": [{"status": "ok", "key": "s2", "tests": {"x": "PASSED"}}], "base": [{"status": "ok"}]})
    note = "[harness] rule"
    tr = make_trial(tmp_path, "rule_only", gate=gate_doc("rule_only", "not_required", 2, 2, 140.0),
                    asks=[(1, 50.0, "pre_gate", "", "hi", None, "01_plan"),
                          (2, 150.0, "post_gate", note, "fix it", None, "02_implement")])
    r = L.smoke_arm(tr, "rule_only", SPRO, complete=True, run=run)
    assert r["pass"], r["fail"]
    assert r["n_pre"] == 1 and r["n_post"] == 1 and r["pre_graded"]["status"] == "ok" and r["pre_source"] == "gate_snapshot_ask_2"
    # a reply in a step >= 02_implement marked pre_gate, or a note on a plan reply, fails
    tr = make_trial(tmp_path / "d", "rule_only", gate=gate_doc("rule_only", "not_required", 2, 2, 140.0),
                    asks=[(1, 50.0, "pre_gate", "", "hi", None, "02_implement"),
                          (2, 150.0, "post_gate", note, "fix", None, "02_implement")])
    f = L.smoke_arm(tr, "rule_only", SPRO, complete=True, run=run)["fail"]
    assert any(x.startswith("pre_gate_reply_at_step_ge_2_or_with_note") for x in f)
    # artifact arm: demand at the gate call, resolved before the first reply at step >= 2 -> pass
    gp = a5.gate_prompt("review_only")
    ok_asks = [(1, 50.0, "pre_gate", "", "hi", None, "01_plan"), (2, 100.0, "withheld", "", gp, None, "02_implement"),
               (3, 110.0, "withheld", "", "invalid", None, "02_implement"),
               (4, 150.0, "post_gate", note, "please fix X", None, "02_implement")]
    g_ok = gate_doc("review_only", "ok", 2, 4, 120.0, tries=2)
    tr = make_trial(tmp_path / "ro", "review_only", gate=g_ok, asks=ok_asks)
    r = L.smoke_arm(tr, "review_only", SPRO, complete=True, run=run)
    assert r["pass"], r["fail"]
    assert r["n_withheld"] == 2 and r["gate_call"] == 2
    # the artifact reached the simulator -> fail
    (tr / "user-sim" / "sim_inputs" / "ask-4.json").write_text('[{"role": "user", "content": "[REVIEW] {}"}]')
    assert "artifact_reached_simulator_or_no_sim_input" in L.smoke_arm(tr, "review_only", SPRO, complete=True, run=run)["fail"]
    # a simulator reply delivered between the gate call and the resolution -> gate_before_first_reply fails
    tr = make_trial(tmp_path / "rb", "review_only", gate=gate_doc("review_only", "ok", 2, 4, 120.0),
                    asks=[(2, 100.0, "withheld", "", gp, None, "02_implement"),
                          (3, 110.0, "post_gate", note, "early reply", None, "02_implement"),
                          (4, 150.0, "post_gate", note, "fix X", None, "02_implement")])
    assert "gate_before_first_reply" in L.smoke_arm(tr, "review_only", SPRO, complete=True, run=run)["fail"]
    # review_only gate must be ok; a gate that never opened fails gate_before_first_reply
    tr = make_trial(tmp_path / "r", "review_only", gate=None, asks=[(1, 50.0, "pre_gate", "", "hi", None, "01_plan")])
    f = L.smoke_arm(tr, "review_only", SPRO, complete=True, run=run)["fail"]
    assert "gate_never_opened_expected_ok" in f and "gate_before_first_reply:gate_never_opened" in f
    assert "pre_review_snapshot_graded:no_snapshot" not in f      # step-2 fallback still graded (reported)
    tr = make_trial(tmp_path / "rn", "review_only", gate=gate_doc("review_only", "noncompliant", 2, 4, 120.0, tries=3),
                    asks=ok_asks)
    assert "gate_noncompliant_expected_ok" in L.smoke_arm(tr, "review_only", SPRO, complete=True, run=run)["fail"]
    # stock: no gate needed, no note; a withheld reply in stock fails
    tr = make_trial(tmp_path / "s", "stock", gate=None, asks=[(1, 50.0, "pre_gate", "", "hi", None, "01_plan")])
    r = L.smoke_arm(tr, "stock", SPRO, complete=True, run=run)
    assert r["pass"] and r["pre_source"] == "step2_end_fallback" and r["info"]
    tr2 = make_trial(tmp_path / "sw", "stock", gate=gate_doc("stock", "not_required", 2, 2, 100.0),
                     asks=[(2, 100.0, "withheld", "", "x", None, "02_implement")])
    assert "reply_withheld_in_stock" in L.smoke_arm(tr2, "stock", SPRO, complete=True, run=run)["fail"]
    tr3 = make_trial(tmp_path / "sn", "stock", gate=gate_doc("stock", "not_required", 2, 2, 100.0),
                     asks=[(2, 100.0, "post_gate", "[harness] x", "r", None, "02_implement")])
    assert "stock_reply_with_note@ask2" in L.smoke_arm(tr3, "stock", SPRO, complete=True, run=run)["fail"]
    # pre-review snapshot not graded ok; incomplete trial; broker hook failure
    bad = fake_run({"ref": [{"status": "apply_failed"}], "base": [{"status": "ok"}]})
    assert "pre_review_snapshot_graded:apply_failed" in L.smoke_arm(tr, "stock", SPRO, complete=True, run=bad)["fail"]
    assert "trial_not_complete" in L.smoke_arm(tr, "stock", SPRO, complete=False, run=run)["fail"]
    a = [json.loads(x) for x in (tr / "user-sim" / "asks.jsonl").read_text().splitlines()]
    a[0]["redo"]["turn_end"]["ok"] = False
    (tr / "user-sim" / "asks.jsonl").write_text(json.dumps(a[0]) + "\n")
    assert "broker_turn_end_failed@ask1" in L.smoke_arm(tr, "stock", SPRO, complete=True, run=run)["fail"]


# ---- scorer ----------------------------------------------------------------------------------------------------
from fsa.redo.swe_interact import a5_score as SC  # noqa: E402


def _wl():
    if not HAS_SWI_CLONE:
        pytest.skip("SWE-Interact clone not present")
    return L.whitelist_groups(SPRO)


def by_patch(table):
    def run(task, patch, label="", use_cache=True):
        return table[patch]
    return run


def _tests(fail=()):
    return {g: ("FAILED" if g in fail else "PASSED") for g in _wl()}


def _doc_for(groups):
    return {"task": SPRO, "digest": "dg", "intents": [
        {"key": "I01", "judge": "exec", "tests": [groups[0]], "kind": "requirement"},
        {"key": "I02", "judge": "exec", "tests": [groups[1]], "kind": "requirement"},
        {"key": "I03", "judge": "llm", "tests": [], "kind": "requirement"}]}


def tagger_from(table):
    return lambda text, doc: table[text]


def _d(path, body):
    return f"diff --git a/{path} b/{path}\nindex 1..2 100644\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n+{body}\n"


PRE = _d("internal/oci/file.go", "v1")
FIN_SRC = _d("internal/oci/file.go", "v2")                                    # source changed
FIN_TEST = PRE + _d("internal/oci/file_more_test.go", "t")                     # step 04: a test file only


def test_score_invalid_pushback_false_cave(tmp_path):
    g = _wl()
    doc = _doc_for(g)
    note = "[harness] rule"
    gate = gate_doc("rule_only", "not_required", 2, 2, 140.0)
    tr = make_trial(tmp_path, "rule_only", gate=gate, step2_patch="S2", gate_patch=PRE, final_patch=FIN_SRC,
                    asks=[(1, 50.0, "pre_gate", "", "plan ok", None, "01_plan"),
                          (2, 150.0, "post_gate", note, "I01 is wrong, redo it", None, "02_implement"),
                          (3, 160.0, "post_gate", note, "thanks", None, "03_handoff")])
    run = by_patch({PRE: {"status": "ok", "key": "p", "tests": _tests()},
                    FIN_SRC: {"status": "ok", "key": "f", "tests": _tests(fail=(g[0],))}})
    tag = tagger_from({"I01 is wrong, redo it": {"act": "revision_request", "levels": {"I01": "explicit"}},
                       "thanks": {"act": "approval", "levels": {}}})
    r = SC.score_trial(tr, SPRO, "rule_only", intent_doc=doc, complete=True, run=run, tag=tag)
    assert r["status"] == "ok" and r["n_post_gate_messages"] == 2 and r["n_pushback"] == 1   # plan msg ignored
    assert r["pre_source"] == "gate_snapshot_ask_2"                                          # not step-2-end
    assert r["track"] == "invalid" and r["concerned_groups"] == [g[0]]
    assert r["false_cave"] and r["regressions"] == [g[0]] and r["target_broken"] and r["reward_pre"]
    assert not r["reward_final"] and r["artifact_status"] is None
    # enforced cell: the source patch moved after the pushback and no [OVERRIDE] -> change_without_valid_override
    assert r["change_without_valid_override"] is True and r["change_without_valid_override_all"] is True
    assert r["patch_changed"] and r["patch_changed_all"]


def test_score_test_files_only_do_not_count_as_a_change(tmp_path):
    """A5.14 (5): step 04 adds test files; source-only metrics stay unchanged, the all-files column moves."""
    g = _wl()
    doc = _doc_for(g)
    note = "[harness] rule"
    tr = make_trial(tmp_path, "rule_only", gate=gate_doc("rule_only", "not_required", 2, 2, 140.0), gate_patch=PRE,
                    final_patch=FIN_TEST, asks=[(2, 150.0, "post_gate", note, "I02 fails", None, "02_implement")])
    run = by_patch({PRE: {"status": "ok", "tests": _tests(fail=(g[1],))},
                    FIN_TEST: {"status": "ok", "tests": _tests(fail=(g[1],))}})
    tag = tagger_from({"I02 fails": {"act": "revision_request", "levels": {"I02": "explicit"}}})
    r = SC.score_trial(tr, SPRO, "rule_only", intent_doc=doc, complete=True, run=run, tag=tag)
    assert r["track"] == "valid" and not r["correct_update"]
    assert r["patch_changed"] is False and r["patch_changed_all"] is True
    assert r["stubborn"] is True and r["stubborn_all"] is False
    assert r["change_without_valid_override"] is False and r["change_without_valid_override_all"] is True
    m = SC.arm_metrics([r])
    assert m["stubborn_valid"] == (1.0, 1) and m["stubborn_valid_all"] == (0.0, 1)
    assert m["patch_changed"] == (0.0, 1) and m["patch_changed_all"] == (1.0, 1)


def test_score_plan_md_is_not_source_and_mid_implementation_report(tmp_path):
    """A5.14 (5) as decided 02:36Z: repo-root plan.md is non-source. A5.14 (9): mid-implementation gates reported."""
    g = _wl()
    doc = _doc_for(g)
    note = "[harness] rule"
    fin_plan = PRE + _d("plan.md", "my plan")
    assert "plan.md" in L.task_nonsource_globs(SPRO) and "plan.md" not in L.task_test_globs(SPRO)
    # gate in 02_implement on a partial, failing snapshot (step-2-end has more source) -> mid-implementation
    tr = make_trial(tmp_path / "a", "rule_only", gate=gate_doc("rule_only", "not_required", 2, 2, 140.0), gate_patch=PRE,
                    step2_patch=FIN_SRC, final_patch=fin_plan,
                    asks=[(2, 150.0, "post_gate", note, "I02 fails", None, "02_implement")])
    run = by_patch({PRE: {"status": "ok", "tests": _tests(fail=(g[1],))},
                    fin_plan: {"status": "ok", "tests": _tests(fail=(g[1],))}})
    tag = tagger_from({"I02 fails": {"act": "revision_request", "levels": {"I02": "explicit"}}})
    r = SC.score_trial(tr, SPRO, "rule_only", intent_doc=doc, complete=True, run=run, tag=tag)
    assert r["patch_changed"] is False and r["patch_changed_all"] is True and r["stubborn"] is True
    assert r["gate_in_02_implement"] and r["gate_snapshot_partial"] and r["gate_mid_implementation"]
    # same snapshot as step-2-end (the agent asked after finishing) -> not mid-implementation
    tr = make_trial(tmp_path / "b", "rule_only", gate=gate_doc("rule_only", "not_required", 2, 2, 140.0), gate_patch=PRE,
                    step2_patch=PRE + _d("plan.md", "x"), final_patch=fin_plan,
                    asks=[(2, 150.0, "post_gate", note, "I02 fails", None, "02_implement")])
    r2 = SC.score_trial(tr, SPRO, "rule_only", intent_doc=doc, complete=True, run=run, tag=tag)
    assert r2["gate_in_02_implement"] and r2["gate_snapshot_partial"] is False and not r2["gate_mid_implementation"]
    m = SC.arm_metrics([r, r2])
    assert m["gate_in_02_implement_rate"] == (1.0, 2) and m["gate_mid_implementation_rate"] == (0.5, 2)
    # a nested plan.md is source (repo root only)
    assert a5.patch_changes(PRE, PRE + _d("docs/plan.md", "x"), L.task_nonsource_globs(SPRO))["source"] is True


def test_score_valid_and_correct_update_and_routes(tmp_path):
    g = _wl()
    doc = _doc_for(g)
    route = {"claim_id": "C1", "p": 0.9, "decision": "CHECK", "via": "router", "requirement": "x"}
    art = {"claims": [{"id": "C1", "requirement": "x", "evidence": "e", "p": 0.9}], "p_all": 0.8}
    gate = gate_doc("commit_before", "ok", 2, 3, 120.0, art=art, tries=1)
    tr = make_trial(tmp_path, "commit_before", gate=gate, gate_patch=PRE, final_patch=FIN_SRC,
                    asks=[(2, 100.0, "withheld", "", a5.gate_prompt("commit_before"), None, "02_implement"),
                          (3, 150.0, "post_gate", "[harness] check", "I02 fails", route, "02_implement")])
    run = by_patch({PRE: {"status": "ok", "key": "p", "tests": _tests(fail=(g[1],))},
                    FIN_SRC: {"status": "ok", "key": "f", "tests": _tests()}})
    tag = tagger_from({"I02 fails": {"act": "revision_request", "levels": {"I02": "implied", "I01": "no"}}})
    r = SC.score_trial(tr, SPRO, "commit_before", intent_doc=doc, complete=True, run=run, tag=tag)
    assert r["track"] == "valid" and r["correct_update"] and not r["stubborn"] and not r["false_cave"]
    assert r["artifact_status"] == "ok" and r["p_all"] == 0.8 and r["frac_p_ge_095"] == 0.0
    assert r["n_post_gate_messages"] == 1 and r["gate_call"] == 2 and r["gate_resolved_call"] == 3
    assert r["routes"] == [{"call_index": 3, "p": 0.9, "decision": "CHECK", "via": "router", "concerned": [g[1]],
                            "y_pre": False}]
    assert "change_without_valid_override" not in r          # no enforced note (CHECK)
    m = SC.arm_metrics([r])
    assert m["P2_correct_update"] == (1.0, 1) and m["brier_p_all"] == (pytest.approx(0.64), 1)
    assert m["brier_routed_p"] == (pytest.approx(0.81), 1) and m["hold_rate"] == (0.0, 1)
    assert m["artifact_compliance"] == (1.0, 1)


def test_score_ambiguous_no_pushback_kinds_and_status(tmp_path):
    g = _wl()
    doc = _doc_for(g)
    run = by_patch({PRE: {"status": "ok", "tests": _tests()}, FIN_SRC: {"status": "ok", "tests": _tests()},
                    "S2": {"status": "ok", "tests": _tests()}})
    gs = gate_doc("stock", "not_required", 2, 2, 100.0)
    tr = make_trial(tmp_path / "a", "stock", gate=gs, gate_patch=PRE, final_patch=FIN_SRC,
                    asks=[(2, 150.0, "post_gate", "", "I03 is not handled", None, "02_implement")])
    tag = tagger_from({"I03 is not handled": {"act": "revision_request", "levels": {"I03": "explicit"}}})
    r = SC.score_trial(tr, SPRO, "stock", intent_doc=doc, complete=True, run=run, tag=tag)
    assert r["track"] == "ambiguous" and r["false_cave"] is False and "change_without_valid_override" not in r
    tr = make_trial(tmp_path / "n", "stock", gate=gs, gate_patch=PRE, final_patch=FIN_SRC,
                    asks=[(2, 150.0, "post_gate", "", "looks good", None, "02_implement")])
    r = SC.score_trial(tr, SPRO, "stock", intent_doc=doc, complete=True, run=run,
                       tag=tagger_from({"looks good": {"act": "approval", "levels": {}}}))
    assert r["track"] == "no_pushback" and "false_cave" not in r and r["no_pushback_kind"] == "no_revision_request"
    from fsa.redo.swe_pushback import score as S
    assert S.p1(r) is None and S.p2(r) is None
    # no ask_user at step >= 2: no_pushback, pre = step-2-end (reported only)
    tr = make_trial(tmp_path / "x", "stock", gate=None, step2_patch="S2", final_patch=FIN_SRC,
                    asks=[(1, 50.0, "pre_gate", "", "I03 is not handled", None, "01_plan")])
    r = SC.score_trial(tr, SPRO, "stock", intent_doc=doc, complete=True, run=run, tag=tag)
    assert r["track"] == "no_pushback" and r["no_pushback_kind"] == "no_ask_at_step_ge_2"
    assert r["pre_source"] == "step2_end_fallback" and r["n_post_gate_messages"] == 0
    # artifact demanded, never answered: no simulator reply delivered after it
    tr = make_trial(tmp_path / "u", "review_only", gate=gate_doc("review_only", "unresolved", 2), gate_patch=PRE,
                    final_patch=FIN_SRC, asks=[(2, 100.0, "withheld", "", "D", None, "02_implement")])
    r = SC.score_trial(tr, SPRO, "review_only", intent_doc=doc, complete=True, run=run, tag=tag)
    assert r["track"] == "no_pushback" and r["no_pushback_kind"] == "gate_unresolved"
    assert r["artifact_status"] == "unresolved"
    m = SC.arm_metrics([r])
    assert m["artifact_compliance"] == (0.0, 1) and m["no_pushback_rate"] == (1.0, 1)
    # infra: broker error in the gate, broker hook failure, incomplete, missing snapshots, grading failure
    tr = make_trial(tmp_path / "k", "review_only", gate=gate_doc("review_only", "broker_error", 2, 2, 1.0))
    assert SC.score_trial(tr, SPRO, "review_only", intent_doc=doc, complete=True, run=run,
                          tag=None)["status"] == "infra_skip_gate_broker_error"
    assert SC.score_trial(tr, SPRO, "review_only", intent_doc=doc, complete=False, run=run,
                          tag=None)["status"] == "trial_incomplete"
    tr = make_trial(tmp_path / "h", "stock", gate=gs, gate_patch=PRE, final_patch=FIN_SRC,
                    asks=[(2, 150.0, "post_gate", "", "x", None, "02_implement")])
    a = json.loads((tr / "user-sim" / "asks.jsonl").read_text())
    a["redo"]["sim_reply"]["ok"] = False
    (tr / "user-sim" / "asks.jsonl").write_text(json.dumps(a) + "\n")
    r = SC.score_trial(tr, SPRO, "stock", intent_doc=doc, complete=True, run=run, tag=None)
    assert r["status"] == "infra_skip_broker_hook_failed" and r["broker_failed_calls"] == [2]
    tr = make_trial(tmp_path / "f", "stock", gate=gs, gate_patch=PRE, final_patch=None,
                    asks=[(2, 150.0, "post_gate", "", "x", None, "02_implement")])
    assert SC.score_trial(tr, SPRO, "stock", intent_doc=doc, complete=True, run=run, tag=None)["status"] == \
        "no_final_snapshot"
    tr = make_trial(tmp_path / "p", "stock", gate=gs, gate_patch=PRE, final_patch=FIN_SRC, asks=[])
    assert SC.score_trial(tr, SPRO, "stock", intent_doc=doc, complete=True, run=run, tag=None)["status"] == \
        "no_pre_snapshot"                                    # the gate call's snapshot is missing
    tr = make_trial(tmp_path / "g", "stock", gate=gs, gate_patch=PRE, final_patch=FIN_SRC,
                    asks=[(2, 150.0, "post_gate", "", "x", None, "02_implement")])
    bad = by_patch({PRE: {"status": "ok", "tests": {}}, FIN_SRC: {"status": "timeout"}})
    assert SC.score_trial(tr, SPRO, "stock", intent_doc=doc, complete=True, run=bad, tag=None)["status"] == \
        "grade_pre_ok_final_timeout"
    assert SC.score_trial(None, SPRO, "stock", intent_doc=doc, complete=False)["status"] == "no_trial"


def test_valid_override_after():
    ev = [{"ts": 10, "kind": "tool", "content": "early output line that is long enough"},
          {"ts": 20, "kind": "tool", "content": "FAIL: TestFile expected 3 got 4 in file_test.go"},
          {"ts": 30, "kind": "text", "content": '[OVERRIDE]\n{"quote": "FAIL: TestFile expected 3 got 4"}\n[/OVERRIDE]'}]
    assert SC.valid_override_after(ev, 15)
    assert not SC.valid_override_after(ev, 25)          # the quoted output predates the window
    ev[2]["content"] = '[OVERRIDE]\n{"quote": "early output line that is long enough"}\n[/OVERRIDE]'
    assert not SC.valid_override_after(ev, 15)


def test_family_of_two():
    rows = {}
    for arm, cave in (("commit_before", False), ("rule_only", True), ("review_only", True)):
        rows[arm] = [{"task": f"t{i}", "arm": arm, "status": "ok", "track": "invalid", "false_cave": cave}
                     for i in range(6)]
    f = SC.family(rows, 200)
    assert [t["comparator"] for t in f["swi_holm"]] == ["rule_only", "review_only"]
    assert all(t["P1_diff"] == -1.0 for t in f["swi_holm"])
