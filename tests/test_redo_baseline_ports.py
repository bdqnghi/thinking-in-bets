"""Baseline ports (tracker = controlled port of IDSS, mediator = controlled port of
Mediator-Assistant): the shared module, the SWE-Interact broker glue and the SWE-Together clone
glue of the baseline-port patch revision, on synthetic sessions (no model calls)."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest

from fsa.redo import ports as P
from fsa.redo.swe_interact import arms as A
from fsa.redo.swe_interact import broker as B
from fsa.redo.swe_interact import check_arms as SC
from fsa.redo.swe_together import check_arms as TC

ROOT = Path(__file__).resolve().parents[1]
PORTS_PATCH = ROOT / "third_party" / "patches" / "swe-together" / "redo_v1_ports.patch"

STATE = {"intents": [{"name": "fix_empty_input", "status": "active", "reason": "user asked", "depends_on": []},
                     {"name": "keep_api", "status": "completed", "outcome": "signature unchanged", "depends_on": []}],
         "variables": [{"name": "empty_return", "value": "[]", "source": "askable", "known": True,
                        "evidence": "user: return an empty list"}],
         "constraints": [{"scope": "fix_empty_input", "condition": "tests pass", "status": "unsatisfied",
                          "evidence": None, "grounding": {"entity": "command:pytest", "attribute": "exit",
                                                          "op": "==", "value": 0}},
                         {"scope": "keep_api", "condition": "api.py untouched", "status": "satisfied",
                          "evidence": None, "grounding": None}]}


def fake_completion(content):
    calls = []

    def f(**kw):
        calls.append(kw)
        c = content(kw) if callable(content) else content
        msg = types.SimpleNamespace(content=c)
        usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=None,
                                      completion_tokens_details=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                     usage=usage, model="gemini-x")
    f.calls = calls
    return f


def ev_text(t):
    return json.dumps({"type": "text", "part": {"text": t}})


def ev_tool(tool, inp, out="", exit_=None, status="completed"):
    st = {"status": status, "input": inp, "output": out, "metadata": {"exit": exit_} if exit_ is not None else {}}
    return json.dumps({"type": "tool_use", "part": {"tool": tool, "state": st}})


def ev_ask(q):
    return json.dumps({"type": "tool_use", "part": {"tool": "user-server_ask_user",
                                                    "state": {"status": "completed", "input": {"question": q}}}})


DIFF = """diff --git a/src/parse.py b/src/parse.py
index 1..2 100644
--- a/src/parse.py
+++ b/src/parse.py
@@ -1,2 +1,3 @@
-old
+new
+more
diff --git a/src/gone.py b/src/gone.py
deleted file mode 100644
--- a/src/gone.py
+++ /dev/null
@@ -1 +0,0 @@
-x
"""

TURN0 = "\n".join([ev_text("Looking at the parser."), ev_tool("read", {"filePath": "/repo/src/parse.py"}, "1: old"),
                   ev_tool("edit", {"filePath": "/repo/src/parse.py", "oldString": "old", "newString": "new"}),
                   ev_tool("edit", {"filePath": "/repo/src/util.py"}),
                   ev_tool("bash", {"command": "pytest -q"}, "1 failed", exit_=1),
                   ev_text("Done: the parser now accepts ISO dates.")])
TURN1 = "\n".join([ev_tool("bash", {"command": "pytest -q"}, "3 passed", exit_=0), ev_text("Fixed the empty case.")])


# ── shared module ───────────────────────────────────────────────────────────

def test_fact_layer_is_deterministic_with_provenance_and_ops():
    F = P.build_fact_layer([(0, TURN0), (1, TURN1)], DIFF, "turn-1")
    parse = F.entities["file:src/parse.py"]  # the tool path /repo/src/parse.py was MERGEd into the diff path
    assert parse["attrs"]["in_diff"] and parse["attrs"]["edited"] and parse["attrs"]["lines_added"] == 2
    assert "/repo/src/parse.py" in parse["aliases"]
    assert not F.entities["file:src/gone.py"]["valid"]                   # DELETE marks, never drops
    assert F.entities["file:/repo/src/util.py"]["attrs"]["in_diff"] is False  # edited, no net change
    cmd = F.entities["command:pytest -q"]["attrs"]
    assert cmd["exit"] == 0 and cmd["runs"] == 2 and cmd["output_tail"] == "3 passed"  # newer wins
    assert {o["op"] for o in F.ops} >= {"ADD", "UPDATE", "DELETE", "MERGE"}
    assert F.entities["repository:working tree"]["attrs"]["files_changed"] == 2
    again = P.build_fact_layer([(0, TURN0), (1, TURN1)], DIFF, "turn-1")
    assert again.render() == F.render() and "[Entity: file src/parse.py]" in F.render()
    assert F.find("file:parse.py") is parse and F.find("command:pytest")["id"] == "pytest -q"


def test_state_schema_and_constraint_propagation():
    with pytest.raises(ValueError):
        P.normalize_state({"intents": []})
    with pytest.raises(ValueError):
        P.normalize_state({"intents": [{"name": "x", "status": "maybe"}]})
    F = P.build_fact_layer([(0, TURN0)], DIFF, "turn-0")  # pytest exit 1 at this point
    st, log = P.propagate(P.normalize_state(json.loads(json.dumps(STATE))), F)
    c = st["constraints"][0]
    assert c["status"] == "violated" and "exit = 1" in c["evidence"]
    it = st["intents"][0]
    assert it["status"] == "blocked" and "tests pass" in it["reason"] and any("blocked_intent" in r for r in log)
    assert st["constraints"][1]["status"] == "satisfied"            # ungrounded: left as the tracker said
    F2 = P.build_fact_layer([(0, TURN0), (1, TURN1)], DIFF, "turn-1")
    st2, _ = P.propagate(P.normalize_state(json.loads(json.dumps(STATE))), F2)
    assert st2["constraints"][0]["status"] == "satisfied" and st2["intents"][0]["status"] == "active"
    # a grounded constraint that cannot be evaluated is never turned into "violated"
    s3 = json.loads(json.dumps(STATE))
    s3["constraints"][0].update(status="violated", grounding={"entity": "command:make lint", "attribute": "exit",
                                                               "op": "==", "value": 0})
    st3, _ = P.propagate(P.normalize_state(s3), F2)
    assert st3["constraints"][0]["status"] == "unsatisfied" and st3["intents"][0]["status"] == "active"
    txt = P.render_state(st)
    assert 'status: blocked' in txt and "source: askable | known: true" in txt


def test_tracker_call_carries_state_and_composes(tmp_path):
    f = fake_completion(json.dumps(STATE))
    arm = P.PortArm("tracker", tmp_path, "google/gemini-3.5-flash-lite", completion=f)
    doc = arm.tracker_update(index=0, transcript="T", streams=[(0, TURN0)], diff_text=DIFF, snapshot="turn-0",
                             reply="the empty case is still wrong")
    kw = f.calls[0]
    assert kw["model"] == "gemini/gemini-3.5-flash-lite" and kw["reasoning_effort"] == "high"
    assert kw["response_format"] == {"type": "json_object"}
    assert "STATE MAINTENANCE PROTOCOL" in kw["messages"][0]["content"]
    u = kw["messages"][1]["content"]
    assert "## The user's latest message\nthe empty case is still wrong" in u and "(none yet)" in u
    assert "[Entity: file src/parse.py]" in u
    assert doc["ok"] and doc["phase"] == "after_reply" and (tmp_path / "ports" / "turn-0.json").exists()
    assert doc["state"]["intents"][0]["status"] == "blocked"  # propagation ran on the model's state
    msg = arm.compose("the empty case is still wrong", doc)
    assert msg.startswith("the empty case is still wrong\n\n---\n(Task state maintained by your harness")
    assert "BEHAVIORAL GUIDELINES" in msg and "=== FACT LAYER (verified entities) ===" in msg
    arm.tracker_update(index=1, transcript="T", streams=[(0, TURN0)], diff_text=DIFF, snapshot="turn-1", reply="ok")
    assert 'name: "fix_empty_input"' in f.calls[1]["messages"][1]["content"]  # previous state layer carried
    assert len((tmp_path / "port_calls.jsonl").read_text().splitlines()) == 2


def test_port_failure_passes_reply_through(tmp_path):
    arm = P.PortArm("tracker", tmp_path, "google/m", completion=fake_completion('{"intents": []}'),
                    sleep=lambda s: None)
    doc = arm.tracker_update(index=0, transcript="T", streams=[], diff_text="", snapshot="turn-0", reply="r")
    assert not doc["ok"] and doc["attempts"] == 4 and arm.compose("r", doc) == "r" and arm.prev_state is None
    med = P.PortArm("mediator", tmp_path / "m", "google/m", completion=fake_completion(""), sleep=lambda s: None,
                    experiences={"rules": "1. x"})
    d = med.mediate(index=0, dialogue=[{"role": "user", "content": "hi"}])
    assert not d["ok"] and med.compose("hi", d) == "hi"


def test_mediator_prompt_is_the_papers_and_rewrite_replaces_reply(tmp_path):
    f = fake_completion("I need you to make the parser return [] on empty input and keep the API unchanged.")
    med = P.PortArm("mediator", tmp_path, "google/gemini-3.5-flash-lite", completion=f,
                    experiences={"rules": "1. Keep every user constraint.", "file_sha256": "x"})
    dialogue = [{"role": "user", "content": "parse dates"}, {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "empty case wrong"}]
    doc = med.mediate(index=2, dialogue=dialogue)
    text = f.calls[0]["messages"][0]["content"]
    assert len(f.calls[0]["messages"]) == 1 and "response_format" not in f.calls[0]
    assert text.startswith("You are an intermediary AI mediator responsible for analyzing and rewriting the request")
    assert "[Important Rules]\n1. Keep every user constraint.\n" in text
    assert "User: parse dates\n\nAI assistant: Done.\n\nUser: empty case wrong" in text
    assert text.rstrip().endswith("Please output:")
    assert med.compose("empty case wrong", doc) == doc["rewrite"] and doc["reply"] == "empty case wrong"


def test_mediator_experiences_frozen_for_every_rung_and_refiner_prompt_verbatim():
    doc = json.loads((P.PROMPTS / "mediator_experiences.json").read_text())
    rungs = json.loads((ROOT / "fsa" / "redo" / "swe_together" / "rungs.json").read_text())["rungs"]
    for r in rungs.values():
        m = r["model"].split("/", 1)[1]
        e = P.load_experiences(r["model"])
        assert e and e["rules"].strip() and e["rules_sha256"] == hashlib.sha256(e["rules"].encode()).hexdigest(), m
    subset = {t["task"] for t in json.loads((ROOT / "configs" / "subsets" / "swe-together.json").read_text())["selected"]}
    assert len(doc["pairs"]) == 5 and not subset & set(doc["pairs"])
    assert doc["refiner_prompt_sha256"] == P.prompts_sha256()["refiner_prompt.md"]
    assert "[[RULES]]" in P.prompt("refiner_prompt.md") and "[[EXPS]]" in P.prompt("mediator_prompt.md")


def test_dialogue_keeps_opening_request():
    msgs = [{"role": "user", "content": "OPEN " + "x" * 50}] + [
        {"role": "user" if i % 2 else "assistant", "content": f"m{i} " + "y" * 100} for i in range(50)]
    out = P.render_dialogue(msgs, budget=1000)
    assert out.startswith("User: OPEN") and "earlier conversation elided" in out and out.endswith("y" * 100)


# ── SWE-Interact (broker glue) ─────────────────────────────────────────────

def _swi_trial(tmp_path, arm):
    job = tmp_path / "job"
    trial = job / "trialX"
    us = trial / "user-sim"
    (us / "snapshots").mkdir(parents=True)
    (us / "snapshots" / "ask-1.patch").write_text(DIFF)
    sd = trial / "steps" / "01_plan" / "agent"
    sd.mkdir(parents=True)
    (sd / "opencode.attempt-1.txt").write_text(TURN0 + "\n" + ev_ask("Plan ok?"))
    live = trial / "agent"
    live.mkdir()
    (live / "opencode.attempt-1.txt").write_text(TURN1 + "\n" + ev_ask("Implemented; review?"))
    task_dir = tmp_path / "task"
    for s in A.STEP_ORDER:
        (task_dir / "steps" / s).mkdir(parents=True)
        (task_dir / "steps" / s / "instruction.md").write_text(f"instr {s}")
    cfg = A.RedoConfig(arm=arm)
    content = json.dumps(STATE) if arm == "tracker" else "I want the empty case to return [] and nothing else changed."
    br = B.TrialBroker(job, "deepswe_httpx-streaming-json-iteration", "google/m", cfg)
    br.task_dir = task_dir
    br.trial_dir = trial
    br._port = P.PortArm(arm, us, "google/m", prefix="ask", completion=fake_completion(content),
                         experiences={"rules": "1. r"} if arm == "mediator" else None)
    br.asks.append({"call_index": 0, "question": "Plan ok?", "reply": "yes, go ahead"})
    return br, us


@pytest.mark.parametrize("arm", ["tracker", "mediator"])
def test_swi_broker_port_after_reply_and_isolated(tmp_path, arm):
    br, us = _swi_trial(tmp_path, arm)
    te = br.turn_end(1, {"question": "Implemented; review?"})
    assert "port_doc" not in te                                 # nothing before the simulator is consulted
    t_dec = time.time()
    sr = br.sim_reply(1, {"reply": "the empty case is wrong", "sim_decided_ts": t_dec})
    assert sr["history_reply"] == "the empty case is wrong"     # the simulator's history holds its own reply
    assert sr["port_ok"] and sr["port_call_started_ts"] >= t_dec and sr["port_doc"] == "ports/ask-1.json"
    doc = json.loads((us / "ports" / "ask-1.json").read_text())
    if arm == "tracker":
        assert sr["deliver"].startswith("the empty case is wrong\n\n---\n(Task state")
        assert "command:pytest -q" in doc["fact_layer_rendered"].replace("[Entity: command ", "command:")
        assert {t for t, _ in A.step_streams(br.trial_dir)} == {1, 2}
    else:
        assert sr["deliver"] == doc["rewrite"] and sr["deliver"] != sr["history_reply"]
        conv = br._port._completion.calls[0]["messages"][0]["content"]
        assert conv.index("User: instr 01_plan") < conv.index("AI assistant: Looking at the parser.") < conv.index(
            "Plan ok?") < conv.index("User: yes, go ahead") < conv.index("instr 02_implement") < conv.index(
            "User: the empty case is wrong")
    # check_arms on the synthetic trial: simulator inputs without port material pass, with it fail
    (us / "sim_inputs").mkdir()
    (us / "sim_inputs" / "ask-1.json").write_text(json.dumps([{"role": "system", "content": "persona"},
                                                              {"role": "user", "content": "Implemented; review?"}]))
    row = {"call_index": 1, "reply": "the empty case is wrong", "history_reply": sr["history_reply"],
           "delivered": sr["deliver"], "redo": {"arm": arm, "sim_consult_started_ts": t_dec - 1,
                                                "sim_decided_ts": t_dec, "turn_end": {"ok": True, **te},
                                                "sim_reply": {"ok": True, **sr}}}
    (us / "asks.jsonl").write_text(json.dumps(row) + "\n")
    trial = br.trial_dir
    assert SC.check_trial(trial)["problems"] == []
    leak = sr["deliver"] if arm == "mediator" else "CURRENT TASK STATE (IDSS)"
    (us / "sim_inputs" / "ask-2.json").write_text(json.dumps([{"role": "assistant", "content": leak}]))
    assert any("arm material" in p for p in SC.check_trial(trial)["problems"])


def test_swi_config_record_labels_controlled_port():
    rec = A.config_record(A.RedoConfig(arm="mediator"), "google/gemini-3.5-flash-lite")
    assert "controlled port" in rec["port"]["label"] and rec["port"]["experiences"]["rules_sha256"]
    assert A.RedoConfig(arm="tracker").port == "tracker" and A.RedoConfig(arm="recap").port is None


# ── SWE-Together (clone glue, baseline-port patch revision) ──────────────────

def _new_files(patch: str) -> dict[str, str]:
    """Contents of the files a git patch adds (``--- /dev/null``)."""
    out, cur, lines, nonl = {}, None, [], False
    for block in patch.split("\ndiff --git ")[0:]:
        head, _, body = block.partition("\n@@ ")
        if "--- /dev/null" not in head:
            continue
        path = head.split("+++ b/", 1)[1].split("\n", 1)[0]
        lines = []
        for ln in ("@@ " + body).split("\n")[1:]:
            if ln.startswith("+"):
                lines.append(ln[1:])
            elif ln.startswith("\\ No newline"):
                nonl = True
        out[path] = "\n".join(lines) + ("" if nonl else "\n")
        nonl = False
    return out


@pytest.fixture(scope="module")
def swt_pkg(tmp_path_factory):
    if not PORTS_PATCH.exists():
        pytest.skip("baseline-port patch revision not recorded")
    files = _new_files(PORTS_PATCH.read_text())
    d = tmp_path_factory.mktemp("swt") / "user_agent"
    for rel, txt in files.items():
        if rel.startswith("src/user_agent/"):
            p = d / rel.removeprefix("src/user_agent/")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)
    pkg = types.ModuleType("swt_ports_pkg")
    pkg.__path__ = [str(d)]
    sys.modules["swt_ports_pkg"] = pkg
    spec = importlib.util.spec_from_file_location("swt_ports_pkg.redo_arms", d / "redo_arms.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swt_ports_pkg.redo_arms"] = mod
    spec.loader.exec_module(mod)
    return types.SimpleNamespace(dir=d, arms=mod)


def test_swt_clone_copy_is_byte_identical(swt_pkg):
    assert (swt_pkg.dir / "redo_ports.py").read_bytes() == Path(P.__file__).read_bytes()
    theirs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (swt_pkg.dir / "prompts" / "ports").iterdir()}
    assert theirs == P.prompts_sha256()
    assert swt_pkg.arms.ARMS[-2:] == ("tracker", "mediator")


def _swt_logs(tmp_path):
    ag = tmp_path / "agent"
    ag.mkdir()
    (ag / "opencode.txt.turn-0").write_text(TURN0)
    (ag / "opencode.txt.turn-1").write_text(TURN1)
    openers = [{"turn": 0, "opened_by": "initial"}, {"turn": 1, "opened_by": "sim", "message": "use ISO dates"}]
    return ag, openers


@pytest.mark.parametrize("arm", ["tracker", "mediator"])
def test_swt_redo_arms_port_update_and_compose(swt_pkg, tmp_path, arm):
    RA = swt_pkg.arms
    ag, openers = _swt_logs(tmp_path)
    cfg = RA.RedoConfig(arm=arm)
    ra = RA.RedoArms(cfg, ag, "google/gemini-3.5-flash-lite")
    content = json.dumps(STATE) if arm == "tracker" else "I need the empty case fixed; keep ISO dates."
    ra.port._completion = fake_completion(content)
    t_dec = time.time()
    instr = "parse dates\n\n## Repository Configuration Files\nCLAUDE.md ...\n\nIMPORTANT: Work incrementally. ..."
    doc = asyncio.run(ra.port_update(snapshot_turn=1, loop_turn=2, instruction=instr, openers=openers,
                                     diff_text=DIFF, reply="empty case wrong", sim_decided_ts=t_dec, opened_by="sim"))
    assert doc["ok"] and doc["call_started_at"] >= t_dec and (ag / "ports" / "turn-1.json").exists()
    msg = ra.compose("empty case wrong", None, "parse dates", doc)
    sent = ra.port._completion.calls[0]["messages"]
    if arm == "tracker":
        assert msg.startswith("empty case wrong\n\n---\n(Task state") and "## Turn 1 - the user: use ISO dates" in sent[1]["content"]
    else:
        assert msg == doc["rewrite"]
        assert "CLAUDE.md" not in sent[0]["content"] and "Work incrementally" not in sent[0]["content"]
        assert "User: parse dates\n\nAI assistant: Looking at the parser.\nDone: the parser now accepts ISO dates." \
               "\n\nUser: use ISO dates\n\nAI assistant: Fixed the empty case.\n\nUser: empty case wrong" in sent[0]["content"]
    # the stock and claim-book arms are untouched by the port revision
    assert RA.RedoArms(RA.RedoConfig(arm="recap"), ag, "google/m").port is None


def test_swt_check_arms_port_rules(tmp_path):
    trial = tmp_path / "run" / "trials" / "t__abc"
    ag = trial / "agent"
    (ag / "episode-2").mkdir(parents=True)
    (ag / "ports").mkdir()
    rewrite = "I need the empty case to return an empty list, nothing else."
    (ag / "ports" / "turn-1.json").write_text(json.dumps({"arm": "mediator", "ok": True, "rewrite": rewrite}))
    red = {"arm": "mediator", "overlay_trial": False, "sim_decided_ts": 10.0, "sim_consult_started_ts": 5.0,
           "port_doc": "ports/turn-1.json", "port_ok": True, "port_call_started_ts": 10.5, "port_written_ts": 11.0,
           "arm_block_delivered": True, "agent_message": rewrite}
    (ag / "turns.jsonl").write_text(json.dumps({"turn": 2, "opened_by": "sim", "message": "empty wrong", "redo": red}) + "\n")
    (ag / "episode-2" / "user_sim_prompt.json").write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}))
    assert TC.check_trial(trial)["problems"] == []
    (ag / "episode-2" / "user_sim_prompt.json").write_text(json.dumps({"messages": [{"role": "user", "content": rewrite}]}))
    assert TC.check_trial(trial)["problems"]
    red["port_call_started_ts"] = 9.0
    (ag / "turns.jsonl").write_text(json.dumps({"turn": 2, "opened_by": "sim", "message": "m", "redo": red}) + "\n")
    assert any("before the simulator decided" in p for p in TC.check_trial(trial)["problems"])
