"""Gemini-only seats (2026-09-27): seat config, the structured-call client, runner environments,
and the final-pass rule without the dropped codex judge. No network."""
from __future__ import annotations

import json
import types

import pytest

from fsa.redo import llm_seats
from fsa.redo import metrics as M


def _fake_chat(outputs):
    """A chat.completions stand-in returning (or raising) the given items in order."""
    calls = []

    def create(**kw):
        calls.append(kw)
        o = outputs[len(calls) - 1]
        if isinstance(o, Exception):
            raise o
        msg = types.SimpleNamespace(content=o)
        u = types.SimpleNamespace(prompt_tokens=5, completion_tokens=1, total_tokens=9,
                                  completion_tokens_details=None, prompt_tokens_details=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")], usage=u,
                                     model="gemini-3.5-flash", id=f"r{len(calls)}")
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))), calls


def test_judge_seat_default_and_override(monkeypatch):
    monkeypatch.delenv("REDO_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("REDO_JUDGE_EFFORT", raising=False)
    s = llm_seats.judge_seat()
    assert (s["model"], s["reasoning_effort"]) == ("gemini-3.5-flash", "high")
    assert llm_seats.backend(s["model"]) == "gemini"
    monkeypatch.setenv("REDO_JUDGE_MODEL", "gemini-x")
    assert llm_seats.judge_seat()["model"] == "gemini-x" and llm_seats.judge_seat()["source"] == "env"


def test_seat_file_has_no_openai_seat():
    doc = llm_seats.seats_doc()
    models = [v.get("model") for v in doc["seats"].values()]
    assert not any(m and "gpt" in m for m in models)
    assert doc["seats"]["swe_together_correctness_judge"]["status"] == "dropped"


def test_json_call_patient_retry_and_invalid_output():
    client, calls = _fake_chat([RuntimeError("Error code: 429 RESOURCE_EXHAUSTED"),
                                RuntimeError("503 UNAVAILABLE high demand"), "not json", '{"a": 1}'])
    slept, logs = [], []
    r = llm_seats.json_call("s", "u", {"type": "object"}, name="t", model="gemini-3.5-flash", effort="high",
                            client=client, sleep=slept.append, log=logs.append)
    assert r["output"] == {"a": 1} and r["attempts"] == 4
    assert len(slept) == 3 and slept[0] >= 10  # transient backoff starts at 10 s
    assert [l["ok"] for l in logs] == [False, False, False, True]
    assert calls[0]["reasoning_effort"] == "high" and calls[0]["response_format"]["type"] == "json_schema"
    assert logs[-1]["usage"]["reasoning_tokens"] == 3


def test_json_call_does_not_retry_bad_request():
    client, calls = _fake_chat([RuntimeError("Error code: 400 invalid argument")])
    with pytest.raises(RuntimeError):
        llm_seats.json_call("s", "u", None, name="t", model="gemini-3.5-flash", effort=None, client=client,
                            sleep=lambda s: None)
    assert len(calls) == 1


def test_structured_call_uses_judge_seat(tmp_path):
    from fsa.redo.swe_interact import llm
    client, calls = _fake_chat(['{"ok": true}'])
    r = llm.structured_call("s", "u", {"type": "object"}, name="x", log_dir=tmp_path, client=client)
    assert r["output"] == {"ok": True} and calls[0]["model"] == llm.MODEL == "gemini-3.5-flash"
    assert len(list((tmp_path / "calls").glob("*.json"))) == 1


def test_swe_interact_seat_env_is_gemini_only(monkeypatch):
    from fsa.redo.swe_interact import run as swi_run
    from fsa.redo.swe_interact.config import TASKS, load_rungs, subset_tasks
    if not (TASKS.is_dir() and (TASKS.parents[1] / ".git").exists()):
        pytest.skip("no SWE-Interact checkout")
    monkeypatch.setenv("GEMINI_API_KEY", "gkey")
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    cfg = load_rungs()
    env = swi_run.seat_env(cfg, subset_tasks()[0])
    assert env["SIM_USER_MODEL"] == "gemini-3.1-pro-preview"
    assert env["LITELLM_BASE_URL"] == "https://generativelanguage.googleapis.com/v1beta/openai#"
    # the sidecar appends /v1; after '#' it is a URL fragment, never sent
    assert (env["LITELLM_BASE_URL"] + "/v1").split("#")[0] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert env["LITELLM_API_KEY"] == "gkey" and env["OPENAI_API_KEY"] == "gkey"
    assert env["EVAL_MODEL"] == "gemini-3.5-flash"
    assert env["OPENAI_API_BASE"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert "real-openai-key" not in env.values()
    assert "GEMINI_API_KEY" not in env and env["GOOGLE_GENERATIVE_AI_API_KEY"] == "gkey"


def test_swe_together_seat_env_drops_codex_judge(monkeypatch):
    from fsa.redo.swe_together import run as swt_run
    from fsa.redo.swe_together.config import load_rungs
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    cfg = load_rungs()
    assert not swt_run.judge_seat_active(cfg)
    env = swt_run.seat_env(cfg, "docker")
    assert "OPENAI_API_KEY" not in env and "SWT_JUDGE_BACKEND" not in env and "SWT_CODEX_AUTH_JSON" not in env


def test_final_pass_from_verifier_reward_and_frozen_intents(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "_REF_REWARD", {"t1": 0.64})
    td = tmp_path / "cell" / "trials" / "t1__x"
    (td / "agent" / "snapshots").mkdir(parents=True)
    (td / "config.json").write_text(json.dumps({"task": {"path": "/x/tasks/t1"}}))
    (td / "agent" / "turns.jsonl").write_text(json.dumps({"turn": 0, "opened_by": "instruction"}) + "\n")
    (td / "agent" / "snapshots" / "turn-0.patch").write_text("diff --git a/a b/a\n+x\n")
    (td / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 0.64}}}))
    jd = {"durable": ["open"], "snapshots": {"999": {"revealed": ["open", "0"],
                                                     "verdicts": {"open": "satisfied", "0": "not_satisfied"}}}}
    row = M.trial_metrics(td, jd, None, {"rung": "W", "arm": "stock", "overlay": False})
    assert row["tests_ok"] and row["intents_ok"] and row["final_pass"]  # intent 0 is not durable
    jd["durable"] = None
    row = M.trial_metrics(td, jd, None, {"rung": "W", "arm": "stock", "overlay": False})
    assert row["tests_ok"] and not row["intents_ok"] and not row["final_pass"]
    (td / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 0.5}}}))
    jd["durable"] = ["open"]
    row = M.trial_metrics(td, jd, None, {"rung": "W", "arm": "stock", "overlay": False})
    assert not row["tests_ok"] and not row["final_pass"] and row["judged"]
