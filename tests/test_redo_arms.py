"""Unit tests for the clone-side arm module (third_party/swe-together/src/user_agent/redo_arms.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsa.redo.swe_together.config import CLONE
from fsa.redo.swe_together.run import redo_arms_module, redo_env

pytestmark = pytest.mark.skipif(not (CLONE / "src" / "user_agent" / "redo_arms.py").exists(),
                                reason="SWE-Together clone with the redo_v1 patch not present")

BOOK = {"claims": [{"id": "C1", "claim": "x uses io.Discard", "p": 0.8, "evidence": "diff"},
                   {"id": "C2", "claim": "tests unchanged", "p": 0.3, "evidence": "none (guess)"}],
        "rejected": [{"option": "nil streams", "turn": 0, "reason": "leaks output"}]}


@pytest.fixture(scope="module")
def ra():
    return redo_arms_module()


def test_stock_env_is_empty_and_config_defaults(ra, monkeypatch):
    assert redo_env("stock") == {}
    for k in list(__import__("os").environ):
        if k.startswith("SWT_REDO_"):
            monkeypatch.delenv(k)
    cfg = ra.RedoConfig.from_env()
    assert cfg.arm == "stock" and not cfg.active and cfg.tau == 0.5


def test_claim_timing(ra):
    assert ra.CLAIM_TIMING == {"commit_before": "before", "claims_no_p": "before", "claims_after": "after"}


def test_compose_commit_before_and_no_p(ra, tmp_path):
    arms = ra.RedoArms(ra.RedoConfig(arm="commit_before"), tmp_path, "google/gemini-3.5-flash-lite")
    assert arms.claim_model == "gemini/gemini-3.5-flash-lite"
    msg = arms.compose("no, wrong file", {"ok": True, "book": BOOK}, "fix it")
    assert msg.startswith("no, wrong file\n\n---")
    assert "C1 (p=0.80) x uses io.Discard" in msg and "p is now below 0.5" in msg
    assert "nil streams (turn 0)" in msg
    nop = ra.RedoArms(ra.RedoConfig(arm="claims_no_p"), tmp_path, "google/m")
    m2 = nop.compose("no", {"ok": True, "book": BOOK}, "fix")
    assert "(p=" not in m2 and "claims the message contradicts" in m2
    # failed claim call: the reply goes through unchanged
    assert arms.compose("hi", {"ok": False, "book": None}, "x") == "hi"


def test_compose_recap_and_minimal(ra, tmp_path):
    rec = ra.RedoArms(ra.RedoConfig(arm="recap"), tmp_path, "google/m")
    rec.compose("first", None, "opening")
    m = rec.compose("second", None, "opening")
    assert "1. (opening request) opening" in m and "2. first" in m and "3. second" in m
    mini = ra.RedoArms(ra.RedoConfig(arm="minimal_edit"), tmp_path, "google/m")
    assert "smallest change" in mini.compose("x", None, "o")


def test_valid_book(ra):
    assert ra._valid_book(BOOK, True)
    assert not ra._valid_book({"claims": [{"claim": "x", "p": 1.5}]}, True)
    assert ra._valid_book({"claims": [{"claim": "x"}], "rejected": []}, False)
    assert not ra._valid_book({"claims": []}, False)


def test_transcript_newest_first_budget(ra, tmp_path):
    for t in range(3):
        (tmp_path / f"opencode.txt.turn-{t}").write_text(
            json.dumps({"type": "text", "part": {"text": f"work {t} " + "x" * 200}}) + "\n")
    openers = [{"turn": 0, "opened_by": "instruction"}, {"turn": 1, "opened_by": "sim", "message": "m1"},
               {"turn": 2, "opened_by": "noop_continue", "message": "continue"}]
    full = ra.session_transcript(tmp_path, "open req", openers, 100_000)
    assert "Turn 1 - the user: m1" in full and "work 2" in full
    short = ra.session_transcript(tmp_path, "open req", openers, 400)
    assert "work 2" in short and "work 0" not in short and "elided" in short


def test_overlay_decide_sequence(ra, tmp_path):
    task = tmp_path / "cli-task-x"
    task.mkdir()
    ov = ra.Overlay(ra.RedoConfig(overlay=True, overlay_seed=7), tmp_path, task)
    no = {"all_revealed_satisfied": False, "labels": {}, "snapshot_turn": 0, "texts": {}}
    assert ov.decide(1, no) is None
    yes = {"all_revealed_satisfied": True, "snapshot_turn": 1, "labels": {"open": "satisfied", "0": "satisfied",
                                                                          "1": "not_applicable"},
           "texts": {"open": "o", "0": "Increase the timeout to 120s.", "1": "q"}}
    first = ov.decide(2, yes)
    assert first["kind"] == "unjustified" and first["message"] in ov.templates["unjustified"]
    assert ov.decide(2, yes) is None  # not at the same turn
    second = ov.decide(3, yes)
    assert second["kind"] == "misaimed" and second["target_intent"] == "0"
    assert "Increase the timeout to 120s" in second["message"]
    assert ov.done and ov.decide(4, yes) is None
    # seeded: a fresh overlay with the same seed and task makes the same choices
    ov2 = ra.Overlay(ra.RedoConfig(overlay=True, overlay_seed=7), tmp_path, task)
    assert ov2.decide(2, yes)["message"] == first["message"]


def test_prompts_are_recorded(ra):
    shas = ra.prompts_sha256()
    for name in ("claim_book_system.md", "claim_book_system_no_p.md", "claim_book_request.md", "rules_p.md", "rules_no_p.md",
                 "recap.md", "minimal_edit.md", "overlay_templates.json"):
        assert name in shas
    # commit_before and claims_after share the same claim-book system prompt; only the reply section differs
    assert (Path(ra.PROMPTS) / "claim_book_system.md").read_text().count("{") >= 1
