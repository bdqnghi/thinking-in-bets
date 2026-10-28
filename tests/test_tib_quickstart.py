"""Offline unit tests for the standalone `tib` Python package (Algorithm 1)."""
from __future__ import annotations

from types import SimpleNamespace
import pytest

from tib import (
    AlwaysOnSession,
    Bet,
    ClaimChecklist,
    OnDemandSession,
    RejectedOption,
    draft_first_solve,
)


class MockCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out))])


class MockClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=MockCompletions(responses))


def test_claim_checklist_roundtrip_and_render():
    raw = """```json
    {
      "claims": [
        {"id": "C1", "claim": "Digest auth preserves cookies", "p": 0.96, "evidence": "pytest test_auth.py passed"},
        {"id": "C2", "claim": "Empty body returns []", "p": 0.40, "evidence": "none (guess)"}
      ],
      "rejected": [
        {"option": "drop cookie jar on redirect", "turn": 1, "reason": "breaks session persistence"}
      ]
    }
    ```"""
    cl = ClaimChecklist.from_json(raw)
    assert len(cl.claims) == 2
    assert cl.claims[0] == Bet(id="C1", claim="Digest auth preserves cookies", p=0.96, evidence="pytest test_auth.py passed")
    assert cl.rejected == [RejectedOption(option="drop cookie jar on redirect", turn=1, reason="breaks session persistence")]
    rendered = cl.render(include_p=True)
    assert "(p=0.96)" in rendered and "rejected:" in rendered
    rendered_nop = cl.render(include_p=False)
    assert "(p=" not in rendered_nop

    with pytest.raises(ValueError):
        Bet(id="C9", claim="bad probability", p=1.5)


def test_always_on_session_seals_checklist_before_feedback():
    book_1 = '{"claims": [{"id": "C1", "claim": "153/153 tests pass", "p": 0.98, "evidence": "pytest"}], "rejected": []}'
    book_2 = '{"claims": [{"id": "C1", "claim": "153/153 tests pass", "p": 0.98, "evidence": "pytest"}], "rejected": []}'
    client = MockClient([
        "Implemented DigestAuth; all 153 tests pass.",
        book_1,
        "All 153 tests pass on my side — could you share the failing test or input?",
        book_2,
    ])
    session = AlwaysOnSession(model="gemini-3.8-flash", tau=0.5, client=client)
    t1 = session.start("Implement digest auth with cookie persistence.")
    assert "153 tests" in t1
    # Checklist was written BEFORE any user criticism arrived
    assert len(session.checklist.claims) == 1 and session.checklist.claims[0].p == 0.98
    assert "wrong" not in client.chat.completions.calls[1]["messages"][1]["content"].lower()

    t2 = session.respond("No, that's wrong. Redo it.")
    assert "failing test" in t2
    # Private note with recorded checklist and tau=0.5 was injected alongside criticism
    sent_to_agent = client.chat.completions.calls[2]["messages"][-1]["content"]
    assert "No, that's wrong. Redo it." in sent_to_agent
    assert "(p=0.98)" in sent_to_agent and "below 0.5" in sent_to_agent


def test_on_demand_session_gates_off_when_confident():
    client = MockClient([
        "Drafted helper function.",
        '{"p": 0.95, "why": "all unit tests pass"}',
        "Handled follow-up directly.",
        '{"p": 0.60, "why": "edge case untested"}',
        '{"claims": [{"id": "C1", "claim": "handles None", "p": 0.45, "evidence": "none (guess)"}], "rejected": []}',
    ])
    session = OnDemandSession(model="gemini-3.8-flash", tau_gate=0.9, client=client)
    session.start("Write helper")
    assert session.gated_off is True and session.last_gate_p == 0.95
    assert len(session.checklist_history) == 0

    session.respond("What about None input?")
    # Turn 1 gated off -> raw user feedback delivered unchanged without checklist overhead
    assert client.chat.completions.calls[2]["messages"][-1]["content"] == "What about None input?"
    # After Turn 2, gate=0.60 < 0.9 -> full checklist written for next turn
    assert session.gated_off is False and len(session.checklist.claims) == 1


def test_draft_first_solve_extracts_bets_and_confidence():
    sample = (
        "Phase 1: candidate answer is 42.\n"
        "[CLAIM_BOOK]\n"
        "C1: base identity | evidence: verified by substitution | p=0.95\n"
        "C2: boundary case n=0 | evidence: checked directly | p=0.90\n"
        "Explanation: Verified both steps.\n"
        "Answer: 42\n"
        "Confidence: 86%"
    )
    client = MockClient([sample])
    res = draft_first_solve("Compute f(6).", model="gemini-3.8-flash", client=client)
    assert res["claim_probabilities"] == [0.95, 0.90]
    assert res["min_claim_p"] == 0.90
    assert res["confidence"] == 0.86
