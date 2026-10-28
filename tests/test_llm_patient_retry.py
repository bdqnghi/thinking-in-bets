"""Provider overload (429/5xx) is retried on a wall-clock budget, not an attempt cap."""
import pytest

from test_llm_usage_evidence import complete_usage, fake_client, sdk_response


class Overloaded(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


def test_overload_bursts_longer_than_the_attempt_cap_are_ridden_out(monkeypatch, tmp_path):
    llm, create, events = fake_client(monkeypatch, tmp_path, complete_usage(), max_retries=6, patient_budget_seconds=3600)
    calls = {"n": 0}
    good = create.return_value
    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 9:  # three more failures than the attempt cap allows
            raise Overloaded(503)
        return good
    create.side_effect = flaky
    slept = []
    monkeypatch.setattr("fsa.llm.client.time.sleep", slept.append)
    response = llm.chat([{"role": "user", "content": "q"}])
    assert response.text and calls["n"] == 10
    assert llm.usage.failed_requests == 9 and llm.usage.requests == 10
    errors = [e for e in events if e["event"] == "request_error"]
    assert len(errors) == 9 and all(e["patient"] and e["status_code"] == 503 for e in errors)
    assert max(slept) <= 150 and len(slept) == 9


def test_patient_budget_exhaustion_fails_loudly_and_ordinary_transients_keep_the_cap(monkeypatch, tmp_path):
    llm, create, _ = fake_client(monkeypatch, tmp_path, complete_usage(), max_retries=3, patient_budget_seconds=0)
    create.side_effect = Overloaded(503)
    monkeypatch.setattr("fsa.llm.client.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="patient budget exhausted"):
        llm.chat([{"role": "user", "content": "q"}])
    assert create.call_count == 1
    llm2, create2, _ = fake_client(monkeypatch, tmp_path, complete_usage(), max_retries=3, patient_budget_seconds=3600)
    create2.side_effect = Overloaded(408)  # transient but not overload: attempt cap applies
    with pytest.raises(RuntimeError, match="after 3 retries"):
        llm2.chat([{"role": "user", "content": "q2"}])
    assert create2.call_count == 3
    llm3, create3, _ = fake_client(monkeypatch, tmp_path, complete_usage(), max_retries=3)
    create3.side_effect = Overloaded(400)  # client error: never retried
    with pytest.raises(Overloaded):
        llm3.chat([{"role": "user", "content": "q3"}])
    assert create3.call_count == 1


def test_provider_internal_error_bursts_are_patient(monkeypatch, tmp_path):
    # Gemini 3.8 answered "500 Internal error encountered" inside a 503 window;
    # the burst is ridden out on the wall-clock budget like the other overload statuses.
    llm, create, _ = fake_client(monkeypatch, tmp_path, complete_usage(), max_retries=3, patient_budget_seconds=0)
    create.side_effect = Overloaded(500)
    monkeypatch.setattr("fsa.llm.client.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="patient budget exhausted"):
        llm.chat([{"role": "user", "content": "q"}])
    assert create.call_count == 1
