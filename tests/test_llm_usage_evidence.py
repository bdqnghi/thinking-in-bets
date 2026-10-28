"""Fake-provider tests only; no network, native environment or old-cache edits."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletion

from fsa.llm.client import Cache, LLM
from fsa.llm.usage_evidence import (
    CONTRACT, EVIDENCE_FIELDS, cache_usage_evidence, revalidate_usage_evidence, usage_evidence,
)


def complete_usage(**changes):
    return {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 10}, "prompt_cache_hit_tokens": 10,
            "prompt_cache_miss_tokens": 90,
            "completion_tokens_details": {"reasoning_tokens": 7}, **changes}


def sdk_response(raw, model="gemini-3.5-flash-lite"):
    return ChatCompletion.model_validate({
        "id": "fake-response", "model": model, "created": 1, "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}],
        "usage": raw,
    })


def fake_client(monkeypatch, tmp_path, raw, *, model="gemini-3.5-flash-lite", **kwargs):
    create = Mock(return_value=sdk_response(raw, model))
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    monkeypatch.setattr("fsa.llm.client.OpenAI", lambda **_: fake)
    events = []
    llm = LLM(model, cache_path=str(tmp_path / "responses.sqlite"), event_callback=events.append,
              thinking="on", reasoning_effort="medium", **kwargs)
    return llm, create, events


def test_complete_explicit_counts_reconcile_without_double_counting_reasoning():
    raw = complete_usage(extra_provider_field={"kept": True})
    before = copy.deepcopy(raw)
    evidence = usage_evidence(raw)
    assert evidence["usage_contract"] == CONTRACT
    assert evidence["usage_status"] == evidence["reasoning_breakdown_status"] == "available"
    assert evidence["usage_unknown_reasons"] == [] and evidence["total_token_residual"] == 0
    assert evidence["reported_usage"] == {
        "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
        "provider_cached_prompt_tokens": 10, "reasoning_tokens": 7,
    }
    assert raw == before and evidence["raw_usage"] == raw
    evidence["raw_usage"]["extra_provider_field"]["kept"] = False
    assert raw == before


def test_sdk_raw_usage_preserves_unknown_fields_and_omits_unset_fields():
    raw = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "provider_extra": [1, 2]}
    evidence = usage_evidence(sdk_response(raw).usage)
    assert evidence["raw_usage"] == raw
    assert "completion_tokens_details" not in evidence["raw_usage"]
    assert evidence["usage_status"] == "unavailable"
    assert "cache_attribution_missing" in evidence["usage_unknown_reasons"]


def test_missing_reasoning_does_not_make_an_observed_zero_or_block_coherent_price_usage():
    raw = complete_usage()
    raw.pop("completion_tokens_details")
    evidence = usage_evidence(raw)
    assert evidence["usage_status"] == "available"
    assert evidence["reasoning_breakdown_status"] == "unavailable"
    assert evidence["reasoning_unknown_reasons"] == ["reasoning_tokens_missing"]
    assert "reasoning_tokens" not in evidence["reported_usage"]


def test_explicit_zeros_are_known_and_distinct_from_missing_fields():
    raw = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
           "prompt_tokens_details": {"cached_tokens": 0},
           "completion_tokens_details": {"reasoning_tokens": 0}}
    evidence = usage_evidence(raw)
    assert evidence["usage_status"] == evidence["reasoning_breakdown_status"] == "available"
    assert evidence["reported_usage"]["reasoning_tokens"] == 0
    assert evidence["reported_usage"]["provider_cached_prompt_tokens"] == 0
    assert evidence["total_token_residual"] == 0


@pytest.mark.parametrize("value", [None, True, -1, 1.5, float("nan"), float("inf"), "10"])
@pytest.mark.parametrize("key", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_missing_or_invalid_core_counts_are_never_known_zero(key, value):
    evidence = usage_evidence(complete_usage(**{key: value}))
    assert evidence["usage_status"] == "unavailable"
    assert key not in evidence["reported_usage"]
    assert key + "_missing_or_invalid" in evidence["usage_unknown_reasons"]


@pytest.mark.parametrize("total,residual", [(121, 1), (1500, 1380), (119, -1)])
def test_residual_stays_unknown_even_with_explicit_cache_and_reasoning(total, residual):
    evidence = usage_evidence(complete_usage(total_tokens=total))
    assert evidence["total_token_residual"] == residual
    assert evidence["usage_status"] == "unavailable"
    assert evidence["usage_unknown_reasons"] == ["total_tokens_do_not_reconcile"]
    assert evidence["reported_usage"]["completion_tokens"] == 20  # no speculative repair


@pytest.mark.parametrize("change,reason", [
    ({"prompt_cache_hit_tokens": 11}, "cache_hit_sources_disagree"),
    ({"prompt_tokens_details": {"cached_tokens": 0}}, "cache_hit_sources_disagree"),
    ({"prompt_tokens_details": {"cached_tokens": True}}, "detail_cached_tokens_invalid"),
    ({"prompt_cache_hit_tokens": -1}, "prompt_cache_hit_tokens_invalid"),
    ({"prompt_cache_miss_tokens": 89}, "cache_hit_miss_breakdown_does_not_reconcile"),
    ({"prompt_cache_miss_tokens": True}, "prompt_cache_miss_tokens_invalid"),
    ({"prompt_tokens_details": {"cached_tokens": 101}, "prompt_cache_hit_tokens": 101}, "cache_hits_exceed_prompt_tokens"),
    ({"prompt_tokens_details": []}, "prompt_tokens_details_invalid"),
])
def test_cache_breakdown_and_duplicate_sources_fail_closed(change, reason):
    evidence = usage_evidence(complete_usage(**change))
    assert evidence["usage_status"] == "unavailable" and reason in evidence["usage_unknown_reasons"]


def test_cache_misses_alone_do_not_invent_a_cache_hit_measurement():
    raw = complete_usage()
    raw.pop("prompt_tokens_details"); raw.pop("prompt_cache_hit_tokens")
    evidence = usage_evidence(raw)
    assert evidence["usage_status"] == "unavailable"
    assert "provider_cached_prompt_tokens" not in evidence["reported_usage"]


@pytest.mark.parametrize("value", [True, -1, 21, "7", float("nan"), None])
def test_reported_reasoning_invalid_or_larger_than_completion_invalidates_usage(value):
    evidence = usage_evidence(complete_usage(completion_tokens_details={"reasoning_tokens": value}))
    assert evidence["usage_status"] == evidence["reasoning_breakdown_status"] == "unavailable"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_raw_usage_produces_journal_safe_unavailable_evidence(value):
    evidence = usage_evidence(complete_usage(completion_tokens=value))
    assert evidence["raw_usage"] is None and evidence["reported_usage"] == {}
    assert "usage_serialization_unavailable" in evidence["usage_unknown_reasons"]
    json.dumps(evidence, allow_nan=False)
    again = revalidate_usage_evidence(evidence)
    assert again["usage_status"] == "unavailable"
    assert "usage_serialization_unavailable" in again["usage_unknown_reasons"]


def test_revalidation_never_trusts_stored_available_flag_or_legacy_normalized_counts():
    result = usage_evidence(complete_usage(total_tokens=121))
    result.update(usage_status="available", usage_unknown_reasons=[])
    checked = revalidate_usage_evidence(result)
    assert checked["usage_status"] == "unavailable"
    assert "total_tokens_do_not_reconcile" in checked["usage_unknown_reasons"]
    assert "stored_usage_evidence_does_not_reconcile" in checked["usage_unknown_reasons"]
    legacy = revalidate_usage_evidence({"usage": {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 0}})
    assert legacy["usage_status"] == "unavailable" and legacy["raw_usage"] is None
    assert "legacy_or_missing_usage_contract" in legacy["usage_unknown_reasons"]


def test_response_event_cache_roundtrip_preserves_raw_evidence_and_cache_counts(monkeypatch, tmp_path):
    raw = complete_usage(provider_extra={"future": 3})
    llm, create, events = fake_client(monkeypatch, tmp_path, raw)
    live = llm.chat([{"role": "user", "content": "hello"}])
    cached = llm.chat([{"role": "user", "content": "hello"}])
    assert create.call_count == 1 and not live.cached and cached.cached
    assert live.raw_usage == cached.raw_usage == raw
    assert live.usage_status == cached.usage_status == "available"
    assert live.usage.provider_cached_prompt_tokens == cached.usage.provider_cached_prompt_tokens == 10
    assert live.total_token_residual == cached.total_token_residual == 0
    assert live.reported_usage["total_tokens"] == 120
    for event in events[1:]:
        assert event["raw_usage"] == raw
        assert revalidate_usage_evidence(event)["usage_status"] == "available"
    live.raw_usage["provider_extra"]["future"] = 999
    assert cached.raw_usage["provider_extra"]["future"] == 3
    assert events[1]["raw_usage"]["provider_extra"]["future"] == 3


def test_unknown_residual_survives_provider_event_and_cache_without_requery(monkeypatch, tmp_path):
    llm, create, events = fake_client(monkeypatch, tmp_path, complete_usage(total_tokens=420))
    response = llm.chat([{"role": "user", "content": "hello"}])
    replay = llm.chat([{"role": "user", "content": "hello"}])
    assert response.total_token_residual == replay.total_token_residual == 300
    assert response.usage_status == replay.usage_status == "unavailable"
    assert create.call_count == 1 and events[-1]["usage_status"] == "unavailable"


def legacy_payload(llm, messages):
    return dict(model=llm.model, messages=messages, temperature=0.0, max_tokens=256,
                stop=None, sample_id=0, tools=None, reasoning_effort=llm.reasoning_effort,
                extra_body=llm.extra_body, response_format=None, cache_namespace=llm.cache_namespace, cache_schema=2)


def test_legacy_cache_is_not_upgraded_overwritten_or_requeried(monkeypatch, tmp_path):
    llm, create, events = fake_client(monkeypatch, tmp_path, complete_usage())
    messages = [{"role": "user", "content": "legacy"}]
    key = Cache.key(legacy_payload(llm, messages))
    saved = {"text": "old", "prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 0,
             "model_version": llm.model, "response_id": "old-response"}
    llm.cache.put(key, saved)
    response = llm.chat(messages)
    assert create.call_count == 0 and response.cached and response.text == "old"
    assert llm.cache.get(key) == saved
    assert response.raw_usage is None and response.reported_usage == {}
    assert response.usage.prompt_tokens == 100  # preserved compatibility counter only
    assert response.usage_status == "unavailable"
    assert "legacy_cache_missing_usage_evidence" in response.usage_unknown_reasons
    assert events[0]["event"] == "cache_hit" and events[0]["usage_status"] == "unavailable"
    checked = revalidate_usage_evidence(events[0])
    assert "legacy_cache_missing_usage_evidence" in checked["usage_unknown_reasons"]
    assert checked["usage_status"] == "unavailable"


def test_forged_cache_availability_or_counter_disagreement_stays_unknown():
    record = {**usage_evidence(complete_usage()), "prompt_tokens": 100, "completion_tokens": 20,
              "reasoning_tokens": 7, "provider_cached_prompt_tokens": 10}
    record["completion_tokens"] = 999
    evidence = cache_usage_evidence(record)
    assert evidence["usage_status"] == "unavailable"
    assert "cached_normalized_counter_disagrees:completion_tokens" in evidence["usage_unknown_reasons"]
    record["usage_contract"] = "unrecognized_future_contract"
    assert "legacy_cache_missing_usage_evidence" in cache_usage_evidence(record)["usage_unknown_reasons"]


def test_missing_provider_usage_preserved_unavailable_and_failed_requests_explicit(monkeypatch, tmp_path):
    llm, create, events = fake_client(monkeypatch, tmp_path, None, use_cache=False, max_retries=2)
    create.side_effect = [TimeoutError("fake timeout"), sdk_response(None)]
    monkeypatch.setattr("fsa.llm.client.time.sleep", lambda _: None)
    result = llm.chat([{"role": "user", "content": "hello"}])
    assert create.call_count == 2 and result.raw_usage is None and result.usage_status == "unavailable"
    failure = next(e for e in events if e["event"] == "request_error")
    assert failure["raw_usage"] is None and failure["usage_status"] == "unavailable"
    assert result.reported_usage == {} and result.usage.prompt_tokens == 0


def test_bad_sdk_usage_serialization_does_not_hide_or_repeat_paid_response(monkeypatch, tmp_path):
    llm, create, events = fake_client(monkeypatch, tmp_path, complete_usage(), use_cache=False)
    broken_usage = SimpleNamespace(model_dump=Mock(side_effect=RuntimeError("fake serialization problem")))
    returned = sdk_response(complete_usage())
    returned.usage = broken_usage
    create.return_value = returned
    result = llm.chat([{"role": "user", "content": "hello"}])
    assert create.call_count == 1 and result.text == "answer"
    assert events[-1]["event"] == "response" and events[-1]["raw_usage"] is None
    assert "usage_serialization_unavailable" in events[-1]["usage_unknown_reasons"]
    json.dumps(events[-1], allow_nan=False)
