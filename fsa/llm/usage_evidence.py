"""Preserved provider usage and conservative, provider-neutral reconciliation.

This contract describes reported token evidence, not a billing receipt. Never
interpret an unexplained total residual as thinking, bill it speculatively, or
turn an absent cache/reasoning field into an observed zero. The legacy Usage
counter object may still use numeric placeholders for compatibility; consumers
must gate pricing on this independently revalidated evidence.
"""
from __future__ import annotations

import copy
import json

CONTRACT = "provider_usage_evidence_v1"
EVIDENCE_FIELDS = (
    "usage_contract", "raw_usage", "reported_usage", "usage_status",
    "usage_unknown_reasons", "total_token_residual", "reasoning_breakdown_status",
    "reasoning_unknown_reasons",
)


def _count(value):
    return type(value) is int and value >= 0


def _preserve(value):
    if value is not None and hasattr(value, "model_dump"):
        # exclude_unset preserves the distinction between absent fields and
        # explicit provider nulls; unknown provider fields remain in model_extra.
        value = value.model_dump(mode="json", exclude_unset=True)
    # Copy through JSON so the callback/cache never aliases a mutable SDK value.
    # Nonfinite JSON cannot be durably written by strict journal encoders. The
    # caller records explicit serialization-unavailable evidence instead of
    # letting a paid response disappear or inventing a replacement token count.
    return json.loads(json.dumps(value, allow_nan=False))


def usage_evidence(raw_usage):
    """Return the fixed evidence contract from an SDK usage object/dict/None.

    Complete price evidence requires explicit nonnegative integer prompt,
    completion and total counts, an exact total reconciliation and coherent
    explicitly reported cache hits. Missing reasoning detail alone leaves
    price evidence available, but its own breakdown explicitly unavailable.
    Invalid reported reasoning does invalidate price evidence.
    """
    issues, reasoning_issues, reported = [], [], {}
    try:
        raw = _preserve(raw_usage)
    except Exception:
        raw = None
        issues.append("usage_serialization_unavailable")
    if type(raw) is not dict:
        issues.append("usage_object_unavailable_or_invalid")
        source = {}
    else:
        source = raw

    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if _count(source.get(name)):
            reported[name] = source[name]
        else:
            issues.append(name + "_missing_or_invalid")
    residual = None
    if all(k in reported for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
        residual = reported["total_tokens"] - reported["prompt_tokens"] - reported["completion_tokens"]
        if residual:
            issues.append("total_tokens_do_not_reconcile")

    hits = []
    details = source.get("prompt_tokens_details")
    if details is not None and type(details) is not dict:
        issues.append("prompt_tokens_details_invalid")
    if type(details) is dict and "cached_tokens" in details:
        value = details["cached_tokens"]
        if _count(value):
            hits.append(value)
        else:
            issues.append("detail_cached_tokens_invalid")
    if "prompt_cache_hit_tokens" in source:
        value = source["prompt_cache_hit_tokens"]
        if _count(value):
            hits.append(value)
        else:
            issues.append("prompt_cache_hit_tokens_invalid")
    if not hits:
        issues.append("cache_attribution_missing")
    elif len(set(hits)) > 1:
        issues.append("cache_hit_sources_disagree")
    else:
        reported["provider_cached_prompt_tokens"] = hits[0]
        if "prompt_tokens" in reported and hits[0] > reported["prompt_tokens"]:
            issues.append("cache_hits_exceed_prompt_tokens")
    if "prompt_cache_miss_tokens" in source:
        misses = source["prompt_cache_miss_tokens"]
        if not _count(misses):
            issues.append("prompt_cache_miss_tokens_invalid")
        elif "prompt_tokens" in reported:
            if misses > reported["prompt_tokens"]:
                issues.append("cache_misses_exceed_prompt_tokens")
            if "provider_cached_prompt_tokens" in reported and misses + reported["provider_cached_prompt_tokens"] != reported["prompt_tokens"]:
                issues.append("cache_hit_miss_breakdown_does_not_reconcile")

    detail = source.get("completion_tokens_details")
    if detail is not None and type(detail) is not dict:
        reasoning_issues.append("completion_tokens_details_invalid")
        issues.append("completion_tokens_details_invalid")
    if type(detail) is dict and "reasoning_tokens" in detail:
        reasoning = detail["reasoning_tokens"]
        if not _count(reasoning):
            reasoning_issues.append("reasoning_tokens_invalid")
            issues.append("reasoning_tokens_invalid")
        else:
            reported["reasoning_tokens"] = reasoning
            if "completion_tokens" not in reported:
                reasoning_issues.append("reasoning_completion_bound_unavailable")
            elif reasoning > reported["completion_tokens"]:
                reasoning_issues.append("reasoning_tokens_exceed_completion_tokens")
                issues.append("reasoning_tokens_exceed_completion_tokens")
    else:
        reasoning_issues.append("reasoning_tokens_missing")
    return {
        "usage_contract": CONTRACT, "raw_usage": raw, "reported_usage": reported,
        "usage_status": "unavailable" if issues else "available",
        "usage_unknown_reasons": issues, "total_token_residual": residual,
        "reasoning_breakdown_status": "unavailable" if reasoning_issues else "available",
        "reasoning_unknown_reasons": reasoning_issues,
    }


def revalidate_usage_evidence(record):
    """Recompute from raw evidence; missing/legacy contracts never upgrade.

    Use the returned reported_usage for pricing, rather than trusting stored
    status fields or a normalized Usage object. Preserve the original record
    separately; this function does not mutate it or repair an old cache.
    """
    source = record if type(record) is dict else {}
    result = usage_evidence(source.get("raw_usage"))
    if source.get("usage_contract") != CONTRACT:
        result["usage_unknown_reasons"].append("legacy_or_missing_usage_contract")
    elif any(name not in source or source[name] != result[name] for name in EVIDENCE_FIELDS):
        result["usage_unknown_reasons"].append("stored_usage_evidence_does_not_reconcile")
    # Keep explicit historical/serialization limitations visible through later
    # role-journal revalidation. These reasons may only restrict availability;
    # they never override a freshly detected inconsistency or confer validity.
    saved_reasons = source.get("usage_unknown_reasons")
    if type(saved_reasons) is list:
        for reason in saved_reasons:
            if type(reason) is str and reason and reason not in result["usage_unknown_reasons"]:
                result["usage_unknown_reasons"].append(reason)
    if result["usage_unknown_reasons"]:
        result["usage_status"] = "unavailable"
    return result


def cache_usage_evidence(record):
    """Read existing cache evidence without altering or upgrading its bytes."""
    result = revalidate_usage_evidence(record)
    if record.get("usage_contract") != CONTRACT or "raw_usage" not in record:
        result["usage_unknown_reasons"].append("legacy_cache_missing_usage_evidence")
        result["usage_status"] = "unavailable"
    for name in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "provider_cached_prompt_tokens"):
        expected = result["reported_usage"].get(name, 0)
        if record.get("usage_contract") == CONTRACT and (type(record.get(name)) is not int or record[name] != expected):
            result["usage_unknown_reasons"].append("cached_normalized_counter_disagrees:" + name)
            result["usage_status"] = "unavailable"
    return result


def evidence_fields(record):
    """Independent copy of the eight evidence fields for an event or response."""
    return {name: copy.deepcopy(record[name]) for name in EVIDENCE_FIELDS}
