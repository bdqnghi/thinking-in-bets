"""One LLM interface for every provider, with a sqlite cache and usage accounting.

Providers are selected by model-name prefix. Everything goes through the
OpenAI-compatible chat API (DeepSeek native; Gemini via its OpenAI endpoint),
so no provider-specific features are used anywhere in the agents.
"""
from __future__ import annotations

import hashlib
import random
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .usage_evidence import cache_usage_evidence, evidence_fields, usage_evidence

PROVIDERS = {
    "gemini": dict(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", key_env="GEMINI_API_KEY"),
    "gemma": dict(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", key_env="GEMINI_API_KEY"),
    "deepseek": dict(base_url="https://api.deepseek.com", key_env="DEEPSEEK_API_KEY"),
}


def provider_for(model: str) -> str:
    for prefix in PROVIDERS:
        if model.startswith(prefix):
            return prefix
    raise ValueError(f"no provider for model {model!r}; known prefixes: {list(PROVIDERS)}")


@dataclass
class Usage:
    """Compatibility counters, not proof of complete reported/billable usage.

    Unknown token fields retain numeric placeholders here. Consult each
    response/event's raw usage evidence before pricing or interpreting zeros.
    """
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    wall_seconds: float = 0.0
    requests: int = 0
    failed_requests: int = 0
    unpriced_failed_requests: int = 0
    provider_cached_prompt_tokens: int = 0

    def add(self, other: "Usage") -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class LLMResponse:
    text: str
    usage: Usage
    cached: bool
    raw_finish: str | None = None
    tool_calls: list[dict[str, Any]] | None = None   # [{"id":..., "name":..., "arguments": {...}}]
    assistant_message: dict[str, Any] | None = None
    model_version: str | None = None
    response_id: str | None = None
    usage_contract: str | None = None
    raw_usage: Any = None
    reported_usage: dict[str, int] = field(default_factory=dict)
    usage_status: str = "unavailable"
    usage_unknown_reasons: list[str] = field(default_factory=lambda: ["usage_evidence_not_supplied"])
    total_token_residual: int | None = None
    reasoning_breakdown_status: str = "unavailable"
    reasoning_unknown_reasons: list[str] = field(default_factory=lambda: ["usage_evidence_not_supplied"])


class Cache:
    """Thread-safe sqlite cache keyed by a hash of the full request."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        con = self._connect()
        con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT NOT NULL, ts REAL)")
        con.commit()
        con.close()

    def _connect(self) -> sqlite3.Connection:
        # Several run processes share one cache file: WAL + a long busy timeout instead of "database is locked".
        con = sqlite3.connect(self.path, timeout=60)
        try:
            con.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        return con

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, k: str) -> dict[str, Any] | None:
        for attempt in range(5):
            try:
                with self._lock:
                    con = self._connect()
                    row = con.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
                    con.close()
                return json.loads(row[0]) if row else None
            except sqlite3.OperationalError:
                time.sleep(0.5 * (attempt + 1))
        return None                       # a cache miss is always safe

    def put(self, k: str, v: dict[str, Any]) -> None:
        for attempt in range(5):
            try:
                with self._lock:
                    con = self._connect()
                    con.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (k, json.dumps(v), time.time()))
                    con.commit()
                    con.close()
                return
            except sqlite3.OperationalError:
                time.sleep(0.5 * (attempt + 1))


# Provider overload statuses retried on a wall-clock budget (see LLM.__init__).
PATIENT_STATUSES = (429, 500, 502, 503, 504)


class LLM:
    """Chat-completion client. `sample_id` distinguishes otherwise-identical calls
    so that repeated trials at temperature>0 are not collapsed by the cache."""

    def __init__(self, model: str, cache_path: str = "results/llm_cache.sqlite", use_cache: bool = True,
                 max_retries: int = 16, reasoning_effort: str | None = None, timeout: float = 120.0,
                 thinking: str = "off", cache_namespace: str = "legacy", event_callback=None,
                 patient_budget_seconds: float | None = None):
        """thinking: "off" disables provider reasoning where the API allows it (DeepSeek: thinking.type=disabled;
        Gemini: reasoning_effort=none); "on" leaves provider defaults; reasoning_effort overrides when set."""
        self.model = model
        self.provider = provider_for(model)
        cfg = PROVIDERS[self.provider]
        base_url, key = cfg["base_url"], os.environ.get(cfg["key_env"])
        # Per-machine endpoint profile (fsa.redo.machine): the public Gemini API by default (nothing
        # changes); another profile swaps in its OpenAI-compatible base and key, and namespaces the
        # cache so responses of different endpoints are never mixed.
        self.endpoint_profile: str | None = None
        if self.provider in ("gemini", "gemma"):
            from fsa.redo import machine
            prof = machine.active()
            if not prof.public:
                base_url, key, self.endpoint_profile = prof.openai_base, prof.api_key(), prof.name
                if not key:
                    raise RuntimeError(f"machine profile {prof.name!r}: no API key ({prof.key_source()})")
        if not key:
            raise RuntimeError(f"{cfg['key_env']} not set")
        # Per-request HTTP timeout; FSA_LLM_TIMEOUT_SECONDS overrides the caller's value (slow frontier
        # checkpoints can exceed the 120 s default on long judge prompts). Unset: unchanged behaviour.
        timeout = float(os.environ.get("FSA_LLM_TIMEOUT_SECONDS") or timeout)
        self.timeout = timeout
        self.client = OpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=0)
        self.cache = Cache(cache_path) if use_cache else None
        self.max_retries = max_retries
        # Provider overload (429/502/503/504) is retried on wall-clock budget rather
        # than attempt count, so a multi-minute outage burst does not kill a cell
        # that can resume the moment the provider serves again. Every attempt is
        # still accounted as a failed request. FSA_PATIENT_RETRY_SECONDS overrides.
        self.patient_budget_seconds = (float(os.environ.get("FSA_PATIENT_RETRY_SECONDS", "1800"))
                                       if patient_budget_seconds is None else float(patient_budget_seconds))
        self.reasoning_effort = reasoning_effort
        self.effective_reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.cache_namespace = cache_namespace
        self.event_callback = event_callback
        self.extra_body: dict[str, Any] = {}
        if self.provider == "deepseek" and (reasoning_effort or thinking == "on"):
            self.extra_body = {"thinking": {"type": "enabled"}}
            self.effective_reasoning_effort = {"medium": "high", "xhigh": "high"}.get(reasoning_effort, reasoning_effort or "high")
        if thinking == "off" and not reasoning_effort:
            if self.provider == "deepseek":
                self.extra_body = {"thinking": {"type": "disabled"}}
            elif self.provider == "gemini" and "lite" not in model:
                if model.startswith("gemini-3.8"):
                    raise ValueError("Gemini 3.8 does not support thinking off; select an explicit supported effort")
                self.extra_body = {"reasoning_effort": "none"}
        self.usage = Usage()
        self._ulock = threading.Lock()

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.0, max_tokens: int = 256,
             stop: list[str] | None = None, sample_id: int | str = 0, tools: list[dict[str, Any]] | None = None,
             response_format: dict[str, Any] | None = None) -> LLMResponse:
        # Keep request-key schema 2 so reading an old cache does not silently
        # trigger a paid replacement call. New values carry their own versioned
        # usage evidence; old values remain explicitly incomplete and unchanged.
        payload: dict[str, Any] = dict(model=self.model, messages=messages, temperature=temperature,
                                       max_tokens=max_tokens, stop=stop, sample_id=sample_id, tools=tools,
                                       reasoning_effort=self.reasoning_effort, extra_body=self.extra_body,
                                       response_format=response_format, cache_namespace=self.cache_namespace, cache_schema=2)
        if self.endpoint_profile:  # only non-public endpoints: public keys are unchanged
            payload["endpoint_profile"] = self.endpoint_profile
        k = Cache.key(payload)
        if self.cache is not None:
            hit = self.cache.get(k)
            if hit is not None:
                evidence = cache_usage_evidence(hit)
                # Replaying historical counters is not another physical request
                # and cannot upgrade a legacy cache's missing usage evidence.
                counters = {name: hit[name] if type(hit.get(name)) is int and hit[name] >= 0 else 0
                            for name in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "provider_cached_prompt_tokens")}
                u = Usage(calls=1, cached_calls=1, **counters)
                with self._ulock:
                    self.usage.add(u)
                if self.event_callback:
                    self.event_callback({"event": "cache_hit", "request_hash": k, "usage": u.as_dict(),
                                         "model_version": hit.get("model_version"), "response_id": hit.get("response_id"),
                                         **evidence_fields(evidence)})
                return LLMResponse(text=hit["text"], usage=u, cached=True, raw_finish=hit.get("finish"), tool_calls=hit.get("tool_calls"),
                                   assistant_message=hit.get("assistant_message"), model_version=hit.get("model_version"), response_id=hit.get("response_id"),
                                   **evidence_fields(evidence))

        kwargs: dict[str, Any] = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = tools
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if self.provider == "deepseek" and self.extra_body.get("thinking", {}).get("type") == "enabled":
            kwargs.pop("temperature", None)  # ignored by the provider in thinking mode
        if response_format:
            kwargs["response_format"] = response_format
        last_err: Exception | None = None
        attempt, first_started = 0, time.time()
        while True:
            started = time.time()
            if self.event_callback:
                self.event_callback({"event": "request_started", "request_hash": k, "attempt": attempt})
            try:
                r = self.client.chat.completions.create(**kwargs)
            except Exception as e:  # rate limits, transient 5xx, timeouts
                last_err = e
                status = getattr(e, "status_code", None)
                patient = status in PATIENT_STATUSES
                elapsed = time.time() - first_started
                failed = Usage(requests=1, failed_requests=1, unpriced_failed_requests=1, wall_seconds=time.time() - started)
                with self._ulock:
                    self.usage.add(failed)
                if self.event_callback:
                    self.event_callback({"event": "request_error", "request_hash": k, "attempt": attempt,
                                         "error_type": type(e).__name__, "status_code": status, "usage": failed.as_dict(),
                                         "patient": patient, "elapsed_seconds": elapsed, **usage_evidence(None)})
                if status is not None and status not in (408, 409, 429) and status < 500:
                    raise
                if patient:
                    if elapsed >= self.patient_budget_seconds:
                        raise RuntimeError(f"LLM call failed after {attempt + 1} attempts over {elapsed:.0f}s of provider "
                                           f"overload (HTTP {status}); patient budget exhausted: {last_err}")
                    delay = min(2 ** min(attempt, 7), 120) * (1 + random.random() * .25)
                elif attempt + 1 >= self.max_retries:
                    raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")
                else:
                    delay = min(2 ** attempt, 60)
                time.sleep(delay)
                attempt += 1
                continue
            break

        text = (r.choices[0].message.content or "") if r.choices else ""
        finish = r.choices[0].finish_reason if r.choices else None
        tcs = None
        if r.choices and getattr(r.choices[0].message, "tool_calls", None):
            tcs = []
            for tc in r.choices[0].message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {"_raw": tc.function.arguments}
                extra = None
                try:
                    extra = tc.model_extra.get("extra_content") if tc.model_extra else None
                except Exception:
                    extra = None
                tcs.append(dict(id=tc.id, name=tc.function.name, arguments=args, extra_content=extra))
        evidence = usage_evidence(getattr(r, "usage", None))
        reported = evidence["reported_usage"]
        pt = reported.get("prompt_tokens", 0)
        ct = reported.get("completion_tokens", 0)
        rt = reported.get("reasoning_tokens", 0)
        cached_pt = reported.get("provider_cached_prompt_tokens", 0)
        message = r.choices[0].message.model_dump(exclude_none=True) if r.choices else {"role": "assistant", "content": ""}
        version = getattr(r, "model", None)
        response_id = getattr(r, "id", None)
        u = Usage(calls=1, requests=1, prompt_tokens=pt, completion_tokens=ct, reasoning_tokens=rt,
                  provider_cached_prompt_tokens=cached_pt, wall_seconds=time.time() - started)
        with self._ulock:
            self.usage.add(u)
        if self.cache is not None:
            self.cache.put(k, dict(text=text, finish=finish, prompt_tokens=pt, completion_tokens=ct, reasoning_tokens=rt, tool_calls=tcs,
                                  provider_cached_prompt_tokens=cached_pt,
                                  assistant_message=message, model_version=version, response_id=response_id, **evidence_fields(evidence)))
        if self.event_callback:
            self.event_callback({"event": "response", "request_hash": k, "attempt": attempt, "usage": u.as_dict(),
                                 "model_version": version, "response_id": response_id, "finish": finish, **evidence_fields(evidence)})
        return LLMResponse(text=text, usage=u, cached=False, raw_finish=finish, tool_calls=tcs,
                           assistant_message=message, model_version=version, response_id=response_id, **evidence_fields(evidence))
