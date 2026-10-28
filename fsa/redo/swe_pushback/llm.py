"""Chat calls for SWE-Pushback-Hidden through the local bridge, inside a shared per-model quota slot (P0.3)."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

from fsa.redo import quota

BASE_URL = os.environ.get("TIB_PUSHBACK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai"
TRANSIENT_BACKOFF = (10, 20, 40, 60, 120, 180, 300, 300, 300, 300, 300, 300)  # ~37 min patient retry
_TRANSIENT = ("429", "rate", "quota", "resource_exhausted", "500", "502", "503", "504", "overload", "unavailable",
              "timeout", "timed out", "connection", "internal", "server", "prefill", "empty response")

_clients: dict = {}


def _client():
    import openai
    if "c" not in _clients:
        tok = (os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ((Path.home() / ".config" / "tib" / "api_key").read_text().strip() if (Path.home() / ".config" / "tib" / "api_key").is_file() else ""))
        _clients["c"] = openai.OpenAI(base_url=BASE_URL, api_key=tok, timeout=900, max_retries=0)
    return _clients["c"]


class LLMError(RuntimeError):
    pass


EMPTY_RETRIES = 4  # empty reply (e.g. finish MALFORMED_FUNCTION_CALL): quick re-sample, then hand back empty


def chat(model: str, messages: list[dict], *, effort: str | None = "low", max_tokens: int = 32768,
         temperature: float | None = None, json_mode: bool = False, tools: list | None = None,
         allow_empty: bool = False, sleep=time.sleep) -> dict:
    """Returns {"content", "message" (assistant dict to append, tool calls included), "tool_calls"
    [(id, name, arguments_str)], "finish", "usage", "attempts", "empty_retries", "errors"}.
    Transient errors: patient backoff, then LLMError. Empty replies (no text, no tool call): up to EMPTY_RETRIES quick
    re-samples; then LLMError, or the empty reply if ``allow_empty``."""
    kw: dict = dict(model=model, messages=messages, max_tokens=max_tokens)
    if effort:
        kw["reasoning_effort"] = effort
    if temperature is not None:
        kw["temperature"] = temperature
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    if tools:
        kw["tools"] = tools
    errs: list[str] = []
    transient = 0
    empty = 0
    attempts = 0
    wait_s = 0.0  # infrastructure wait: quota-slot queueing + transient backoff (not agent time; A5.9)
    while True:
        attempts += 1
        t0 = time.time()
        try:
            tq = time.time()
            with quota.slot(model):
                wait_s += time.time() - tq
                r = _client().chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"[:500]
            errs.append(msg)
            if transient >= len(TRANSIENT_BACKOFF) or not any(m in msg.lower() for m in _TRANSIENT):
                raise LLMError(msg) from e
            tb = time.time()
            sleep(TRANSIENT_BACKOFF[transient] * (1 + 0.2 * random.random()))
            wait_s += time.time() - tb
            transient += 1
            continue
        ch = r.choices[0] if r.choices else None
        m = ch.message if ch else None
        content = (m.content or "") if m else ""
        tcs = list(m.tool_calls or []) if m else []
        finish = getattr(ch, "finish_reason", None) if ch else None
        u = r.usage
        usage = dict(prompt=getattr(u, "prompt_tokens", 0) or 0, completion=getattr(u, "completion_tokens", 0) or 0,
                     total=getattr(u, "total_tokens", 0) or 0) if u else {}
        if not content.strip() and not tcs:
            errs.append(f"empty reply (finish={finish})")
            if empty < EMPTY_RETRIES:
                empty += 1
                sleep(2 + 2 * random.random())
                continue
            if not allow_empty:
                raise LLMError(f"empty reply x{empty + 1} (finish={finish})")
        msg_d = m.model_dump(exclude_none=True) if m else {"role": "assistant", "content": ""}
        msg_d["role"] = "assistant"
        msg_d.setdefault("content", content)
        for k in ("refusal", "audio", "function_call", "annotations"):
            msg_d.pop(k, None)
        calls = [(tc.id, tc.function.name, tc.function.arguments or "") for tc in tcs]
        return dict(content=content, message=msg_d, tool_calls=calls, finish=finish, usage=usage, attempts=attempts,
                    empty_retries=empty, latency_s=round(time.time() - t0, 2), errors=errs, wait_s=wait_s)

