"""Model seats and one structured-output call for every judge-like seat of the redo study.

Seat values come from ``configs/seats/redo_v1_seats.json`` (host-side config, read when a seat is
first used; the in-trial SWE-Together overlay trigger reads it when its judge is built at trial
start). ``REDO_JUDGE_MODEL`` / ``REDO_JUDGE_EFFORT`` override the judge seat and are recorded.

Backends, chosen by model id:

* ``gemini-*`` - Gemini's OpenAI-compatible endpoint (``openai`` SDK, ``chat.completions``,
  ``response_format`` json_schema strict or json_object, ``reasoning_effort`` -> thinkingLevel),
  base URL and key from the machine's endpoint profile (``fsa.redo.machine``; default: the public
  endpoint ``GEMINI_BASE_URL`` with ``GEMINI_API_KEY``). Gemini reports thinking tokens only inside
  ``total_tokens``; ``reasoning_tokens`` = total - prompt - completion.
* ``gpt-*`` / ``o*`` - the OpenAI Responses API (kept only so historical records stay
  reproducible; no current seat uses it).

``json_call`` retries patiently: transient errors (429/5xx/overload/timeouts) up to 9 times with
10..300 s backoff; invalid output (bad JSON, ``validate`` raising ValueError) up to 4 times.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from pathlib import Path

from . import machine

ROOT = Path(__file__).resolve().parents[2]
SEATS_FILE = ROOT / "configs" / "seats" / "redo_v1_seats.json"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  # the public profile's

TRANSIENT_BACKOFF_SEC = (10, 20, 40, 60, 120, 180, 300, 300, 300)
INVALID_BACKOFF_SEC = (5, 15, 30, 60)
_TRANSIENT = ("429", "rate limit", "ratelimit", "rate_limit", "resource_exhausted", "quota", "503", "502", "500", "504",
              "overloaded", "high demand", "timeout", "timed out", "connection", "unavailable", "internal", "server_error")

_lock = threading.Lock()
_clients: dict[tuple, object] = {}


def seats_doc() -> dict:
    return json.loads(SEATS_FILE.read_text())


def judge_seat() -> dict:
    """{model, reasoning_effort, source} of the judge seat (config, then environment override)."""
    s = seats_doc()["seats"]["judge"]
    model = os.environ.get("REDO_JUDGE_MODEL") or s["model"]
    effort = os.environ.get("REDO_JUDGE_EFFORT") or s["reasoning_effort"]
    src = "env" if (os.environ.get("REDO_JUDGE_MODEL") or os.environ.get("REDO_JUDGE_EFFORT")) else \
        str(SEATS_FILE.relative_to(ROOT))
    return {"model": model, "reasoning_effort": effort, "source": src}


def backend(model: str) -> str:
    m = model.split("/", 1)[-1]
    return "openai" if m.startswith(("gpt-", "o1", "o3", "o4")) else "gemini"


def gemini_endpoint() -> str:
    """The OpenAI-compatible Gemini base URL of the active machine profile."""
    return machine.openai_base()


def client_for(model: str, timeout: float = 900):
    """Cached ``openai.OpenAI`` client for the model's backend (no SDK-level retries)."""
    import openai
    b = backend(model)
    prof = machine.active() if b == "gemini" else None
    key = (b, timeout, prof.name if prof else None)
    with _lock:
        if key not in _clients:
            if b == "gemini":
                api_key = prof.api_key()
                if not api_key:
                    raise RuntimeError(f"no Gemini API key for machine profile {prof.name!r} "
                                       f"({prof.key_source()}; Gemini judge seat)")
                _clients[key] = openai.OpenAI(api_key=api_key, base_url=prof.openai_base, timeout=timeout,
                                              max_retries=0)
            else:
                _clients[key] = openai.OpenAI(timeout=timeout, max_retries=0)
        return _clients[key]


def transient(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    if "input token count exceeds" in s:
        return False
    return any(m in s for m in _TRANSIENT)


def _chat_usage(u) -> dict:
    if u is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None, "cached_tokens": None}
    pt, ct, tt = getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None), getattr(u, "total_tokens", None)
    ctd, ptd = getattr(u, "completion_tokens_details", None), getattr(u, "prompt_tokens_details", None)
    rt = getattr(ctd, "reasoning_tokens", None) if ctd is not None else None
    if rt is None and None not in (pt, ct, tt):
        rt = max(0, tt - pt - ct)  # Gemini: thoughts are counted in total_tokens only
    return {"input_tokens": pt, "output_tokens": ct, "reasoning_tokens": rt,
            "cached_tokens": getattr(ptd, "cached_tokens", None) if ptd is not None else None, "total_tokens": tt}


def raw_call(client, model: str, effort: str | None, system: str, user: str, schema: dict | None, name: str,
             extra: dict | None = None) -> tuple[str, dict, str, str | None, str | None]:
    """One attempt. Returns (output text, usage, resolved model, response id, finish reason).
    ``schema=None`` asks for a JSON object without a schema."""
    if backend(model) == "openai":
        kw = {"reasoning": {"effort": effort}} if effort else {}
        fmt = ({"type": "json_schema", "name": name, "schema": schema, "strict": True} if schema is not None
               else {"type": "json_object"})
        r = client.responses.create(model=model, input=[{"role": "system", "content": system},
                                                        {"role": "user", "content": user}],
                                    text={"format": fmt}, **kw)
        u = r.usage
        usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "reasoning_tokens": getattr(u.output_tokens_details, "reasoning_tokens", None),
                 "cached_tokens": getattr(u.input_tokens_details, "cached_tokens", None)}
        return r.output_text, usage, r.model, r.id, None
    rf = ({"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
          if schema is not None else {"type": "json_object"})
    kw = dict(extra or {})
    if effort:
        kw["reasoning_effort"] = effort
    r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system},
                                                              {"role": "user", "content": user}],
                                       response_format=rf, **kw)
    ch = r.choices[0] if r.choices else None
    text = (ch.message.content if ch is not None else None) or ""
    return text, _chat_usage(r.usage), getattr(r, "model", model), getattr(r, "id", None), \
        getattr(ch, "finish_reason", None)


def strip_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def json_call(system: str, user: str, schema: dict | None, *, name: str, model: str, effort: str | None,
              validate=None, log=None, meta: dict | None = None, client=None, sleep=time.sleep,
              extra: dict | None = None) -> dict:
    """Structured call with patient retry. ``log(rec)`` receives every attempt (ok or not).
    Returns ``{"output", "usage", "response_id", "resolved_model", "attempts"}``."""
    n_trans = n_bad = 0
    attempt = 0
    last = None
    while True:
        attempt += 1
        t0 = time.time()
        rec = {"ts": t0, "model": model, "effort": effort, "backend": backend(model), "schema_name": name,
               "attempt": attempt, **(meta or {})}
        try:
            c = client if client is not None else client_for(model)
            text, usage, resolved, rid, finish = raw_call(c, model, effort, system, user, schema, name, extra)
            rec.update(latency_sec=round(time.time() - t0, 2), resolved_model=resolved, usage=usage,
                       output_text=text, response_id=rid, finish_reason=finish)
            try:
                out = json.loads(strip_fence(text))
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON output ({finish}): {e}") from e
            if validate is not None:
                validate(out)
            rec["ok"] = True
            if log:
                log(rec)
            return {"output": out, "usage": usage, "response_id": rid, "resolved_model": resolved,
                    "attempts": attempt}
        except Exception as e:  # noqa: BLE001
            last = e
            rec.update(ok=False, latency_sec=round(time.time() - t0, 2), error=f"{type(e).__name__}: {str(e)[:500]}")
            if log:
                log(rec)
            if isinstance(e, (ValueError, KeyError, TypeError)) and not transient(e):
                n_bad += 1
                if n_bad > len(INVALID_BACKOFF_SEC):
                    break
                sleep(INVALID_BACKOFF_SEC[n_bad - 1] * (1 + 0.2 * random.random()))
            elif transient(e):
                n_trans += 1
                if n_trans > len(TRANSIENT_BACKOFF_SEC):
                    break
                sleep(TRANSIENT_BACKOFF_SEC[n_trans - 1] * (1 + 0.2 * random.random()))
            else:
                break
    raise RuntimeError(f"{name}: {model} call failed after {attempt} attempts: {last}")


def seat_record(model: str, effort: str | None, **more) -> dict:
    """Manifest record of a seat: settings plus the provider's model record."""
    gem = backend(model) == "gemini"
    rec = {"model": model, "reasoning_effort": effort, "backend": backend(model),
           "endpoint": gemini_endpoint() if gem else "https://api.openai.com/v1", **more}
    if gem:
        rec["endpoint_profile"] = machine.active().name
        from .swe_together.run import gemini_model_record
        rec["provider_record"] = gemini_model_record(model.split("/", 1)[-1])
    return rec
