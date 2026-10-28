"""One structured call on the judge seat (strict JSON schema) with patient retries; every attempt
is kept as one JSON file under ``<log_dir>/calls/`` (the seat's token record). Shared by the
SWE-Interact intent builder, revelation tagger and rubric judge and by the SWE-Together mapper /
judge v2. Seat: ``configs/seats/redo_v1_seats.json`` (``gemini-3.5-flash`` since 2026-09-27;
``gpt-5.5`` on the OpenAI Responses API before). Backend and retry: ``fsa.redo.llm_seats``."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .. import llm_seats

_SEAT = llm_seats.judge_seat()
MODEL = _SEAT["model"]
EFFORT = _SEAT["reasoning_effort"]


def client():
    return llm_seats.client_for(MODEL)


def structured_call(system: str, user: str, schema: dict, *, name: str, log_dir: Path, effort: str | None = None,
                    model: str | None = None, meta: dict | None = None, validate=None, sleep=time.sleep,
                    client=None) -> dict:
    """Returns ``{"output": <parsed json>, "usage": {...}, "response_id", "resolved_model", "attempts"}``.
    ``validate(parsed)`` may raise ValueError to force a retry (counted as an attempt)."""
    log_dir = Path(log_dir) / "calls"
    log_dir.mkdir(parents=True, exist_ok=True)

    def log(rec):
        (log_dir / f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.json").write_text(json.dumps(rec, indent=1))
    return llm_seats.json_call(system, user, schema, name=name, model=model or MODEL, effort=effort or EFFORT,
                               validate=validate, log=log, meta=meta, sleep=sleep, client=client)
