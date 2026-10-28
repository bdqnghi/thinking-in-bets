"""Revelation tagger for SWE-Interact: which intents a simulator message reveals ("revealed by turn t").

SWE-Interact's simulator holds the whole task block from turn 0 and discloses requirements at
review time, so there is no ground-truth cursor as in SWE-Together. One judge-seat call per
simulator message (``prompts/revelation_tagger_system.md``, strict schema; seat model and effort from
``configs/seats/redo_v1_seats.json``: gemini-3.5-flash / high since 2026-09-27, gpt-5.5 / medium before) sees
only the frozen intent list and that one message - never the repository or the agent's words -
and labels every intent explicit | implied | no, plus the message's act
(revision_request | approval | answer | other). Context-free by construction, so results are
cached per sha256(prompt, model, message, intent-set digest).

Definitions (``revealed_sets``), for the messages delivered up to the one that opened turn t:

* ``tagger``      - intents tagged explicit or implied in some message, plus the headline intent
                    once any simulator message was delivered (the user's first reply states the task);
* ``conservative`` - intents tagged explicit in some message or in the agreed plan
                    (``/tmp/plan.md`` saved at the end of 01_plan), plus the headline.

Overlay messages and simulator replies the overlay replaced were never "said" by the benchmark's
user and reveal nothing; the act of a message is also the correction label used by the metrics
(correction = ``revision_request``).
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from .config import RESULTS
from .llm import EFFORT, MODEL, structured_call

PROMPT = Path(__file__).with_name("prompts") / "revelation_tagger_system.md"
CACHE = RESULTS / "swi_tagger_cache"
LEVELS = ("explicit", "implied", "no")
ACTS = ("revision_request", "approval", "answer", "other")


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def prompt_sha() -> str:
    return _sha(PROMPT.read_bytes())


def config_record() -> dict:
    return {"model": MODEL, "effort": EFFORT, "prompt_file": "fsa/redo/swe_interact/prompts/revelation_tagger_system.md",
            "prompt_sha256": prompt_sha(), "levels": LEVELS, "acts": ACTS}


def _schema(keys: list[str]) -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["act", "intents"],
            "properties": {"act": {"type": "string", "enum": list(ACTS)},
                           "intents": {"type": "array", "items": {
                               "type": "object", "additionalProperties": False, "required": ["key", "level"],
                               "properties": {"key": {"type": "string", "enum": keys},
                                              "level": {"type": "string", "enum": list(LEVELS)}}}}}}


def tag_message(message: str, intent_doc: dict, cache: Path = CACHE) -> dict:
    """{"act": ..., "levels": {key: explicit|implied|no}, "cached": bool}."""
    its = intent_doc["intents"]
    k = _sha(json.dumps([MODEL, EFFORT, prompt_sha(), intent_doc["digest"], message]))
    p = Path(cache) / k[:2] / f"{k}.json"
    if p.exists():
        return {**json.loads(p.read_text()), "cached": True}
    keys = [i["key"] for i in its]

    def check(out):
        got = {x["key"] for x in out["intents"]}
        if set(keys) - got:
            raise ValueError(f"tagger missing keys {sorted(set(keys) - got)}")
    user = ("## Intents\n" + "\n".join(json.dumps({"key": i["key"], "intent": i["text"]}, ensure_ascii=False)
                                       for i in its) + f"\n\n## The user's message\n{message}\n")
    res = structured_call(PROMPT.read_text(), user, _schema(keys), name="revelation", log_dir=Path(cache),
                          model=MODEL, effort=EFFORT, meta={"kind": "tag", "task": intent_doc["task"], "message_sha256": _sha(message)},
                          validate=check)
    levels = {x["key"]: x["level"] for x in res["output"]["intents"]}
    rec = {"act": res["output"]["act"], "levels": levels, "message_sha256": _sha(message),
           "intent_digest": intent_doc["digest"], "usage": res["usage"], "resolved_model": res["resolved_model"]}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, p)
    return {**rec, "cached": False}


def headline(intent_doc: dict) -> str | None:
    return next((i["key"] for i in intent_doc["intents"] if i.get("headline")), None)


def revealed_sets(tags: list[dict], intent_doc: dict, plan_tag: dict | None = None) -> dict[str, list[str]]:
    """``tags``: tag records of the delivered simulator messages so far (in order)."""
    hl = headline(intent_doc)
    tagger, cons = set(), set()
    for t in tags:
        for k, lvl in t["levels"].items():
            if lvl in ("explicit", "implied"):
                tagger.add(k)
            if lvl == "explicit":
                cons.add(k)
    if plan_tag:
        cons |= {k for k, lvl in plan_tag["levels"].items() if lvl == "explicit"}
    if tags and hl:
        tagger.add(hl)
        cons.add(hl)
    order = [i["key"] for i in intent_doc["intents"]]
    return {"tagger": [k for k in order if k in tagger], "conservative": [k for k in order if k in cons]}
