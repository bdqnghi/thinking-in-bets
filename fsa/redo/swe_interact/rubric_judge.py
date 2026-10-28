"""Per-item rubric judge for SWE-Interact refactoring (rf_*) tasks on any repository snapshot.

This is the benchmark's own rubric grader (``tests/evaluate_rubrics.py``) run from the host on
one snapshot instead of only on the final patch: the task's ``rubrics_system_prompt.txt`` and
``rubrics_user_prompt_template.txt``, the problem statement ``prompt.txt``, the snapshot diff as
the response and one rubric title per call; YES/NO; a negative item passes on NO. Seat as in the
final verifier (recorded deviation from Claude Opus 4.5): the judge seat (``gemini-3.5-flash`` since 2026-09-27,
``gpt-5.5`` before), chat completions, ``response_format=json_object``, ``max_completion_tokens=16384``,
``reasoning_effort=high``; patient retry (``fsa.redo.llm_seats``). The judge never sees the dialogue.
An unparseable answer (upstream: IGNORED) is ``error`` here. Verdicts are cached per
sha256(model, settings, prompts, task, item id + title, patch).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import llm_seats
from .config import RESULTS, TASKS

MODEL = llm_seats.judge_seat()["model"]
SETTINGS = {"max_completion_tokens": 16384, "reasoning_effort": "high"}
CACHE = RESULTS / "swi_rubric_cache"


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def _parse(text: str):
    """Same tolerant parse as evaluate_rubrics._parse_llm_response."""
    if not text:
        return None
    text = text.strip()
    if "```json" in text:
        after = text[text.find("```json") + 7:]
        end = after.find("```")
        if end != -1:
            text = after[:end].strip()
    if not text.startswith("{"):
        start = text.find('{"ratings"')
        if start == -1:
            start = text.find('{ "ratings"')
        if start != -1:
            text = text[start:]
            depth = 0
            for i, ch in enumerate(text):
                depth += (ch == "{") - (ch == "}")
                if depth == 0:
                    text = text[:i + 1]
                    break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class RubricJudge:
    def __init__(self, cache_dir: Path = CACHE, workers: int = 8, client=None):
        self.cache_dir = Path(cache_dir)
        (self.cache_dir / "calls").mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self._client = client
        self._lock = threading.Lock()
        self.stats = {"cache_hits": 0, "calls": 0, "failed_calls": 0, "input_tokens": 0, "output_tokens": 0,
                      "reasoning_tokens": 0}

    @property
    def client(self):
        if self._client is None:
            self._client = llm_seats.client_for(MODEL)
        return self._client

    @staticmethod
    def prompts(task: str) -> tuple[str, str, str]:
        d = TASKS / task / "steps" / "05_test_handoff" / "tests"
        return ((d / "rubrics_system_prompt.txt").read_text(), (d / "rubrics_user_prompt_template.txt").read_text(),
                (d / "prompt.txt").read_text())

    def key(self, task: str, item: dict, patch: str) -> str:
        sysp, tmpl, ps = self.prompts(task)
        return _sha(json.dumps([MODEL, SETTINGS, _sha(sysp), _sha(tmpl), _sha(ps), task, item["rubric_id"],
                                item["title"], _sha(patch.encode("utf-8", "surrogateescape"))]))

    def _path(self, k: str) -> Path:
        return self.cache_dir / k[:2] / f"{k}.json"

    def _one(self, task: str, item: dict, patch: str) -> dict:
        k = self.key(task, item, patch)
        p = self._path(k)
        if p.exists():
            with self._lock:
                self.stats["cache_hits"] += 1
            return {**json.loads(p.read_text()), "cached": True}
        sysp, tmpl, ps = self.prompts(task)
        user = tmpl.format(problem_statement=ps, model_answer=patch, title=json.dumps(item["title"]))
        rec = None
        state = {}

        def validate(parsed):
            status = None
            if isinstance(parsed, dict) and isinstance(parsed.get("ratings"), list) and parsed["ratings"]:
                status = str(parsed["ratings"][0].get("status") or "").strip().upper()
            if status not in ("YES", "NO"):
                raise ValueError(f"unparseable rating: {json.dumps(parsed)[:200]}")
            state["status"], state["just"] = status, parsed["ratings"][0].get("justification")

        def log(call):
            call.update(task=task, rubric_id=item["rubric_id"], **SETTINGS)
            with self._lock:
                if call.get("ok"):
                    u = call.get("usage") or {}
                    self.stats["calls"] += 1
                    self.stats["input_tokens"] += u.get("input_tokens") or 0
                    self.stats["output_tokens"] += u.get("output_tokens") or 0
                    self.stats["reasoning_tokens"] += u.get("reasoning_tokens") or 0
                else:
                    self.stats["failed_calls"] += 1
            self._log(call)
        try:
            r = llm_seats.json_call(sysp, user, None, name="rubric_item", model=MODEL,
                                    effort=SETTINGS["reasoning_effort"], validate=validate, log=log,
                                    client=self.client,
                                    extra={"max_completion_tokens": SETTINGS["max_completion_tokens"]})
            status = state["status"]
            passes = (status == "NO") if item["kind"] == "rubric_negative" else (status == "YES")
            rec = {"verdict": "satisfied" if passes else "not_satisfied", "status": status,
                   "evidence": state["just"], "rubric_id": item["rubric_id"], "patch_sha256": _sha(patch.encode(
                       "utf-8", "surrogateescape")), "resolved_model": r["resolved_model"], "judged_at": time.time()}
        except RuntimeError:
            rec = None
        if rec is None:
            return {"verdict": "error", "evidence": "rubric judge failed", "cached": False}
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, p)
        return {**rec, "cached": False}

    def _log(self, call: dict) -> None:
        (self.cache_dir / "calls" / f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.json").write_text(
            json.dumps(call, indent=1))

    def judge(self, task: str, patch: str, items: list[dict]) -> dict[str, dict]:
        """key -> verdict record for rubric intents (``kind`` rubric_positive / rubric_negative)."""
        if not items:
            return {}
        with ThreadPoolExecutor(max_workers=min(self.workers, len(items))) as ex:
            res = list(ex.map(lambda it: self._one(task, it, patch), items))
        return {it["key"]: r for it, r in zip(items, res)}
