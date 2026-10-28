"""Intent -> executable-check and intent -> frozen-goal maps for SWE-Together (per-intent judge v2).

    python -u -m fsa.redo.swe_together.intent_map build [--tasks a,b]   # judge-seat proposal per task
    python -u -m fsa.redo.swe_together.intent_map freeze                # configs/intents/swe-together/exec_map.json

Eligible checks of a task (``eligible_checks``), from the reference and unmodified-base runs of
``exec_judge`` (``results/redo_v1/swt_exec_cache/refbase_summary.json``):

* ``gate:<id>`` emitted explicitly by the verifier (auto-bridged gates carry no per-gate result);
* ``test:<name>`` of a SWE-rebench-style verifier that exists in the unmodified repository (status
  not ``MISSING`` on the base); a FAIL_TO_PASS test that only exists once the agent writes a test
  of exactly that name cannot decide a behaviour intent and is not offered.

The judge seat (``gemini-3.5-flash``, effort high, ``prompts/intent_map_system.md``) proposes, per intent, the checks that
decide it (``full`` / ``partial`` / ``none``) and the goals of ``canonical_goals.json`` that
express it, plus the user turn each goal comes from. Only ``full`` mappings whose checks all pass
on the reference are used by the executable judge; the reviewed map is frozen (with digest) in
``configs/intents/swe-together/exec_map.json``; review edits are recorded per entry (``review``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from ..intent_judge import load_task_intents
from ..swe_interact.llm import MODEL, structured_call
from .config import CLONE, RESULTS, ROOT, subset_tasks

PROMPT = Path(__file__).with_name("prompts") / "intent_map_system.md"
BUILD_DIR = RESULTS / "swt_intent_map_build"
REFBASE = RESULTS / "swt_exec_cache" / "refbase_summary.json"
MAP_FILE = ROOT / "configs" / "intents" / "swe-together" / "exec_map.json"
EFFORT = "high"


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def manifest_gates(task: str) -> dict[str, dict]:
    p = CLONE / "tasks" / task / "tests" / "test_manifest.yaml"
    if not p.exists():
        return {}
    m = yaml.safe_load(p.read_text()) or {}
    return {g["id"]: g for g in (m.get("gates") or []) if isinstance(g, dict) and g.get("id")}


def eligible_checks(task: str, refbase: dict) -> list[dict]:
    rb = refbase[task]
    ref = (rb.get("reference") or {}).get("checks") or {}
    base = (rb.get("base") or {}).get("checks") or {}
    gates = manifest_gates(task)
    out = []
    for cid in sorted(set(ref) | set(base)):
        r, b = ref.get(cid) or {}, base.get(cid) or {}
        if r.get("bridged") or b.get("bridged"):
            continue
        if cid.startswith("test:") and b.get("status") in (None, "MISSING"):
            continue
        g = gates.get(cid.split(":", 1)[1], {}) if cid.startswith("gate:") else {}
        kind = g.get("kind", "TEST" if cid.startswith("test:") else "F2P")
        out.append({"id": cid, "kind": kind, "weight": g.get("weight"), "description": g.get("description", ""),
                    "command": str(g.get("command", ""))[:1500],
                    "reference": None if not r else ("pass" if r.get("passed") else "fail"),
                    "base": None if not b else ("pass" if b.get("passed") else "fail")})
    return out


def _schema(keys: list[str], goals: list[str], checks: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "intents": {"type": "array", "items": {
                "type": "object",
                "properties": {"key": {"type": "string", "enum": keys},
                               "checks": {"type": "array", "items": {"type": "string", "enum": checks or ["-"]}},
                               "check_coverage": {"type": "string", "enum": ["full", "partial", "none"]},
                               "check_note": {"type": "string"},
                               "goals": {"type": "array", "items": {"type": "string", "enum": goals or ["-"]}},
                               "goal_relation": {"type": "string", "enum": ["same", "broader", "narrower", "none"]},
                               "goal_note": {"type": "string"}},
                "required": ["key", "checks", "check_coverage", "check_note", "goals", "goal_relation", "goal_note"],
                "additionalProperties": False}},
            "goals": {"type": "array", "items": {
                "type": "object",
                "properties": {"id": {"type": "string", "enum": goals or ["-"]}, "source_turn": {"type": "integer"},
                               "note": {"type": "string"}},
                "required": ["id", "source_turn", "note"], "additionalProperties": False}},
        },
        "required": ["intents", "goals"], "additionalProperties": False,
    }


def build(task: str, refbase: dict) -> dict:
    td = CLONE / "tasks" / task
    intents = load_task_intents(td)
    goals = json.loads((td / "canonical_goals.json").read_text())["completeness_goals"]
    checks = eligible_checks(task, refbase)
    user = "\n\n".join([
        f"# Task {task}",
        "## Intents\n" + "\n".join(json.dumps({"key": i.key, "source_turn": i.source_turn, "kind": i.kind,
                                               "text": i.text}, ensure_ascii=False) for i in intents),
        "## Goals (canonical_goals.json)\n" + "\n".join(json.dumps(
            {k: g.get(k) for k in ("id", "tier", "weight", "goal", "rationale")}, ensure_ascii=False) for g in goals),
        "## Eligible executable checks\n" + ("\n".join(json.dumps(c, ensure_ascii=False) for c in checks)
                                             if checks else "(none)"),
    ])
    keys = [i.key for i in intents]
    gids = [g["id"] for g in goals]
    cids = [c["id"] for c in checks]

    def validate(out):
        got = [x["key"] for x in out["intents"]]
        if sorted(got) != sorted(keys):
            raise ValueError(f"intent keys mismatch: {sorted(set(keys) ^ set(got))}")
        if sorted(x["id"] for x in out["goals"]) != sorted(gids):
            raise ValueError("goal ids mismatch")
    r = structured_call(PROMPT.read_text(), user, _schema(keys, gids, cids), name="swt_intent_map",
                        log_dir=BUILD_DIR, model=MODEL, effort=EFFORT, meta={"task": task}, validate=validate)
    doc = {"task": task, "prompt_sha256": _sha(PROMPT.read_text()), "model": MODEL, "effort": EFFORT,
           "resolved_model": r["resolved_model"], "response_id": r["response_id"], "usage": r["usage"],
           "eligible_checks": checks, "proposal": r["output"]}
    (BUILD_DIR / f"{task}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    return doc


def load_map(path: Path = MAP_FILE, verify: bool = True) -> dict:
    doc = json.loads(Path(path).read_text())
    if verify:
        d = map_digest(doc)
        if d != doc.get("digest"):
            raise SystemExit(f"{path}: digest mismatch ({d} != {doc.get('digest')}); the map was edited after freezing")
    return doc


def map_digest(doc: dict) -> str:
    body = {k: doc[k] for k in ("schema", "version", "tasks")}
    return _sha(json.dumps(body, sort_keys=True, ensure_ascii=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    refbase = json.loads(REFBASE.read_text())
    tasks = a.tasks.split(",") if a.tasks else subset_tasks()

    def one(t):
        try:
            d = build(t, refbase)
            full = sum(1 for x in d["proposal"]["intents"] if x["check_coverage"] == "full")
            print(f"{t}: {len(d['eligible_checks'])} checks, {full}/{len(d['proposal']['intents'])} intents full",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {t}: {e}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
