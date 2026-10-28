"""Offline Refiner for the intent-mediator port (controlled port of Mediator-Assistant,
arXiv 2602.07338 §4.2; not a native reproduction).

    python -u -m fsa.redo.ports_refine pairs                     # build + print the contrastive pairs
    python -u -m fsa.redo.ports_refine refine --models gemini-3.5-flash-lite,gemini-3.6-flash,gemini-3.8-flash

The paper distils "experiences" E from contrastive pairs (a failed multi-turn transcript C_fail and
the successful single-turn request I_succ of the same instance; 5 instances per task domain, removed
from the test set) with the "Prompt For Refiner" (App. B, verbatim here), and injects E into the
Mediator's prompt ([[EXPS]]). Our pairs (recorded deviation): 5 SWE-Together tasks *outside* the
frozen 20-task subset and the first 3 replacement candidates of each stratum, drawn once with the subset seed; C = the task's real recorded session
(``original_session.json``: the user's messages and the original assistant's text replies, tool
traffic dropped), I = a fully specified first-person request made deterministically from the
opening request and every oracle intent of the session (``oracle_intents.json``). E is distilled
once per rung model (the arm's own seat) and frozen in ``fsa/redo/prompts/ports/mediator_experiences.json``;
at full scale these 5 tasks are to be excluded from mediator comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from . import ports as P

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "third_party" / "swe-together" / "tasks"
SUBSET = ROOT / "configs" / "subsets" / "swe-together.json"
SEED = 20260926
N_PAIRS = 5
REPLACEMENT_GUARD = 3
OUT = P.PROMPTS / "mediator_experiences.json"
PAIR_BUDGET = 24_000


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def session_dialogue(task_dir: Path) -> list[dict]:
    msgs = json.loads((task_dir / "original_session.json").read_text())["messages"]
    out = []
    for m in msgs:
        t = _text(m.get("content")).strip()
        if not t or t.startswith("<local-command") or t.startswith("<command-name>/clear"):
            continue
        role = "user" if m.get("role") == "user" else "assistant"
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n" + t
        else:
            out.append({"role": role, "content": t})
    return out


def ground_truth(task_dir: Path) -> str:
    instr = (task_dir / "instruction.md").read_text().strip()
    its = json.loads((task_dir / "oracle_intents.json").read_text())["intents"]
    lines = [instr[:4000], "", "To be complete, here is everything I need from you in this session:"]
    lines += [f"{k}. {i['text']}" for k, i in enumerate(its, 1)]
    return "\n".join(lines)


def held_out_tasks() -> list[str]:
    sub = json.loads(SUBSET.read_text())
    chosen = {t["task"] for t in sub["selected"]}
    for v in (sub.get("replacements") or {}).values():  # the next candidates of each stratum's replacement list
        chosen |= set(v[:REPLACEMENT_GUARD])             # (the subset's infrastructure replacements) are excluded
    pool = sorted(d.name for d in TASKS.iterdir()
                  if (d / "original_session.json").exists() and (d / "oracle_intents.json").exists()
                  and d.name not in chosen)
    return sorted(random.Random(SEED).sample(pool, N_PAIRS))


def pairs() -> list[dict]:
    return [{"task": t, "transcript": P.render_dialogue(session_dialogue(TASKS / t), budget=PAIR_BUDGET),
             "ground_truth": ground_truth(TASKS / t)} for t in held_out_tasks()]


def refiner_input(ps: list[dict]) -> str:
    ex = []
    for k, p in enumerate(ps, 1):
        ex.append(f"Example {k}\n1. Conversation Transcript:\n{p['transcript']}\n\n"
                  f"2. Ground Truth Rewritten Request:\n{p['ground_truth']}")
    return P.prompt("refiner_prompt.md").replace("[[CONVERSATIONS]]", "\n\n".join(ex))


def parse_rules(content: str) -> str:
    if "[[RULES]]" not in content:
        raise ValueError("invalid json refiner output: no [[RULES]] section")  # 'invalid json' = retried
    body = content.split("[[RULES]]", 1)[1].split("[[END]]", 1)[0].strip()
    if not body:
        raise ValueError("empty response rules")
    return body


def refine(models: list[str], out_dir: Path) -> dict:
    ps = pairs()
    text = refiner_input(ps)
    doc = json.loads(OUT.read_text()) if OUT.exists() else {"schema": "redo_v1/mediator-experiences/1", "by_model": {}}
    doc.update({"method": P.PORT_LABEL["mediator"], "refiner_prompt_sha256": P.prompts_sha256()["refiner_prompt.md"],
                "pairs": [p["task"] for p in ps], "pair_seed": SEED, "pair_budget_chars": PAIR_BUDGET,
                "refiner_input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "pair_source": "SWE-Together tasks outside configs/subsets/swe-together.json; transcript = "
                               "original_session.json (user + assistant text), ground truth = instruction.md + "
                               "oracle_intents.json (deterministic template)"})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "refiner_input.txt").write_text(text)
    for m in models:
        arm = P.PortArm.__new__(P.PortArm)  # only the retrying call helper is needed
        arm.arm, arm.out_dir, arm.model, arm.reasoning = "mediator_refiner", out_dir, P.litellm_model(f"google/{m}"), "high"
        arm._completion, arm._sleep = None, time.sleep
        rules, ok, recs = arm._call(0, [{"role": "user", "content": text}], json_mode=False, parse=parse_rules)
        if not ok:
            raise SystemExit(f"refiner failed for {m}: {recs[-1].get('error')}")
        last = recs[-1]
        doc["by_model"][m] = {"rules": rules, "rules_sha256": hashlib.sha256(rules.encode()).hexdigest(),
                              "refiner_model": arm.model, "reasoning_effort": "high", "pairs": [p["task"] for p in ps],
                              "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "usage": {k: last.get(k) for k in ("prompt_tokens", "completion_tokens",
                                                                   "reasoning_tokens", "resolved_model")},
                              "attempts": len(recs)}
        print(f"{m}: {len(rules)} chars of rules ({len(recs)} attempt(s))", flush=True)
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc


def main(argv: list[str] | None = None) -> int:
    from fsa.redo import machine
    machine.apply_process_env()  # this machine's endpoint profile, for this process and its children
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pairs")
    r = sub.add_parser("refine")
    r.add_argument("--models", required=True)
    r.add_argument("--out-dir", default=str(ROOT / "results" / "redo_v1" / "baseline_ports_smoke" / "refiner"))
    a = ap.parse_args(argv)
    if a.cmd == "pairs":
        for p in pairs():
            print(json.dumps({"task": p["task"], "transcript_chars": len(p["transcript"]),
                              "ground_truth": p["ground_truth"][:600]}, ensure_ascii=False))
        return 0
    refine(a.models.split(","), Path(a.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
