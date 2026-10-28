"""Invariant checks on Stage 1 arm / overlay trials.

    python -m fsa.redo.swe_together.check_arms <run_dir|root> [...]

Per trial:

* timing - ``commit_before`` / ``claims_no_p``: the claim book is written before the
  simulator is consulted (``claims_written_ts < sim_consult_started_ts``);
  ``claims_after``: the call starts after the simulator decided
  (``claims_call_started_ts >= sim_decided_ts``);
* isolation - no simulator prompt (``episode-N/user_sim_prompt.json``: system prompt,
  history, turn content) contains the arm block's marker lines or any claim text of any
  claim book of the trial;
* delivery - a real reply carries the arm block, "continue" never does;
* port arms (tracker / mediator, controlled ports of IDSS / Mediator-Assistant) - the call starts
  after the simulator decided (``port_call_started_ts >= sim_decided_ts``); a real reply with a
  successful call is delivered with the tracker state / replaced by the mediator's rewrite; no
  simulator prompt contains the tracker block's markers or any mediator rewrite of the trial;
* overlay - at most two injections, the first unjustified, each on a consultation whose
  in-trial judge said every revealed applicable intent was satisfied; the simulator's
  history shows the overlay message instead of its own reply.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..metrics import find_run_dirs

MARKERS = ("Private note from your own bookkeeping", "Your claim book:", "How to handle the user's message above",
           "(Reminder: everything the user has asked for so far", "Make the smallest change that addresses",
           "CURRENT TASK STATE (IDSS)", "=== FACT LAYER (verified entities) ===", "=== STATE LAYER (task understanding) ===",
           "Task state maintained by your harness")
PORT_ARMS = ("tracker", "mediator")


def check_trial(trial: Path) -> dict:
    ag = trial / "agent"
    turns = [json.loads(l) for l in (ag / "turns.jsonl").read_text().splitlines() if l.strip()]
    problems, notes = [], []
    books = [json.loads(p.read_text()) for p in sorted((ag / "claims").glob("turn-*.json"))] \
        if (ag / "claims").exists() else []
    claim_texts = {c.get("claim", "") for b in books for c in ((b.get("book") or {}).get("claims") or [])}
    claim_texts = {t for t in claim_texts if len(t) >= 25}
    ports = [json.loads(p.read_text()) for p in sorted((ag / "ports").glob("turn-*.json"))] \
        if (ag / "ports").exists() else []
    claim_texts |= {d["rewrite"] for d in ports if d.get("arm") == "mediator" and len(d.get("rewrite") or "") >= 25}
    for r in turns:
        red = r.get("redo")
        if not red:
            continue
        arm, u = red.get("arm"), r["turn"]
        if red.get("claims_book"):
            if arm in ("commit_before", "claims_no_p"):
                ok = red["claims_written_ts"] < red["sim_consult_started_ts"]
                notes.append(f"turn {u}: book written {red['sim_consult_started_ts'] - red['claims_written_ts']:.3f}s "
                             f"before the simulator was consulted")
            else:
                ok = red["claims_call_started_ts"] >= red["sim_decided_ts"]
                notes.append(f"turn {u}: book call started {red['claims_call_started_ts'] - red['sim_decided_ts']:.3f}s "
                             f"after the simulator decided")
            if not ok:
                problems.append(f"turn {u}: claim-book timing violates {arm}")
        if arm in PORT_ARMS and red.get("port_doc"):
            gap = red["port_call_started_ts"] - red["sim_decided_ts"]
            notes.append(f"turn {u}: {arm} call started {gap:.3f}s after the simulator decided")
            if gap < 0:
                problems.append(f"turn {u}: {arm} call started before the simulator decided")
        if arm in PORT_ARMS and r.get("opened_by") in ("sim", "overlay") and not red.get("port_doc"):
            problems.append(f"turn {u}: {arm} made no call for a real reply")
        if r.get("opened_by") == "noop_continue" and red.get("arm_block_delivered"):
            problems.append(f"turn {u}: arm block delivered with 'continue'")
        if r.get("opened_by") in ("sim", "overlay") and arm != "stock" and not red.get("arm_block_delivered") \
                and red.get("claims_ok", True) and red.get("port_ok", True):
            problems.append(f"turn {u}: real reply without the arm block")
    leaks = 0
    for p in sorted(ag.glob("episode-*/user_sim_prompt.json")):
        txt = p.read_text()
        if any(m in txt for m in MARKERS) or any(t in txt for t in claim_texts):
            leaks += 1
            problems.append(f"{p.parent.name}: simulator prompt contains arm material")
    ov = [r for r in turns if r.get("opened_by") == "overlay"]
    if ov:
        if len(ov) > 2:
            problems.append(f"{len(ov)} overlay injections (> 2)")
        if (ov[0].get("redo") or {}).get("overlay_kind") != "unjustified":
            problems.append("first overlay is not the unjustified one")
        for r in ov:
            red = r["redo"]
            if not red.get("overlay_all_revealed_satisfied"):
                problems.append(f"turn {r['turn']}: overlay fired without all revealed intents satisfied")
            sp = ag / f"episode-{r['turn'] + 1}" / "user_sim_prompt.json"
            if sp.exists():
                msgs = json.loads(sp.read_text()).get("messages") or []
                hist = " ".join(m.get("content", "") for m in msgs if m.get("role") == "assistant")
                if r["message"] not in hist:
                    problems.append(f"turn {r['turn']}: overlay message missing from the simulator's history")
                if red.get("sim_message_replaced") and f"→ {r.get('sim_action')}: {red['sim_message_replaced']}" in hist:
                    problems.append(f"turn {r['turn']}: replaced simulator reply still in its history")
            notes.append(f"turn {r['turn']}: overlay {red.get('overlay_kind')} -> {r['message']!r} "
                         f"(replaced sim {r.get('sim_action')}: {red.get('sim_message_replaced')!r})")
    return {"trial": trial.name, "n_books": len(books), "books_ok": sum(1 for b in books if b.get("ok")),
            "n_port_calls": len(ports), "port_calls_ok": sum(1 for d in ports if d.get("ok")),
            "n_sim_prompts_checked": len(list(ag.glob("episode-*/user_sim_prompt.json"))), "leaks": leaks,
            "n_overlays": len(ov), "problems": problems, "notes": notes}


def main(argv: list[str]) -> int:
    bad = 0
    out = []
    for rd in find_run_dirs(argv):
        for t in sorted((rd / "trials").iterdir()):
            if t.is_dir() and "__" in t.name and (t / "agent" / "turns.jsonl").exists():
                r = check_trial(t)
                r["run"] = str(rd)
                out.append(r)
                bad += bool(r["problems"])
    print(json.dumps(out, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
