"""Invariant checks on SWE-Interact arm / overlay trials (mirror of fsa/redo/swe_together/check_arms.py).

    python -m fsa.redo.swe_interact.check_arms <run_dir|root> [...]

Per trial (``user-sim/asks.jsonl`` rows with a ``redo`` record):

* hooks answered - both broker calls returned ``ok`` (else the plain reply went through; recorded);
* timing - ``commit_before`` / ``claims_no_p``: the claim book was written before the simulator
  was consulted (``claims_written_ts`` < ``sim_consult_started_ts``); ``claims_after``: the call
  started after the simulator decided (``claims_call_started_ts`` >= ``sim_decided_ts``);
* isolation - no simulator input (``user-sim/sim_inputs/ask-N.json``: system prompt, its own
  history, the automatic repository snapshot) contains the arm block's marker lines or any claim
  text of the trial's claim books; the simulator's history entry (``history_reply``) never carries
  the block. (The agent's questions are the agent's words: if the agent quotes its own book in a
  question, that is counted as an echo, not a harness leak.)
* delivery - claim-book arms with a valid book, recap and minimal_edit deliver reply + block;
  stock (overlay trials) delivers the reply unchanged;
* overlay - at most two, the first unjustified, each at a consultation whose in-trial judge said
  every revealed intent was satisfied; the simulator's next input holds the overlay message as
  its own reply and not the reply it replaced;
* port arms (tracker / mediator, controlled ports of IDSS / Mediator-Assistant) - the call started
  after the simulator decided; a successful call's reply is delivered with the tracker state /
  replaced by the mediator's rewrite; no simulator input holds the tracker markers or a rewrite;
* MCP - no ``ask_user`` call ended on opencode's MCP timeout (the agent would miss the reply).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .metrics import find_run_dirs, trials_of

MARKERS = ("Private note from your own bookkeeping", "Your claim book:", "How to handle the user's message above",
           "(Reminder: everything the user has asked for so far", "Make the smallest change that addresses",
           "CURRENT TASK STATE (IDSS)", "=== FACT LAYER (verified entities) ===", "=== STATE LAYER (task understanding) ===",
           "Task state maintained by your harness")
PORT_ARMS = ("tracker", "mediator")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def check_trial(trial: Path) -> dict:
    us = trial / "user-sim"
    asks = _jsonl(us / "asks.jsonl")
    problems, notes = [], []
    books = [json.loads(p.read_text()) for p in sorted((us / "claims").glob("ask-*.json"))] if (us / "claims").exists() else []
    claim_texts = {c.get("claim", "") for b in books for c in ((b.get("book") or {}).get("claims") or [])}
    claim_texts = {t for t in claim_texts if len(t) >= 25}
    ports = [json.loads(p.read_text()) for p in sorted((us / "ports").glob("ask-*.json"))] if (us / "ports").exists() else []
    claim_texts |= {d["rewrite"] for d in ports if d.get("arm") == "mediator" and len(d.get("rewrite") or "") >= 25}
    n_ov = 0
    ov_rows = []
    for a in asks:
        red = a.get("redo")
        if not red:
            continue
        n = a["call_index"]
        te, sr = red.get("turn_end") or {}, red.get("sim_reply") or {}
        arm = red.get("arm")
        for name, r in (("turn_end", te), ("sim_reply", sr)):
            if not r.get("ok"):
                problems.append(f"ask {n}: broker {name} failed: {r.get('error')}")
        if arm in ("commit_before", "claims_no_p") and te.get("claims_written_ts"):
            gap = red["sim_consult_started_ts"] - te["claims_written_ts"]
            notes.append(f"ask {n}: book written {gap:.3f}s before the simulator was consulted")
            if gap <= 0:
                problems.append(f"ask {n}: claim book not written before the simulator was consulted")
        if arm == "claims_after" and sr.get("claims_call_started_ts"):
            gap = sr["claims_call_started_ts"] - red["sim_decided_ts"]
            notes.append(f"ask {n}: book call started {gap:.3f}s after the simulator decided")
            if gap < 0:
                problems.append(f"ask {n}: claims_after call started before the simulator decided")
        if arm in PORT_ARMS:
            if not sr.get("port_doc"):
                problems.append(f"ask {n}: {arm} made no call")
            else:
                gap = sr["port_call_started_ts"] - red["sim_decided_ts"]
                notes.append(f"ask {n}: {arm} call started {gap:.3f}s after the simulator decided")
                if gap < 0:
                    problems.append(f"ask {n}: {arm} call started before the simulator decided")
                if sr.get("port_ok") and (a.get("delivered") or "") == (a.get("history_reply") or ""):
                    problems.append(f"ask {n}: {arm} reply delivered unchanged despite a successful call")
        hist = a.get("history_reply") or ""
        if any(m in hist for m in MARKERS):
            problems.append(f"ask {n}: arm block in the simulator's history")
        delivered = a.get("delivered") or ""
        if arm in ("recap", "minimal_edit") or (arm in ("commit_before", "claims_after", "claims_no_p")
                                                and (te.get("claims_ok") or sr.get("claims_ok"))):
            if delivered == hist:
                problems.append(f"ask {n}: reply delivered without the arm block")
        if arm == "stock" and delivered != hist:
            problems.append(f"ask {n}: stock delivered something else than the reply")
        if sr.get("opened_by") == "overlay":
            n_ov += 1
            ov_rows.append((n, sr, a))
    echoes = 0
    for p in sorted((us / "sim_inputs").glob("ask-*.json")) if (us / "sim_inputs").exists() else []:
        for m in json.loads(p.read_text()):
            txt = str(m.get("content") or "")
            hit = any(k in txt for k in MARKERS) or any(t in txt for t in claim_texts)
            if hit and m.get("role") == "user":
                echoes += 1  # the agent's own question quoting its book: agent behaviour, not a harness leak
            elif hit:
                problems.append(f"{p.name}: simulator input ({m.get('role')}) contains arm material")
    if echoes:
        notes.append(f"{echoes} agent question(s) quote claim-book material (the agent's own words)")
    if n_ov > 2:
        problems.append(f"{n_ov} overlay injections (> 2)")
    for i, (n, sr, a) in enumerate(ov_rows):
        if i == 0 and sr.get("overlay_kind") != "unjustified":
            problems.append("first overlay is not the unjustified one")
        if not sr.get("overlay_all_revealed_satisfied"):
            problems.append(f"ask {n}: overlay fired without all revealed intents satisfied")
        nxt = us / "sim_inputs" / f"ask-{n + 1}.json"
        if nxt.exists():
            msgs = json.loads(nxt.read_text())
            said = [m.get("content") for m in msgs if m.get("role") == "assistant"]
            if a.get("history_reply") not in said:
                problems.append(f"ask {n}: overlay message missing from the simulator's history")
            if a.get("reply") in said and a.get("reply") != a.get("history_reply"):
                problems.append(f"ask {n}: replaced simulator reply still in its history")
        notes.append(f"ask {n}: overlay {sr.get('overlay_kind')} -> {a.get('history_reply')!r} "
                     f"(replaced: {(a.get('reply') or '')[:120]!r})")
    mcp = 0
    for f in (trial / "steps").glob("*/agent/opencode.attempt-*.txt"):
        for line in f.open(errors="replace"):
            if "ask_user" in line and "Request timed out" in line:
                mcp += 1
    if mcp:
        problems.append(f"{mcp} ask_user call(s) ended on the MCP timeout")
    return {"trial": trial.name, "job": trial.parent.name, "n_asks": len(asks), "n_books": len(books),
            "n_port_calls": len(ports), "port_calls_ok": sum(1 for d in ports if d.get("ok")),
            "books_ok": sum(1 for b in books if b.get("ok")),
            "n_sim_inputs_checked": len(list((us / "sim_inputs").glob("ask-*.json"))) if (us / "sim_inputs").exists() else 0,
            "n_overlays": n_ov, "mcp_timeouts": mcp, "problems": problems, "notes": notes}


def main(argv: list[str]) -> int:
    out, bad = [], 0
    for rd in find_run_dirs(argv):
        for t in trials_of(rd):
            r = check_trial(t)
            r["run"] = str(rd)
            out.append(r)
            bad += bool(r["problems"])
    print(json.dumps(out, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
