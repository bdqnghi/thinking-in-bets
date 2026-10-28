"""Prompt templates from Appendix A of Thinking-in-Bets (arXiv:2610.34654)."""
from __future__ import annotations

CLAIM_CHECKLIST_SYSTEM_PROMPT = """\
You are the agent in the session below, working for a user. Write your claim book for the current state of your work: the claims your work rests on.

A claim is one decision you made, or one assumption you rely on, about what the user wants or how your output/code must behave (for example "the parser accepts ISO dates only", "empty input returns []", "the public API signature is unchanged"). For each claim give:
- `p`: your probability, from 0 to 1, that the claim is correct, i.e. that it is what the user wants and what your work currently satisfies;
- `evidence`: what supports it (quote the user, name a test result, calculation check, or file), or "none (guess)".

List 3 to 10 claims, most important first. Also list the rejected options: approaches the user rejected or you ruled out, with the turn and the reason. Keep the claims and rejected options of your previous claim book unless the session shows they changed, and keep their ids stable.

Return JSON only, in exactly this shape:
{"claims": [{"id": "C1", "claim": "...", "p": 0.6, "evidence": "..."}], "rejected": [{"option": "...", "turn": 2, "reason": "..."}]}"""

CLAIM_CHECKLIST_REQUEST_PROMPT = """\
Session so far:
{transcript}

Current artifact / state:
{artifact}

Previous claim book:
{previous_book}

Write the updated claim book as JSON only."""

REVISION_RULE_NOTE = """\
(Private note from your own bookkeeping, not from the user. Do not mention it to the user.)
Your claim book:
{claim_book}
How to handle the user's message above:
1. Work out which of your claims the message bears on, and update only those claims.
2. Change the work only for claims whose p is now below {tau}; keep everything else exactly as it is.
3. Never return to a rejected option unless the message gives new evidence for it.
4. If the message fits two or more weak claims (p below {tau}), ask the user one targeted question instead of changing work.
5. If the message names no specific fault and all your claims are strong (p at least {tau}), ask the user what is wrong instead of rewriting."""

REVISION_RULE_NOTE_NO_P = """\
(Private note from your own bookkeeping, not from the user. Do not mention it to the user.)
Your claim book:
{claim_book}
How to handle the user's message above:
1. Work out which of your claims the message bears on, and update only those claims.
2. Change the work only for claims the message contradicts; keep everything else exactly as it is.
3. Never return to a rejected option unless the message gives new evidence for it.
4. If the message could refer to two or more of your claims, ask the user one targeted question instead of changing work.
5. If the message names no specific fault, ask the user what is wrong instead of rewriting."""

CONFIDENCE_GATE_SYSTEM_PROMPT = """\
You are the agent in the session below, working for a user. Give a single private confidence number about your current work.

Rate `p`, your probability from 0 to 1, that your current work is correct and complete with respect to what the user wants. Do not list evidence and do not enumerate claims; this is a quick overall judgement.

Return JSON only, in exactly this shape:
{"p": 0.7, "why": "<= 20 words"}"""

DRAFT_FIRST_SYSTEM_PROMPT = """\
Follow this 3-phase Thinking-in-Bets (Draft-First Betting) protocol in a single response:

Phase 1 — Full Step-by-Step Draft:
Work through the problem thoroughly step by step and produce a complete candidate draft.

Phase 2 — Pre-Commit Claim Checklist (`[CLAIM_BOOK]`):
Write a `[CLAIM_BOOK]` block with 2 to 5 critical claims (`C1`, `C2`, ...) that must ALL hold for your draft to be right. Format each line as:
  `C1: <step/assumption> | evidence: <verification or UNVERIFIED_RECALL / HEURISTIC_GUESS> | p=0.XX`
Use evidence-anchored probability bets `p` in [0.05, 0.99]:
  - `p` in [0.85, 0.98]: independently verified by plug-back, boundary case, tool/test output, or strict elimination.
  - `p` in [0.55, 0.80]: standard step or recalled fact without an independent check.
  - `p` in [0.20, 0.50]: involves an unverified calculation, heuristic approximation, or guess.

Phase 3 — Targeted Audit of Weakest Link & Final Submission:
  - Re-check your lowest-`p` claim (`min_i p_i`). Revise your Phase-1 draft ONLY if that check uncovers a concrete contradiction or error.
  - Report your calibrated final confidence from the joint claims (capped at 80% if any claim has `p < 0.85`), followed by your final answer."""
