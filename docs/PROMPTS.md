# Claim Checklist Schema & Drop-In Prompt Templates

This document provides the structured **Claim Checklist** format, the **2×2 Selective Revision Rule**, and the exact Appendix A prompt templates from [`tib/prompts.py`](../tib/prompts.py) so you can drop **Thinking-in-Bets** into any custom agent harness (LangGraph, OpenCode, Mini-SWE-Agent, Claude Code hooks, or bespoke tool loops).

---

## 1. Private Claim Checklist Schema

At turn $t$, after producing an output $y_t$ and *before* reading user reply $u_{t+1}$, the agent records a private **Claim Checklist**:

$$\mathcal{B}_t = \bigl(\{(c_i, p_i, e_i)\}_{i=1}^{K},\; \mathcal{R}_t\bigr), \qquad p_i \in [0, 1]$$

```json
{
  "claims": [
    {
      "id": "C1",
      "claim": "Empty input returns [] without mutating the caller's list",
      "p": 0.95,
      "evidence": "Verified by pytest tests/test_parser.py::test_empty_input (passed)"
    },
    {
      "id": "C2",
      "claim": "Legacy ISO-8601 timestamps without timezone offsets default to UTC",
      "p": 0.40,
      "evidence": "none (guess — user prompt did not specify naive timestamp behavior)"
    }
  ],
  "rejected": [
    {
      "option": "Raise ValueError on empty input",
      "turn": 1,
      "reason": "Contradicts docstring example in parser.py:L42"
    }
  ]
}
```

---

## 2. Selective Revision Matrix ($\tau = 0.5$)

When the user's message arrives, the agent maps it to the affected claim(s) and follows a 2×2 action policy governed by pre-committed bet strength ($p_i \ge \tau$ vs. $p_i < \tau$) and feedback specificity:

| Pre-feedback Bet Strength | Vague Complaint (*"This looks wrong, redo it"*) | Specific Contradiction / Concrete Counter-Evidence |
| :--- | :--- | :--- |
| **Strong bet ($p_i \ge 0.5$)** | **Keep & Ask:** Preserve current work; ask one targeted clarifying question. | **Re-check:** Inspect/test the cited fault; revise *only* if the new evidence lowers $p_i < 0.5$. |
| **Weak bet ($p_i < 0.5$)** | **Ask (if $\ge 2$ weak claims)** or **Fix:** Clarify ambiguity instead of guessing blindly. | **Fix:** Revise the targeted claim while keeping unrelated claims untouched and avoiding rejected options $\mathcal{R}_t$. |

---

## 3. Prompt Templates

### A. Seal the Claim Checklist Before Reading Feedback (`CLAIM_CHECKLIST_SYSTEM_PROMPT`)

```text
You are the agent in the session below, working for a user. Write your claim book for the current state of your work: the claims your work rests on.

A claim is one decision you made, or one assumption you rely on, about what the user wants or how your output/code must behave (for example "the parser accepts ISO dates only", "empty input returns []", "the public API signature is unchanged"). For each claim give:
- `p`: your probability, from 0 to 1, that the claim is correct, i.e. that it is what the user wants and what your work currently satisfies;
- `evidence`: what supports it (quote the user, name a test result, calculation check, or file), or "none (guess)".

List 3 to 10 claims, most important first. Also list the rejected options: approaches the user rejected or you ruled out, with the turn and the reason. Keep the claims and rejected options of your previous claim book unless the session shows they changed, and keep their ids stable.

Return JSON only, in exactly this shape:
{"claims": [{"id": "C1", "claim": "...", "p": 0.6, "evidence": "..."}], "rejected": [{"option": "...", "turn": 2, "reason": "..."}]}
```

### B. Private Adjudication Note Prepended on User Reply (`REVISION_RULE_NOTE`)

```text
(Private note from your own bookkeeping, not from the user. Do not mention it to the user.)
Your claim book:
{claim_book}
How to handle the user's message above:
1. Work out which of your claims the message bears on, and update only those claims.
2. Change the work only for claims whose p is now below 0.5; keep everything else exactly as it is.
3. Never return to a rejected option unless the message gives new evidence for it.
4. If the message fits two or more weak claims (p below 0.5), ask the user one targeted question instead of changing work.
5. If the message names no specific fault and all your claims are strong (p at least 0.5), ask the user what is wrong instead of rewriting.
```

### C. Lightweight On-Demand Confidence Gate (`CONFIDENCE_GATE_SYSTEM_PROMPT`, $\tau_{\text{gate}} = 0.9$)

```text
You are the agent in the session below, working for a user. Give a single private confidence number about your current work.

Rate `p`, your probability from 0 to 1, that your current work is correct and complete with respect to what the user wants. Do not list evidence and do not enumerate claims; this is a quick overall judgement.

Return JSON only, in exactly this shape:
{"p": 0.7, "why": "<= 20 words"}
```

### D. Single-Pass Draft-First Betting (`DRAFT_FIRST_SYSTEM_PROMPT`)

```text
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
  - Report your calibrated final confidence from the joint claims (capped at 80% if any claim has `p < 0.85`), followed by your final answer.
```
