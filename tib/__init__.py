"""Thinking-in-Bets (TiB): Training-free claim checklists at test time.

Implements the Claim Checklist data structures, prompt templates, and three
recording schedules from:

    Nghi D. Q. Bui (2026). "Thinking-in-Bets at Test Time Reduces Sycophantic
    Revisions in LLM Agents." arXiv:2610.34654.
"""
from __future__ import annotations

from .checklist import Bet, ClaimChecklist, RejectedOption
from .prompts import (
    CLAIM_CHECKLIST_REQUEST_PROMPT,
    CLAIM_CHECKLIST_SYSTEM_PROMPT,
    CONFIDENCE_GATE_SYSTEM_PROMPT,
    DRAFT_FIRST_SYSTEM_PROMPT,
    REVISION_RULE_NOTE,
    REVISION_RULE_NOTE_NO_P,
)
from .schedules import AlwaysOnSession, OnDemandSession, draft_first_solve

__all__ = [
    "AlwaysOnSession",
    "Bet",
    "CLAIM_CHECKLIST_REQUEST_PROMPT",
    "CLAIM_CHECKLIST_SYSTEM_PROMPT",
    "CONFIDENCE_GATE_SYSTEM_PROMPT",
    "ClaimChecklist",
    "DRAFT_FIRST_SYSTEM_PROMPT",
    "OnDemandSession",
    "REVISION_RULE_NOTE",
    "REVISION_RULE_NOTE_NO_P",
    "RejectedOption",
    "draft_first_solve",
]
