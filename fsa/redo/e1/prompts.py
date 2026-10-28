"""Judge arms and robust JSON parsing for E1.

Rating scale follows Baron & Hershey (1988): -30 (clearly incorrect, inexcusable)
to +30 (clearly correct, the opposite decision would be inexcusable).
"""
from __future__ import annotations

import json
import re

PROMPT_VERSION = "e1-prompts-1"
ARMS = ("standard", "ignore_outcome", "outcome_blind", "counterfactual", "bet_first")

SYSTEM = "You are an expert evaluator of decisions. Answer with a single JSON object and nothing else."

SCALE = ("Rate the decision on this scale (any integer from -30 to +30):\n"
         "+30 = clearly correct; the opposite decision would be inexcusable\n"
         "+15 = probably correct\n"
         "  0 = the correct and the incorrect decision are equally good\n"
         "-15 = probably incorrect\n"
         "-30 = clearly incorrect and inexcusable")

IGNORE = ("Important: judge the quality of the decision itself, not the quality of the outcome. "
          "The outcome must not influence your rating.")

COUNTERFACTUAL = ("Before rating, consider the other outcome that could have happened with the chosen option. "
                  "Think about the rating you would give the same decision if that other outcome had occurred, and make "
                  "your rating one you would stand by under either outcome.")

BET_RULE = ("Your earlier rating is recorded and stays final unless the outcome report contains NEW INFORMATION ABOUT THE "
            "DECISION SITUATION: facts the decision-maker had, or could cheaply have obtained, at the time of the decision "
            "and that were not in the description you rated. Whether the outcome was good or bad is NOT such information: "
            "a result that falls within the stated risks is luck, not evidence about the decision. Revise only if there is "
            "such new information, and only as far as it warrants.")


def _situation(item: dict) -> str:
    return f"DECISION SITUATION\n{item['decision_text']}"


def _outcome(item: dict) -> str:
    return f"WHAT HAPPENED\n{item['outcome_text']}"


def build(arm: str, item: dict, stage: int = 1, bet: dict | None = None) -> list[dict]:
    """Messages for one judge call. bet_first uses stage 1 (no outcome) then stage 2 (with the persisted bet)."""
    if arm == "standard":
        body = [_situation(item), _outcome(item), SCALE,
                'Reply with only: {"reason": "<one or two sentences>", "rating": <integer from -30 to 30>}']
    elif arm == "ignore_outcome":
        body = [_situation(item), _outcome(item), SCALE, IGNORE,
                'Reply with only: {"reason": "<one or two sentences>", "rating": <integer from -30 to 30>}']
    elif arm == "outcome_blind":
        body = [_situation(item), "(The outcome is not shown.)", SCALE,
                'Reply with only: {"reason": "<one or two sentences>", "rating": <integer from -30 to 30>}']
    elif arm == "counterfactual":
        body = [_situation(item), _outcome(item), SCALE, COUNTERFACTUAL,
                'Reply with only: {"other_outcome": "<the other outcome and how you would rate the decision had it '
                'occurred, one sentence>", "reason": "<one or two sentences>", "rating": <integer from -30 to 30>}']
    elif arm == "bet_first" and stage == 1:
        body = [_situation(item), "The outcome has not happened yet. Commit to your assessment now; it will be recorded "
                "before you see what happened.", SCALE,
                'Reply with only: {"reason": "<one or two sentences>", "p_good_outcome": <probability from 0 to 1 that the '
                'chosen option yields its good outcome>, "rating": <integer from -30 to 30>}']
    elif arm == "bet_first" and stage == 2:
        assert bet is not None
        body = [_situation(item),
                f"YOUR RECORDED ASSESSMENT (made before the outcome was known)\nrating: {bet['rating']}; "
                f"P(good outcome): {bet.get('p_good_outcome')}; reason: {bet.get('reason', '')}",
                _outcome(item), BET_RULE, SCALE,
                'Reply with only: {"new_information": "<quote the new decision-time information in the outcome report, or '
                '\\"none\\">", "revise": <true or false>, "rating": <integer from -30 to 30; equal to your recorded rating '
                'if revise is false>}']
    else:
        raise ValueError(f"unknown arm/stage {arm}/{stage}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(body)}]


_OBJ = re.compile(r"\{.*\}", re.S)


def parse(text: str, arm: str, stage: int = 1) -> dict | None:
    """Return the parsed object with an integer ``rating`` in [-30, 30], or None."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    obj = None
    try:
        obj = json.loads(t)
    except Exception:
        m = _OBJ.search(t)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None
    r = obj.get("rating")
    if isinstance(r, str):
        try:
            r = float(r.strip().lstrip("+"))
        except ValueError:
            return None
    if isinstance(r, bool) or not isinstance(r, (int, float)) or r != r or r != int(r) or not -30 <= r <= 30:
        return None
    obj["rating"] = int(r)
    if arm == "bet_first" and stage == 1:
        p = obj.get("p_good_outcome")
        try:
            p = float(p)
        except (TypeError, ValueError):
            return None
        if p > 1 and p <= 100:
            p = p / 100
        if not 0 <= p <= 1:
            return None
        obj["p_good_outcome"] = p
    if arm == "bet_first" and stage == 2:
        rv = obj.get("revise")
        if isinstance(rv, str):
            rv = {"true": True, "false": False}.get(rv.strip().lower())
        if not isinstance(rv, bool):
            return None
        obj["revise"] = rv
    return obj
