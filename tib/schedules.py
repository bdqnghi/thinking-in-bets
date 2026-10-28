"""Execution schedules for Thinking-in-Bets (Algorithm 1, Section 3.2):

- ``AlwaysOnSession``: Writes/updates a private Claim Checklist before reading every user reply.
- ``OnDemandSession``: Runs a lightweight confidence gate (tau_gate = 0.9) before reading a reply,
  writing the full Claim Checklist only when uncertain (p_gate < tau_gate).
- ``draft_first_solve``: Single-pass Draft-First Betting for non-interactive tasks without user feedback.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from .checklist import ClaimChecklist
from .prompts import (
    CLAIM_CHECKLIST_REQUEST_PROMPT,
    CLAIM_CHECKLIST_SYSTEM_PROMPT,
    CONFIDENCE_GATE_SYSTEM_PROMPT,
    DRAFT_FIRST_SYSTEM_PROMPT,
    REVISION_RULE_NOTE,
)

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def default_client(model: str, api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client (auto-selecting Gemini's OpenAI endpoint for gemini-* models)."""
    resolved_base = base_url or os.environ.get("OPENAI_BASE_URL")
    if not resolved_base and model.startswith(("gemini-", "gemma-")):
        resolved_base = GEMINI_OPENAI_BASE
    resolved_key = api_key or (
        os.environ.get("GEMINI_API_KEY") if model.startswith(("gemini-", "gemma-")) else None
    ) or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not resolved_key:
        raise RuntimeError("Set GEMINI_API_KEY or OPENAI_API_KEY, or pass `client=` / `api_key=`.")
    return OpenAI(api_key=resolved_key, base_url=resolved_base)


class AlwaysOnSession:
    """Interactive multi-turn session using Always-On Betting (tau = 0.5)."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        system_prompt: str = "You are a helpful, rigorous software engineering and reasoning assistant.",
        tau: float = 0.5,
        client: OpenAI | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.tau = tau
        self.temperature = temperature
        self.client = client or default_client(model, api_key=api_key, base_url=base_url)
        self.messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self.public_history: list[tuple[str, str]] = []
        self.checklist: ClaimChecklist = ClaimChecklist()
        self.checklist_history: list[ClaimChecklist] = []

    def _transcript_text(self) -> str:
        return "\n\n".join(f"[{role.upper()}]: {text}" for role, text in self.public_history)

    def _record_checklist(self, artifact: str = "") -> ClaimChecklist:
        last_assistant = next((t for r, t in reversed(self.public_history) if r == "assistant"), "")
        req = CLAIM_CHECKLIST_REQUEST_PROMPT.format(
            transcript=self._transcript_text(),
            artifact=artifact or last_assistant,
            previous_book=self.checklist.to_json() if self.checklist.claims else "(empty)",
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CLAIM_CHECKLIST_SYSTEM_PROMPT},
                {"role": "user", "content": req},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        self.checklist = ClaimChecklist.from_json(content, require_p=True)
        self.checklist_history.append(self.checklist)
        return self.checklist

    def start(self, task: str, *, artifact: str = "") -> str:
        """Run Turn 1 on the user's initial task and seal the pre-feedback Claim Checklist."""
        self.messages.append({"role": "user", "content": task})
        self.public_history.append(("user", task))
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=list(self.messages),
            temperature=self.temperature,
        )
        reply = resp.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        self.public_history.append(("assistant", reply))
        self._prepare_for_next_feedback(artifact=artifact)
        return reply

    def _prepare_for_next_feedback(self, artifact: str = "") -> None:
        self._record_checklist(artifact=artifact)

    def _compose_feedback_message(self, user_feedback: str) -> str:
        note = REVISION_RULE_NOTE.format(claim_book=self.checklist.render(include_p=True), tau=self.tau)
        return f"{user_feedback}\n\n---\n{note}"

    def respond(self, user_feedback: str, *, artifact: str = "") -> str:
        """Adjudicate user feedback against the sealed Claim Checklist, respond, and update the checklist."""
        augmented = self._compose_feedback_message(user_feedback)
        self.messages.append({"role": "user", "content": augmented})
        self.public_history.append(("user", user_feedback))
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=list(self.messages),
            temperature=self.temperature,
        )
        reply = resp.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        self.public_history.append(("assistant", reply))
        self._prepare_for_next_feedback(artifact=artifact)
        return reply


class OnDemandSession(AlwaysOnSession):
    """Interactive multi-turn session using On-Demand Betting (tau_gate = 0.9, tau = 0.5)."""

    def __init__(self, *args: Any, tau_gate: float = 0.9, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tau_gate = tau_gate
        self.last_gate_p: float | None = None
        self.gated_off: bool = False

    def _check_confidence_gate(self, artifact: str = "") -> float:
        last_assistant = next((t for r, t in reversed(self.public_history) if r == "assistant"), "")
        prompt = f"Session so far:\n{self._transcript_text()}\n\nCurrent work:\n{artifact or last_assistant}"
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CONFIDENCE_GATE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        m = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(m.group(0) if m else raw)
        p_val = float(data["p"])
        if p_val > 1.0 and p_val <= 100.0:
            p_val /= 100.0
        return max(0.0, min(1.0, p_val))

    def _prepare_for_next_feedback(self, artifact: str = "") -> None:
        p_gate = self._check_confidence_gate(artifact=artifact)
        self.last_gate_p = p_gate
        if p_gate >= self.tau_gate:
            self.gated_off = True
        else:
            self.gated_off = False
            self._record_checklist(artifact=artifact)

    def _compose_feedback_message(self, user_feedback: str) -> str:
        if self.gated_off or not self.checklist.claims:
            return user_feedback
        return super()._compose_feedback_message(user_feedback)


def draft_first_solve(
    question: str,
    *,
    model: str = "gemini-2.5-flash",
    system_prompt: str | None = None,
    client: OpenAI | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run Draft-First Betting on a single-pass task without user feedback."""
    cli = client or default_client(model, api_key=api_key, base_url=base_url)
    sys_msg = f"{system_prompt}\n\n{DRAFT_FIRST_SYSTEM_PROMPT}" if system_prompt else DRAFT_FIRST_SYSTEM_PROMPT
    resp = cli.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
    )
    text = resp.choices[0].message.content or ""
    probs = [float(x) for x in re.findall(r"\bp\s*=\s*(0\.\d+|1\.0+)", text)]
    conf_match = re.findall(r"Confidence\s*:\s*(\d{1,3})\s*%?", text, flags=re.I)
    reported_conf = int(conf_match[-1]) / 100.0 if conf_match else None
    return {
        "response": text,
        "claim_probabilities": probs,
        "min_claim_p": min(probs) if probs else None,
        "confidence": reported_conf,
    }
