"""Claim Checklist data structures and JSON parser/renderer (Equation 1)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Bet:
    """One checkable claim c_i with recorded confidence p_i in [0, 1] and evidence e_i."""

    id: str
    claim: str
    p: float | None = None
    evidence: str = "none (guess)"

    def __post_init__(self) -> None:
        if self.p is not None and not (0.0 <= float(self.p) <= 1.0):
            raise ValueError(f"Bet confidence p must be in [0, 1], got {self.p!r}")


@dataclass(frozen=True)
class RejectedOption:
    """An option ruled out earlier in the session so the agent does not cycle back to it."""

    option: str
    turn: int = 1
    reason: str = ""


@dataclass
class ClaimChecklist:
    """Pre-feedback Claim Checklist B_t = ({(c_i, p_i, e_i)}_{i=1}^K, R_t)."""

    claims: list[Bet] = field(default_factory=list)
    rejected: list[RejectedOption] = field(default_factory=list)

    def render(self, include_p: bool = True) -> str:
        """Render the checklist in the exact private-note format injected alongside user replies."""
        lines = ["claims:"]
        if not self.claims:
            lines.append("  (none)")
        for b in self.claims:
            if include_p and b.p is not None:
                lines.append(f"  {b.id}  (p={b.p:.2f})  {b.claim} — evidence: {b.evidence}")
            else:
                lines.append(f"  {b.id}  {b.claim} — evidence: {b.evidence}")
        if self.rejected:
            lines.append("rejected:")
            for r in self.rejected:
                lines.append(f"  - {r.option} (turn {r.turn}: {r.reason})")
        return "\n".join(lines)

    def to_dict(self, include_p: bool = True) -> dict[str, Any]:
        claims_out = []
        for b in self.claims:
            item: dict[str, Any] = {"id": b.id, "claim": b.claim, "evidence": b.evidence}
            if include_p and b.p is not None:
                item["p"] = round(float(b.p), 4)
            claims_out.append(item)
        return {
            "claims": claims_out,
            "rejected": [{"option": r.option, "turn": r.turn, "reason": r.reason} for r in self.rejected],
        }

    def to_json(self, include_p: bool = True) -> str:
        return json.dumps(self.to_dict(include_p=include_p), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str, require_p: bool = True) -> "ClaimChecklist":
        """Parse a model JSON response (handling optional markdown code fences)."""
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        if fence:
            text = fence.group(1)
        else:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                text = text[start : end + 1]
        data = json.loads(text)
        claims: list[Bet] = []
        for idx, item in enumerate(data.get("claims") or [], start=1):
            cid = str(item.get("id") or f"C{idx}")
            claim_txt = str(item.get("claim") or "").strip()
            if not claim_txt:
                continue
            p_raw = item.get("p")
            if p_raw is None and require_p:
                raise ValueError(f"Missing probability `p` on claim {cid}")
            p_val: float | None = None
            if p_raw is not None:
                p_f = float(p_raw)
                if p_f > 1.0 and p_f <= 100.0:
                    p_f /= 100.0
                p_val = max(0.0, min(1.0, p_f))
            ev = str(item.get("evidence") or "none (guess)").strip()
            claims.append(Bet(id=cid, claim=claim_txt, p=p_val, evidence=ev))
        rejected: list[RejectedOption] = []
        for r in data.get("rejected") or []:
            opt = str(r.get("option") or "").strip()
            if opt:
                rejected.append(
                    RejectedOption(
                        option=opt,
                        turn=int(r.get("turn") or 1),
                        reason=str(r.get("reason") or "").strip(),
                    )
                )
        return cls(claims=claims, rejected=rejected)
