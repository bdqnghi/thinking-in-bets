"""E2 prompts (templates in ``fsa/redo/e2/prompts/``) and robust parsing.

Conditions (one call each per item; the context block is byte-identical across them):

* ``real``        (a) the real user reply to turn t, then rate every requirement;
* ``unjustified`` (b) an unjustified reply ("this is wrong, redo it"), same wrapper as (a);
* ``none``        (c) no reply ("The user has not replied yet."), then rate;
* ``bet``         (d, stage 1) no reply; the rating is a commitment recorded before any reply;
* ``bet_real`` / ``bet_unjustified`` (d, stage 2) the recorded bet plus the real / unjustified
  reply and the revision rule: revise a requirement only for new information about it.

Every rating is per requirement: ``satisfied`` (bool) and ``p_satisfied`` (integer 0-100; the
primary scale). Requirements are shown as R1..Rn; the judge's labels are never shown.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PROMPT_VERSION = "e2-prompts-1"
DIR = Path(__file__).with_name("prompts")
FILES = ("system.md", "context.md", "reply.md", "no_reply.md", "rate.md", "bet_stage1.md", "bet_stage2.md")
STAGE1 = ("real", "unjustified", "none", "bet")
STAGE2 = ("bet_real", "bet_unjustified")
CONDITIONS = STAGE1 + STAGE2


def _t(name: str) -> str:
    return (DIR / name).read_text()


def prompts_sha256() -> str:
    h = hashlib.sha256()
    for f in FILES:
        h.update(f.encode() + b"\0" + (DIR / f).read_bytes() + b"\0")
    return h.hexdigest()


def requirements_block(item: dict) -> str:
    return "\n".join(f"{i['rid']}: {i['text']}" for i in item["intents"])


def context(item: dict) -> str:
    return (_t("context.md").replace("{requirements}", requirements_block(item))
            .replace("{agent_message}", item.get("agent_message") or "(no message)")
            .replace("{diff}", item["diff"].rstrip("\n"))).rstrip()


def reply_for(cond: str, item: dict) -> str:
    return item["real_reply"] if cond in ("real", "bet_real") else item["unjustified_reply"]


def render_bet(item: dict, bet: dict) -> str:
    lines = []
    for i in item["intents"]:
        b = bet[i["rid"]]
        lines.append(f"{i['rid']}: satisfied={str(b['satisfied']).lower()}; p_satisfied={b['p_satisfied']}; "
                     f"reason: {b.get('reason', '')}")
    return "\n".join(lines)


def build(cond: str, item: dict, bet: dict | None = None) -> list[dict]:
    ctx = context(item)
    if cond in ("real", "unjustified"):
        body = [ctx, _t("reply.md").replace("{reply}", reply_for(cond, item)).rstrip(), _t("rate.md").rstrip()]
    elif cond == "none":
        body = [ctx, _t("no_reply.md").rstrip(), _t("rate.md").rstrip()]
    elif cond == "bet":
        body = [ctx, _t("bet_stage1.md").rstrip()]
    elif cond in STAGE2:
        assert bet is not None
        body = [ctx, _t("bet_stage2.md").replace("{bet}", render_bet(item, bet))
                .replace("{reply}", reply_for(cond, item)).rstrip()]
    else:
        raise ValueError(f"unknown condition {cond}")
    return [{"role": "system", "content": _t("system.md").strip()}, {"role": "user", "content": "\n\n".join(body)}]


_OBJ = re.compile(r"\{.*\}", re.S)


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return {"true": True, "false": False, "yes": True, "no": False}.get(v.strip().lower())
    return None


def _p(v):
    if isinstance(v, str):
        v = v.strip().rstrip("%")
        try:
            v = float(v)
        except ValueError:
            return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        return None
    if 0 < v < 1 and v != int(v):      # a probability given on the 0-1 scale
        v = v * 100
    if not 0 <= v <= 100:
        return None
    return int(round(v))


def load_obj(text: str):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t)
    except Exception:
        m = _OBJ.search(t)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def parse(text: str, cond: str, rids: list[str]) -> dict | None:
    """``{rid: {"satisfied", "p_satisfied", ...}}`` covering exactly ``rids``, or None."""
    obj = load_obj(text)
    if isinstance(obj, dict):
        obj = obj.get("requirements")
    if not isinstance(obj, list):
        return None
    out = {}
    for e in obj:
        if not isinstance(e, dict):
            return None
        rid = str(e.get("id", "")).strip()
        if rid not in rids or rid in out:
            return None
        sat, p = _bool(e.get("satisfied")), _p(e.get("p_satisfied"))
        if sat is None or p is None:
            return None
        rec = dict(satisfied=sat, p_satisfied=p, reason=e.get("reason"),
                   inconsistent=(sat and p < 50) or (not sat and p > 50))
        if cond in STAGE2:
            rv = _bool(e.get("revise"))
            if rv is None:
                return None
            rec.update(revise=rv, new_information=e.get("new_information"))
        out[rid] = rec
    if set(out) != set(rids):
        return None
    return out
