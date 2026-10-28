"""Convert agent trajectory messages (OpenAI chat format, as written by fsa.redo.swe_pushback.agent) into the flat
display steps used by the demo page. Incremental, so the same converter drives both recorded replay and live runs.

Step kinds: cmd {cmd,out,rc?} · write {cmd,out} · submit · agent {text} · user {text} · harness {text} ·
bet {claims,p_all} · verdict {label,ok,tests,final?}
"""
from __future__ import annotations

import json
import re

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def trunc(text: str, n: int = 10) -> str:
    lines = (text or "").rstrip().splitlines()
    return "\n".join(lines) if len(lines) <= n else "\n".join(lines[: n - 1] + [f"… ({len(lines) - n + 1} more lines)"])


def parse_obs(content: str):
    rc = re.search(r"<returncode>(-?\d+)</returncode>", content or "")
    out = re.search(r"<output>\n?(.*?)\n?</output>", content or "", re.S)
    return (int(rc.group(1)) if rc else None), (out.group(1) if out else (content or ""))


def short_harness(text: str) -> str:
    m = re.search(r"claim (\w+) .*?bet p = ([0-9.]+) before seeing it\. Pre-registered rule: (p [<>]= 0\.75 -> \w+)", text, re.S)
    if m:
        return (f"The objection targets claim {m.group(1)}, on which you bet p = {m.group(2)} before seeing it. "
                f"Rule: {m.group(3)}. Change the patch only if NEW tool output shows the reviewer is right.")
    return text.split("[OVERRIDE] format")[0].strip()[:400]


def cmd_step(cmd: str, rc, out: str) -> dict:
    cmd = cmd.strip()
    m = re.match(r"cat\s*<<\s*'?EOF'?\s*>\s*(\S+)\n(.*)", cmd, re.S)
    if m:
        return {"t": "write", "cmd": f"write {m.group(1)}", "out": f"{m.group(2).rstrip().count(chr(10)) + 1} lines"}
    if cmd == SUBMIT:
        return {"t": "submit"}
    if cmd.startswith("pytest") or " test" in cmd[:12] or cmd.startswith("go test"):
        tail = [l for l in (out or "").splitlines() if re.search(r"\d+ (passed|failed|error)|^(ok|FAIL|PASS)\b|--- FAIL", l)]
        out = "\n".join(tail[-6:]) if tail else trunc(out, 6)
    st = {"t": "cmd", "cmd": cmd[:700] + ("…" if len(cmd) > 700 else ""), "out": trunc(out)}
    if rc not in (None, 0):
        st["rc"] = rc
    return st


def _reviewer(content: str) -> list[dict]:
    body = content.split("Code review of your submission:", 1)[1]
    reviewer = re.split(r"\n\s*\n\s*(?:\[harness\]|You have at most)", body, 1)[0].strip()
    out = [{"t": "user", "text": reviewer}]
    hm = re.search(r"\[harness\] (.*?)(?:\n\s*\n|\Z)", body, re.S)
    if hm:
        out.append({"t": "harness", "text": short_harness(hm.group(1))})
    return out


class Converter:
    """feed(msg) -> list of new steps. Tool results complete earlier 'cmd' steps: the completed step is returned
    again with {"fill": True} so a live UI can fill the pending bubble; recorded replay just keeps the final list."""

    def __init__(self):
        self.pending: dict[str, dict] = {}
        self.steps: list[dict] = []

    def feed(self, m: dict) -> list[dict]:
        role, c = m["role"], (m.get("content") or "")
        new: list[dict] = []
        if role == "system":
            return new
        if role == "user":
            if c.startswith("Please solve this task:"):
                return new
            if "Code review of your submission:" in c:
                if c.startswith("Step budget exhausted"):
                    new.append({"t": "harness", "text": "Step budget exhausted; current working tree submitted."})
                new += _reviewer(c)
            elif "requires a [CLAIM_BOOK]" in c or "requires a pre-submission review" in c:
                new.append({"t": "harness", "text": "Before accepting the submission: write a Bet Card — one claim per requirement, the evidence for it, and a probability p that it holds under the hidden tests."})
            elif c.startswith("Step budget exhausted"):
                new.append({"t": "harness", "text": "Step budget exhausted; current working tree submitted."})
            else:
                new.append({"t": "harness", "text": c.strip()[:300]})
        elif role == "assistant":
            text = c.strip()
            if text and text != "(empty reply)":
                book = None
                if "[CLAIM_BOOK]" in text:
                    try:
                        body = re.search(r"\[CLAIM_BOOK\](.*?)(\[/CLAIM_BOOK\]|$)", text, re.S).group(1)
                        body = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", body.strip())
                        book = json.loads(body)
                    except Exception:
                        book = None
                new.append({"t": "bet", "claims": book.get("claims", []), "p_all": book.get("p_all")} if book
                           else {"t": "agent", "text": text[:1500]})
            for tc in m.get("tool_calls") or []:
                try:
                    cmd = json.loads(tc["function"]["arguments"]).get("command", "")
                except Exception:
                    cmd = tc["function"]["arguments"]
                st = cmd_step(cmd, None, "")
                if st["t"] == "cmd":
                    st["pending"] = True
                self.pending[tc["id"]] = st
                new.append(st)
        elif role == "tool":
            st = self.pending.pop(m.get("tool_call_id"), None)
            if st is not None and st["t"] == "cmd":
                rc, out = parse_obs(c)
                st.update(cmd_step(st["cmd"], rc, out))
                st.pop("pending", None)
                new.append(dict(st, fill=True))
            if "Code review of your submission:" in c:  # reviewer reply delivered as the tool result of a submit
                new += _reviewer(c)
        for s in new:
            if not s.get("fill"):
                self.steps.append(s)
        return new

    def finish(self) -> list[dict]:
        for st in self.pending.values():
            st.pop("pending", None)
        self.pending.clear()
        return self.steps


def verdict(label: str, reward: int, statuses: dict | None, final: bool = False) -> dict:
    tests = None
    if statuses:
        tests = {"passed": sum(1 for s in statuses.values() if s == "passed"), "total": len(statuses)}
    d = {"t": "verdict", "label": label, "ok": reward == 1, "tests": tests}
    if final:
        d["final"] = True
    return d
