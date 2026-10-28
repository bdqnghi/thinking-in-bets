"""Controlled ports of two published inference-time methods (redo_v1, PLAN.md §8).

    tracker   Situation tracker, a CONTROLLED PORT of IDSS (Intent-Driven Situation States,
              arXiv 2608.15755, Tao et al. 2026). Not a native reproduction.
    mediator  Intent mediator, a CONTROLLED PORT of the Mediator-Assistant framework
              (arXiv 2602.07338, Liu et al. 2026). Not a native reproduction.

This file is stdlib-only and exists twice, byte-identical: ``fsa/redo/ports.py`` (tib; used by
the SWE-Interact broker) and ``src/user_agent/redo_ports.py`` in the SWE-Together clone (baseline
port patch revision); its prompts live beside it in ``prompts/ports/`` (both copies; a test compares
the hashes). The benchmark-specific glue (where the hook runs, what the transcript is made of) is in
``fsa/redo/swe_interact/{arms,broker}.py`` and the clone's ``redo_arms.py`` / ``user_enabled_opencode.py``.

Both ports act at the turn boundary the benchmark controls, like the other redo arms: after the
simulated user's reply is known (both methods consume the user's message), before it is delivered
to the agent. The simulator never sees their output. Their model calls use the rung's own Gemini
model (LiteLLM ``gemini/<model>``, ``reasoning_effort=high``, patient retry 10..300 s) and every
attempt is logged in ``port_calls.jsonl`` (their own token seat).

tracker (IDSS §3)
    Situation state S = (F, T): a fact layer F of provenance-aware entities parsed
    *deterministically* from the agent's tool calls and the repository diff (§3.2.1: ADD / UPDATE /
    DELETE / MERGE operations, newer tool observations overwrite older ones), and a state layer
    T = (intents, variables, constraints) (§3.2.2) with the paper's statuses (intents active /
    pending / completed / blocked; variables askable / retrievable / derivable, known / unknown;
    constraints satisfied / unsatisfied / violated). After each user message one call of the agent's
    own model applies the paper's STATE MAINTENANCE PROTOCOL (App. B.1) to the previous state layer,
    the fact layer, the session and the message. Cross-layer constraint propagation (§3.3) is
    deterministic: constraints with a grounding (entity, attribute, op, value) are re-evaluated
    against F; ungrounded or unevaluable ones are never turned into "violated"; a violated
    constraint blocks its intent with the factual basis recorded. The state (F rendered as in
    App. B.3, T as in App. B.2) is delivered with the user's message together with the paper's
    BEHAVIORAL GUIDELINES (App. B.1).

mediator (Mediator-Assistant §4)
    One call per user message: the paper's "Prompt For Mediator" (App. B, verbatim) over the
    conversation transcript between the user and the assistant, with the Refiner-distilled
    experiences E in [[EXPS]]; the output, a first-person, fully specified request, replaces the
    user's message for the agent (Eq. 7: the assistant acts on the rewritten instruction). E is
    distilled offline, once per rung model, by the paper's "Prompt For Refiner" (verbatim) from
    contrastive pairs of held-out SWE-Together sessions (``fsa/redo/ports_refine.py``) and frozen in
    ``prompts/ports/mediator_experiences.json``.

Deviations from the papers are listed in docs/results/REDO_BASELINE_PORTS.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

PROMPTS = Path(__file__).with_name("prompts") / "ports"
PORT_ARMS = ("tracker", "mediator")
PORT_LABEL = {"tracker": "situation tracker - controlled port of IDSS (arXiv 2608.15755)",
              "mediator": "intent mediator - controlled port of Mediator-Assistant (arXiv 2602.07338)"}

INTENT_STATUS = ("active", "pending", "completed", "blocked")
VAR_SOURCE = ("askable", "retrievable", "derivable")
CONSTRAINT_STATUS = ("satisfied", "unsatisfied", "violated")
GROUND_OPS = ("==", "!=", "contains", "not_contains", "exists", "absent")
FILE_TOOLS = {"read": "read", "edit": "edit", "write": "write", "patch": "edit", "multiedit": "edit"}

_TRANSIENT = ("429", "rate limit", "ratelimit", "resource_exhausted", "resource has been exhausted", "quota",
              "503", "unavailable", "overloaded", "500", "502", "504", "internal error", "internalservererror",
              "timeout", "timed out", "connection", "high demand", "empty response", "invalid json")
BACKOFF_SEC = (10, 20, 40, 60, 90, 120, 180, 240, 300)
FACT_RENDER_CHARS = 12_000
MAX_COMMANDS = 25          # fact-layer compression: newest command entities rendered
MAX_READ_ONLY_FILES = 30   # files only read (not edited, not in the diff) rendered


def prompt(name: str) -> str:
    return (PROMPTS / name).read_text()


def prompts_sha256() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PROMPTS.iterdir()) if p.is_file()}


def module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def litellm_model(agent_model: str) -> str:
    """opencode names the native Gemini provider ``google/``; LiteLLM calls it ``gemini/``."""
    prov, _, name = (agent_model or "").partition("/")
    return f"gemini/{name}" if prov == "google" else agent_model


def bare_model(agent_model: str) -> str:
    return (agent_model or "").split("/", 1)[-1]


# ── opencode event streams ──────────────────────────────────────────────────

def _json_lines(raw: str):
    for line in (raw or "").split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            yield json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue


def stream_items(raw: str) -> list[dict]:
    """Ordered items of an ``opencode run --format=json`` stream: ``{"kind": "text", "text"}``,
    ``{"kind": "ask", "text"}`` (an ask_user MCP call) or ``{"kind": "tool", "tool", "input",
    "status", "exit", "output"}``."""
    out = []
    for e in _json_lines(raw):
        part = e.get("part") or {}
        if e.get("type") == "text":
            t = part.get("text")
            if isinstance(t, str) and t.strip():
                out.append({"kind": "text", "text": t.strip()})
        elif e.get("type") == "tool_use":
            st = part.get("state") if isinstance(part.get("state"), dict) else {}
            name = part.get("tool") or part.get("name") or "?"
            args = st.get("input") or part.get("input") or {}
            if "ask_user" in name:
                q = args.get("question") if isinstance(args, dict) else None
                out.append({"kind": "ask", "text": str(q if q is not None else args)})
                continue
            meta = st.get("metadata") if isinstance(st.get("metadata"), dict) else {}
            res = st.get("output") or part.get("output") or ""
            out.append({"kind": "tool", "tool": name, "input": args if isinstance(args, dict) else {"raw": args},
                        "status": st.get("status"), "exit": meta.get("exit"),
                        "output": res if isinstance(res, str) else json.dumps(res)})
    return out


# ── fact layer (IDSS §3.2.1) ────────────────────────────────────────────────

def diff_files(diff_text: str) -> dict[str, dict]:
    """Per-file change of a cumulative ``git diff`` (deterministic)."""
    files: dict[str, dict] = {}
    cur = None
    for line in (diff_text or "").split("\n"):
        m = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if m:
            cur = {"change": "modified", "lines_added": 0, "lines_removed": 0}
            files[m.group(2)] = cur
            continue
        if cur is None:
            continue
        if line.startswith("new file mode"):
            cur["change"] = "added"
        elif line.startswith("deleted file mode"):
            cur["change"] = "deleted"
        elif line.startswith("rename from"):
            cur["change"] = "renamed"
        elif line.startswith("+") and not line.startswith("+++"):
            cur["lines_added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            cur["lines_removed"] += 1
    return files


def _one_line(s, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


class FactLayer:
    """Entity store F = {(type, id, attributes, source)}. Updated only through operations
    (ADD, UPDATE, DELETE, MERGE); every operation is kept in ``ops`` (provenance)."""

    def __init__(self):
        self.entities: dict[str, dict] = {}
        self.ops: list[dict] = []
        self._seq = 0

    def _touch(self, e: dict) -> None:
        self._seq += 1
        e["seq"] = self._seq

    def apply(self, op: str, key: str, attrs: dict | None = None, source: str | None = None, into: str | None = None):
        attrs = attrs or {}
        if op in ("ADD", "UPDATE"):
            e = self.entities.get(key)
            if e is None:
                typ, _, ident = key.partition(":")
                e = self.entities[key] = {"type": typ, "id": ident, "attrs": {}, "source": source, "valid": True}
                op = "ADD"
            e["attrs"].update(attrs)  # attribute-level merge; newer observation of the same source type wins
            if source:
                e["source"] = source
            e["valid"] = True
            self._touch(e)
        elif op == "DELETE":
            e = self.entities.get(key)
            if e is None:
                return
            e["valid"] = False
            e["attrs"].update(attrs)
            if source:
                e["source"] = source
            self._touch(e)
        elif op == "MERGE":
            src, dst = self.entities.pop(key, None), self.entities.get(into)
            if src is None or dst is None:
                return
            older, newer = (src, dst) if src.get("seq", 0) <= dst.get("seq", 0) else (dst, src)
            dst["attrs"] = {**older["attrs"], **newer["attrs"]}
            dst["source"] = newer.get("source") or older.get("source")
            dst.setdefault("aliases", []).append(src["id"])
            self._touch(dst)
        else:
            raise ValueError(op)
        self.ops.append({"op": op, "key": key, **({"into": into} if into else {}), "attrs": attrs,
                         **({"source": source} if source else {})})

    def find(self, ref: str) -> dict | None:
        """Entity by name: exact key, else (files) path-suffix match, else (commands) the newest
        command containing the given text."""
        if not ref:
            return None
        if ref in self.entities:
            return self.entities[ref]
        typ, _, ident = ref.partition(":")
        if typ == "file":
            ident = ident[2:] if ident.startswith("./") else ident
            hits = [e for k, e in self.entities.items() if e["type"] == "file"
                    and (e["id"].endswith("/" + ident) or ident.endswith("/" + e["id"]) or e["id"] == ident)]
        elif typ == "command":
            hits = [e for e in self.entities.values() if e["type"] == "command" and ident.strip() and ident.strip() in e["id"]]
        else:
            hits = []
        return max(hits, key=lambda e: e.get("seq", 0)) if hits else None

    def to_dict(self) -> dict:
        return {"entities": self.entities, "n_ops": len(self.ops)}

    def render(self, budget: int = FACT_RENDER_CHARS) -> str:
        """App. B.3 layout: ``[Entity: ...]`` blocks with type, id, source and attributes."""
        ents = list(self.entities.values())
        repo = [e for e in ents if e["type"] == "repository"]
        files = [e for e in ents if e["type"] == "file"]
        changed = sorted((e for e in files if e["attrs"].get("in_diff") or e["attrs"].get("edited")),
                         key=lambda e: e["id"])
        read_only = sorted((e for e in files if e not in changed), key=lambda e: -e.get("seq", 0))[:MAX_READ_ONLY_FILES]
        cmds = sorted((e for e in ents if e["type"] == "command"), key=lambda e: -e.get("seq", 0))[:MAX_COMMANDS]
        blocks = []
        for e in repo + changed + read_only + cmds:
            lines = [f"[Entity: {e['type']} {_one_line(e['id'], 160)}]", f" type: {e['type']}",
                     f" id: {_one_line(e['id'], 300)}", f" source: {_one_line(e.get('source') or '?', 200)}"]
            if not e.get("valid", True):
                lines.append(" status: invalid (deleted)")
            for k, v in e["attrs"].items():
                lines.append(f" {k}: {_one_line(json.dumps(v) if isinstance(v, (dict, list)) else v, 400)}")
            blocks.append("\n".join(lines))
        out, used = [], 0
        for b in blocks:
            if used + len(b) > budget:
                out.append(f"[... {len(blocks) - len(out)} more entities omitted ...]")
                break
            out.append(b)
            used += len(b) + 2
        return "\n\n".join(out) if out else "(no entities yet)"


def build_fact_layer(streams: list[tuple[int, str]], diff_text: str, snapshot: str) -> FactLayer:
    """Deterministic parser: the agent's tool calls (turn order) -> file / command entities; the
    cumulative diff at ``snapshot`` -> file change attributes (newest observation) and one
    ``repository`` entity. Re-running it over the whole session gives the same store as applying
    the operations turn by turn (the parser is incremental in order)."""
    F = FactLayer()
    for turn, raw in streams:
        for it in stream_items(raw):
            if it["kind"] != "tool":
                continue
            tool, inp = it["tool"], it["input"]
            if tool in FILE_TOOLS:
                path = inp.get("filePath") or inp.get("path") or inp.get("file_path")
                if not path:
                    continue
                path = str(path)
                prev = (F.entities.get(f"file:{path}") or {}).get("attrs", {})
                attrs = {"last_tool": tool, "last_turn": turn,
                         "edited": bool(prev.get("edited")) or FILE_TOOLS[tool] in ("edit", "write")}
                if it.get("status") and it["status"] != "completed":
                    attrs["last_tool_status"] = it["status"]
                F.apply("UPDATE", f"file:{path}", attrs, source=f"{tool}({path}) in turn {turn}")
            elif tool == "bash":
                cmd = str(inp.get("command") or inp.get("raw") or "").strip()
                if not cmd:
                    continue
                key = f"command:{cmd[:300]}"
                prev = (F.entities.get(key) or {}).get("attrs", {})
                out = (it.get("output") or "").strip()
                F.apply("UPDATE", key, {"exit": it.get("exit"), "status": it.get("status"),
                                        "output_tail": out[-300:] if out else "(no output)",
                                        "last_turn": turn, "runs": int(prev.get("runs") or 0) + 1},
                        source=f"bash in turn {turn}")
    files = diff_files(diff_text)
    for path, ch in files.items():
        key = f"file:{path}"
        for k in [k for k, e in F.entities.items() if e["type"] == "file" and k != key
                  and (e["id"].endswith("/" + path))]:
            if key not in F.entities:
                F.apply("ADD", key, {}, source=f"repository diff at {snapshot}")
            F.apply("MERGE", k, into=key)
        attrs = {"in_diff": True, "change": ch["change"], "lines_added": ch["lines_added"],
                 "lines_removed": ch["lines_removed"]}
        if ch["change"] == "deleted":
            F.apply("UPDATE", key, attrs, source=f"repository diff at {snapshot}")
            F.apply("DELETE", key, {}, source=f"repository diff at {snapshot}")
        else:
            F.apply("UPDATE", key, attrs, source=f"repository diff at {snapshot}")
    for k, e in list(F.entities.items()):
        if e["type"] == "file" and e["attrs"].get("edited") and e["id"] not in files \
                and not any(e["id"].endswith("/" + p) for p in files):
            F.apply("UPDATE", k, {"in_diff": False}, source=f"repository diff at {snapshot} (no net change)")
    F.apply("UPDATE", "repository:working tree",
            {"files_changed": len(files), "lines_added": sum(c["lines_added"] for c in files.values()),
             "lines_removed": sum(c["lines_removed"] for c in files.values())},
            source=f"repository diff at {snapshot}")
    return F


# ── state layer (IDSS §3.2.2) and propagation (§3.3) ─────────────────────────

def _low(x) -> str:
    return str(x or "").strip().lower()


def normalize_state(d) -> dict:
    """Validate and normalise the tracker's JSON; raises ValueError (schema)."""
    if not isinstance(d, dict):
        raise ValueError("state is not an object")
    intents, variables, constraints = d.get("intents"), d.get("variables", []), d.get("constraints", [])
    if not isinstance(intents, list) or not intents:
        raise ValueError("no intents")
    if not isinstance(variables, list) or not isinstance(constraints, list):
        raise ValueError("variables/constraints not lists")
    out = {"intents": [], "variables": [], "constraints": []}
    for i in intents:
        if not isinstance(i, dict) or not i.get("name") or _low(i.get("status")) not in INTENT_STATUS:
            raise ValueError(f"bad intent {str(i)[:120]}")
        dep = i.get("depends_on") or []
        out["intents"].append({"name": str(i["name"]), "status": _low(i["status"]),
                               "reason": i.get("reason"), "outcome": i.get("outcome"),
                               "depends_on": [str(x) for x in dep] if isinstance(dep, list) else [str(dep)]})
    for v in variables:
        if not isinstance(v, dict) or not v.get("name"):
            raise ValueError(f"bad variable {str(v)[:120]}")
        src = _low(v.get("source"))
        out["variables"].append({"name": str(v["name"]), "value": v.get("value"),
                                 "source": src if src in VAR_SOURCE else "askable",
                                 "known": bool(v.get("known")), "evidence": v.get("evidence")})
    for c in constraints:
        if not isinstance(c, dict) or not c.get("condition") or _low(c.get("status")) not in CONSTRAINT_STATUS:
            raise ValueError(f"bad constraint {str(c)[:120]}")
        g = c.get("grounding")
        if isinstance(g, dict) and g.get("entity") and _low(g.get("op")) in GROUND_OPS:
            g = {"entity": str(g["entity"]), "attribute": g.get("attribute"), "op": _low(g["op"]), "value": g.get("value")}
        else:
            g = None
        out["constraints"].append({"scope": str(c.get("scope") or ""), "condition": str(c["condition"]),
                                   "status": _low(c["status"]), "evidence": c.get("evidence"), "grounding": g})
    return out


def _eval_ground(g: dict, F: FactLayer):
    """True / False / None (not evaluable) and the fact it rests on."""
    e = F.find(g["entity"])
    op, attr = g["op"], g.get("attribute")
    if op in ("exists", "absent"):
        present = e is not None and e.get("valid", True) and (not attr or attr in e["attrs"])
        return (present if op == "exists" else not present), (f"{g['entity']} {'present' if present else 'absent'}")
    if e is None or not attr or attr not in e["attrs"]:
        return None, None
    val = e["attrs"][attr]
    fact = f"{e['type']}:{_one_line(e['id'], 120)}.{attr} = {json.dumps(val)}"
    want = g.get("value")
    if op in ("==", "!="):
        eq = str(val).strip().lower() == str(want).strip().lower()
        return (eq if op == "==" else not eq), fact
    has = str(want) in str(val)
    return (has if op == "contains" else not has), fact


def propagate(state: dict, F: FactLayer) -> tuple[dict, list[dict]]:
    """Re-evaluate grounded constraints against F; block intents whose constraint is violated
    (factual basis and constraint chain recorded). Ungrounded / unevaluable constraints are never
    set to violated by propagation (unevaluable grounded ones become unsatisfied)."""
    log = []
    by_name = {i["name"]: i for i in state["intents"]}
    for c in state["constraints"]:
        g = c.get("grounding")
        if not g:
            continue
        ok, fact = _eval_ground(g, F)
        before = c["status"]
        if ok is None:
            c["status"] = "unsatisfied" if before == "violated" else before
            if before == "violated":
                c["evidence"] = "grounding not evaluable against the fact layer (kept unsatisfied)"
        else:
            c["status"] = "satisfied" if ok else "violated"
            c["evidence"] = fact
        log.append({"constraint": c["condition"], "scope": c["scope"], "grounding": g, "result": ok, "fact": fact,
                    "status_before": before, "status_after": c["status"]})
        if c["status"] == "violated":
            it = by_name.get(c["scope"])
            if it is not None and it["status"] != "blocked":
                log.append({"blocked_intent": it["name"], "status_before": it["status"],
                            "because": f"constraint '{c['condition']}' violated: {fact}"})
                it["status"] = "blocked"
                it["reason"] = f"constraint '{c['condition']}' violated ({fact})"
    return state, log


def render_state(state: dict) -> str:
    """App. B.2 layout (YAML-like ``<state>`` body)."""
    q = lambda s: json.dumps(s, ensure_ascii=False) if s is not None else "null"  # noqa: E731
    lines = ["intents:"]
    for i in state["intents"]:
        lines += [f"- name: {q(i['name'])}", f"  status: {i['status']}"]
        if i.get("reason"):
            lines.append(f"  reason: {q(i['reason'])}")
        if i.get("outcome"):
            lines.append(f"  outcome: {q(i['outcome'])}")
        lines.append(f"  depends_on: {json.dumps(i.get('depends_on') or [], ensure_ascii=False)}")
    lines.append("variables:" + ("" if state["variables"] else " []"))
    for v in state["variables"]:
        lines += [f"- name: {q(v['name'])}", f"  value: {q(v.get('value'))}",
                  f"  source: {v['source']} | known: {str(v['known']).lower()}"]
        if v.get("evidence"):
            lines.append(f"  evidence: {q(v['evidence'])}")
    lines.append("constraints:" + ("" if state["constraints"] else " []"))
    for c in state["constraints"]:
        lines += [f"- scope: {q(c['scope'])}", f"  condition: {q(c['condition'])}", f"  status: {c['status']}",
                  f"  evidence: {q(c.get('evidence'))}"]
    return "\n".join(lines)


# ── mediator transcript ─────────────────────────────────────────────────────

def render_dialogue(messages: list[dict], budget: int = 60_000) -> str:
    """``User: ...`` / ``AI assistant: ...`` blocks (the paper's "Conversation Transcript between
    the User and the AI assistant"). The opening message is always kept; after it, the newest
    messages are kept first when the budget binds (the middle is elided)."""
    blocks = []
    for i, m in enumerate(messages):
        who = "User" if m["role"] == "user" else "AI assistant"
        cap = 8000 if i == 0 else (4000 if m["role"] == "user" else 2500)
        txt = (m.get("content") or "").strip()
        if len(txt) > cap:
            txt = txt[:cap] + " [...]"
        blocks.append(f"{who}: {txt}")
    if not blocks:
        return ""
    kept, used = [], len(blocks[0])
    for b in reversed(blocks[1:]):
        if used + len(b) > budget:
            kept.append("[... earlier conversation elided ...]")
            break
        kept.append(b)
        used += len(b)
    return "\n\n".join([blocks[0], *reversed(kept)])


def load_experiences(model: str) -> dict | None:
    p = PROMPTS / "mediator_experiences.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    e = (doc.get("by_model") or {}).get(bare_model(model))
    return {**e, "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest()} if e else None


# ── the arm ─────────────────────────────────────────────────────────────────

class PortArm:
    """Per-trial state of one port arm (tracker: the previous state layer; nothing carries across
    trials or tasks). Outputs in ``out_dir``: ``ports/<prefix>-N.json`` and ``port_calls.jsonl``.
    ``completion`` is a synchronous LiteLLM-like callable (``litellm.completion`` by default)."""

    def __init__(self, arm: str, out_dir: Path, agent_model: str, *, prefix: str = "turn",
                 reasoning: str = "high", completion=None, sleep=time.sleep, experiences: dict | None = None):
        if arm not in PORT_ARMS:
            raise ValueError(f"{arm!r} is not a port arm {PORT_ARMS}")
        self.arm = arm
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.model = litellm_model(agent_model)
        self.reasoning = reasoning
        self.prev_state: dict | None = None
        self._completion = completion
        self._sleep = sleep
        self.experiences = experiences
        if arm == "mediator" and self.experiences is None:
            self.experiences = load_experiences(agent_model)
            if self.experiences is None:
                raise ValueError(f"no Refiner experiences for {bare_model(agent_model)!r} in "
                                 f"{PROMPTS / 'mediator_experiences.json'} (run fsa.redo.ports_refine)")

    def _complete(self, **kw):
        if self._completion is not None:
            return self._completion(**kw)
        import litellm
        return litellm.completion(**kw)

    def _call(self, index: int, messages: list[dict], *, json_mode: bool, parse) -> tuple[object, bool, list[dict]]:
        out, ok, records = None, False, []
        for attempt in range(1, len(BACKOFF_SEC) + 2):
            t0 = time.time()
            rec = {"ts": t0, "arm": self.arm, "index": index, "attempt": attempt, "model": self.model,
                   "reasoning_effort": self.reasoning, "prompt_chars": sum(len(m["content"]) for m in messages)}
            try:
                kw = dict(model=self.model, messages=messages, reasoning_effort=self.reasoning, timeout=600,
                          num_retries=0)
                if json_mode:
                    kw["response_format"] = {"type": "json_object"}
                r = self._complete(**kw)
                u = getattr(r, "usage", None)
                ptd = getattr(u, "prompt_tokens_details", None)
                ctd = getattr(u, "completion_tokens_details", None)
                content = (r.choices[0].message.content or "").strip()
                rec.update(latency_sec=round(time.time() - t0, 2), resolved_model=getattr(r, "model", None),
                           finish_reason=getattr(r.choices[0], "finish_reason", None),
                           prompt_tokens=getattr(u, "prompt_tokens", None),
                           completion_tokens=getattr(u, "completion_tokens", None),
                           reasoning_tokens=getattr(ctd, "reasoning_tokens", None),
                           cached_tokens=getattr(ptd, "cached_tokens", None), raw=content[:20000])
                if not content:
                    raise ValueError("empty response")
                out = parse(content)
                ok = True
                rec["ok"] = True
                records.append(rec)
                break
            except Exception as e:  # noqa: BLE001
                rec.update(ok=False, latency_sec=round(time.time() - t0, 2),
                           error=f"{type(e).__name__}: {str(e)[:300]}")
                records.append(rec)
                txt = f"{type(e).__name__} {e}".lower()
                transient = any(m in txt for m in _TRANSIENT) or isinstance(e, json.JSONDecodeError)
                if not transient or attempt > len(BACKOFF_SEC) or (isinstance(e, ValueError) and attempt >= 4):
                    break
                self._sleep(BACKOFF_SEC[attempt - 1])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "port_calls.jsonl", "a") as f:
            for r_ in records:
                f.write(json.dumps(r_, ensure_ascii=False) + "\n")
        return out, ok, records

    def _write(self, index: int, doc: dict) -> None:
        d = self.out_dir / "ports"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{self.prefix}-{index}.json.tmp"
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str))
        os.replace(tmp, d / f"{self.prefix}-{index}.json")

    # tracker ──

    def tracker_update(self, *, index: int, transcript: str, streams: list[tuple[int, str]], diff_text: str,
                       snapshot: str, reply: str, meta: dict | None = None) -> dict:
        t_start = time.time()
        F = build_fact_layer(streams, diff_text, snapshot)
        facts = F.render()
        user = (transcript
                + "\n\n## Fact Layer (verified entities, parsed from your tool calls and the repository diff)\n"
                + facts
                + "\n\n## Previous state layer\n"
                + (render_state(self.prev_state) if self.prev_state else "(none yet)")
                + f"\n\n## The user's latest message\n{reply}\n\n" + prompt("tracker_request.md"))
        msgs = [{"role": "system", "content": prompt("tracker_system.md")}, {"role": "user", "content": user}]

        def parse(content):
            d = json.loads(content.removeprefix("```json").removesuffix("```").strip())
            try:
                return normalize_state(d)
            except ValueError as e:  # "invalid json" marks it retryable (at most 4 attempts, as the claim book)
                raise ValueError(f"invalid json state (schema): {e}") from None

        state, ok, records = self._call(index, msgs, json_mode=True, parse=parse)
        prop = []
        if ok:
            state, prop = propagate(state, F)
            self.prev_state = state
        doc = {"schema": "redo_v1/port-tracker/1", "arm": "tracker", "label": PORT_LABEL["tracker"], "index": index,
               "phase": "after_reply", "reply": reply, "snapshot": snapshot, "call_started_at": t_start,
               "written_at": time.time(), "model": self.model, "ok": ok, "attempts": len(records),
               "state": state, "propagation": prop, "fact_layer": F.to_dict(), "fact_ops": F.ops,
               "fact_layer_rendered": facts, **(meta or {})}
        self._write(index, doc)
        return doc

    # mediator ──

    def mediate(self, *, index: int, dialogue: list[dict], meta: dict | None = None) -> dict:
        t_start = time.time()
        conv = render_dialogue(dialogue)
        text = prompt("mediator_prompt.md").replace("[[EXPS]]", self.experiences["rules"].strip()).replace(
            "[[CONVERSATION]]", conv)
        msgs = [{"role": "user", "content": text}]
        rewrite, ok, records = self._call(index, msgs, json_mode=False, parse=lambda c: c.strip())
        doc = {"schema": "redo_v1/port-mediator/1", "arm": "mediator", "label": PORT_LABEL["mediator"],
               "index": index, "phase": "after_reply", "reply": dialogue[-1]["content"] if dialogue else None,
               "call_started_at": t_start, "written_at": time.time(), "model": self.model, "ok": ok,
               "attempts": len(records), "rewrite": rewrite, "n_messages": len(dialogue),
               "experiences_sha256": self.experiences.get("file_sha256"), **(meta or {})}
        self._write(index, doc)
        return doc

    # delivery ──

    def compose(self, reply: str, doc: dict | None) -> str:
        """The message the agent receives for a real user reply. A failed call passes the reply
        through unchanged (recorded in the doc and ``port_calls.jsonl``)."""
        if not (doc and doc.get("ok")):
            return reply
        if self.arm == "mediator":
            return doc["rewrite"]
        block = prompt("tracker_block.md").replace("{fact_layer_snapshot}", doc["fact_layer_rendered"]).replace(
            "{state_layer_snapshot}", render_state(doc["state"]))
        return f"{reply}\n\n{block.rstrip()}"


def config_record(arm: str, agent_model: str) -> dict:
    rec = {"arm": arm, "label": PORT_LABEL[arm], "model": litellm_model(agent_model), "reasoning_effort": "high",
           "prompts_sha256": prompts_sha256(), "module_sha256": module_sha256(), "timing": "after_reply"}
    if arm == "mediator":
        e = load_experiences(agent_model)
        rec["experiences"] = {k: e.get(k) for k in ("file_sha256", "rules_sha256", "refiner_model", "pairs")} if e else None
    return rec
