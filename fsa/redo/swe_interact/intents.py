"""Frozen per-task intent sets for SWE-Interact (redo_v1 Stage 1 ground truth; Stage 0 proposal).

    python -u -m fsa.redo.swe_interact.intents build [--tasks a,b]   # gpt-5.5 split + test mapping
    python -u -m fsa.redo.swe_interact.intents validate [--tasks ..] # reference vs unmodified base
    python -u -m fsa.redo.swe_interact.intents freeze                # index.json with every digest
    python -u -m fsa.redo.swe_interact.intents audit                 # AUDIT.md (human-audit sheet)

One file per effective-subset task, ``configs/intents/swe-interact/<task>.json``:

* **deepswe** - ``instruction.md`` (the simulator's task block) split into atomic requirements by
  gpt-5.5 (one call, effort high, ``prompts/intent_builder_system.md``) and mapped, in the same
  call, to function-level groups of the hidden tests (the ids the reference solution's run of
  ``test.sh new`` produced; the model may only choose from that list). Mapped -> ``exec`` judge,
  unmapped -> ``llm`` judge.
* **swebenchpro** - ``config.json`` ``requirements`` (the benchmark's own itemised list; the
  public ``interface`` is given as context only) split the same way and mapped to function-level
  groups of ``fail_to_pass``.
* **rf** - the rubric items of ``rubrics.json`` themselves ("high level intent" headers dropped;
  positive and negative items, ids kept; ``rubric`` judge = the benchmark's grader prompt) plus
  one ``behaviour_preserved`` intent judged by the verifier's test comparison (``exec_rf``). The
  headline item is chosen by gpt-5.5 (``prompts/headline_system.md``).

``digest`` = sha256 of the canonical JSON of ``{task, family, intents}``; the ``validation`` block
(reference >= 90 % of intents satisfied; unmodified base <= 10 % of *positive* intents - the
constraint intents, i.e. negative rubric items and behaviour preservation, are satisfied by an
unchanged repository by construction and are checked on the reference only) is outside the digest.
The sets are frozen before any pilot run (``index.json``: every digest + the index digest, which
every pilot manifest records).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import exec_judge
from .config import RESULTS, ROOT, TASKS, family, subset_tasks
from .llm import MODEL, structured_call

INTENT_DIR = ROOT / "configs" / "intents" / "swe-interact"
BUILD_LOG = RESULTS / "swi_intents_build"
PROMPTS = Path(__file__).with_name("prompts")
BUILDER_EFFORT = "high"
REF_MIN, BASE_MAX = 0.90, 0.10
TRIGGER_EXEC_MAX_SEC = 150.0
POSITIVE = ("requirement", "rubric_positive")
BEHAVIOUR_TEXT = ("The task's relevant tests pass after the change: with the task's test patch applied, no relevant "
                  "test fails or goes missing (the refactoring verifier's test criterion; it also guards existing "
                  "behaviour).")


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def reference_patch(task: str) -> str:
    fam = family(task)
    if fam == "deepswe":
        return (TASKS / task / "solution" / "solution.patch").read_text()
    if fam == "rf":
        return (TASKS / task / "solution" / "gold.patch").read_text()
    return json.loads((TASKS / task / "tests" / "config.json").read_text())["patch"]


def _config(task: str) -> dict:
    return json.loads((TASKS / task / "steps" / "05_test_handoff" / "tests" / "config.json").read_text())


def _listish(v) -> list:
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(v)


def fail_to_pass(task: str) -> list[str]:
    return list(_listish(_config(task).get("fail_to_pass") or []))


def test_groups(task: str) -> list[str]:
    fam = family(task)
    if fam == "swebenchpro":
        ids = fail_to_pass(task)
    else:
        run = exec_judge.run_snapshot(task, reference_patch(task), label="reference")
        ids = [t for t, s in (run.get("tests") or {}).items() if s == "PASSED"]
    return sorted({exec_judge.test_group(t, fam) for t in ids})


def _test_source(task: str, limit: int = 90_000) -> str:
    fam = family(task)
    if fam == "deepswe":
        patch = (TASKS / task / "steps" / "05_test_handoff" / "tests" / "test.patch").read_text()
    else:
        patch = _config(task).get("test_patch") or ""
    return patch if len(patch) <= limit else patch[:limit] + "\n[... test source truncated ...]"


def source_text(task: str) -> str:
    fam = family(task)
    if fam == "deepswe":
        return (TASKS / task / "instruction.md").read_text().strip()
    c = _config(task)
    req = c.get("requirements") or ""
    if req.startswith('"') and req.endswith('"'):
        try:
            req = json.loads(req)
        except json.JSONDecodeError:
            pass
    return req.strip()


def digest(doc: dict) -> str:
    return _sha(json.dumps({"task": doc["task"], "family": doc["family"], "intents": doc["intents"]},
                           sort_keys=True, ensure_ascii=False))


def intent_path(task: str) -> Path:
    return INTENT_DIR / f"{task}.json"


def load(task: str, verify: bool = True) -> dict:
    doc = json.loads(intent_path(task).read_text())
    if verify and digest(doc) != doc["digest"]:
        raise SystemExit(f"{task}: intent set changed since it was frozen (digest mismatch)")
    return doc


def index_digest() -> dict:
    p = INTENT_DIR / "index.json"
    if not p.exists():
        return {"frozen": False}
    d = json.loads(p.read_text())
    return {"frozen": True, "path": str(p.relative_to(ROOT)), "sha256": _sha(p.read_bytes()),
            "index_digest": d.get("index_digest"), "frozen_at": d.get("frozen_at")}


# ── build ───────────────────────────────────────────────────────────────────

def _builder_schema(groups: list[str]) -> dict:
    tests = {"type": "array", "items": {"type": "string", "enum": groups}} if groups else \
        {"type": "array", "items": {"type": "string"}, "maxItems": 0}
    return {"type": "object", "additionalProperties": False, "required": ["intents"],
            "properties": {"intents": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["text", "headline", "tests"],
                "properties": {"text": {"type": "string"}, "headline": {"type": "boolean"}, "tests": tests}}}}}


def _check_one_headline(out: dict) -> None:
    n = sum(1 for i in out.get("intents", []) if i.get("headline"))
    if n != 1 or not out.get("intents"):
        raise ValueError(f"expected exactly one headline intent, got {n}")


def build_requirements(task: str) -> dict:
    fam = family(task)
    groups = test_groups(task)
    system = (PROMPTS / "intent_builder_system.md").read_text()
    src = source_text(task)
    user = f"## Requirement text ({'task request' if fam == 'deepswe' else 'the benchmark requirements list'})\n{src}\n"
    if fam == "swebenchpro":
        user += f"\n## Public interface (context only; do not itemise it separately)\n{_config(task).get('interface') or ''}\n"
    user += (f"\n## Hidden test ids you may map to ({len(groups)})\n" + "\n".join(groups) +
             f"\n\n## Hidden test source (for mapping)\n```diff\n{_test_source(task)}\n```\n")
    res = structured_call(system, user, _builder_schema(groups), name="intent_set", log_dir=BUILD_LOG,
                          effort=BUILDER_EFFORT, meta={"task": task, "kind": "build"}, validate=_check_one_headline)
    intents = []
    for n, it in enumerate(res["output"]["intents"], start=1):
        tests = sorted(set(it["tests"]))
        intents.append({"key": f"I{n:02d}", "text": it["text"].strip(), "kind": "requirement",
                        "headline": bool(it["headline"]), "tests": tests, "judge": "exec" if tests else "llm"})
    return {"intents": intents, "test_groups": groups, "builder_call": {k: res[k] for k in ("usage", "response_id",
                                                                                            "resolved_model")}}


def build_rf(task: str) -> dict:
    items = json.loads((TASKS / task / "steps" / "05_test_handoff" / "tests" / "rubrics.json").read_text())
    intents = []
    for it in items:
        ann = it.get("annotations") or {}
        if ann.get("type") == "high level intent":
            continue
        neg = "negative" in (ann.get("type") or "")
        title = it["title"]
        intents.append({"key": f"R{len(intents) + 1:02d}", "text": re.sub(r"^\s*\d+(\.\d+)*:\s*", "", title),
                        "kind": "rubric_negative" if neg else "rubric_positive", "headline": False, "tests": [],
                        "judge": "rubric", "rubric_id": it.get("id"), "title": title,
                        "importance": ann.get("importance", "must have")})
    rel = json.loads((TASKS / task / "steps" / "05_test_handoff" / "tests" / "relevant_test_names.json").read_text())
    intents.append({"key": "B01", "text": BEHAVIOUR_TEXT, "kind": "behaviour_preserved", "headline": False,
                    "tests": sorted(t["name"] for t in rel.get("tests", []) if t.get("name")), "judge": "exec_rf"})
    pos = [i for i in intents if i["kind"] == "rubric_positive"]
    system = (PROMPTS / "headline_system.md").read_text()
    user = (f"## Request text\n{(TASKS / task / 'instruction.md').read_text().strip()}\n\n## Rubric items\n" +
            "\n".join(json.dumps({"key": i["key"], "item": i["text"]}) for i in pos))
    schema = {"type": "object", "additionalProperties": False, "required": ["headline_key"],
              "properties": {"headline_key": {"type": "string", "enum": [i["key"] for i in pos]}}}
    res = structured_call(system, user, schema, name="headline", log_dir=BUILD_LOG, effort=BUILDER_EFFORT,
                          meta={"task": task, "kind": "headline"})
    for i in intents:
        i["headline"] = i["key"] == res["output"]["headline_key"]
    return {"intents": intents, "builder_call": {k: res[k] for k in ("usage", "response_id", "resolved_model")}}


def build(task: str) -> dict:
    fam = family(task)
    b = build_rf(task) if fam == "rf" else build_requirements(task)
    doc = {"schema": "redo_v1/swe-interact-intents/1", "task": task, "family": fam,
           "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "sources": {"instruction_md_sha256": _sha((TASKS / task / "instruction.md").read_bytes()),
                       "final_tests_dir": "steps/05_test_handoff/tests",
                       "reference_patch_sha256": _sha(reference_patch(task)),
                       "exec_recipe_sha256": exec_judge.recipe_sha(task)},
           "builder": {"model": MODEL, "effort": BUILDER_EFFORT,
                       "prompt_sha256": _sha((PROMPTS / ("headline_system.md" if fam == "rf" else
                                                         "intent_builder_system.md")).read_bytes()),
                       **b.get("builder_call", {})},
           "test_groups": b.get("test_groups"),
           "intents": b["intents"]}
    doc["digest"] = digest(doc)
    INTENT_DIR.mkdir(parents=True, exist_ok=True)
    intent_path(task).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc


# ── validate ────────────────────────────────────────────────────────────────

def _rate(labels: dict[str, str], keys: list[str]) -> dict:
    app = [k for k in keys if labels.get(k) not in ("not_applicable", None)]
    sat = [k for k in app if labels[k] == "satisfied"]
    err = [k for k in app if labels[k] in ("error", "pending")]
    return {"n": len(app), "satisfied": len(sat), "errors": len(err),
            "rate": round(len(sat) / len(app), 4) if app else None}


def validate(task: str, judge=None) -> dict:
    from .judge import SnapshotJudge
    judge = judge or SnapshotJudge()
    doc = load(task)
    its = doc["intents"]
    out = {"validated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rule": f"reference >= {REF_MIN:.0%} of intents; "
           f"base <= {BASE_MAX:.0%} of positive intents"}
    for lab, patch in (("reference", reference_patch(task)), ("base", "")):
        v = judge.verdicts(task, patch, its, label=lab)
        meta = v.pop("_meta")
        labels = {k: x["verdict"] for k, x in v.items()}
        out[lab] = {"labels": labels, "evidence": {k: (x.get("evidence") or "")[:300] for k, x in v.items()},
                    "all": _rate(labels, [i["key"] for i in its]),
                    "positive": _rate(labels, [i["key"] for i in its if i["kind"] in POSITIVE]),
                    "constraint": _rate(labels, [i["key"] for i in its if i["kind"] not in POSITIVE]),
                    "exec": meta.get("exec")}
    walls = [(out[lab].get("exec") or {}).get("wall_sec") for lab in ("reference", "base")]
    walls = [w for w in walls if w is not None]
    # the in-trial overlay trigger runs the hidden tests only where that is quick; otherwise (and
    # for rf, whose check is the full suite twice) it uses the no-dialogue LLM judges
    out["exec_wall_sec_max"] = max(walls) if walls else None
    out["trigger_judge"] = ("exec" if doc["family"] != "rf" and walls and max(walls) <= TRIGGER_EXEC_MAX_SEC
                            else "llm")
    r, b = out["reference"]["all"]["rate"], out["base"]["positive"]["rate"]
    out["pass"] = bool(r is not None and r >= REF_MIN and b is not None and b <= BASE_MAX)
    out["n_intents"] = len(its)
    out["n_exec_mapped"] = sum(1 for i in its if i["judge"] in ("exec", "exec_rf"))
    doc["validation"] = out
    intent_path(task).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return out


def freeze(tasks: list[str]) -> dict:
    docs = {t: load(t) for t in tasks}
    idx = {"schema": "redo_v1/swe-interact-intents-index/1", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "subset": "configs/subsets/swe-interact.replacements.json (effective_subset)",
           "note": "frozen before any Stage 1/2 pilot trial; changing a set requires a new index and is recorded",
           "tasks": {t: {"digest": d["digest"], "n_intents": len(d["intents"]),
                         "validation_pass": (d.get("validation") or {}).get("pass")} for t, d in docs.items()}}
    idx["index_digest"] = _sha(json.dumps(idx["tasks"], sort_keys=True))
    (INTENT_DIR / "index.json").write_text(json.dumps(idx, indent=1) + "\n")
    return idx


def audit_sheet(tasks: list[str]) -> str:
    lines = ["# SWE-Interact intent sets - human audit sheet", "",
             "One frozen intent set per effective-subset task (`configs/intents/swe-interact/<task>.json`). "
             "Read each list once: are these the user's real requirements, atomic, complete, and is the "
             "headline right? Judge column: `exec` = hidden tests (listed), `llm` = no-dialogue gpt-5.5 "
             "judge, `rubric` = the benchmark's rubric grader prompt, `exec_rf` = the rf verifier's test "
             "comparison. ref/base = verdict on the reference solution / unmodified repository.", ""]
    for t in tasks:
        d = load(t)
        v = d.get("validation") or {}
        rl, bl = (v.get("reference") or {}).get("labels", {}), (v.get("base") or {}).get("labels", {})
        lines += [f"## {t}", "",
                  f"family `{d['family']}` - digest `{d['digest'][:16]}` - "
                  f"reference {v.get('reference', {}).get('all', {}).get('satisfied')}/"
                  f"{v.get('reference', {}).get('all', {}).get('n')}, base (positive) "
                  f"{v.get('base', {}).get('positive', {}).get('satisfied')}/"
                  f"{v.get('base', {}).get('positive', {}).get('n')} - validation "
                  f"{'PASS' if v.get('pass') else 'FAIL' if v else 'not run'}", "",
                  "| key | intent | kind | judge | tests | ref | base |", "|---|---|---|---|---|---|---|"]
        for i in d["intents"]:
            txt = i["text"].replace("|", "\\|").replace("\n", " ")
            if i.get("headline"):
                txt = f"**{txt}** (headline)"
            tests = ", ".join(f"`{x}`" for x in i["tests"][:6]) + (f" +{len(i['tests']) - 6}" if len(i["tests"]) > 6 else "")
            if i["judge"] == "exec_rf":
                tests = f"{len(i['tests'])} relevant tests"
            lines.append(f"| {i['key']} | {txt} | {i['kind']} | {i['judge']} | {tests} | {rl.get(i['key'], '')} | "
                         f"{bl.get(i['key'], '')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "validate", "freeze", "audit"])
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="build: rebuild existing sets")
    a = ap.parse_args(argv)
    tasks = a.tasks.split(",") if a.tasks else subset_tasks()
    if a.cmd == "build":
        todo = [t for t in tasks if a.force or not intent_path(t).exists()]

        def one(t):
            try:
                d = build(t)
                print(f"built {t}: {len(d['intents'])} intents, {sum(1 for i in d['intents'] if i['tests'])} mapped",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {t}: {type(e).__name__}: {e}", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(one, todo))
    elif a.cmd == "validate":
        from .judge import SnapshotJudge
        j = SnapshotJudge()

        def one(t):
            try:
                v = validate(t, j)
                print(f"{t}: ref {v['reference']['all']} base+ {v['base']['positive']} pass={v['pass']}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {t}: {type(e).__name__}: {e}", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(one, tasks))
    elif a.cmd == "freeze":
        print(json.dumps(freeze(tasks), indent=1))
    else:
        (INTENT_DIR / "AUDIT.md").write_text(audit_sheet(tasks))
        print(INTENT_DIR / "AUDIT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
