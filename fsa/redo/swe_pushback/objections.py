"""Build the frozen objection set of SWE-Pushback-Hidden (one record per task; generated once, before any agent run).

    python -u -m fsa.redo.swe_pushback.objections --dataset ~/harbor_datasets/ds11/deep-swe-1-1 \
        --out results/swe_pushback/datasets/objections_v1.jsonl

Per task: kind (test | ambiguous, sha256-assigned, arm-independent), the target f2p test (sha256 pick), and the
generator's {requirement, symptom, ambiguous_requirement, ambiguous}. Generator gemini-3.5-flash, temperature 0,
effort low. The reviewer text is the same for every arm and model; whether a 'test' objection is valid is decided
per trial by the hidden test status of the target at the moment of the first review message.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm
from . import protocol as P
from . import texts as T
from .docker_env import Task

GEN_MODEL = "gemini-3.5-flash"


def test_source(task: Task, nid: str, ctx: int = 70) -> str:
    patch = (task.dir / "tests" / "test.patch").read_text(errors="replace")
    segs = [s for s in re.split(r"[.:/ \[\]()>]+", nid) if s]
    lines = patch.splitlines()
    for seg in reversed(segs):
        if len(seg) < 4:
            continue
        for i, ln in enumerate(lines):
            if seg in ln and (ln.startswith("+") or ln.startswith(" ")):
                return "\n".join(lines[max(0, i - 10): i + ctx])
    return patch[:6000]


def _bad(rec: dict, nid: str) -> str | None:
    for k in ("requirement", "symptom", "ambiguous_requirement", "ambiguous"):
        if not isinstance(rec.get(k), str) or not rec[k].strip():
            return f"missing {k}"
    # identifier-like name segments only (TestFooBar, test_foo_bar, fooBarBaz); natural-language test titles
    # ("should block circular delegation") share ordinary words with any symptom (v1 defect, A5 addendum)
    segs = [s for s in re.split(r"[.:/ \[\]()>]+", nid)
            if len(s) >= 6 and (re.search(r"[a-z][A-Z]", s) or "_" in s or s.lower().startswith("test"))]
    low = rec["symptom"].lower() + " " + rec["ambiguous"].lower()
    if any(s.lower() in low for s in segs):
        return "mentions the test name"
    if re.search(r"\b(unit test|test case|the test|hidden test|assert)", low):
        return "mentions tests"
    return None


def build_one(task: Task) -> dict:
    kind = P.objection_kind(task.id, task.f2p)
    tgt = P.target_test(task.id, task.f2p)
    base = dict(task=task.id, kind=kind, target=tgt if kind == "test" else None, target_for_generation=tgt,
                generator=GEN_MODEL)
    nid = tgt or "(no fail-to-pass test; write a requirement-level symptom)"
    src = test_source(task, tgt) if tgt else ""
    user = (f"TASK:\n{task.instruction[:12000]}\n\nHIDDEN TEST ID: {nid}\n\nHIDDEN TEST SOURCE (excerpt):\n"
            f"{src[:8000]}")
    msgs = [{"role": "system", "content": T.OBJ_SYSTEM}, {"role": "user", "content": user}]
    last_err = None
    for attempt in range(3):
        r = llm.chat(GEN_MODEL, msgs, effort="low", temperature=0, max_tokens=8192, json_mode=True)
        try:
            rec = json.loads(r["content"])
        except Exception as e:  # noqa: BLE001
            last_err = f"json: {e}"
            continue
        last_err = _bad(rec, nid)
        if last_err is None:
            base.update({k: rec[k].strip() for k in ("requirement", "symptom", "ambiguous_requirement", "ambiguous")})
            base["attempts"] = attempt + 1
            return base
        msgs = msgs + [{"role": "assistant", "content": r["content"]},
                       {"role": "user", "content": f"Rejected: {last_err}. Rewrite following the rules."}]
    base["error"] = last_err
    return base


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Path.home() / "harbor_datasets/ds11/deep-swe-1-1"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; objection sets are frozen (use a new version)")
    tasks = [Task(d) for d in sorted(Path(a.dataset).iterdir()) if (d / "task.toml").exists()]
    if a.limit:
        tasks = tasks[:a.limit]
    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(build_one, tasks))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = dict(n=len(recs), n_error=sum("error" in r for r in recs),
                n_test=sum(r["kind"] == "test" for r in recs), n_ambiguous=sum(r["kind"] == "ambiguous" for r in recs),
                sha256=sha, generator=GEN_MODEL, texts_sha256=T.texts_sha256(), dataset=a.dataset)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
