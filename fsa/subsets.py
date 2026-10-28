"""Seeded, stratified, frozen task subsets for the redo-robustness study (docs/plans/PLAN.md §5.3).

Every benchmark gets one 20-task subset, drawn once and written to
``configs/subsets/<benchmark>.json`` before any run. Allocation across strata is
proportional (largest remainder, ties broken by the seeded order). Within a
stratum tasks are taken in a seeded random order; the rest of that order is kept
as the pre-declared replacement list, used only when a selected task is excluded
for an infrastructure reason (e.g. its arm64 image fails gold validation), never
because of a result.

Usage: python -m fsa.subsets swe-together|swe-interact [--size 20] [--seed 20260926] [--write]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "swe-together": ROOT / "third_party" / "swe-together",
    "swe-interact": ROOT / "third_party" / "swe-interact",
}
CATEGORY = {"bugfix": "bugfix", "feature": "feature", "feature-implementation": "feature", "implementation": "feature",
            "refactor": "refactor", "refactoring": "refactor"}


def intent_bucket(n: int) -> str:
    return "1-2" if n <= 2 else "3-4" if n <= 4 else "5+"


def swe_together_tasks(src: Path) -> list[dict]:
    plan = json.loads((src / "canonical_full109.json").read_text())
    out = []
    for tid in plan["tasks"]:
        meta = tomllib.loads((src / "tasks" / tid / "task.toml").read_text())["metadata"]
        intents = json.loads((src / "tasks" / tid / "oracle_intents.json").read_text())["intents"]
        cat = CATEGORY[meta["category"]]
        out.append({"task": tid, "category": cat, "n_intents": len(intents),
                    "stratum": f"{cat}/{intent_bucket(len(intents))}"})
    return out


def swe_interact_tasks(src: Path) -> list[dict]:
    out = []
    for d in sorted((src / "data" / "multiturn").iterdir()):
        meta = tomllib.loads((d / "task.toml").read_text()).get("metadata", {})
        family = d.name.split("_", 1)[0]
        out.append({"task": d.name, "family": family, "language": meta.get("language"), "stratum": family})
    return out


def allocate(sizes: dict[str, int], total: int, order: list[str]) -> dict[str, int]:
    n = sum(sizes.values())
    quota = {s: total * k / n for s, k in sizes.items()}
    alloc = {s: int(q) for s, q in quota.items()}
    rank = sorted(sizes, key=lambda s: (-(quota[s] - alloc[s]), order.index(s)))
    for s in rank[: total - sum(alloc.values())]:
        alloc[s] += 1
    return alloc


def draw(tasks: list[dict], size: int, seed: int) -> dict:
    rng = random.Random(seed)
    by = defaultdict(list)
    for t in sorted(tasks, key=lambda t: t["task"]):
        by[t["stratum"]].append(t)
    strata = sorted(by)
    order = strata[:]
    rng.shuffle(order)
    alloc = allocate({s: len(by[s]) for s in strata}, size, order)
    selected, replacements = [], {}
    for s in strata:
        pool = by[s][:]
        rng.shuffle(pool)
        selected += pool[: alloc[s]]
        replacements[s] = [t["task"] for t in pool[alloc[s]:]]
    return {"selected": sorted(selected, key=lambda t: (t["stratum"], t["task"])), "allocation": alloc,
            "stratum_sizes": {s: len(by[s]) for s in strata}, "replacements": replacements}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("benchmark", choices=sorted(SOURCES))
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--write", action="store_true", help="write configs/subsets/<benchmark>.json (refuses to overwrite)")
    a = ap.parse_args(argv)
    src = SOURCES[a.benchmark]
    tasks = swe_together_tasks(src) if a.benchmark == "swe-together" else swe_interact_tasks(src)
    result = draw(tasks, a.size, a.seed)
    import subprocess
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    doc = {"benchmark": a.benchmark, "source_commit": commit, "seed": a.seed, "size": a.size,
           "population": len(tasks), "strata": "category x intent-count bucket" if a.benchmark == "swe-together" else "task family",
           "replacement_rule": "a selected task is replaced only for an infrastructure reason recorded before any result "
                               "(e.g. arm64 image fails gold validation), by the next task in its stratum's replacement list",
           **result}
    doc["digest"] = hashlib.sha256(json.dumps([t["task"] for t in result["selected"]]).encode()).hexdigest()
    text = json.dumps(doc, indent=1) + "\n"
    if a.write:
        out = ROOT / "configs" / "subsets" / f"{a.benchmark}.json"
        if out.exists():
            raise SystemExit(f"{out} exists; subsets are frozen once written")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out}")
    print(json.dumps({"allocation": result["allocation"], "selected": [t["task"] for t in result["selected"]]}, indent=1))


if __name__ == "__main__":
    main()
