"""#13 (L-O): intent-to-treat vs per-protocol for the agentic (Harbor) runs (W8).

For every Harbor job pair (same bench + model rung; stock vs each non-stock arm), on the tasks both arms were
assigned:

* **ITT**: pass rate by *assigned* arm, one trial per task chosen without looking at outcomes (``rule="first"``:
  the earliest-started scored-or-timed-out trial; errored trials count as failures). Also shown with the published
  best-of-N rule (``rule="collect"``, mirrors ``scripts/harbor_collect.py``) so the size of that choice is visible.
* **Compliance**: share of treated trials in which a Bet Card / claim book was actually written
  (``claim_book`` feature from ``harbor_trials.trajectory_features``: a ``[CLAIM_BOOK]`` header with >= 2 claim ids).
* **Per-protocol**: the paired comparison restricted to tasks where the treated trial complied. This is a
  *selected* subgroup (compliance can depend on task difficulty), reported only next to ITT.

Arms detected from the job name: ``<bench>_<rung>_<arm>[_p<part>]`` for bench in {ds11, ds11m, ds11r2, tb4}.
Reads ``~/jobs`` read-only. Output: ``results/analysis/itt_per_protocol.json`` + markdown on stdout.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import harbor_trials as H  # noqa: E402

# Broaden the job pattern locally (the module default only knows stock|tib); harbor_trials itself is unchanged.
H.JOB_RE = re.compile(r"^(tb4|ds11m|ds11r2|ds11)_([WFMSG])_([a-z0-9]+?)(?:_p(\d+))?$")
H.HARNESS = dict(H.HARNESS, ds11r2="opencode")


def load() -> list[dict]:
    """All finished trials, including the ones the repair scripts moved aside (``_attempts_<job>``, ``_dups_<job>``):
    without them the outcome-blind ``first`` rule could only see the surviving (re-run) trial."""
    rows = []
    for jd in sorted(H.JOBS.iterdir()):
        if not jd.is_dir():
            continue
        name, origin = jd.name, "main"
        for pre in ("_attempts_", "_dups_"):
            if name.startswith(pre):
                name, origin = name[len(pre):], pre.strip("_")
        if name.startswith("_") or not H.JOB_RE.match(name):
            continue
        for d in sorted(jd.glob("*__*")):
            if d.is_dir():
                r = H.trial_row(d, name)
                if r:
                    r["origin"] = origin
                    rows.append(r)
    return rows


def sign_p(w: int, l: int) -> float | None:
    n = w + l
    if n == 0:
        return None
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(stock: dict, arm: dict, tasks: list[str]) -> dict:
    w = l = t = 0
    for k in tasks:
        a, b = bool(stock[k]["passed"]), bool(arm[k]["passed"])
        if b and not a:
            w += 1
        elif a and not b:
            l += 1
        else:
            t += 1
    n = len(tasks)
    return dict(n=n, stock_pass=round(100 * sum(stock[k]["passed"] for k in tasks) / n, 1) if n else None,
                arm_pass=round(100 * sum(arm[k]["passed"] for k in tasks) / n, 1) if n else None,
                W=w, T=t, L=l, sign_p=sign_p(w, l))


def main() -> None:
    rows = load()
    sel = {rule: H.select_per_task(rows, rule=rule) for rule in ("first", "collect")}
    out = {"pairs": {}}
    md = ["| Bench | Rung | Arm | Rule | n | stock pass | arm pass | W/T/L | sign p | compliance | PP n | PP stock | PP arm | PP W/T/L |",
          "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
    base_jobs = sorted(sel["first"])
    for bj in base_jobs:
        bench, rung, arm = bj.split("_", 2)
        if arm == "stock":
            continue
        sj = f"{bench}_{rung}_stock"
        if sj not in sel["first"]:
            continue
        for rule in ("first", "collect"):
            S, A = sel[rule][sj], sel[rule][bj]
            tasks = sorted(set(S) & set(A))
            itt = compare(S, A, tasks)
            comp = [k for k in tasks if A[k].get("claim_book")]
            pp = compare(S, A, comp)
            rec = dict(itt=itt, compliance=round(len(comp) / len(tasks), 3) if tasks else None,
                       n_complied=len(comp), per_protocol=pp,
                       compliance_given_pass=round(sum(1 for k in comp if A[k]["passed"]) /
                                                   max(1, sum(1 for k in tasks if A[k]["passed"])), 3),
                       errored_arm=sum(1 for k in tasks if A[k]["exc_type"] and not A[k]["scored"]),
                       errored_stock=sum(1 for k in tasks if S[k]["exc_type"] and not S[k]["scored"]))
            out["pairs"].setdefault(bj, {})[rule] = rec
            md.append(f"| {bench} | {rung} | {arm} | {rule} | {itt['n']} | {itt['stock_pass']} | {itt['arm_pass']} | "
                      f"{itt['W']}/{itt['T']}/{itt['L']} | {itt['sign_p'] if itt['sign_p'] is None else round(itt['sign_p'], 3)} | "
                      f"{rec['compliance']} | {pp['n']} | {pp['stock_pass']} | {pp['arm_pass']} | "
                      f"{pp['W']}/{pp['T']}/{pp['L']} |")
    out["notes"] = [
        "rule=first: earliest scored-or-timed-out trial per task (no outcome-dependent choice); errored = fail",
        "rule=collect: harbor_collect.py best-of-N over infra retries (the published rule)",
        "compliance applies to arms that are asked to write a claim book; for stock-like arms (review_only, "
        "commit_only) it measures spontaneous claim-book text and is expected to be ~0",
        "jobs still running are partial snapshots",
    ]
    dst = ROOT / "results/analysis/itt_per_protocol.json"
    dst.write_text(json.dumps(out, indent=1))
    print("\n".join(md))
    print(f"\n{len(rows)} trials; wrote {dst}")


if __name__ == "__main__":
    main()
