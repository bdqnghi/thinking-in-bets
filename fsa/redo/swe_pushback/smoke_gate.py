"""Pre-launch smoke gate for a dspush lane (REVIEWER_EXPS_PREREGISTRATION.md A5.10).

Before a waiter launches the G / S / M lane it runs 1 task x 2 arms (``stock``, ``commit_before``) in /tmp through the
same runner (native ``bash`` tool path, A5.7) and calls this check. The lane launches only if the check passes; on a
failure the waiter writes the reasons to ``.logs/dspush/`` and stops. The smoke is not a finding.

Pass iff, for every smoke arm:
  1. the trial wrote ``result.json`` with ``status == "ok"``;
  2. a pre-review verifier outcome and a track exist (``reward_pre`` and ``track`` not null);
  3. the solve phase ran at least one bash command and had fewer format-error turns than command steps (an A5.7-type
     protocol failure - replies without a usable tool call - shows up as format errors / no commands);
  4. artifact arms: ``artifact_status == "ok"`` (the harness accepted the block).

    python -m fsa.redo.swe_pushback.smoke_gate OUT_DIR --arms stock,commit_before
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import texts as T

SMOKE_TASK = "etree-xml-diff-patch"  # the task of the A5.8 F smoke (completed in all 6 arms there)
SMOKE_ARMS = ("stock", "commit_before")


def _solve_counts(traj: dict) -> tuple[int, int]:
    ev = traj.get("events") or []
    cmds = sum(1 for e in ev if e.get("phase") == "solve" and e.get("kind") == "cmd")
    fmt = sum(1 for e in ev if e.get("phase") == "solve" and e.get("kind") == "format_error")
    return cmds, fmt


def check_trial(d: Path, arm: str) -> list[str]:
    """Reasons the trial in ``d`` fails the gate (empty = pass)."""
    rp = d / "result.json"
    if not rp.exists():
        return [f"{arm}: no result.json"]
    try:
        r = json.loads(rp.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [f"{arm}: unreadable result.json ({e})"]
    why = []
    if r.get("status") != "ok":
        why.append(f"{arm}: status={r.get('status')} error={str(r.get('error'))[:300]}")
    if r.get("reward_pre") is None or r.get("track") is None:
        why.append(f"{arm}: no pre-review verifier outcome / track")
    try:
        traj = json.loads((d / "trajectory.json").read_text())
    except (json.JSONDecodeError, OSError):
        traj = {}
    cmds, fmt = _solve_counts(traj)
    if cmds < 1:
        why.append(f"{arm}: solve phase ran no bash command (format errors {fmt})")
    elif fmt >= cmds:
        why.append(f"{arm}: solve phase format errors {fmt} >= command steps {cmds}")
    if T.ARTIFACT.get(arm) and r.get("artifact_status") != "ok":
        why.append(f"{arm}: artifact_status={r.get('artifact_status')}")
    return why


def check(out_dir: Path, task: str = SMOKE_TASK, arms=SMOKE_ARMS, seed: int = 0) -> dict:
    reasons = []
    for arm in arms:
        reasons += check_trial(Path(out_dir) / "trials" / f"{task}__{arm}__s{seed}", arm)
    return {"ok": not reasons, "reasons": reasons, "out_dir": str(out_dir), "task": task, "arms": list(arms)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--task", default=SMOKE_TASK)
    ap.add_argument("--arms", default=",".join(SMOKE_ARMS))
    a = ap.parse_args(argv)
    res = check(Path(a.out_dir), a.task, [x for x in a.arms.split(",") if x])
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
