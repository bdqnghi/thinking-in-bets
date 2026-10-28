"""Unit tests for the dspush pre-launch smoke gate (A5.10)."""
from __future__ import annotations

import json
from pathlib import Path

from fsa.redo.swe_pushback import smoke_gate as G

TASK = G.SMOKE_TASK


def _trial(root: Path, arm: str, *, status="ok", reward_pre=1, track="invalid", artifact_status=None,
           cmds=5, fmt=0, result=True):
    d = root / "trials" / f"{TASK}__{arm}__s0"
    d.mkdir(parents=True)
    ev = [{"phase": "solve", "kind": "cmd"}] * cmds + [{"phase": "solve", "kind": "format_error"}] * fmt
    ev += [{"phase": "push1", "kind": "format_error"}] * 9  # post-review format errors do not count
    (d / "trajectory.json").write_text(json.dumps({"messages": [], "events": ev}))
    if result:
        r = {"status": status, "reward_pre": reward_pre, "track": track}
        if artifact_status is not None:
            r["artifact_status"] = artifact_status
        (d / "result.json").write_text(json.dumps(r))
    return d


def test_pass(tmp_path):
    _trial(tmp_path, "stock")
    _trial(tmp_path, "commit_before", artifact_status="ok")
    assert G.check(tmp_path) == {"ok": True, "reasons": [], "out_dir": str(tmp_path), "task": TASK,
                                 "arms": list(G.SMOKE_ARMS)}


def test_missing_and_bad_status(tmp_path):
    _trial(tmp_path, "stock", result=False)
    _trial(tmp_path, "commit_before", status="llm_error", artifact_status="ok")
    res = G.check(tmp_path)
    assert not res["ok"]
    assert any("stock: no result.json" in r for r in res["reasons"])
    assert any("status=llm_error" in r for r in res["reasons"])


def test_protocol_failure_and_artifact(tmp_path):
    _trial(tmp_path, "stock", cmds=0, fmt=4)                       # A5.7-type: no usable tool call
    _trial(tmp_path, "commit_before", cmds=3, fmt=3, artifact_status="noncompliant")
    res = G.check(tmp_path)
    assert not res["ok"]
    joined = " | ".join(res["reasons"])
    assert "stock: solve phase ran no bash command" in joined
    assert "format errors 3 >= command steps 3" in joined
    assert "artifact_status=noncompliant" in joined


def test_no_pre_review_outcome(tmp_path):
    _trial(tmp_path, "stock", reward_pre=None, track=None)
    _trial(tmp_path, "commit_before", artifact_status="ok")
    res = G.check(tmp_path)
    assert res["reasons"] == ["stock: no pre-review verifier outcome / track"]


def test_cli_exit_code(tmp_path):
    assert G.main([str(tmp_path)]) == 1
