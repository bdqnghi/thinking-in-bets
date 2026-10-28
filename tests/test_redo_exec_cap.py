"""exec_with_budget (SWE-Together clone, redo_v1 patch): a backend-raised per-exec timeout
(Harbor docker) is a cap like an asyncio timeout, so the wrapper's cap-rescue path runs."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
import types

import pytest

from fsa.redo.swe_together.config import CLONE

PATH = CLONE / "src" / "user_agent" / "exec_helpers.py"
pytestmark = pytest.mark.skipif(not PATH.exists(), reason="SWE-Together clone not present")


@pytest.fixture(scope="module")
def eh():
    spec = importlib.util.spec_from_file_location("swt_exec_helpers", PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swt_exec_helpers"] = mod
    spec.loader.exec_module(mod)
    return mod


class Env:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.seen = None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None):
        self.seen = (command, timeout_sec)
        if self.behaviour == "docker_timeout":  # harbor docker.py:231
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        if self.behaviour == "other_error":
            raise RuntimeError("docker compose exec failed: no such service")
        return types.SimpleNamespace(return_code=0, stdout="ok", stderr="")


def run(eh, behaviour):
    env = Env(behaviour)
    inp = types.SimpleNamespace(command="echo hi", cwd=None, env={})
    return env, asyncio.run(eh.exec_with_budget(env, inp, start_time=time.monotonic()))


def test_docker_backend_timeout_is_a_cap(eh):
    env, (res, timed_out) = run(eh, "docker_timeout")
    assert timed_out is True and res.return_code == -1 and res.stdout == ""
    assert "capped" in res.stderr and "sandbox backend" in res.stderr
    assert env.seen[0].startswith("set -o pipefail; ")


def test_other_runtime_errors_still_raise(eh):
    with pytest.raises(RuntimeError, match="no such service"):
        run(eh, "other_error")


def test_normal_exec_unchanged(eh):
    _, (res, timed_out) = run(eh, "ok")
    assert timed_out is False and res.stdout == "ok"


def test_asyncio_timeout_path_unchanged(eh, monkeypatch):
    class Hang:
        async def exec(self, **kw):
            await asyncio.sleep(10)
    monkeypatch.setattr(eh, "PER_EXEC_CAP_SEC", 0)  # cap_sec = 0 -> wait_for(timeout=30); shorten via patch
    async def go():
        orig = asyncio.wait_for
        async def fast(aw, timeout):
            return await orig(aw, timeout=0.05)
        monkeypatch.setattr(eh.asyncio, "wait_for", fast)
        inp = types.SimpleNamespace(command="x", cwd=None, env={})
        return await eh.exec_with_budget(Hang(), inp, start_time=time.monotonic())
    res, timed_out = asyncio.run(go())
    assert timed_out is True and "abandoning call" in res.stderr
