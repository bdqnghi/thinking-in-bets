"""Hermetic defaults for the test suite."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _public_machine_profile(monkeypatch):
    """Every test sees the public Gemini API profile (``fsa.redo.machine``), whatever this machine's
    ``configs/machines/local.json`` selects; tests of other profiles set ``TIB_MACHINE_PROFILE``."""
    monkeypatch.setenv("TIB_MACHINE_PROFILE", "gemini_api")
    for k in ("GEMINI_API_BASE", "SWT_GOOGLE_BASE_URL", "GOOGLE_BASE_URL"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(autouse=True)
def _hermetic_quota(monkeypatch, tmp_path_factory):
    """Runner code takes ``fsa.redo.quota`` slots around every model call. Tests use a private lock directory and
    generous caps, so they never wait on (or take) the live slots held by running experiment lanes."""
    import json

    import fsa.redo.quota as q
    d = tmp_path_factory.mktemp("quota")
    caps = d / "caps.json"
    caps.write_text(json.dumps({m: 64 for m in q.KNOWN_MODELS}))
    monkeypatch.setattr(q, "LOCK_ROOT", d / "locks")
    monkeypatch.setattr(q, "CAPS_PATH", caps)
    monkeypatch.setattr(q, "_caps_at", 0.0)
    monkeypatch.setattr(q, "_caps", {})
