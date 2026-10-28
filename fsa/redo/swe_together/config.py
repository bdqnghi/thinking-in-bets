"""Paths and the rung/seat configuration for the SWE-Together runner."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# SWT_CLONE_DIR / SWT_PATCH_FILE select another copy of the clone and its recorded patch revision
# (e.g. the baseline-port revision developed beside a running pilot); unset = the pinned pair.
CLONE = Path(os.environ.get("SWT_CLONE_DIR") or ROOT / "third_party" / "swe-together").resolve()
PATCH_FILE = Path(os.environ.get("SWT_PATCH_FILE") or ROOT / "third_party" / "patches" / "swe-together" / "redo_v1.patch").resolve()
RUNGS_FILE = Path(__file__).with_name("rungs.json")
RESULTS = ROOT / "results" / "redo_v1"
PY = CLONE / ".venv" / "bin" / "python"
PATCH_DIR = ROOT / "third_party" / "patches" / "swe-together"


@dataclass(frozen=True)
class Revision:
    """One copy of the benchmark clone and the recorded patch revision it must equal (pin + patch)."""
    clone: Path
    patch_file: Path

    @property
    def py(self) -> Path:
        return self.clone / ".venv" / "bin" / "python"

    @property
    def pinned(self) -> bool:
        return self == PINNED

    def env(self) -> dict[str, str]:
        """Trial / subprocess environment that makes tib code inside the trial resolve this copy."""
        return {} if self.pinned else {"SWT_CLONE_DIR": str(self.clone), "SWT_PATCH_FILE": str(self.patch_file)}


PINNED = Revision((ROOT / "third_party" / "swe-together").resolve(), (PATCH_DIR / "redo_v1.patch").resolve())
DEFAULT = Revision(CLONE, PATCH_FILE)  # the env-selected pair (unset env = PINNED)
# Stage 1.2 (2026-09-28): the pinned Stage 1.1 revision plus SWT_REDO_TRIGGER=agentic_v3 (the frozen overlay
# trigger); a separate copy so the running natural-cell launcher's pinned clone is never touched. Overlay cells.
TRIGGER_V3 = Revision((RESULTS / "trigger_v3_work" / "swe-together").resolve(),
                      (PATCH_DIR / "redo_v1_trigger.patch").resolve())


def load_rungs() -> dict:
    return json.loads(RUNGS_FILE.read_text())


def subset_tasks() -> list[str]:
    cfg = load_rungs()
    sub = json.loads((ROOT / cfg["benchmark"]["subset"]).read_text())
    return [t["task"] for t in sub["selected"]]


def official_image(task: str, clone: Path | None = None) -> str:
    """The task's official (amd64) image: ``[environment].docker_image`` in its task.toml."""
    import tomllib
    return tomllib.loads(((clone or CLONE) / "tasks" / task / "task.toml").read_text())["environment"]["docker_image"]
