"""Paths and rung/seat configuration for the SWE-Interact runner (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLONE = ROOT / "third_party" / "swe-interact"
TASKS = CLONE / "data" / "multiturn"
PATCH_DIR = ROOT / "third_party" / "patches" / "swe-interact"
RUNGS_FILE = Path(__file__).with_name("rungs.json")
OVERLAY = Path(__file__).with_name("overlay")
# Results tree. TIB_RESULTS_ROOT points a runner at another tree (the exploratory frontier rung uses
# results/redo_v1_g4, whose HOST.json records its own endpoint profile); unset: the pre-registered tree.
CANONICAL_RESULTS = ROOT / "results" / "redo_v1"
RESULTS = Path(os.environ["TIB_RESULTS_ROOT"]).resolve() if os.environ.get("TIB_RESULTS_ROOT") else CANONICAL_RESULTS
VENV = CLONE / ".venv"
HARBOR = VENV / "bin" / "harbor"


def load_rungs() -> dict:
    return json.loads(RUNGS_FILE.read_text())


def subset_file() -> Path:
    return ROOT / load_rungs()["benchmark"]["subset"]


def subset_tasks() -> list[str]:
    cfg = load_rungs()["benchmark"]
    return list(json.loads(subset_file().read_text())[cfg["subset_key"]])


def subset_digest() -> dict:
    doc = json.loads(subset_file().read_text())
    return {"path": str(subset_file().relative_to(ROOT)),
            "file_sha256": hashlib.sha256(subset_file().read_bytes()).hexdigest(),
            "frozen_subset_digest": doc.get("frozen_subset_digest"),
            "effective_digest": doc.get("effective_digest"),
            "tasks": doc["effective_subset"]}


def family(task: str) -> str:
    return task.split("_", 1)[0]


def repo_root(task: str) -> str:
    """Repository root inside the task container (REPO_EXEC_ROOT in the task's compose; default /app)."""
    m = re.search(r"REPO_EXEC_ROOT=(\S+)", (TASKS / task / "environment" / "docker-compose.yaml").read_text())
    return m.group(1) if m else "/app"


def task_docker_image(task: str) -> str | None:
    """task.toml ``environment.docker_image`` (a prebuilt image), or None (built from the Dockerfile)."""
    import tomllib
    return tomllib.loads((TASKS / task / "task.toml").read_text())["environment"].get("docker_image") or None


# Harbor 0.23.0 (environments/definition.py::environment_content_hash; docker.py names a build
# ``hb__<hash>``, which its name sanitizer leaves unchanged): a task without ``docker_image`` is
# built from ``environment/`` and named by the content of that directory. (Such tasks that also have
# their own docker-compose.yaml - all of SWE-Interact's build-based tasks - are built in each trial by
# ``docker compose build``, tagged ``<compose project>-main``: the same context, so the same cached
# layers.)
_HASH_IGNORE = frozenset({".DS_Store", ".git", "__pycache__"})


def environment_content_hash(env_dir: Path, truncate: int = 32) -> str:
    files = sorted((p.relative_to(env_dir).as_posix(), p) for p in env_dir.rglob("*")
                   if p.is_file() and not p.is_symlink() and not (_HASH_IGNORE & set(p.relative_to(env_dir).parts)))
    if not files:
        return hashlib.sha256(env_dir.name.encode()).hexdigest()[:truncate]
    d = hashlib.sha256()
    for rel, p in files:
        rb, data = rel.encode(), p.read_bytes()
        d.update(len(rb).to_bytes(4, "big"))
        d.update(rb)
        d.update(len(data).to_bytes(4, "big"))
        d.update(data)
    return d.hexdigest()[:truncate]


def task_image(task: str) -> str:
    """The task's main image: task.toml ``docker_image`` (arm64 hosts: the arm64 layer's rebuilds;
    amd64 hosts: the amd64 layer's native builds, ``amd64.py``; the pin alone: the benchmark's published
    images), else the build of the task's ``environment/`` under Harbor's content-addressed name
    (``hb__<environment content hash>``; ``run.build_task_image``), which judges use and which shares
    its layers with each trial's build."""
    return task_docker_image(task) or f"hb__{environment_content_hash(TASKS / task / 'environment')}"
