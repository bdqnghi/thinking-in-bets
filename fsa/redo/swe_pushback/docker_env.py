"""Docker plumbing for SWE-Pushback-Hidden: agent container, working-tree patch, hidden-test verifier.

No Harbor: the agent loop runs on the host and executes commands with ``docker exec`` in the task image
(DeepSWE 1.1 ``environment.docker_image``). The verifier is the task's own ``tests/Dockerfile`` image (hidden tests
baked in, never visible to the agent) run with ``--network none`` on the collected patch, exactly as Harbor's
separate verifier environment does (``/logs/artifacts/model.patch`` -> ``/tests/test.sh`` -> ``reward.json`` +
``ctrf.json``).

Container budget: two flock pools shared by every process on the host, ``agent`` (one slot per live agent
container) and ``verifier`` (one per running verifier container). Caps come from
``configs/swe_pushback_docker_caps.json`` (re-read on every poll) so total new containers <= agent + verifier cap.
Separate pools avoid the deadlock of agent containers all waiting for a verifier slot.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import tomllib
import uuid
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[3]
DOCKER_CAPS = Path(os.environ.get("TIB_PUSHBACK_DOCKER_CAPS", ROOT / "configs" / "swe_pushback_docker_caps.json"))
SLOT_ROOT = Path(os.environ.get("TIB_PUSHBACK_SLOT_DIR", Path.home() / ".local" / "share" / "tib-docker-slots"))
DEFAULT_CAPS = {"agent": 6, "verifier": 2}
OUT_LIMIT = 10000  # observation chars (mini-swe-agent style head/tail truncation)


def _caps() -> dict:
    try:
        d = json.loads(DOCKER_CAPS.read_text())
        return {k: int(d.get(k, v)) for k, v in DEFAULT_CAPS.items()}
    except Exception:
        return dict(DEFAULT_CAPS)


@contextlib.contextmanager
def docker_slot(pool: str) -> Iterator[int]:
    d = SLOT_ROOT / pool
    d.mkdir(parents=True, exist_ok=True)
    while True:
        n = _caps()[pool]
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            fd = os.open(d / f"slot_{i:03d}.lock", os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            try:
                os.ftruncate(fd, 0)
                os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode())
                yield i
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            return
        time.sleep(1.0 + random.random())


class Task:
    def __init__(self, task_dir: Path):
        self.dir = Path(task_dir)
        self.id = self.dir.name
        t = tomllib.loads((self.dir / "task.toml").read_text())
        self.toml = t
        self.image = t["environment"]["docker_image"]
        self.base = t["metadata"]["base_commit_hash"]
        self.agent_timeout = float(t.get("agent", {}).get("timeout_sec", 5400))
        self.verifier_timeout = float(t.get("verifier", {}).get("timeout_sec", 1800))
        self.instruction = (self.dir / "instruction.md").read_text()
        self.config = json.loads((self.dir / "tests" / "config.json").read_text())
        self.f2p = list(self.config.get("f2p_node_ids") or [])
        self.p2p = list(self.config.get("p2p_node_ids") or [])

    @property
    def verifier_image(self) -> str:
        return f"tib/dspush-verify:{self.id}"


def _run(cmd: list[str], timeout: float | None = None, input_: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, input=input_)


def ensure_verifier_image(task: Task) -> None:
    if _run(["docker", "image", "inspect", task.verifier_image]).returncode == 0:
        return
    r = _run(["docker", "build", "-q", "-t", task.verifier_image, str(task.dir / "tests")], timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"verifier build failed for {task.id}: {r.stderr.decode()[-2000:]}")


def ensure_image(image: str) -> None:
    if _run(["docker", "image", "inspect", image]).returncode == 0:
        return
    r = _run(["docker", "pull", image], timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"pull failed for {image}: {r.stderr.decode()[-2000:]}")


def truncate(s: str, limit: int = OUT_LIMIT) -> str:
    if len(s) <= limit:
        return s
    h = limit // 2
    return (s[:h] + f"\n<... {len(s) - limit} characters elided ...>\n" + s[-h:])


class Container:
    def __init__(self, task: Task, tag: str):
        self.task = task
        self.name = f"dspush-{tag}-{uuid.uuid4().hex[:6]}"[:120]
        self.alive = False

    def start(self) -> None:
        ensure_image(self.task.image)
        r = _run(["docker", "run", "-d", "--name", self.name, "--cpus", "2", "--memory", "8g", "-w", "/app",
                  "--label", "tib=swe_pushback", self.task.image, "sleep", "infinity"], timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed: {r.stderr.decode()[-2000:]}")
        self.alive = True
        self.exec("git config --global --add safe.directory /app", 60)

    def exec(self, cmd: str, timeout: int) -> tuple[int, str]:
        full = ["docker", "exec", "-w", "/app", self.name, "timeout", "-k", "5", str(timeout), "bash", "-c", cmd]
        try:
            r = _run(full, timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            return 124, f"<command timed out after {timeout}s>"
        out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
        if r.returncode == 124:
            out += f"\n<command timed out after {timeout}s>"
        return r.returncode, out

    def patch(self) -> bytes:
        """Working-tree diff vs base (tracked changes + new non-ignored files) via a throwaway index, so the
        agent's own index and commits are untouched."""
        cmd = (f"cd /app && export GIT_INDEX_FILE=$(mktemp -u /tmp/.dspush_idx.XXXX) && git read-tree HEAD && "
               f"git add -A . >/dev/null 2>&1; git diff --cached --binary {self.task.base}; rc=$?; "
               f"rm -f $GIT_INDEX_FILE; exit $rc")
        r = _run(["docker", "exec", self.name, "bash", "-c", cmd], timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"patch collection failed: {r.stderr.decode()[-1000:]}")
        return r.stdout

    def stop(self) -> None:
        if self.alive:
            _run(["docker", "rm", "-f", self.name], timeout=300)
            self.alive = False


def verify(task: Task, patch: bytes, out_dir: Path) -> dict:
    """Run the hidden tests on ``patch``. Returns {reward, statuses{node_id: passed|failed|skipped}, ...}.
    Infrastructure failure -> {"error": ...} (never silently a pass)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.patch").write_bytes(patch)
    ensure_verifier_image(task)
    tmp = Path(tempfile.mkdtemp(prefix="dspush_v_"))
    (tmp / "artifacts").mkdir()
    (tmp / "verifier").mkdir()
    (tmp / "artifacts" / "model.patch").write_bytes(patch)
    os.chmod(tmp / "verifier", 0o777)
    name = f"dspush-v-{task.id[:40]}-{uuid.uuid4().hex[:6]}"
    t0 = time.time()
    res: dict = {}
    with docker_slot("verifier"):
        try:
            r = _run(["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", "2", "--memory", "8g",
                      "--label", "tib=swe_pushback",
                      "-v", f"{tmp / 'artifacts'}:/logs/artifacts", "-v", f"{tmp / 'verifier'}:/logs/verifier",
                      task.verifier_image, "bash", "-c",
                      "bash /tests/test.sh; rc=$?; chmod -R a+rwX /logs/verifier /logs/artifacts; exit $rc"],
                     timeout=task.verifier_timeout + 300)
            stdout = (r.stdout + r.stderr).decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            _run(["docker", "rm", "-f", name], timeout=120)
            stdout = "<verifier timeout>"
    res["verifier_sec"] = round(time.time() - t0, 1)
    (out_dir / "test-stdout.txt").write_text(stdout[-200000:])
    rj = tmp / "verifier" / "reward.json"
    cj = tmp / "verifier" / "ctrf.json"
    if rj.exists():
        reward = json.loads(rj.read_text())
        shutil.copy(rj, out_dir / "reward.json")
        res.update(reward=int(reward.get("reward", 0)), reward_doc=reward,
                   apply_failed=bool(reward.get("apply_failed")))
        statuses: dict[str, str] = {}
        if cj.exists():
            shutil.copy(cj, out_dir / "ctrf.json")
            for t in (json.loads(cj.read_text()).get("results") or {}).get("tests") or []:
                nm = str(t.get("name", ""))
                # grader names rows "[f2p] <id>" / "[p2p] <id>"
                if nm.startswith("[") and "] " in nm:
                    nm = nm.split("] ", 1)[1]
                statuses[nm] = t.get("status", "failed")
        for nid in task.f2p + task.p2p:  # absence == failure (grader rule)
            statuses.setdefault(nid, "failed")
        res["statuses"] = statuses
    else:
        res["error"] = "no reward.json (verifier infrastructure failure)"
    shutil.rmtree(tmp, ignore_errors=True)
    (out_dir / "verify.json").write_text(json.dumps(res, indent=1))
    return res
