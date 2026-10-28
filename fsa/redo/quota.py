"""Shared per-model concurrency slots across processes (P0.3, PARALLEL_REVIEWER_EXPERIMENTS.md §2/§3).

Every new reviewer-experiment process wraps its single LLM-call function in ``slot(model)``. A slot is an
``fcntl.flock`` on one of ``cap`` lock files under ``~/.local/share/tib-quota/<model>/``, so the total number of
in-flight requests per model is bounded across *all* processes, and a crashed process frees its slots
automatically (the kernel drops the flock with the file descriptor).

Caps live in ``configs/quota_caps.json`` (keys: model ids; values: non-negative ints), validated with
``fsa.redo.caps.read_caps_file`` and re-read every ``REFRESH_S`` seconds, so an operator edit (halve a lane on
429s, ramp +2 after a clean 2 h) takes effect without restarting anything. A file that does not validate keeps
the caps in force (and the last good caps are used). A model missing from the file gets ``DEFAULT_CAP``.
A ``model@effort`` id (effort bridge suffix) shares the base model's slots.

Lowering a cap does not interrupt in-flight calls; slots with index >= new cap simply stop being handed out.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import random
import threading
import time
from pathlib import Path
from typing import Iterator

from fsa.redo.caps import read_caps_file

ROOT = Path(__file__).resolve().parents[2]
CAPS_PATH = Path(os.environ.get("TIB_QUOTA_CAPS", ROOT / "configs" / "quota_caps.json"))
LOCK_ROOT = Path(os.environ.get("TIB_QUOTA_DIR", Path.home() / ".local" / "share" / "tib-quota"))
KNOWN_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-4.0-argon",
)
DEFAULT_CAP = 2
REFRESH_S = 10.0
POLL_S = 0.25

_lock = threading.Lock()
_caps: dict[str, int] = {}
_caps_at = 0.0
_caps_error: str | None = None


def base_model(model: str) -> str:
    return model.split("@", 1)[0]


def _refresh() -> None:
    global _caps, _caps_at, _caps_error
    now = time.time()
    if now - _caps_at < REFRESH_S:
        return
    _caps_at = now
    try:
        got = read_caps_file(CAPS_PATH, KNOWN_MODELS)
    except ValueError as e:  # keep the caps in force
        _caps_error = str(e)
        return
    _caps_error = None
    if got is not None:
        _caps = got


def cap(model: str) -> int:
    with _lock:
        _refresh()
        return _caps.get(base_model(model), DEFAULT_CAP)


def caps_status() -> dict:
    with _lock:
        _refresh()
        return dict(caps=dict(_caps), error=_caps_error, path=str(CAPS_PATH))


def _model_dir(model: str) -> Path:
    d = LOCK_ROOT / base_model(model).replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextlib.contextmanager
def slot(model: str, timeout: float | None = None) -> Iterator[int]:
    """Hold one of the model's ``cap`` slots for the duration of the block; yields the slot index.

    Blocks (polling) until a slot is free. ``timeout`` (seconds) raises TimeoutError; None waits forever.
    A cap of 0 pauses the lane: callers wait until the cap is raised.
    """
    d = _model_dir(model)
    t0 = time.time()
    while True:
        n = cap(model)
        order = list(range(n))
        random.shuffle(order)  # spread contention over the slot files
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
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
            return
        if timeout is not None and time.time() - t0 > timeout:
            raise TimeoutError(f"no free quota slot for {model} (cap={n}) within {timeout}s")
        time.sleep(POLL_S * (0.5 + random.random()))


def in_flight(model: str, upto: int = 64) -> int:
    """Number of currently held slots for ``model`` (probe; for monitoring only)."""
    d = _model_dir(model)
    held = 0
    for i in range(upto):
        p = d / f"slot_{i:03d}.lock"
        if not p.exists():
            continue
        fd = os.open(p, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            held += 1
        finally:
            os.close(fd)
    return held


if __name__ == "__main__":
    import json

    st = caps_status()
    st["in_flight"] = {m: in_flight(m) for m in KNOWN_MODELS}
    print(json.dumps(st, indent=1))
