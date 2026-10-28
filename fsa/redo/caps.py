"""Operator caps files for the pilot launchers (``--caps-file``).

A caps file is a JSON object whose keys are a subset of the launcher's cap names (per-rung
``W``/``M``/``S`` and global limits such as ``max_total``) and whose values are non-negative
integers. Launchers read it at start and poll it while trials run, so concurrency can follow
the provider quota without draining running trials. A file that does not parse or validate
is rejected as a whole (the caps in force stay unchanged).
"""
from __future__ import annotations

import json
from pathlib import Path


def read_caps_file(path: Path, keys: tuple[str, ...]) -> dict[str, int] | None:
    """The caps in ``path`` (subset of ``keys``, non-negative ints); None when the file is
    absent. Raises ValueError for anything else, so a bad edit never changes the caps."""
    try:
        spec = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"unreadable: {e}") from e
    if not isinstance(spec, dict) or not spec or set(spec) - set(keys):
        raise ValueError(f"expected an object with keys from {keys}, got {spec!r}")
    out = {}
    for k, v in spec.items():
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"{k}: expected a non-negative integer, got {v!r}")
        out[k] = v
    return out
