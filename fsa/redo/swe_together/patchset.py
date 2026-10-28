"""Record / verify the local SWE-Together patch.

    python -m fsa.redo.swe_together.patchset write    # clone diff -> patch file
    python -m fsa.redo.swe_together.patchset verify   # clone == pinned commit + patch file
    python -m fsa.redo.swe_together.patchset apply    # fresh clone at the pin -> apply patch
    ... [--clone-dir DIR --patch-file FILE]            # another copy / revision (default: SWT_CLONE_DIR /
                                                       # SWT_PATCH_FILE, unset = the pinned pair)

The patch is ``git diff <pin>`` of tracked files plus ``git diff --no-index`` for
every untracked, non-ignored file, so new modules are included.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys

from pathlib import Path

from .config import DEFAULT, Revision, load_rungs


def _git(*args: str, check: bool = True, rev: Revision | None = None) -> str:
    r = subprocess.run(["git", "-C", str((rev or DEFAULT).clone), *args], capture_output=True, text=True)
    if check and r.returncode not in (0, 1):
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr}")
    return r.stdout


def current_diff(rev: Revision | None = None) -> str:
    pin = load_rungs()["benchmark"]["commit"]
    out = _git("diff", "--binary", pin, rev=rev)
    for f in sorted(_git("ls-files", "--others", "--exclude-standard", rev=rev).split()):
        out += _git("diff", "--no-index", "--binary", "/dev/null", f, rev=rev)
    return out


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def head(rev: Revision | None = None) -> str:
    return _git("rev-parse", "HEAD", rev=rev).strip()


def verify(rev: Revision | None = None) -> dict:
    """Return provenance; raise if the clone copy is not exactly pin + its recorded patch revision."""
    rev = rev or DEFAULT
    PATCH_FILE = rev.patch_file
    pin = load_rungs()["benchmark"]["commit"]
    h = head(rev)
    if h != pin:
        raise SystemExit(f"swe-together clone HEAD {h} != pinned {pin}")
    recorded = PATCH_FILE.read_text() if PATCH_FILE.exists() else ""
    live = current_diff(rev)
    if live != recorded:
        raise SystemExit(
            f"swe-together working tree differs from {PATCH_FILE.name} "
            f"(live {sha256(live)[:12]} vs recorded {sha256(recorded)[:12]}); "
            "run `python -m fsa.redo.swe_together.patchset write` and commit it, or revert the clone"
        )
    out = {"benchmark_commit": h, "patch_file": str(PATCH_FILE.relative_to(PATCH_FILE.parents[3])),
           "patch_sha256": sha256(recorded)}
    if not rev.pinned:  # the pinned pair keeps the historical record shape
        out["clone_dir"] = str(rev.clone)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "verify", "apply"])
    ap.add_argument("--clone-dir", default=None)
    ap.add_argument("--patch-file", default=None)
    a = ap.parse_args(argv)
    rev = Revision(Path(a.clone_dir or DEFAULT.clone).resolve(), Path(a.patch_file or DEFAULT.patch_file).resolve())
    PATCH_FILE = rev.patch_file
    if a.cmd == "write":
        PATCH_FILE.parent.mkdir(parents=True, exist_ok=True)
        d = current_diff(rev)
        PATCH_FILE.write_text(d)
        print(f"wrote {PATCH_FILE} ({len(d)} bytes, sha256 {sha256(d)})")
    elif a.cmd == "verify":
        print(verify(rev))
    else:
        r = subprocess.run(["git", "-C", str(rev.clone), "apply", "--whitespace=nowarn", str(PATCH_FILE)])
        return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
