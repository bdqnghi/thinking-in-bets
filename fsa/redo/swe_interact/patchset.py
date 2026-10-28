"""Record / verify the local SWE-Interact patches (two layers) and the Harbor venv.

    python -m fsa.redo.swe_interact.patchset write    # clone diff over (pin + arm64.patch) -> redo_v1.patch (arm64 hosts)
    python -m fsa.redo.swe_interact.patchset verify   # clone == pin + layer 1 + redo_v1.patch; harbor pinned + patched
    python -m fsa.redo.swe_interact.patchset apply    # fresh clone at the pin -> apply layer 1, then layer 2
    python -m fsa.redo.swe_interact.patchset write-server  # any host: re-splice only the user-server/server.py chunks

Layer 1 depends on the host architecture (never an emulated image):

* arm64 hosts: ``arm64.patch``, a verbatim copy of swe_benchmark_arm's ``swe-interact-arm64.patch``
  (task.toml ``docker_image = bdqnghi/swe-interact:<tag>``, arm64 Dockerfiles); its sha256 is
  pinned in ``rungs.json``.
* amd64 hosts: ``amd64.patch``, generated from the pin by ``amd64.py`` (task.toml ``docker_image``
  of ``amd64.patch_tasks()`` - the Stage-1 subset + the 25 swebenchpro tasks, A5.13 (1) - -> native images
  built from the published images plus the task Dockerfiles' multi-turn additions, see that module);
  ``verify`` checks the file equals a fresh generation and the sha256 pinned in ``rungs.json``.

Layer 2, ``redo_v1.patch``: our changes on top (``tree.py``), recorded against a scratch index
holding pin + ``arm64.patch``, so it never repeats layer 1 (New files are included, ``git diff
--no-index``). The same file applies unchanged on pin + ``amd64.patch`` (its hunks do not touch the
lines either layer 1 changes), so on amd64 ``verify`` checks that the working tree equals
pin + ``amd64.patch`` + ``redo_v1.patch`` (no residual diff against a scratch index holding exactly
that). Every run manifest records both layers and the Harbor provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import machine
from .config import CLONE, PATCH_DIR, ROOT, VENV, load_rungs

ARM64 = PATCH_DIR / "arm64.patch"
AMD64 = PATCH_DIR / "amd64.patch"
REDO = PATCH_DIR / "redo_v1.patch"
SITE = VENV / "lib" / "python3.12" / "site-packages"
HARBOR_MULTISTEP = SITE / "harbor" / "trial" / "multi_step.py"
HARBOR_VERIFIER = SITE / "harbor" / "verifier" / "verifier.py"


def _git(*args: str, env: dict | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(CLONE), *args], capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    if check and r.returncode not in (0, 1):
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr}")
    return r.stdout


def sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def layer1(arch: str | None = None) -> Path:
    """The architecture layer for this host: ``arm64.patch`` on arm64, ``amd64.patch`` on amd64."""
    return ARM64 if (arch or machine.host_arch()) == "arm64" else AMD64


def _diff_over(patches: list[Path]) -> str:
    """Working tree (incl. untracked, non-ignored files) against a scratch index = pin + ``patches``."""
    pin = load_rungs()["benchmark"]["commit"]
    with tempfile.TemporaryDirectory() as td:
        env = {"GIT_INDEX_FILE": str(Path(td) / "index")}
        _git("read-tree", pin, env=env)
        for i, p in enumerate(patches):
            r = subprocess.run(["git", "-C", str(CLONE), "apply", "--cached", "--whitespace=nowarn", str(p)],
                               capture_output=True, text=True, env={**os.environ, **env})
            if r.returncode:
                raise SystemExit(f"{p.name} does not apply to the pin (+ {[q.name for q in patches[:i]]}): {r.stderr}")
        out = _git("diff", "--binary", env=env)
        for f in sorted(_git("ls-files", "--others", "--exclude-standard", env=env).split("\n")):
            if f:
                out += _git("diff", "--no-index", "--binary", "/dev/null", f)
    return out


def redo_diff() -> str:
    """Layer 2 as recorded: the working tree against pin + ``arm64.patch`` (arm64 layout)."""
    return _diff_over([ARM64])


def splice_server_chunks(text: str) -> tuple[str, int]:
    """``redo_v1.patch`` with its ``*/environment/user-server/server.py`` chunks replaced by fresh
    ``git diff --binary <pin> -- <those paths>`` of the clone (after ``tree apply``). Neither layer 1 touches
    server.py, so these chunks are relative to the pin on both layouts; the other chunks are kept byte for byte (A5.14:
    the sidecar gate changed ``overlay/server.py`` on an amd64 host, where ``write`` refuses). Returns (patch, n)."""
    import re
    pin = load_rungs()["benchmark"]["commit"]
    parts = re.split(r"(?m)^(?=diff --git )", text)
    head, chunks = parts[0], parts[1:]
    paths = [re.match(r"diff --git a/(\S+) b/", c).group(1) for c in chunks]
    srv = [q for q in paths if q.endswith("/environment/user-server/server.py")]
    fresh = _git("diff", "--binary", pin, "--", *srv) if srv else ""
    fmap = {re.match(r"diff --git a/(\S+) b/", c).group(1): c for c in re.split(r"(?m)^(?=diff --git )", fresh)[1:]}
    missing = sorted(set(srv) - set(fmap))
    if missing:
        raise SystemExit(f"server.py unchanged vs the pin in the clone (run `tree apply` first): {missing[:3]}")
    return head + "".join(fmap.get(q, c) for q, c in zip(paths, chunks)), len(srv)


def harbor_provenance() -> dict:
    cfg = load_rungs()["harbor"]
    exe = VENV / "bin" / "python"
    ver = subprocess.run([str(exe), "-c", "import importlib.metadata as m; print(m.version('harbor'))"],
                         capture_output=True, text=True).stdout.strip()
    ms = HARBOR_MULTISTEP.read_text() if HARBOR_MULTISTEP.exists() else ""
    vf = HARBOR_VERIFIER.read_text() if HARBOR_VERIFIER.exists() else ""
    return {"harbor_version": ver, "harbor_venv": str(VENV.relative_to(ROOT)),
            "harbor_multistep_patch_applied": "prepare_logs_for_host()" in ms,
            "harbor_redo_patch_applied": ms.count("[tib redo_v1]") == 1 and vf.count("[tib redo_v1]") == 1,
            "harbor_multistep_py_sha256": sha256(ms), "harbor_verifier_py_sha256": sha256(vf),
            "harbor_patch_sha256": sha256((ROOT / cfg["patch"]).read_bytes()),
            "harbor_patches_sha256": {p["file"]: sha256((ROOT / p["file"]).read_bytes()) for p in cfg.get("patches", [])},
            "harbor_freeze_sha256": sha256((ROOT / cfg["freeze"]).read_bytes())}


def verify() -> dict:
    cfg = load_rungs()
    pin = cfg["benchmark"]["commit"]
    head = _git("rev-parse", "HEAD").strip()
    if head != pin:
        raise SystemExit(f"swe-interact clone HEAD {head} != pinned {pin}")
    recorded = REDO.read_text() if REDO.exists() else ""
    l1 = layer1()
    if l1 == ARM64:
        arm = l1.read_bytes()
        if sha256(arm) != cfg["benchmark"]["arm64_patch_sha256"]:
            raise SystemExit("arm64.patch differs from the sha256 pinned in rungs.json")
        live = redo_diff()
        if live != recorded:
            raise SystemExit(f"swe-interact working tree differs from {REDO.name} (live {sha256(live)[:12]} vs recorded "
                             f"{sha256(recorded)[:12]}); run `python -m fsa.redo.swe_interact.patchset write` and commit it")
        layers = {"layer1": str(l1.relative_to(ROOT)), "layer1_sha256": sha256(arm), "arm64_patch_sha256": sha256(arm),
                  "layer2_check": "working tree - (pin + arm64.patch) == redo_v1.patch"}
    else:
        from .amd64 import expected_patch
        amd = l1.read_text() if l1.exists() else ""
        if sha256(amd) != cfg["benchmark"].get("amd64_patch_sha256"):
            raise SystemExit("amd64.patch is missing or differs from the sha256 pinned in rungs.json "
                             "(benchmark.amd64_patch_sha256)")
        if expected_patch() != amd:
            raise SystemExit("amd64.patch differs from a fresh generation (python -m fsa.redo.swe_interact.amd64 patch)")
        residual = _diff_over([AMD64, REDO])
        if residual:
            raise SystemExit(f"swe-interact working tree != pin + {AMD64.name} + {REDO.name} (amd64 layout); residual "
                             f"diff {sha256(residual)[:12]}, {len(residual)} bytes. Rebuild: checkout the pin, "
                             "`python -m fsa.redo.swe_interact.patchset apply`")
        layers = {"layer1": str(l1.relative_to(ROOT)), "layer1_sha256": sha256(amd), "arm64_patch_sha256": None,
                  "layer2_check": "working tree == pin + amd64.patch + redo_v1.patch (native amd64 images, amd64.py)"}
    hp = harbor_provenance()
    if (hp["harbor_version"] != cfg["harbor"]["version"] or not hp["harbor_multistep_patch_applied"]
            or not hp["harbor_redo_patch_applied"]):
        raise SystemExit(f"harbor venv not as pinned: {hp}")
    return {"benchmark_commit": head, "host_arch": machine.host_arch(), **layers,
            "patch_file": str(REDO.relative_to(ROOT)), "patch_sha256": sha256(recorded), **hp}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "verify", "apply", "write-server"])
    a = ap.parse_args(argv)
    if a.cmd == "write":
        if layer1() != ARM64:
            raise SystemExit("layer 2 is recorded on the arm64 layout (pin + arm64.patch); on this amd64 host "
                             "`verify` checks the tree against pin + amd64.patch + redo_v1.patch instead")
        d = redo_diff()
        REDO.write_text(d)
        print(f"wrote {REDO} ({len(d)} bytes, sha256 {sha256(d)})")
    elif a.cmd == "verify":
        print(verify())
    elif a.cmd == "write-server":
        d, n = splice_server_chunks(REDO.read_text())
        REDO.write_text(d)
        print(f"wrote {REDO} ({n} server.py chunks re-spliced, {len(d)} bytes, sha256 {sha256(d)})")
    else:
        for p in (layer1(), REDO):
            r = subprocess.run(["git", "-C", str(CLONE), "apply", "--whitespace=nowarn", str(p)])
            if r.returncode:
                return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
