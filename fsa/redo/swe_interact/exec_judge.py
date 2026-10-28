"""Executable per-snapshot test runs for SWE-Interact (redo_v1 Stage 1 ground truth).

A repository snapshot (the cumulative diff after an agent turn, ``user-sim/snapshots/ask-N.patch``
or ``steps/<step>/agent/redo/step-K-end.patch``) is applied to a FRESH container of the task's
arm64 image and the task's hidden tests are run with the final-step verifier's own recipe
(``steps/05_test_handoff/tests`` uploaded as ``/tests``, exactly what Harbor uploads):

* ``deepswe``: canonical_test.sh steps 1-2 (reset the files the hidden ``test.patch`` touches to
  HEAD, apply ``test.patch``), then ``test.sh new``. For per-test visibility the ``new`` script
  is run in a *per-test variant* (``PER_TEST_EDITS``): ``set -e`` -> ``set +e`` and ``-x`` removed
  (so one failure does not hide the rest), ``go test -v``, mocha ``--reporter tap``, pytest
  ``-rA`` (``PYTEST_ADDOPTS``). Pass/fail of every individual test is unchanged by these edits.
  (``test.sh base`` - the pre-existing suite - is not run: it maps to no intent.)
* ``swebenchpro``: ``canonical_test.sh`` verbatim (gold test-file checkout from
  ``before_repo_set_cmd``, ``run_script.sh`` on ``selected_test_files_to_run``, the task's own
  ``parser.py``) -> per-test statuses from ``/tmp/output.json``.
* ``rf``: ``canonical_test.sh`` verbatim without grader credentials (the rubric part is graded
  by ``rubric_judge`` instead): master validator ``--mode agent --relax`` -> ``tests_reward``
  (no relevant pass->fail / missing->fail test) and the filtered comparison.

The snapshot must have been taken against the image's HEAD (the anchor the agent wrapper
records before step 1); the run records ``head`` so this is checkable (``anchor_mismatch``).
Results are cached by sha256(recipe version, task, patch bytes): identical snapshots run once.
Every run keeps its full log next to the cache entry. Containers are labelled
``tib.swi_exec=1`` and always removed; nothing else on the docker host is touched.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import tomllib
import uuid
from pathlib import Path

from .config import RESULTS, TASKS, family, repo_root, task_image

RECIPE_VERSION = "swi-exec/1"
CACHE = RESULTS / "swi_exec_cache"
FINAL_STEP = "05_test_handoff"
DEFAULT_TIMEOUT = {"deepswe": 1800, "swebenchpro": 1800, "rf": 5400}
RF_CPUS = 8          # rf tasks ask for 16 CPU; the judge uses 8 (recorded) so it can run beside trials
_LOCAL = threading.local()

PER_TEST_EDITS = [
    # (regex, replacement) applied to test.sh before `new` runs; recorded in every result
    (r"(?m)^\s*set -e[a-z]*( -o pipefail)?\s*$", "set +e"),
    (r"(?m)^\s*set -euo pipefail\s*$", "set +e"),
    (r"(?<=\s)(-x|--exitfirst)(?=\s|$)", ""),
    (r"\bgo test(?! -v)\b", "go test -v"),
    (r"(node_modules/\.bin/mocha)\b", r"\1 --reporter tap"),
]

DEEPSWE_SCRIPT = r'''
set -uo pipefail
mkdir -p /logs/agent /logs/artifacts /logs/verifier /tmp/swi
cd "$SWI_ROOT"
git config --global --add safe.directory '*' 2>/dev/null || true
echo "HEAD=$(git rev-parse HEAD)" > /tmp/swi/head.txt
if [ -s /tmp/swi/snap.patch ]; then
  git apply --binary --whitespace=nowarn /tmp/swi/snap.patch || { echo SWI_APPLY_FAILED; exit 90; }
fi
python3 -c '
import re
patch = open("/tests/test.patch", encoding="utf-8").read()
files = set()
for line in patch.splitlines():
    m = re.match(r"^diff --git \"?a/.+ \"?b/(.+?)\"?$", line)
    if m:
        files.add(m.group(1))
for f in sorted(files):
    print(f)
' | while IFS= read -r f; do
  if git checkout HEAD -- "$f" 2>/dev/null; then :; else
    git rm -r --cached --ignore-unmatch -- "$f" >/dev/null 2>&1 || true
    [ -e "$f" ] && rm -rf "$f"
  fi
done
git apply --whitespace=nowarn /tests/test.patch || { echo SWI_TEST_PATCH_FAILED; exit 91; }
cp /tmp/swi/test_pertest.sh ./test.sh && chmod +x ./test.sh
echo SWI_NEW_BEGIN
PYTEST_ADDOPTS="-rA" timeout "$SWI_TIMEOUT" bash ./test.sh new
echo "SWI_NEW_RC=$?"
'''

CANONICAL_SCRIPT = r'''
set -uo pipefail
mkdir -p /logs/agent /logs/artifacts /logs/verifier /tmp/swi
cd "$SWI_ROOT"
git config --global --add safe.directory '*' 2>/dev/null || true
echo "HEAD=$(git rev-parse HEAD)" > /tmp/swi/head.txt
if [ -s /tmp/swi/snap.patch ]; then
  git apply --binary --whitespace=nowarn /tmp/swi/snap.patch || { echo SWI_APPLY_FAILED; exit 90; }
fi
echo SWI_CANONICAL_BEGIN
timeout "$SWI_TIMEOUT" bash /tests/canonical_test.sh
echo "SWI_CANONICAL_RC=$?"
'''


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


def tests_dir(task: str) -> Path:
    return TASKS / task / "steps" / FINAL_STEP / "tests"


def task_resources(task: str) -> tuple[int, int]:
    env = tomllib.loads((TASKS / task / "task.toml").read_text())["environment"]
    return int(env.get("cpus") or 2), int(env.get("memory_mb") or 8192)


def per_test_script(task: str) -> str:
    text = _extract_test_sh(task)
    for pat, rep in PER_TEST_EDITS:
        text = re.sub(pat, rep, text)
    return text


def _extract_test_sh(task: str) -> str:
    """test.sh as added by the hidden test.patch (a new file in every deepswe task)."""
    patch = (tests_dir(task) / "test.patch").read_text()
    m = re.search(r"diff --git a/test\.sh b/test\.sh\n(?:.*\n)*?@@[^\n]*\n((?:[+ \\].*\n?)*)", patch)
    if not m:
        raise ValueError(f"{task}: test.patch adds no test.sh")
    lines = [l[1:] for l in m.group(1).splitlines() if l.startswith("+")]
    return "\n".join(lines) + "\n"


def recipe_sha(task: str) -> str:
    fam = family(task)
    parts = [RECIPE_VERSION, fam]
    if fam == "deepswe":
        parts += [DEEPSWE_SCRIPT, per_test_script(task), json.dumps(PER_TEST_EDITS)]
    else:
        parts += [CANONICAL_SCRIPT]
    for p in sorted(tests_dir(task).rglob("*")):
        if p.is_file():
            parts.append(f"{p.relative_to(tests_dir(task))}:{_sha(p.read_bytes())}")
    return _sha("\n".join(parts))


# ── parsing ─────────────────────────────────────────────────────────────────

_PYTEST = re.compile(r"^(PASSED|FAILED|ERROR|XPASS|XFAIL) (.+?::.+?)(?: - .*)?$")
_GO = re.compile(r"^\s*--- (PASS|FAIL|SKIP): (\S+)")
_TAP = re.compile(r"^(ok|not ok) \d+ (.*?)(\s+# (SKIP|TODO).*)?$")


def parse_new_output(log: str) -> dict[str, str]:
    """test id -> PASSED | FAILED | SKIPPED from a per-test ``test.sh new`` run
    (pytest -rA summary lines, ``go test -v`` result lines, mocha TAP)."""
    out: dict[str, str] = {}
    body = log.split("SWI_NEW_BEGIN", 1)[-1]
    for line in body.splitlines():
        line = line.rstrip()
        m = _PYTEST.match(line)
        if m and "::" in m.group(2):
            st = {"PASSED": "PASSED", "XPASS": "PASSED"}.get(m.group(1), "FAILED")
            if m.group(1) == "XFAIL":
                st = "SKIPPED"
            out[m.group(2)] = st if out.get(m.group(2)) != "FAILED" else "FAILED"
            continue
        m = _GO.match(line)
        if m:
            st = {"PASS": "PASSED", "FAIL": "FAILED", "SKIP": "SKIPPED"}[m.group(1)]
            name = m.group(2)
            if out.get(name) != "FAILED":  # same name in two packages: a failure anywhere wins
                out[name] = st
            continue
        m = _TAP.match(line)
        if m:
            name = m.group(2).strip()
            st = "SKIPPED" if m.group(4) else ("PASSED" if m.group(1) == "ok" else "FAILED")
            if out.get(name) != "FAILED":
                out[name] = st
    return out


def test_group(test_id: str, fam: str) -> str:
    """Function-level group of a test id (parameters / subtests stripped)."""
    if fam in ("deepswe", "swebenchpro") and "::" in test_id:
        return re.sub(r"\[.*\]$", "", test_id)
    if "/" in test_id and re.match(r"^Test\w+/", test_id):  # go subtest
        return test_id.split("/", 1)[0]
    return test_id


# ── container run ───────────────────────────────────────────────────────────

def _docker(*args: str, timeout: int | None = None, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, timeout=timeout, input=input_bytes)


def cache_path(task: str, key: str) -> Path:
    return CACHE / task / f"{key}.json"


def cache_key(task: str, patch: str) -> str:
    return _sha(f"{recipe_sha(task)}\n{task}\n") + _sha(patch.encode("utf-8", "surrogateescape"))[:16]


def _reparse(rec: dict) -> dict:
    """deepswe: per-test statuses are re-derived from the kept log (parser fixes apply to old runs)."""
    if rec.get("family") == "deepswe" and rec.get("log"):
        lp = RESULTS / rec["log"]
        if lp.exists():
            rec["tests"] = parse_new_output(lp.read_text(errors="replace"))
    return rec


def cached(task: str, patch: str) -> dict | None:
    p = cache_path(task, cache_key(task, patch))
    if p.exists():
        try:
            return _reparse(json.loads(p.read_text()))
        except json.JSONDecodeError:
            return None
    return None


def run_snapshot(task: str, patch: str, *, label: str = "", timeout: int | None = None,
                 use_cache: bool = True) -> dict:
    """Apply ``patch`` to a fresh container of the task image and run the hidden tests.
    Returns (and caches) ``{status: ok|apply_failed|infra_error|timeout, tests: {...}, ...}``."""
    fam = family(task)
    key = cache_key(task, patch)
    out_p = cache_path(task, key)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    lock = open(out_p.with_suffix(".lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)  # one run per (task, patch) across threads and processes
    try:
        if use_cache and out_p.exists():
            try:
                return _reparse(json.loads(out_p.read_text()))
            except json.JSONDecodeError:
                pass
        rec = _run(task, fam, patch, key, label, timeout or DEFAULT_TIMEOUT[fam])
        tmp = out_p.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, out_p)
        return rec
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _run(task: str, fam: str, patch: str, key: str, label: str, timeout: int) -> dict:
    img = task_image(task)
    cpus, mem = task_resources(task)
    if fam == "rf":
        cpus = min(cpus, RF_CPUS)
    root = repo_root(task)
    name = f"swi-exec-{re.sub(r'[^a-z0-9]+', '-', task.lower())[:40]}-{uuid.uuid4().hex[:8]}"
    rec: dict = {"schema": "redo_v1/swi-exec/1", "task": task, "family": fam, "key": key, "label": label,
                 "recipe_version": RECIPE_VERSION, "recipe_sha256": recipe_sha(task), "image": img,
                 "patch_sha256": _sha(patch.encode("utf-8", "surrogateescape")), "patch_bytes": len(patch),
                 "cpus": cpus, "memory_mb": mem, "repo_root": root, "started_at": time.time()}
    log_p = cache_path(task, key).with_suffix(".log")
    t0 = time.time()
    try:
        r = _docker("run", "-d", "--name", name, "--label", "tib.swi_exec=1", "--cpus", str(cpus),
                    "--memory", f"{mem}m", "--entrypoint", "sleep", img, "infinity", timeout=300)
        if r.returncode:
            raise RuntimeError(f"docker run: {r.stderr.decode(errors='replace')[:500]}")
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "snap.patch").write_bytes(patch.encode("utf-8", "surrogateescape"))
            script = DEEPSWE_SCRIPT if fam == "deepswe" else CANONICAL_SCRIPT
            (tdp / "judge.sh").write_text(script)
            if fam == "deepswe":
                (tdp / "test_pertest.sh").write_text(per_test_script(task))
            for src, dst in ((str(tests_dir(task)) + "/.", "/tests"), (str(tdp) + "/.", "/tmp/swi")):
                _docker("exec", name, "mkdir", "-p", dst, timeout=60)
                c = _docker("cp", src, f"{name}:{dst}", timeout=600)
                if c.returncode:
                    raise RuntimeError(f"docker cp {dst}: {c.stderr.decode(errors='replace')[:300]}")
        try:
            ex = _docker("exec", "-w", root, "-e", f"SWI_ROOT={root}", "-e", f"SWI_TIMEOUT={timeout}",
                         name, "bash", "-c", "bash /tmp/swi/judge.sh", timeout=timeout + 600)
            log = ex.stdout.decode(errors="replace") + "\n--- stderr ---\n" + ex.stderr.decode(errors="replace")
            rc = ex.returncode
        except subprocess.TimeoutExpired as e:
            log = (e.stdout or b"").decode(errors="replace") + "\n[SWI: host timeout]"
            rc = -9
        log_p.write_text(log)
        rec["log"] = str(log_p.relative_to(RESULTS))
        rec["exit_code"] = rc
        head = _docker("exec", name, "cat", "/tmp/swi/head.txt", timeout=60).stdout.decode().strip()
        rec["head"] = head.removeprefix("HEAD=") or None
        if "SWI_APPLY_FAILED" in log:
            rec["status"] = "apply_failed"
        elif "SWI_TEST_PATCH_FAILED" in log:
            rec["status"] = "test_patch_failed"
        elif rc == -9:
            rec["status"] = "timeout"
        else:
            rec["status"] = "ok"
        if fam == "deepswe":
            rec["tests"] = parse_new_output(log)
            m = re.search(r"SWI_NEW_RC=(\d+)", log)
            rec["new_rc"] = int(m.group(1)) if m else None
            if m and int(m.group(1)) == 124:
                rec["status"] = "timeout"
        elif fam == "swebenchpro":
            o = _docker("exec", name, "cat", "/tmp/output.json", timeout=60)
            try:
                rec["tests"] = {t["name"]: t["status"] for t in json.loads(o.stdout.decode()).get("tests", [])}
            except (json.JSONDecodeError, AttributeError):
                rec["tests"] = {}
                if rec["status"] == "ok":
                    rec["status"] = "no_test_output"
        else:  # rf
            cr = _docker("exec", name, "cat", "/logs/verifier/comparison_results.json", timeout=60)
            tr = _docker("exec", name, "cat", "/logs/verifier/tests_reward.txt", timeout=60)
            try:
                comp = json.loads(cr.stdout.decode())
            except json.JSONDecodeError:
                comp = {}
            rec["tests_reward"] = float(tr.stdout.decode().strip()) if tr.returncode == 0 and tr.stdout.strip() else None
            rec["filtered_comparison"] = comp.get("filtered_comparison")
            rec["filtered_summary"] = comp.get("filtered_summary")
            rec["full_p2f_count"], rec["full_m2f_count"] = comp.get("full_p2f_count"), comp.get("full_m2f_count")
            if rec["tests_reward"] is None and rec["status"] == "ok":
                rec["status"] = "no_test_output"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "infra_error"
        rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    finally:
        _docker("rm", "-f", name, timeout=300)
    rec["wall_sec"] = round(time.time() - t0, 1)
    return rec


def image_head(task: str) -> str | None:
    """HEAD of the repository in a fresh container of the task image (the trial anchor)."""
    r = _docker("run", "--rm", "--entrypoint", "git", task_image(task), "-C", repo_root(task), "rev-parse", "HEAD",
                timeout=300)
    s = r.stdout.decode().strip().splitlines()
    return s[-1] if r.returncode == 0 and s else None
