"""Executable per-snapshot checks for SWE-Together (redo_v1 per-intent judge v2).

A repository snapshot (``agent/snapshots/turn-N.patch``: the cumulative diff vs ``harbor-base``,
the same artefact the benchmark judge applies as ``final.patch``) is applied to a FRESH container
of the task's native image (arm64 hosts: the local rebuild ``bdqnghi/swe-together:<task>``; amd64
hosts: the task's official image from ``task.toml``) and the task's own verifier is run,
exactly as the arm64 gold validation (``swe_benchmark_arm/swe_together/validate.py``) and Harbor's
verifier do it:

* apply as root: repo = the recorder's banner hint, else the shallowest ``.git`` under the
  benchmark's well-known roots; candidates from the clone's ``src/patch_normalize.apply_candidates``
  tried with ``git apply --check`` (then ``-C2``); ``chmod -R a+rwX``  (= ``eval/correctness/sandbox.py``);
* upload ``tests/`` to ``/tests`` and run ``/tests/test.sh > /logs/verifier/test-stdout.txt`` as
  root, with the task's cpus / memory limits; never under emulation (the image must match the
  host architecture).

Outputs per run (``checks``):

* ``gate:<id>`` - every gate the verifier itself emitted to ``/logs/verifier/gates.json``. Gates
  the script did not emit are filled by the benchmark's "auto-bridge" (a proportional pass count,
  not a per-gate result); they are recorded as ``bridged`` and are never evidence for an intent.
* ``test:<name>`` - for SWE-rebench-style verifiers (``tests/install_config.json`` + ``log_parser``),
  every FAIL_TO_PASS / PASS_TO_PASS test status parsed from ``test_run.log`` with the task's own
  ``log_parsers.py`` (loaded on the host from the task dir).

Also kept per run: the post-patch text of every file the snapshot touches (``files``; the LLM
judge reads it), ``reward.txt``, the verifier log, the apply log and the image id. Results are
cached by sha256(recipe, task tests digest, image id, patch text): identical snapshots run once.
Containers are labelled ``tib.swt_exec=1`` and always removed; nothing else on the host is touched.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import uuid
from pathlib import Path

from .. import machine
from .config import CLONE, RESULTS, official_image

RECIPE_VERSION = "swt-exec/1"
CACHE = RESULTS / "swt_exec_cache"
IMAGE_TEMPLATE = "bdqnghi/swe-together:{task}"  # arm64 hosts: the gold-validated native rebuilds
MAX_FILE_CHARS = 200_000

sys.path.insert(0, str(CLONE / "src"))
try:
    from patch_normalize import apply_candidates, main_repo_path  # noqa: E402
except ModuleNotFoundError:
    from .patch_normalize import apply_candidates, main_repo_path  # noqa: E402

APPLY_SH = r'''set -e
HINT=%(hint)s
ROOTS="/workspace /opt /home /app /repo /tmp /entire-cli /entireio-cli /no-magic"
EXISTING=""
for r in $ROOTS; do [ -e "$r" ] && EXISTING="$EXISTING $r"; done
if [ -z "$EXISTING" ]; then echo "NO_REPO_ROOTS_EXIST" >&2; exit 1; fi
REPO=""
if [ -n "$HINT" ] && [ -e "$HINT/.git" ]; then REPO="$HINT"; fi
if [ -z "$REPO" ]; then
  REPO=$(find $EXISTING -maxdepth 3 -name .git \( -type d -o -type f \) 2>/dev/null | while IFS= read -r g; do printf "%%d %%s\n" "${#g}" "$g"; done | sort -n | head -1 | cut -d" " -f2- | xargs -I{} dirname {})
fi
if [ -z "$REPO" ]; then echo "NO_GIT_REPO_FOUND" >&2; exit 1; fi
echo "$REPO" > /tmp/swt_repo
cd "$REPO" && echo "applying to $(pwd)"
APPLIED=""
if [ %(n)d -gt 0 ]; then
for flags in "" "-C2"; do
  for i in $(seq 0 %(last)d); do
    if git -c safe.directory="*" apply --check --whitespace=nowarn $flags "/tmp/agent.patch.$i" 2>/dev/null; then
      git -c safe.directory="*" apply --whitespace=nowarn $flags "/tmp/agent.patch.$i" && cp "/tmp/agent.patch.$i" /tmp/agent.patch && APPLIED="$i${flags:+ $flags}"; break 2
    fi
  done
done
if [ -z "$APPLIED" ]; then git -c safe.directory="*" apply --check --whitespace=nowarn /tmp/agent.patch.0; echo SWT_APPLY_FAILED; exit 90; fi
echo "applied candidate $APPLIED"
fi
{ chmod -R a+rwX "$REPO" 2>/dev/null || true; }
'''


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode("utf-8", "surrogateescape") if isinstance(b, str) else b).hexdigest()


def task_dir(task: str) -> Path:
    return CLONE / "tasks" / task


def image(task: str) -> str:
    """The task's native image on this host: the arm64 rebuild on arm64 hosts, else the official image."""
    return IMAGE_TEMPLATE.format(task=task) if machine.host_arch() == "arm64" else official_image(task)


def tests_digest(task: str) -> str:
    td = task_dir(task) / "tests"
    return _sha("\n".join(f"{p.relative_to(td)}:{_sha(p.read_bytes())}" for p in sorted(td.rglob("*")) if p.is_file()))


_IMAGE_IDS: dict[str, tuple[str | None, str | None]] = {}


def image_id(task: str) -> tuple[str | None, str | None]:
    if task not in _IMAGE_IDS:
        r = subprocess.run(["docker", "image", "inspect", image(task), "--format", "{{.Id}} {{.Architecture}}"],
                           capture_output=True, text=True)
        parts = r.stdout.split()
        _IMAGE_IDS[task] = (parts[0], parts[1]) if r.returncode == 0 and len(parts) == 2 else (None, None)
    return _IMAGE_IDS[task]


def recipe_sha(task: str) -> str:
    return _sha("\n".join([RECIPE_VERSION, APPLY_SH, tests_digest(task), image_id(task)[0] or "?"]))


def cache_key(task: str, patch: str) -> str:
    return _sha(f"{recipe_sha(task)}\n{task}\n")[:24] + _sha(patch)[:40]


def cache_path(task: str, key: str) -> Path:
    return CACHE / task / f"{key}.json"


def task_limits(task: str) -> tuple[str | None, str | None, float]:
    cfg = tomllib.loads((task_dir(task) / "task.toml").read_text())
    env = cfg.get("environment", {})
    vt = float(cfg.get("verifier", {}).get("timeout_sec", 600))
    return (str(env["cpus"]) if env.get("cpus") else None, str(env["memory"]) if env.get("memory") else None,
            max(1800.0, 3 * vt))


def patch_files(patch: str) -> list[str]:
    """Post-image paths the (task-repo section of the) snapshot touches."""
    cands = apply_candidates(patch)
    if not cands:
        return []
    out = []
    for m in re.finditer(r"^diff --git a/(.+?) b/(.+)$", cands[0], re.M):
        out.append(m.group(2).strip().strip('"'))
    return sorted(set(out))


# ── parsing ─────────────────────────────────────────────────────────────────

def _gate_passed(r: dict) -> bool:
    """Gate emit formats differ by task: ``passed``/``pass`` booleans or a ``status``/``result`` word."""
    for k in ("passed", "pass", "ok", "success"):
        if k in r:
            v = r[k]
            return v.strip().lower() in ("true", "pass", "passed", "1", "yes") if isinstance(v, str) else bool(v)
    for k in ("status", "result", "verdict", "outcome"):
        if k in r:
            return str(r[k]).strip().upper() in ("PASS", "PASSED", "OK", "SUCCESS", "TRUE")
    return False


def parse_gates(text: str) -> dict[str, dict]:
    """gates.json (JSON lines, or a list / {"gates": [...]}) -> id -> {passed, bridged, detail};
    for a gate emitted twice the last explicit emit wins."""
    rows = []
    t = (text or "").strip()
    if t.startswith("[") or (t.startswith("{") and '"gates"' in t[:200] and "\n{" not in t):
        try:
            d = json.loads(t)
            rows = d.get("gates", []) if isinstance(d, dict) else d
        except json.JSONDecodeError:
            rows = []
    if not rows:
        for line in t.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    out: dict[str, dict] = {}
    for r in rows:
        if not isinstance(r, dict) or not r.get("id"):
            continue
        detail = str(r.get("detail") or "")
        bridged = detail.startswith("auto-bridge")
        if r["id"] in out and not out[r["id"]]["bridged"] and bridged:
            continue
        out[r["id"]] = {"passed": _gate_passed(r), "bridged": bridged, "detail": detail[:300]}
    return out


def parse_tests(task: str, log_text: str) -> dict[str, str] | None:
    """F2P/P2P test statuses of a SWE-rebench-style verifier via the task's own log parser."""
    td = task_dir(task) / "tests"
    cfgp = td / "install_config.json"
    if not (cfgp.exists() and (td / "log_parsers.py").exists()):
        return None
    cfg = json.loads(cfgp.read_text())
    if not cfg.get("log_parser"):
        return None
    sys.path.insert(0, str(td))
    try:
        spec = importlib.util.spec_from_file_location(f"swt_log_parsers_{_sha(task)[:8]}", td / "log_parsers.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parsed = mod.NAME_TO_PARSER[cfg["log_parser"]](log_text)
    except Exception as e:  # noqa: BLE001
        return {"__parse_error__": f"{type(e).__name__}: {e}"[:200]}
    finally:
        sys.path.remove(str(td))
    names = list(cfg.get("FAIL_TO_PASS") or []) + list(cfg.get("PASS_TO_PASS") or [])
    return {n: parsed.get(n, "MISSING") for n in names}


# ── container run ───────────────────────────────────────────────────────────

def _docker(*args: str, timeout: float | None = None, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, timeout=timeout, input=input_bytes)


def cached(task: str, patch: str) -> dict | None:
    p = cache_path(task, cache_key(task, patch))
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def run_snapshot(task: str, patch: str, *, label: str = "", use_cache: bool = True) -> dict:
    """Apply ``patch`` in a fresh container of the task image, run the verifier, return (and cache)
    ``{status: ok|apply_failed|timeout|infra_error, checks, files, reward, ...}``."""
    key = cache_key(task, patch)
    out_p = cache_path(task, key)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    lock = open(out_p.with_suffix(".lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)  # one run per (task, patch) across threads and processes
    try:
        if use_cache and out_p.exists():
            try:
                return json.loads(out_p.read_text())
            except json.JSONDecodeError:
                pass
        rec = _run(task, patch, key, label)
        tmp = out_p.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, out_p)
        return rec
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _read_tar_files(tar_bytes: bytes) -> dict[str, str]:
    out = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                data = tf.extractfile(m).read()
                txt = data.decode("utf-8", errors="replace")
                if len(txt) > MAX_FILE_CHARS:
                    txt = txt[:MAX_FILE_CHARS] + f"\n[... {len(txt) - MAX_FILE_CHARS} characters truncated ...]"
                out[m.name] = txt
    return out


def _run(task: str, patch: str, key: str, label: str) -> dict:
    img = image(task)
    iid, arch = image_id(task)
    cpus, mem, vtimeout = task_limits(task)
    name = f"swt-exec-{re.sub(r'[^a-z0-9]+', '-', task.lower())[:40]}-{uuid.uuid4().hex[:8]}"
    rec: dict = {"schema": "redo_v1/swt-exec/1", "task": task, "key": key, "label": label,
                 "recipe_version": RECIPE_VERSION, "recipe_sha256": recipe_sha(task), "image": img,
                 "image_id": iid, "image_arch": arch, "patch_sha256": _sha(patch), "patch_bytes": len(patch),
                 "cpus": cpus, "memory": mem, "verifier_timeout_sec": vtimeout, "started_at": time.time()}
    log_dir = cache_path(task, key).with_suffix("")
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        host = machine.host_arch()
        if arch != host:
            raise RuntimeError(f"image {img} is {arch}, not {host} (never run under emulation)")
        run = ["run", "-d", "--name", name, "--label", "tib.swt_exec=1", "--platform", f"linux/{host}"]
        if cpus:
            run += ["--cpus", cpus]
        if mem:
            run += ["--memory", mem]
        r = _docker(*run, img, "sh", "-c", "sleep infinity", timeout=300)
        if r.returncode:
            raise RuntimeError(f"docker run: {r.stderr.decode(errors='replace')[-400:]}")
        cands = apply_candidates(patch)
        with tempfile.TemporaryDirectory() as td:
            for i, c in enumerate(cands):
                p = Path(td) / f"agent.patch.{i}"
                p.write_text(c, encoding="utf-8", errors="surrogateescape")
                c_ = _docker("cp", str(p), f"{name}:/tmp/agent.patch.{i}", timeout=300)
                if c_.returncode:
                    raise RuntimeError(f"docker cp patch: {c_.stderr.decode(errors='replace')[-300:]}")
        script = APPLY_SH % {"hint": shlex.quote(main_repo_path(patch) or ""), "n": len(cands),
                             "last": max(len(cands) - 1, 0)}
        a = _docker("exec", "-u", "root", name, "bash", "-c", script, timeout=600)
        apply_log = a.stdout.decode(errors="replace") + a.stderr.decode(errors="replace")
        (log_dir / "apply.log").write_text(apply_log)
        rec["applied"] = next((l for l in apply_log.splitlines() if l.startswith("applied candidate")), None)
        repo = _docker("exec", name, "cat", "/tmp/swt_repo", timeout=60).stdout.decode().strip() or None
        rec["repo"] = repo
        if a.returncode:
            rec["status"] = "apply_failed"
            rec["apply_error"] = apply_log[-600:]
            return rec
        # post-patch text of the touched files (for the LLM judge)
        files = [f for f in patch_files(patch)]
        rec["files"] = {}
        rec["files_missing"] = []
        if files and repo:
            lst = ("\n".join(files) + "\n").encode()
            t = _docker("exec", "-i", "-u", "root", "-w", repo, name, "bash", "-c",
                        "while IFS= read -r f; do [ -f \"$f\" ] && printf '%s\\0' \"$f\"; done | "
                        "tar --null -T - -cf - 2>/dev/null", timeout=300, input_bytes=lst)
            got = _read_tar_files(t.stdout) if t.stdout else {}
            rec["files"] = got
            rec["files_missing"] = [f for f in files if f not in got]  # deleted by the snapshot
        # verifier
        _docker("exec", "-u", "root", name, "bash", "-c", "rm -rf /tests && mkdir -p /logs/verifier /logs/agent",
                timeout=120)
        c = _docker("cp", f"{task_dir(task) / 'tests'}/.", f"{name}:/tests", timeout=600)
        if c.returncode:
            raise RuntimeError(f"docker cp tests: {c.stderr.decode(errors='replace')[-300:]}")
        _docker("exec", "-u", "root", name, "bash", "-c", "chmod -R a+rwX /tests /logs && chmod +x /tests/test.sh",
                timeout=120)
        t1 = time.time()
        try:
            v = _docker("exec", "-u", "root", name, "bash", "-c",
                        "/tests/test.sh > /logs/verifier/test-stdout.txt 2>&1", timeout=vtimeout)
            rec["test_exit"] = v.returncode
            rec["status"] = "ok"
        except subprocess.TimeoutExpired:
            rec["status"] = "timeout"
        rec["test_seconds"] = round(time.time() - t1, 1)
        tb = _docker("exec", "-u", "root", name, "bash", "-c", "tar -C /logs/verifier -cf - .", timeout=300)
        vfiles = _read_tar_files(tb.stdout) if tb.stdout else {}
        vfiles = {k.lstrip("./"): v for k, v in vfiles.items()}
        for k, v in vfiles.items():
            (log_dir / k.replace("/", "_")).write_text(v)
        rec["verifier_files"] = sorted(vfiles)
        try:
            rec["reward"] = float((vfiles.get("reward.txt") or "").strip())
        except ValueError:
            rec["reward"] = None
        gates = parse_gates(vfiles.get("gates.json", ""))
        checks = {f"gate:{g}": {"passed": x["passed"], "bridged": x["bridged"]} for g, x in gates.items()}
        tests = parse_tests(task, vfiles.get("test_run.log", ""))
        if tests:
            for n, st in tests.items():
                checks[f"test:{n}"] = {"passed": st == "PASSED", "status": st, "bridged": False}
        rec["checks"] = checks
        rec["gates_detail"] = {g: x["detail"] for g, x in gates.items()}
        rec["log_dir"] = str(log_dir.relative_to(RESULTS))
    except Exception as e:  # noqa: BLE001
        rec["status"] = "infra_error"
        rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    finally:
        _docker("rm", "-f", name, timeout=300)
        rec["wall_sec"] = round(time.time() - t0, 1)
    return rec


def reference_patch(task: str) -> str:
    p = task_dir(task) / "reference_patch.json"
    return (json.loads(p.read_text()).get("patch") or "") if p.exists() else ""


REFBASE = CACHE / "refbase_summary.json"


def _slim(rec: dict | None) -> dict | None:
    return None if rec is None else {k: v for k, v in rec.items() if k != "files"}


def refbase(tasks: list[str], workers: int = 4) -> dict:
    """The reference patch (``reference_patch.json``) and the empty patch of every task, run on this
    host's native image -> ``swt_exec_cache/refbase_summary.json`` (``{task: {reference, base}}``:
    the final-pass rule's "reward >= the reference reward on the same image", ``metrics.py``) and
    ``refbase_summary.meta.json`` (host, images, time). Restartable: runs are cached by content, and
    tasks already in the summary with an ``ok`` run on the same image are kept."""
    from concurrent.futures import ThreadPoolExecutor
    have = json.loads(REFBASE.read_text()) if REFBASE.exists() else {}

    def fresh(t: str) -> bool:
        v = have.get(t) or {}
        runs = [r for r in (v.get("reference"), v.get("base")) if r is not None]
        return bool(runs) and all(r.get("status") == "ok" and r.get("image_id") == image_id(t)[0] for r in runs)

    def one(t: str) -> tuple[str, dict]:
        if fresh(t):
            return t, have[t]
        ref = reference_patch(t)
        return t, {"reference": _slim(run_snapshot(t, ref, label="reference")) if ref else None,
                   "base": _slim(run_snapshot(t, "", label="base"))}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for t, v in ex.map(one, tasks):
            have[t] = v
            REFBASE.parent.mkdir(parents=True, exist_ok=True)
            tmp = REFBASE.with_suffix(f".{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps(have, indent=1, sort_keys=True))
            os.replace(tmp, REFBASE)
    meta = {"written_at": time.time(), "host_arch": machine.host_arch(), "recipe_version": RECIPE_VERSION,
            "images": {t: {"image": image(t), "image_id": image_id(t)[0], "arch": image_id(t)[1]} for t in sorted(have)},
            "note": "reference / empty patch on this host's native images (arm64 rebuilds on arm64 hosts, the "
                    "tasks' official images on amd64); never compare across architectures"}
    REFBASE.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return have


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config import subset_tasks
    ap = argparse.ArgumentParser(description="Executable checks for SWE-Together snapshots.")
    ap.add_argument("cmd", choices=["refbase"])
    ap.add_argument("--tasks", default=None, help="comma-separated (default: the selected subset)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    tasks = a.tasks.split(",") if a.tasks else subset_tasks()
    out = refbase(tasks, a.workers)
    for t in tasks:
        v = out.get(t) or {}
        r, b = v.get("reference") or {}, v.get("base") or {}
        print(json.dumps({"task": t, "reference": {k: r.get(k) for k in ("status", "reward", "applied")} if r else None,
                          "base": {k: b.get(k) for k in ("status", "reward")}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
