"""SWE-Interact runner (redo_v1): Harbor 0.23.0 ``-e docker`` on the task tree native to this host
(arm64: the arm64 layer's rebuilt images; amd64: native builds of the benchmark's published images
plus the multi-turn additions, ``amd64.py``; ``patchset.layer1``).

    python -u -m fsa.redo.swe_interact.run trials --run-dir results/redo_v1/stage0_swe_interact/W_bringup \
        --rung W --tasks deepswe_httpx-streaming-json-iteration,... [--workers 3]

One Harbor job per (task, attempt): ``<run-dir>/jobs/<task>__<rung>__a<N>/``; Harbor's trial
dir inside it holds ``steps/<step>/{agent,verifier}``, ``user-sim/`` (sidecar instrumentation,
local patch) and ``result.json``. A task with a completed trial (``result.json`` with a verifier
result and no exception) is skipped on re-run; an incomplete one gets a new attempt dir. Nothing is
deleted. Every invocation appends a manifest entry (``<run-dir>/manifest.json``): exact model ids
with provider records, reasoning, seats, benchmark commit, both patch hashes, Harbor provenance,
opencode version, image ids/digests, subset digest, command and non-secret environment.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .. import machine
from . import patchset
from .config import (CANONICAL_RESULTS, HARBOR, RESULTS, ROOT, TASKS, family, load_rungs, repo_root, subset_digest,
                     subset_tasks, task_docker_image, task_image)

ANON_DOCKER_CONFIG = RESULTS / ".docker-anon"
MCP_TIMEOUT_MS = 3_600_000  # opencode experimental.mcp_timeout for every arm (agent.py docstring); 0 = upstream 60 s
SUBNET_LOCKS = CANONICAL_RESULTS / ".swi_subnets"  # host-wide: shared by every runner whatever its results tree
SUBNET_FMT = "10.213.{slot}.0/24"


class SubnetLease:
    """Exclusive 10.213.N.0/24 for one trial's compose network (flock across runner
    processes; skips subnets any existing docker network already uses)."""

    def __enter__(self):
        import fcntl
        SUBNET_LOCKS.mkdir(parents=True, exist_ok=True)
        used = set()
        for n in subprocess.run(["docker", "network", "ls", "-q"], capture_output=True, text=True).stdout.split():
            used |= set(subprocess.run(["docker", "network", "inspect", n, "--format",
                                        "{{range .IPAM.Config}}{{.Subnet}} {{end}}"],
                                       capture_output=True, text=True).stdout.split())
        for slot in range(1, 250):
            subnet = SUBNET_FMT.format(slot=slot)
            if subnet in used:
                continue
            f = open(SUBNET_LOCKS / f"{slot}.lock", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            self._f, self.subnet = f, subnet
            return self
        raise SystemExit("no free 10.213.N.0/24 subnet")

    def __exit__(self, *exc):
        self._f.close()


def _gemini_record(model_id: str) -> dict:
    from ..swe_together.run import gemini_model_record
    return gemini_model_record(model_id)


def judge_seats_record() -> dict:
    """Host-side judge seats of this run (tagger, LLM judge, rubric judge) and the SWE-Interact
    judge freeze in force (``configs/intents/swe-interact/JUDGE_FREEZE.json``)."""
    from .. import llm_seats
    seat = llm_seats.judge_seat()
    rec = {"judge_seat": llm_seats.seat_record(seat["model"], seat["reasoning_effort"], source=seat["source"]),
           "seats_file": str(llm_seats.SEATS_FILE.relative_to(ROOT)),
           "seats_file_sha256": patchset.sha256(llm_seats.SEATS_FILE.read_bytes())}
    fz = ROOT / "configs" / "intents" / "swe-interact" / "JUDGE_FREEZE.json"
    rec["judge_freeze"] = ({"file": str(fz.relative_to(ROOT)), "sha256": patchset.sha256(fz.read_bytes())}
                           if fz.exists() else None)
    return rec


def _image_inspect(img: str) -> dict:
    """Local id / digests of an image (None values when absent)."""
    r = subprocess.run(["docker", "image", "inspect", img, "--format", "{{.Id}} {{json .RepoDigests}}"],
                       capture_output=True, text=True)
    parts = r.stdout.strip().split(" ", 1)
    return {"image": img, "image_id": parts[0] or None,
            "repo_digests": json.loads(parts[1]) if len(parts) > 1 else None}


def _base_image(task: str) -> dict | None:
    """The FROM image of a build-based task's ``environment/Dockerfile`` (last stage) and its local
    digest: the part of a build-based trial image that is not in this repository."""
    df = TASKS / task / "environment" / "Dockerfile"
    froms = [ln.split()[1] for ln in df.read_text().splitlines() if ln.strip().upper().startswith("FROM ") and
             len(ln.split()) > 1] if df.is_file() else []
    if not froms:
        return None
    return _image_inspect(froms[-1])


def _amd64_layout(task: str) -> bool:
    """task.toml points at this task's amd64 layer-1 image (``amd64.patch`` applied)."""
    from . import amd64
    return task_docker_image(task) == amd64.image_tag(task)


def image_record(task: str) -> dict:
    """The task's image (task.toml ``docker_image``: the arm64 rebuild on arm64 hosts, the native
    build of ``amd64.py`` on amd64 hosts - see ``patchset.layer1``; tasks without one: Harbor's build
    of ``environment/``, ``config.task_image``). ``present`` only for this host's architecture (an
    image of another architecture would run emulated, which the study never does) and, for the amd64
    images, only when built from the current recipe (label ``tib.recipe_sha256``)."""
    img = task_image(task)
    r = subprocess.run(["docker", "image", "inspect", img, "--format", "{{.Id}} {{.Architecture}} {{json .RepoDigests}}"],
                       capture_output=True, text=True)
    parts = r.stdout.strip().split(" ", 2)
    arch = parts[1] if len(parts) > 1 else None
    rec = {"image": img, "prebuilt": task_docker_image(task) is not None,
           "present": r.returncode == 0 and arch == machine.host_arch(),
           "image_id": parts[0] if parts[0] else None, "arch": arch, "host_arch": machine.host_arch(),
           "repo_digests": json.loads(parts[2]) if len(parts) > 2 else None}
    if _amd64_layout(task):
        from . import amd64
        rc, st = amd64.recipe(task), amd64.image_state(task)
        rec["amd64_recipe"] = {"kind": rc["kind"], "dockerfile_sha256": rc["dockerfile_sha256"],
                               "image_label_recipe_sha256": st["recipe_sha256"],
                               "base_image": _image_inspect(rc["base"]) if rc["base"] else None,
                               "source": "fsa/redo/swe_interact/amd64.py (rungs.json benchmark.amd64_images)"}
        rec["present"] = rec["present"] and st["recipe_sha256"] == rc["dockerfile_sha256"]
    elif not rec["prebuilt"]:
        rec["base_image"] = _base_image(task)
        rec["trial_image"] = ("built in each trial by Harbor's compose build of environment/ (tag "
                              "<compose project>-main): the same build context and cached layers as `image`, "
                              "which differs only in image labels")
    return rec


def build_task_image(task: str) -> dict:
    """Build the task's image locally: the amd64 layer-1 image (``amd64.build``) when task.toml points
    at it; otherwise a task without a prebuilt image from its ``environment/`` (``hb__<content hash>``,
    Harbor 0.23.0's content-addressed name for it), so judges can use it before the first trial (each
    trial's compose build of the same context then hits the same layer cache). Base images are pulled
    anonymously."""
    img = task_image(task)
    if _amd64_layout(task):
        from . import amd64
        res = amd64.build(task, arch=machine.host_arch())
        return {"image": img, "built_now": res["built"], "recipe_kind": res["kind"],
                "recipe_sha256": res["recipe_sha256"], "build_log_tail": res["log_tail"]}
    ANON_DOCKER_CONFIG.mkdir(parents=True, exist_ok=True)
    (ANON_DOCKER_CONFIG / "config.json").write_text("{}")
    r = subprocess.run(["docker", "build", "--platform", f"linux/{machine.host_arch()}", "-t", img,
                        str(TASKS / task / "environment")], capture_output=True, text=True,
                       env={**os.environ, "DOCKER_CONFIG": str(ANON_DOCKER_CONFIG)})
    return {"image": img, "built_now": r.returncode == 0, "build_log_tail": (r.stdout + r.stderr)[-1500:]}


def sidecar_base_record() -> dict:
    r = subprocess.run(["docker", "image", "inspect", "python:3.12-slim", "--format", "{{.Id}} {{json .RepoDigests}}"],
                       capture_output=True, text=True)
    return {"image": "python:3.12-slim", "inspect": r.stdout.strip() or r.stderr.strip()[:200]}


def _key(ref: str) -> str:
    """``$NAME`` -> the value of environment variable NAME (seat keys are named in rungs.json)."""
    return os.environ[ref[1:]] if ref.startswith("$") else ref


def seat_env(cfg: dict, task: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SWI_", "EVAL_")) and k not in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE")}
    s = cfg["seats"]
    prof = machine.active()
    if prof.public:  # rungs.json endpoints and keys, as recorded there
        sim_base, sim_key = s["user_sim"]["env"]["LITELLM_BASE_URL"], _key(s["user_sim"]["env"]["LITELLM_API_KEY"])
        eval_base, eval_key = (s["rubric_grader"]["env"]["OPENAI_API_BASE"],
                               _key(s["rubric_grader"]["env"]["OPENAI_API_KEY"]))
        agent_key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY") or os.environ["GEMINI_API_KEY"]
    else:  # local endpoint (fsa.redo.machine): the containers reach it at the container-side base
        agent_key = sim_key = eval_key = prof.api_key()
        if not agent_key:
            raise SystemExit(f"machine profile {prof.name!r}: no API key ({prof.key_source()})")
        sim_base, eval_base = machine.sidecar_litellm_base(prof), prof.container_openai_base
    env.update({
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        # simulator seat (sidecar): gemini-3.1-pro-preview on Gemini's OpenAI-compatible endpoint
        # (Gemini-only seats, 2026-09-27; recorded deviation from the official gpt-5.5; rungs.json)
        "SIM_USER_MODEL": s["user_sim"]["env"]["SIM_USER_MODEL"],
        "LITELLM_BASE_URL": sim_base,
        "LITELLM_API_KEY": sim_key,
        "SWI_SIM_RETRY": "1",
        # rubric grader seat (rf_* final verifier; task.toml: EVAL_API_KEY=${OPENAI_API_KEY},
        # EVAL_BASE_URL=${OPENAI_API_BASE}) - the Gemini key under the name the task file reads
        "OPENAI_API_BASE": eval_base,
        "OPENAI_API_KEY": eval_key,
        "EVAL_MODEL": s["rubric_grader"]["env"]["EVAL_MODEL"],
        "EVAL_MAX_COMPLETION_TOKENS": s["rubric_grader"]["env"]["EVAL_MAX_COMPLETION_TOKENS"],
        "EVAL_REASONING_EFFORT": s["rubric_grader"]["env"]["EVAL_REASONING_EFFORT"],
        # agent seat: native Gemini API (opencode provider `google`)
        "GOOGLE_GENERATIVE_AI_API_KEY": agent_key,
        # (GEMINI_API_KEY / GOOGLE_API_KEY are dropped below: Harbor forwards the first of
        # GOOGLE_API_KEY, GEMINI_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY it finds, and opencode's
        # @ai-sdk/google reads only GOOGLE_GENERATIVE_AI_API_KEY)
        # instrumentation
        "SWI_REPO_ROOT": repo_root(task),
        # anonymous Docker Hub pulls (sidecar base image), not the account's shared quota
        "DOCKER_CONFIG": str(ANON_DOCKER_CONFIG),
    })
    env[machine.ENV_VAR] = prof.name
    if not prof.public:
        # the agent: Harbor 0.23.0 resolves GOOGLE_BASE_URL for the `google` provider and writes it
        # to opencode's config (provider.google.options.baseURL); host-side LiteLLM: GEMINI_API_BASE
        env["GOOGLE_BASE_URL"] = prof.container_gemini_base
        env["GEMINI_API_BASE"] = prof.gemini_base
    else:  # public API: a stray shell value must never redirect the run
        for k in machine.BASE_VARS:
            env.pop(k, None)
    env.pop("GEMINI_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)
    return env


def redo_env(arm: str = "stock", *, overlay: bool = False, mcp_timeout_ms: int = MCP_TIMEOUT_MS) -> dict[str, str]:
    """Sidecar arm selection (compose defaults = stock, no overlay) + the agent's MCP timeout."""
    env: dict[str, str] = {}
    if arm != "stock":
        env["SWI_REDO_ARM"] = arm
    if overlay:
        env["SWI_REDO_OVERLAY"] = "1"
    if mcp_timeout_ms:
        env["SWI_MCP_TIMEOUT_MS"] = str(mcp_timeout_ms)
    return env


def redo_record(arm: str, *, tau: float, overlay: bool, overlay_seed: int, mcp_timeout_ms: int,
                overlay_wait: float, agent_model: str | None = None, overlay_mode: str = "standard") -> dict:
    from . import arms as A
    from . import tagger
    from .intents import index_digest
    cfg = A.RedoConfig(arm=arm, tau=tau, overlay=overlay, overlay_seed=overlay_seed, overlay_mode=overlay_mode)
    rec = {"arm": arm, "tau": tau if A.CLAIM_TIMING.get(arm) and arm != "claims_no_p" else None,
           "claim_timing": A.CLAIM_TIMING.get(arm), "overlay": overlay,
           "overlay_mode": overlay_mode if overlay else None,
           "overlay_seed": overlay_seed if overlay else None, "max_overlays": cfg.max_overlays if overlay else None,
           "overlay_wait_sec": overlay_wait if overlay else None,
           "claim_book_seat": ({"model": "agent's own model via host LiteLLM (gemini/<rung model>)",
                                "reasoning_effort": "high (litellm -> thinkingLevel=high)",
                                "response_format": "json_object"} if A.CLAIM_TIMING.get(arm) else None),
           "port_seat": ({"model": "agent's own model via host LiteLLM (gemini/<rung model>)",
                          "reasoning_effort": "high", "label": A._ports.PORT_LABEL[arm],
                          "calls": "user-sim/port_calls.jsonl"} if arm in A.PORT_ARMS else None),
           "arms": A.config_record(cfg, agent_model), "broker": "fsa/redo/swe_interact/broker.py (host, file RPC via user-sim/rpc)",
           "mcp_timeout_ms": mcp_timeout_ms, "env": redo_env(arm, overlay=overlay, mcp_timeout_ms=mcp_timeout_ms),
           "module_sha256": {m: patchset.sha256(Path(__file__).with_name(m).read_bytes())
                             for m in ("arms.py", "broker.py", "judge.py", "exec_judge.py", "tagger.py",
                                       "rubric_judge.py", "intents.py")}
           | {"../ports.py": patchset.sha256((Path(__file__).parent.parent / "ports.py").read_bytes())}}
    if overlay:
        from .judge import config_record
        rec["judge"] = config_record()
        rec["tagger"] = tagger.config_record()
        rec["intents_index"] = index_digest()
    return rec


def public_env(env: dict) -> dict:
    keep = ("SIM_USER_MODEL", "LITELLM_BASE_URL", "SWI_SIM_RETRY", "OPENAI_API_BASE", "EVAL_MODEL",
            "EVAL_MAX_COMPLETION_TOKENS", "EVAL_REASONING_EFFORT", "SWI_REPO_ROOT", "DOCKER_CONFIG",
            "SWI_REDO_ARM", "SWI_REDO_OVERLAY", "SWI_MCP_TIMEOUT_MS", machine.ENV_VAR, "GOOGLE_BASE_URL",
            "GEMINI_API_BASE")
    return {k: env.get(k) for k in keep}


def harbor_cmd(cfg: dict, rung: str, task: str, jobs_dir: Path, job_name: str) -> list[str]:
    h = cfg["harness"]
    r = cfg["rungs"][rung]
    variant = r.get("variant") or h["variant"]  # a rung may override the harness-wide thinking variant
    return [str(HARBOR), "run", "--yes",
            "-p", str((TASKS / task).resolve()),
            "-a", h["agent_import_path"], "-m", r["model"],
            "--ak", f"version={h['opencode_version']}", "--ak", f"variant={variant}",
            "-e", "docker", "--no-force-build", "-k", "1", "-n", "1",
            "-o", str(jobs_dir.resolve()), "--job-name", job_name]


def trial_dirs(job_dir: Path) -> list[Path]:
    return [d for d in job_dir.iterdir() if d.is_dir() and (d / "config.json").exists()] if job_dir.exists() else []


def trial_complete(job_dir: Path) -> bool:
    for d in trial_dirs(job_dir):
        try:
            res = json.loads((d / "result.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if res.get("verifier_result") and not res.get("exception_info"):
            return True
    return False


def write_manifest(run_dir: Path, entry: dict) -> None:
    p = run_dir / "manifest.json"
    doc = json.loads(p.read_text()) if p.exists() else {"schema": "redo_v1/swe-interact-run/1", "invocations": []}
    doc["invocations"].append(entry)
    p.write_text(json.dumps(doc, indent=2))


def base_manifest(cfg: dict, stage: str) -> dict:
    return {"stage": stage, "started_at": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
            "host_arch": platform.machine(),
            "machine": {**machine.record(), "results_marker": machine.check_results_root(RESULTS)},
            "tib_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip(),
            "tib_dirty": bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "fsa/redo"],
                                             capture_output=True, text=True).stdout.strip()),
            "argv": sys.argv, **patchset.verify(),
            "rungs_config_sha256": patchset.sha256(Path(__file__).with_name("rungs.json").read_bytes()),
            "agent_module_sha256": patchset.sha256(Path(__file__).with_name("agent.py").read_bytes()),
            "subset": subset_digest()}


def run_one(cfg: dict, rung: str, task: str, run_dir: Path, *, arm: str = "stock", overlay: bool = False,
            tau: float = 0.5, overlay_seed: int = 0, mcp_timeout_ms: int = MCP_TIMEOUT_MS,
            overlay_wait: float | None = None, overlay_mode: str = "standard") -> dict:
    """One Harbor job (= one trial) of ``task``; with an arm or the overlay, a host-side
    ``TrialBroker`` serves the sidecar hooks for the lifetime of the job."""
    jobs = run_dir / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    stem = f"{task[:80]}__{rung}"
    prior = sorted(jobs.glob(f"{stem}__a*"))
    if any(trial_complete(j) for j in prior):
        return {"task": task, "skipped": "complete"}
    job_name = f"{stem}__a{len(prior) + 1}"
    env = seat_env(cfg, task)
    env.update(redo_env(arm, overlay=overlay, mcp_timeout_ms=mcp_timeout_ms))
    cmd = harbor_cmd(cfg, rung, task, jobs, job_name)
    log = run_dir / "logs" / f"{job_name}.log"
    log.parent.mkdir(exist_ok=True)
    broker = None
    if arm != "stock" or overlay:
        from .arms import RedoConfig
        from .broker import OVERLAY_WAIT_SEC, TrialBroker
        broker = TrialBroker(jobs / job_name, task, cfg["rungs"][rung]["model"],
                             RedoConfig(arm=arm, tau=tau, overlay=overlay, overlay_seed=overlay_seed,
                                        overlay_mode=overlay_mode),
                             overlay_wait=overlay_wait or OVERLAY_WAIT_SEC).start()
    t0 = time.time()
    try:
        with SubnetLease() as lease, open(log, "w") as f:
            env["SWI_SUBNET"] = lease.subnet
            f.write(f"# SWI_SUBNET={lease.subnet}\n")
            f.flush()
            rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    finally:
        if broker is not None:
            broker.stop()
    out = {"task": task, "job": job_name, "rc": rc, "wall_sec": round(time.time() - t0, 1), "log": str(log),
           "subnet": lease.subnet, "complete": trial_complete(jobs / job_name), "arm": arm, "overlay": overlay}
    if broker is not None:
        out["broker_errors"] = broker.errors[:20]
    return out


def cmd_trials(a) -> int:
    cfg = load_rungs()
    rung = cfg["rungs"][a.rung]
    tasks = a.tasks.split(",")
    bad = [t for t in tasks if t not in set(subset_tasks())]
    if bad:
        raise SystemExit(f"tasks outside the frozen effective subset: {bad}")
    workers = min(a.workers, rung["stage0_max_concurrent_trials"]) if a.stage0 else a.workers
    run_dir = Path(a.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ANON_DOCKER_CONFIG.mkdir(parents=True, exist_ok=True)
    (ANON_DOCKER_CONFIG / "config.json").write_text("{}")
    man = base_manifest(cfg, "trials")
    man.update({
        "rung": a.rung,
        "agent": {"model": rung["model"], "thinking": rung["thinking"], "harness": cfg["harness"],
                  "provider_record": _gemini_record(rung["model"].split("/", 1)[1])},
        "user_sim": {**cfg["seats"]["user_sim"], "provider_record": _gemini_record(cfg["seats"]["user_sim"]["model"])},
        "rubric_grader": {**cfg["seats"]["rubric_grader"],
                          "provider_record": _gemini_record(cfg["seats"]["rubric_grader"]["model"])},
        "judge_seats": judge_seats_record(),
        "harbor": cfg["harbor"],
        "tasks": {t: {"family": family(t), "repo_root": repo_root(t), **image_record(t)} for t in tasks},
        "sidecar_base": sidecar_base_record(),
        "workers": workers,
        "commands": {t: harbor_cmd(cfg, a.rung, t, run_dir / "jobs", f"{t[:80]}__{a.rung}__a<N>") for t in tasks},
        "env": public_env(seat_env(cfg, tasks[0])) | {"SWI_REPO_ROOT": {t: repo_root(t) for t in tasks}},
    })
    missing = [t for t, r in man["tasks"].items() if not r["present"]]
    if missing:
        raise SystemExit(f"native {machine.host_arch()} task images missing locally or stale (pull / build them "
                         f"first; amd64: python -m fsa.redo.swe_interact.amd64 build): {missing}")
    write_manifest(run_dir, man)
    print(f"{len(tasks)} task(s) on {a.rung} ({rung['model']}), workers={workers}; run dir {run_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda t: run_one(cfg, a.rung, t, run_dir), tasks))
    for r in results:
        print(json.dumps(r), flush=True)
    write_manifest(run_dir, {"stage": "trials_done", "finished_at": datetime.now(timezone.utc).isoformat(),
                             "results": results})
    return 0


def main(argv: list[str] | None = None) -> int:
    machine.apply_process_env()  # this machine's endpoint profile, for this process and its children
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trials")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--rung", required=True, choices=list(load_rungs()["rungs"]))
    t.add_argument("--tasks", required=True)
    t.add_argument("--workers", type=int, default=3)
    t.add_argument("--no-stage0-cap", dest="stage0", action="store_false")
    a = ap.parse_args(argv)
    return cmd_trials(a)


if __name__ == "__main__":
    sys.exit(main())
