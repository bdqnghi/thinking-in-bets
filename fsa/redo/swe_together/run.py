"""SWE-Together runner wrapper (redo_v1).

    # Stage 1 of the benchmark: agent + simulator trials
    python -u -m fsa.redo.swe_together.run trials --run-dir results/redo_v1/stage0/W_bringup \
        --rung W --tasks cli-task-2026-0,cli-task-cd4662 [--env-type e2b|docker] [--workers 3]

    # Stage 2 of the benchmark: tagger + intent coverage + aggregate (the codex/gpt-5.5
    # correctness judge seat is dropped since 2026-09-27: --skip-correctness)
    python -u -m fsa.redo.swe_together.run judge --run-dir results/redo_v1/stage0/W_bringup

    # Stage 1 arms / overlay (clone module src/user_agent/redo_arms.py; stock = unset)
    ... trials ... --arm commit_before|claims_after|claims_no_p|recap|minimal_edit [--tau 0.5]
                   [--overlay --overlay-seed 0 [--trigger agentic_v3|v1]]

The overlay trigger defaults to the frozen ``agentic_v3`` one
(``configs/intents/swe-together/OVERLAY_TRIGGER_FREEZE.json``, ``SWT_REDO_TRIGGER``); it needs a clone
revision that supports it (Stage 1.2, ``redo_v1_trigger.patch``: select with ``SWT_CLONE_DIR`` /
``SWT_PATCH_FILE``); an overlay run on a revision without it is refused.

Every invocation verifies that the clone is the pinned commit plus the recorded
patch, and appends a manifest (exact model ids + provider-reported versions,
reasoning settings, seat backends, benchmark commit, patch sha256, sandbox and
image refs, command line) to ``<run-dir>/manifest.json``. Raw trial dirs are
never deleted; benchmark-side retries archive failed attempts under
``trials/_failed/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .. import machine
from . import patchset
from .config import CLONE, DEFAULT, PY, RESULTS, ROOT, Revision, load_rungs, subset_tasks

CODEX_HOME = RESULTS / ".secrets" / "codex_apikey_home"


# ── provenance helpers ──────────────────────────────────────────────────────

def _sha256_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gemini_model_record(model_id: str) -> dict:
    """Provider-side record of a Gemini model id (resolves the alias to a version), fetched through
    the machine profile's endpoint (``fsa.redo.machine``; default: the public Gemini API)."""
    return machine.model_record(model_id)


def openai_model_record(model_id: str) -> dict:
    try:
        req = urllib.request.Request(f"https://api.openai.com/v1/models/{model_id}",
                                     headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
        return {k: d.get(k) for k in ("id", "created", "owned_by")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def task_image_refs(tasks: list[str], env_type: str, image_template: str | None) -> dict:
    import tomllib
    sys.path.insert(0, str(CLONE))
    out = {}
    for t in tasks:
        td = CLONE / "tasks" / t
        cfg = tomllib.loads((td / "task.toml").read_text())
        rec = {"official_image": cfg.get("environment", {}).get("docker_image")}
        try:
            from dirhash import dirhash
            rec["e2b_template_alias"] = f"tb-{t}__{dirhash(str(td / 'environment'), 'sha256')[:8]}".replace(".", "-")
        except Exception as e:  # noqa: BLE001
            rec["e2b_template_alias"] = f"unavailable: {e}"
        if env_type == "docker":
            img = image_template.format(task=t) if image_template else rec["official_image"]
            rec["trial_image"] = img
            r = subprocess.run(["docker", "image", "inspect", img, "--format",
                                "{{.Id}} {{.Architecture}} {{json .RepoDigests}}"],
                               capture_output=True, text=True)
            rec["trial_image_local"] = r.stdout.strip() or r.stderr.strip()[:200]
        else:
            rec["trial_image"] = rec["official_image"] + " (E2B template)"
        out[t] = rec
    return out


def ensure_codex_auth() -> Path:
    """API-key auth for the codex judge (never the host's ChatGPT OAuth file)."""
    auth = CODEX_HOME / "auth.json"
    if not auth.exists():
        CODEX_HOME.mkdir(parents=True, exist_ok=True)
        os.chmod(CODEX_HOME, 0o700)
        subprocess.run(["codex", "login", "--with-api-key"], input=os.environ["OPENAI_API_KEY"],
                       text=True, check=True, env={**os.environ, "CODEX_HOME": str(CODEX_HOME)},
                       capture_output=True)
    d = json.loads(auth.read_text())
    if d.get("auth_mode") != "apikey" or d.get("tokens"):
        raise SystemExit(f"{auth} is not an API-key auth file")
    return auth


def redo_arms_module(rev: Revision | None = None):
    """The clone's arm module (stdlib-only at import), loaded by path."""
    import importlib.util
    clone = (rev or DEFAULT).clone
    name = "swt_redo_arms" if clone == CLONE else f"swt_redo_arms_{hashlib.sha256(str(clone).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(name, clone / "src" / "user_agent" / "redo_arms.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TRIGGERS = ("v1", "agentic_v3")
DEFAULT_TRIGGER = "agentic_v3"  # the frozen overlay trigger (OVERLAY_TRIGGER_FREEZE.json), since 2026-09-28


def redo_env(arm: str = "stock", *, tau: float = 0.5, overlay: bool = False, overlay_seed: int = 0,
             trigger: str | None = None) -> dict[str, str]:
    """Environment that selects an arm / the overlay in the clone (empty for stock). ``trigger`` (overlay
    only): ``agentic_v3`` sets ``SWT_REDO_TRIGGER``; ``None`` / ``v1`` leaves the historical v1 trigger."""
    env: dict[str, str] = {}
    if arm != "stock":
        env.update(SWT_REDO_ARM=arm, SWT_REDO_TAU=f"{tau:g}")
    if overlay:
        env.update(SWT_REDO_OVERLAY="1", SWT_REDO_OVERLAY_SEED=str(overlay_seed))
        if trigger not in (None, "v1"):
            if trigger not in TRIGGERS:
                raise SystemExit(f"unknown overlay trigger {trigger!r}")
            env["SWT_REDO_TRIGGER"] = trigger
    if env:
        env["SWT_REDO_TIB_ROOT"] = str(ROOT)
        env["SWT_REDO_JUDGE_CACHE"] = str(RESULTS / "intent_judge_cache")
    return env


def trigger_freeze_record() -> dict:
    """The frozen trigger this tib checkout serves (verified against the judge freeze)."""
    from . import agentic_judge as aj
    cfg = aj.load_trigger_freeze()
    return {"trigger": "agentic_v3", "file": str(aj.TRIGGER_FREEZE_FILE.relative_to(ROOT)),
            "sha256": aj.freeze_digest(aj.TRIGGER_FREEZE_FILE), "judge_freeze_sha256": aj.freeze_digest(),
            "config": cfg, "module": "fsa/redo/swe_together/agentic_judge.py",
            "module_sha256": _sha256_file(ROOT / "fsa" / "redo" / "swe_together" / "agentic_judge.py"),
            "container_slots": "host-wide flock slots <agentic_judge_cache>/.slots (config slots; SWT_JUDGE3_SLOTS "
                               "lowers it), shared with offline v3 judging"}


def check_trigger_support(trigger: str | None, rev: Revision | None = None) -> None:
    """Refuse an agentic trigger on a clone revision whose overlay would silently use the v1 trigger."""
    if trigger in (None, "v1"):
        return
    mod = redo_arms_module(rev)
    if trigger not in getattr(mod, "TRIGGERS", ("v1",)):
        raise SystemExit(f"clone {(rev or DEFAULT).clone} (patch {(rev or DEFAULT).patch_file.name}) does not support "
                         f"SWT_REDO_TRIGGER={trigger}; select the Stage 1.2 revision (redo_v1_trigger.patch)")


def redo_record(arm: str, *, tau: float, overlay: bool, overlay_seed: int, trigger: str | None = None,
                rev: Revision | None = None) -> dict:
    mod = redo_arms_module(rev)
    rec = {"arm": arm, "tau": tau if mod.CLAIM_TIMING.get(arm) and arm != "claims_no_p" else None,
           "claim_timing": mod.CLAIM_TIMING.get(arm), "overlay": overlay,
           "overlay_seed": overlay_seed if overlay else None, "max_overlays": mod.MAX_OVERLAYS if overlay else None,
           "claim_book_seat": ({"model": "agent's own model via host LiteLLM (gemini/<rung model>)",
                                "reasoning_effort": "high (litellm -> thinkingLevel=high)",
                                "response_format": "json_object"} if mod.CLAIM_TIMING.get(arm) else None),
           "clone_module": "src/user_agent/redo_arms.py",
           "prompts_sha256": mod.prompts_sha256(), "env": redo_env(arm, tau=tau, overlay=overlay,
                                                                   overlay_seed=overlay_seed, trigger=trigger),
           "clone_module_sha256": _sha256_file((rev or DEFAULT).clone / "src" / "user_agent" / "redo_arms.py")}
    if overlay and trigger not in (None, "v1"):
        check_trigger_support(trigger, rev)
        rec["overlay_trigger"] = trigger_freeze_record()
    elif overlay:
        from ..intent_judge import judge_config
        rec["overlay_trigger"] = {"trigger": "v1"}
        rec["intent_judge"] = judge_config()
        rec["intent_judge_module_sha256"] = _sha256_file(ROOT / "fsa" / "redo" / "intent_judge.py")
    return rec


def sandbox_key(env_type: str) -> str:
    """rungs.json sandbox of a run: ``e2b``, or local docker with native images for this host's
    architecture (``docker_arm64``: the arm64 rebuild template; ``docker_amd64``: the official
    task images). Never an emulated architecture."""
    if env_type != "docker":
        return "e2b"
    return "docker_arm64" if machine.host_arch() == "arm64" else "docker_amd64"


def seat_env(cfg: dict, env_type: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SWT_REDO_")}
    env.update({
        "PYTHONUNBUFFERED": "1",
        "SWT_SANDBOX": env_type,
        "HARBOR_TEAM_PREFIX": "tb",
    })
    j = cfg["seats"]["judge"]
    if judge_seat_active(cfg):  # codex/gpt-5.5 correctness judge (dropped 2026-09-27)
        env.update({
            "SWT_JUDGE_BACKEND": "codex",
            "CODEX_JUDGE_MODEL": j["model"].split("/", 1)[1],
            "CODEX_CLI_VERSION": j["codex_cli_version"],
            "SWT_CODEX_REASONING_EFFORT": j["reasoning_effort"],
            "SWT_CODEX_AUTH_JSON": str(ensure_codex_auth()),
        })
    else:  # no OpenAI key reaches the benchmark (Gemini-only seats)
        for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "SWT_CODEX_AUTH_JSON"):
            env.pop(k, None)
    if os.environ.get("GEMINI_API_KEY"):
        env.setdefault("GOOGLE_GENERATIVE_AI_API_KEY", os.environ["GEMINI_API_KEY"])
    # endpoint profile of this machine: public Gemini API (nothing changes) or a local endpoint
    # (keys, LiteLLM's GEMINI_API_BASE for the simulator/tagger/claim book on the host, and
    # SWT_GOOGLE_BASE_URL for opencode's google provider inside the trial container)
    machine.apply_env(env)
    sb = cfg["sandboxes"][sandbox_key(env_type)]
    env["TRIAL_BUDGET_SEC"] = str(sb["trial_budget_sec"])
    if env_type == "docker":
        if sb.get("image_template"):
            env["SWT_DOCKER_IMAGE_TEMPLATE"] = sb["image_template"]
        else:  # the task's official image (task.toml), native on this host
            env.pop("SWT_DOCKER_IMAGE_TEMPLATE", None)
        env["SWT_DOCKER_EXEC_USER"] = "root"  # parity with E2B, which execs as root
    else:
        env.pop("SWT_DOCKER_IMAGE_TEMPLATE", None)
        env["SWT_E2B_SANDBOX_TIMEOUT_SEC"] = str(sb["sandbox_timeout_sec"])
    return env


def judge_seat_active(cfg: dict) -> bool:
    return bool((cfg["seats"].get("judge") or {}).get("model")) and cfg["seats"]["judge"].get("status") != "dropped"


def judge_seats_record(cfg: dict) -> dict:
    """The host-side judge seats in force (per-intent judge / overlay trigger / mapper) and the
    SWE-Together judge freeze, recorded in every new manifest."""
    from .. import llm_seats
    seat = llm_seats.judge_seat()
    fz = ROOT / "configs" / "intents" / "swe-together" / "JUDGE_FREEZE.json"
    return {"judge_seat": llm_seats.seat_record(seat["model"], seat["reasoning_effort"], source=seat["source"]),
            "seats_file": str(llm_seats.SEATS_FILE.relative_to(ROOT)),
            "seats_file_sha256": _sha256_file(llm_seats.SEATS_FILE),
            "judge_freeze": ({"file": str(fz.relative_to(ROOT)), "sha256": _sha256_file(fz)} if fz.exists() else None),
            "correctness_judge": cfg["seats"]["judge"]}


def write_manifest(run_dir: Path, entry: dict) -> None:
    p = run_dir / "manifest.json"
    doc = json.loads(p.read_text()) if p.exists() else {"schema": "redo_v1/swe-together-run/1", "invocations": []}
    doc["invocations"].append(entry)
    p.write_text(json.dumps(doc, indent=2))


def base_manifest(cfg: dict, args, stage: str, rev: Revision | None = None) -> dict:
    rev = rev or DEFAULT
    prov = patchset.verify(rev)
    oc_ver = cfg["harness"]["opencode_version"]
    return {
        "stage": stage,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "host_arch": platform.machine(),
        "machine": {**machine.record(), "results_marker": machine.check_results_root(RESULTS)},
        "tib_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
        "argv": sys.argv,
        **prov,
        "rungs_config_sha256": _sha256_file(Path(__file__).with_name("rungs.json")),
        "opencode": {
            "version": oc_ver,
            "binary_sha256": {plat: _sha256_file(rev.clone / "enroot_images" / "tools" / "opencode" / oc_ver / plat / "opencode")
                              for plat in ("linux-x64", "linux-arm64")},
        },
    }


# ── commands ────────────────────────────────────────────────────────────────

def trials_command(cfg: dict, rung_name: str, env_type: str, trials_dir: Path, tasks: list[str], workers: int,
                   tag: str, skip_existing: bool, rev: Revision | None = None) -> list[str]:
    rung = cfg["rungs"][rung_name]
    cmd = [str((rev or DEFAULT).py), "-u", "src/run_eval.py",
           "--model", rung["model"], "--agent-type", cfg["harness"]["agent_type"],
           "--opencode-version", cfg["harness"]["opencode_version"],
           "--reasoning-effort", cfg["harness"]["reasoning_effort_flag"],
           "--user-model", cfg["seats"]["user_sim"]["model"],
           "--agent-timeout", str(cfg["sandboxes"][sandbox_key(env_type)]["agent_timeout_sec"]),
           "--user-context-chars", str(cfg["harness"]["user_context_chars"]),
           "--tag", tag, "--workers", str(workers),
           "--trials-dir", str(trials_dir), "--tasks", ",".join(tasks),
           "--env-type", env_type]
    if skip_existing:
        cmd.append("--skip-existing")
    return cmd


def trials_manifest(cfg: dict, args, rung_name: str, env_type: str, env: dict, tasks: list[str], workers: int,
                    trials_dir: Path, cmd: list[str], redo: dict, cell: dict | None = None,
                    images: dict | None = None, rev: Revision | None = None) -> dict:
    rung = cfg["rungs"][rung_name]
    user_model = cfg["seats"]["user_sim"]["model"]
    man = base_manifest(cfg, args, "trials", rev)
    man.update({
        "rung": rung_name,
        "agent": {"model": rung["model"], "opencode_model": rung["opencode_model"], "thinking": rung["thinking"],
                  "provider_record": gemini_model_record(rung["model"].split("/", 1)[1])},
        "user_sim": {**cfg["seats"]["user_sim"],
                     "provider_record": gemini_model_record(user_model.split("/", 1)[1])},
        "harness": cfg["harness"],
        "sandbox": {"env_type": env_type,
                    **cfg["sandboxes"][sandbox_key(env_type)],
                    "env": {k: env.get(k) for k in ("TRIAL_BUDGET_SEC", "SWT_E2B_SANDBOX_TIMEOUT_SEC",
                                                     "SWT_DOCKER_IMAGE_TEMPLATE", "PER_EXEC_CAP_SEC")}},
        "images": images if images is not None else task_image_refs(tasks, env_type, env.get("SWT_DOCKER_IMAGE_TEMPLATE")),
        "tasks": tasks, "workers": workers, "trials_dir": str(trials_dir),
        "redo": redo,
        "judge_seats": judge_seats_record(cfg),
        "command": cmd,
    })
    if cell is not None:
        man["cell"] = cell
    return man


def cmd_trials(args) -> int:
    cfg = load_rungs()
    rung = cfg["rungs"][args.rung]
    tasks = args.tasks.split(",")
    allowed = set(subset_tasks())
    bad = [t for t in tasks if t not in allowed]
    if bad:
        raise SystemExit(f"tasks outside the frozen subset: {bad}")
    cap = rung["stage0_max_concurrent_trials"] if args.stage0 else None
    workers = min(args.workers, cap) if cap else args.workers
    run_dir = Path(args.run_dir).resolve()
    trials_dir = run_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    env_type = args.env_type
    env = seat_env(cfg, env_type)
    trigger = args.trigger if args.overlay else None
    env.update(redo_env(args.arm, tau=args.tau, overlay=args.overlay, overlay_seed=args.overlay_seed, trigger=trigger))
    env.update(DEFAULT.env())
    cmd = trials_command(cfg, args.rung, env_type, trials_dir, tasks, workers, f"{run_dir.name}_{args.rung}",
                         args.skip_existing)
    redo = redo_record(args.arm, tau=args.tau, overlay=args.overlay, overlay_seed=args.overlay_seed, trigger=trigger)
    write_manifest(run_dir, trials_manifest(cfg, args, args.rung, env_type, env, tasks, workers, trials_dir, cmd, redo))
    log = run_dir / f"trials_{args.rung}_{int(time.time())}.log"
    print(f"running {len(tasks)} task(s) on rung {args.rung} ({rung['model']}), arm={args.arm}, "
          f"overlay={args.overlay}, workers={workers}, env={env_type}; log {log}", flush=True)
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=CLONE, env=env, stdout=f, stderr=subprocess.STDOUT)
    write_manifest(run_dir, {"stage": "trials_done", "finished_at": datetime.now(timezone.utc).isoformat(),
                             "rc": rc, "log": str(log)})
    return rc


def cmd_judge(args) -> int:
    cfg = load_rungs()
    run_dir = Path(args.run_dir).resolve()
    # --trials-root: judge a staging dir of symlinks (pilot `judge --completed-only`) instead of
    # run_dir/trials; verdicts still land in the real trial dirs (the benchmark writes through the link)
    trials_dir = Path(args.trials_root).resolve() if getattr(args, "trials_root", None) else run_dir / "trials"
    env = seat_env(cfg, "e2b")  # the judge always runs on E2B (official image), as the benchmark ships
    s = cfg["seats"]
    n_tasks = len({d.name.rsplit("__", 1)[0] for d in trials_dir.iterdir() if "__" in d.name})
    cmd = [str(PY), "-u", "-m", "eval.run_eval",
           "--trials-root", str(trials_dir), "--tasks-root", "tasks",
           "--output-dir", str(run_dir / "eval"), "--model-tag", run_dir.name,
           "--correctness-workers", str(s["judge"].get("workers") or (s["judge"].get("was") or {}).get("workers", 3)),
           "--tag-model", s["tagger"]["model"], "--tag-workers", str(s["tagger"]["workers"]),
           "--intent-coverage-model", s["intent_coverage"]["model"],
           "--intent-coverage-workers", str(s["intent_coverage"]["workers"]),
           "--denom-tasks", str(args.denom_tasks or n_tasks)]
    if not judge_seat_active(cfg):
        cmd.append("--skip-correctness")
    if args.skip_intent_coverage:
        cmd.append("--skip-intent-coverage")
    if getattr(args, "skip_tag_messages", False):
        cmd.append("--skip-tag-messages")
    man = base_manifest(cfg, args, "judge")
    man.update({"judge": ({**s["judge"], "provider_record": openai_model_record(s["judge"]["model"].split("/", 1)[1])}
                          if judge_seat_active(cfg) else s["judge"]),
                "judge_seats": judge_seats_record(cfg),
                "tagger": {**s["tagger"], "provider_record": gemini_model_record(s["tagger"]["model"].split("/", 1)[1])},
                "intent_coverage": s["intent_coverage"],
                "judge_sandbox": "e2b", "command": cmd, "trials_root": str(trials_dir)})
    write_manifest(run_dir, man)
    log = run_dir / f"judge_{int(time.time())}.log"
    print(f"judging {trials_dir} ; log {log}", flush=True)
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=CLONE, env=env, stdout=f, stderr=subprocess.STDOUT)
    write_manifest(run_dir, {"stage": "judge_done", "finished_at": datetime.now(timezone.utc).isoformat(),
                             "rc": rc, "log": str(log)})
    return rc


def main(argv: list[str] | None = None) -> int:
    machine.apply_process_env()  # this machine's endpoint profile, for this process and its children
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trials")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--rung", required=True, choices=["W", "M", "S"])
    t.add_argument("--tasks", required=True)
    t.add_argument("--env-type", default="e2b", choices=["e2b", "docker"])
    t.add_argument("--workers", type=int, default=3)
    t.add_argument("--no-stage0-cap", dest="stage0", action="store_false")
    t.add_argument("--skip-existing", action="store_true")
    t.add_argument("--arm", default="stock",
                   choices=["stock", "commit_before", "claims_after", "claims_no_p", "recap", "minimal_edit"])
    t.add_argument("--tau", type=float, default=0.5)
    t.add_argument("--overlay", action="store_true")
    t.add_argument("--overlay-seed", type=int, default=0)
    t.add_argument("--trigger", default=DEFAULT_TRIGGER, choices=list(TRIGGERS),
                   help="overlay trigger (default: the frozen agentic_v3; v1 = the historical diff-only judge)")
    j = sub.add_parser("judge")
    j.add_argument("--run-dir", required=True)
    j.add_argument("--denom-tasks", type=int, default=None)
    j.add_argument("--skip-intent-coverage", action="store_true")
    j.add_argument("--skip-tag-messages", action="store_true")
    j.add_argument("--trials-root", default=None, help="judge this dir of trial dirs/symlinks instead of run_dir/trials")
    a = ap.parse_args(argv)
    return cmd_trials(a) if a.cmd == "trials" else cmd_judge(a)


if __name__ == "__main__":
    sys.exit(main())
