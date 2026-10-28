"""A5.11 SWE-Interact lane (REVIEWER_EXPS_PREREGISTRATION.md A5.11, rules in A5.13, gate and smoke rules in A5.14): gold
checks, the A5.11 intent index, launch conditions, the lane runner, the per-task test-file globs and the /tmp smoke
check.

    python -u -m fsa.redo.swe_interact.a5_lane gold [--tasks a,b] [--workers 3]   # (ii)  -> a5_11/gold.json
    python -m fsa.redo.swe_interact.a5_lane freeze                                # A5.13 (4) -> a5_11/index.json
    python -m fsa.redo.swe_interact.a5_lane status                                # launch conditions (i)-(vi)
    python -u -m fsa.redo.swe_interact.a5_lane run --rung F --run-dir results/swe_interact_a5/F --workers N
    python -u -m fsa.redo.swe_interact.a5_lane run --smoke --rung F --run-dir /tmp/swi_a5_smoke/<ts> \\
        --tasks <task> --workers 4                                                # (vi), never scored
    python -m fsa.redo.swe_interact.a5_lane smoke-check --run-dir /tmp/swi_a5_smoke/<ts>

Gold check (A5.13 (2)): the task's amd64 image exists with the current recipe; the reference patch passes every test
of fail_to_pass + pass_to_pass (absent = failed; the canonical_test.sh rule); the unmodified base fails at least one
fail_to_pass test. A run that is not ``ok`` (infra_error / timeout / no_test_output) is re-run once in a fresh
container. Exclusion (A5.13 (4)): gold check failed, or intent set not validated (the 7 Stage-1 sets keep their
Stage-1 validation). Trials: included tasks x ``a5.ARMS``, task by task, one Harbor job per (task, arm) through
``run.run_one`` with ``SWI_REDO_ARM=a5_<arm>``, jobs under ``<run-dir>/<arm>/jobs/``.

Smoke (A5.14 (2)): ``smoke_arm`` checks the sidecar gate (``user-sim/a5/gate.json``, ``a5.Gate``):
``gate_before_first_reply`` (artifact arms: the artifact was demanded at the first ``ask_user`` call at step >=
02_implement and resolved before the first simulator reply at step >= 2 was delivered; no-artifact arms: never
withheld) and ``pre_review_snapshot_graded`` (the sidecar snapshot at the gate call graded ``ok``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import a5
from .config import ROOT, subset_tasks

A5_DIR = ROOT / "configs" / "intents" / "swe-interact" / "a5_11"
GOLD = A5_DIR / "gold.json"
INDEX = A5_DIR / "index.json"
STAGE1_INDEX = ROOT / "configs" / "intents" / "swe-interact" / "index.json"
REGISTERED_TEXTS_SHA256 = "5efff2ee4c3d9d9d5ccf09500622dc3edd4b8cec7457132eba489cc1b0fb999f"   # A5.14 (6); A5.11 (6) was ef794a32...
F_ROWS = ROOT / "results" / "swe_pushback" / "F_low" / "rows.jsonl"
F_MIN_TASKS = 20                                                                               # A5.11 (7)(i)
SMOKE_RECORD = ROOT / ".logs" / "swi_scaleup" / "smoke_A5_14.json"   # A5.14 (2); the A5.13 record failed
RETRY_STATUSES = frozenset({"infra_error", "timeout", "no_test_output"})
FAMILY = "swebenchpro"
NOTE_ARMS = ("rule_only", "review_only", "commit_before", "claims_p_noevid", "stub_3")   # arms with a post-gate note
GATE_FILE = Path("user-sim") / "a5" / "gate.json"


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


def a5_tasks() -> list[str]:
    from .amd64 import swebenchpro_tasks
    return swebenchpro_tasks()


def stage1_tasks() -> list[str]:
    return sorted(set(a5_tasks()) & set(subset_tasks()))


def new_tasks() -> list[str]:
    return sorted(set(a5_tasks()) - set(subset_tasks()))


def required_tests(task: str) -> dict[str, list[str]]:
    from .intents import _config, _listish
    c = _config(task)
    return {"fail_to_pass": list(_listish(c.get("fail_to_pass") or [])),
            "pass_to_pass": list(_listish(c.get("pass_to_pass") or []))}


def task_config(task: str) -> dict:
    """The hidden-test config the verifier uses (``steps/05_test_handoff/tests/config.json``)."""
    from .exec_judge import tests_dir
    return json.loads((tests_dir(task) / "config.json").read_text())


def task_test_globs(task: str) -> list[str]:
    """A5.14 (5): the task's test-file globs = the language's conventions + the exact paths the hidden test_patch
    touches (``a5.test_globs``); recorded per task in every lane manifest."""
    c = task_config(task)
    return a5.test_globs(c.get("repo_language"), c.get("test_patch"))


def task_nonsource_globs(task: str) -> list[str]:
    """A5.14 (5) as decided 02:36Z: the per-task exclusion list of the SOURCE-ONLY metrics = ``task_test_globs`` +
    the files the Stage-2 harness instructs the agent to write in the repo (``a5.HARNESS_FILES``: ``plan.md``)."""
    c = task_config(task)
    return a5.nonsource_globs(c.get("repo_language"), c.get("test_patch"))


def whitelist_groups(task: str) -> list[str]:
    """A5.11 (5): the groups of fail_to_pass + pass_to_pass."""
    from .exec_judge import test_group
    req = required_tests(task)
    return sorted({test_group(t, FAMILY) for t in req["fail_to_pass"] + req["pass_to_pass"]})


# ── (ii) gold check ─────────────────────────────────────────────────────────

def _graded(task: str, patch: str, label: str, run) -> dict:
    """One exec run (cached), re-run once in a fresh container if it did not end ``ok``."""
    r = run(task, patch, label=label)
    rec = {"first_status": r.get("status"), "first_key": r.get("key"), "retried": False}
    if r.get("status") in RETRY_STATUSES:
        r = run(task, patch, label=label + "_retry", use_cache=False)
        rec["retried"] = True
    rec.update(status=r.get("status"), key=r.get("key"), wall_sec=r.get("wall_sec"), head=r.get("head"),
               error=r.get("error"), n_tests=len(r.get("tests") or {}))
    return {"rec": rec, "tests": r.get("tests") or {}}


def gold_check(task: str, *, run=None, image_state=None) -> dict:
    """A5.13 (2). ``run`` = ``exec_judge.run_snapshot`` (injectable); ``image_state`` -> {exists, arch, recipe_ok,
    layout_ok}."""
    from .intents import reference_patch
    if run is None:
        from .exec_judge import run_snapshot as run
    if image_state is None:
        image_state = _image_state
    reasons = []
    img = image_state(task)
    if not (img.get("exists") and img.get("arch") == "amd64" and img.get("recipe_ok") and img.get("layout_ok")):
        reasons.append("image")
    req = required_tests(task)
    required = set(req["fail_to_pass"]) | set(req["pass_to_pass"])
    ref = _graded(task, reference_patch(task), "gold_reference", run)
    passed = {t for t, s in ref["tests"].items() if str(s).upper() == "PASSED"}
    ref["rec"].update(n_required=len(required), missing_or_failing=sorted(required - passed)[:50],
                      n_missing_or_failing=len(required - passed))
    if ref["rec"]["status"] != "ok":
        reasons.append(f"reference_{ref['rec']['status']}")
    elif not required or required - passed:
        reasons.append("reference_not_all_required_pass")
    base = _graded(task, "", "gold_base", run)
    f2p = set(req["fail_to_pass"])
    bpass = {t for t, s in base["tests"].items() if str(s).upper() == "PASSED"}
    base["rec"].update(n_fail_to_pass=len(f2p), n_fail_to_pass_not_passed=len(f2p - bpass))
    if base["rec"]["status"] != "ok":
        reasons.append(f"base_{base['rec']['status']}")
    elif not (f2p - bpass):
        reasons.append("base_passes_all_fail_to_pass")
    return {"task": task, "image": img, "reference": ref["rec"], "base": base["rec"], "pass": not reasons,
            "reasons": reasons, "checked_at": datetime.now(timezone.utc).isoformat()}


def _image_state(task: str) -> dict:
    from . import amd64
    from .config import task_image
    st = amd64.image_state(task)
    return {**st, "recipe_ok": st.get("recipe_sha256") == amd64.recipe(task)["dockerfile_sha256"],
            "layout_ok": task_image(task) == amd64.image_tag(task)}


def write_gold(results: dict[str, dict]) -> dict:
    from .exec_judge import RECIPE_VERSION
    old = json.loads(GOLD.read_text()) if GOLD.exists() else {"tasks": {}}
    doc = {"schema": "a5_11/swi-gold/1", "rule": "A5.13 (2)", "recipe_version": RECIPE_VERSION,
           "written_at": datetime.now(timezone.utc).isoformat(), "tasks": {**old.get("tasks", {}), **results}}
    doc["n_pass"] = sum(1 for r in doc["tasks"].values() if r["pass"])
    A5_DIR.mkdir(parents=True, exist_ok=True)
    GOLD.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


# ── A5.13 (4) exclusion + freeze ────────────────────────────────────────────

def decide(tasks: list[str], gold: dict, stage1_index: dict, load_intents) -> dict:
    """Included / excluded per A5.13 (4). ``load_intents(task)`` -> the (digest-verified) intent doc."""
    inc, exc = {}, {}
    s1 = set(stage1_tasks())
    for t in tasks:
        reasons = []
        g = (gold.get("tasks") or {}).get(t)
        if g is None:
            reasons.append("gold_not_run")
        elif not g.get("pass"):
            reasons.append("gold_failed: " + ",".join(g.get("reasons") or []))
        try:
            doc = load_intents(t)
        except (SystemExit, FileNotFoundError, json.JSONDecodeError) as e:
            exc[t] = {"reasons": reasons + [f"intent_set_unreadable: {e}"]}
            continue
        if t in s1:
            ent = (stage1_index.get("tasks") or {}).get(t) or {}
            val_pass, val_src = ent.get("validation_pass"), "stage1_index"
            if ent.get("digest") != doc["digest"]:
                reasons.append("intent_digest_differs_from_stage1_index")
        else:
            v = doc.get("validation") or {}
            val_pass, val_src = v.get("pass"), "a5_13_validation"
        if val_pass is not True:
            reasons.append("intent_validation_failed" if val_pass is False else "intent_validation_missing")
        ent = {"digest": doc["digest"], "n_intents": len(doc["intents"]),
               "n_exec": sum(1 for i in doc["intents"] if i.get("judge") == "exec"),
               "validation_pass": val_pass, "validation_source": val_src,
               "validation": {k: (doc.get("validation") or {}).get(k) for k in ("reference", "base")}
               if t not in s1 else None,
               "gold_pass": (g or {}).get("pass")}
        if ent["validation"]:
            ent["validation"] = {k: (x or {}).get("all" if k == "reference" else "positive")
                                 for k, x in ent["validation"].items()}
        if reasons:
            exc[t] = {**ent, "reasons": reasons}
        else:
            inc[t] = ent
    return {"included": inc, "excluded": exc}


def freeze() -> dict:
    from .intents import load
    gold = json.loads(GOLD.read_text())
    s1 = json.loads(STAGE1_INDEX.read_text())
    d = decide(a5_tasks(), gold, s1, load)
    idx = {"schema": "a5_11/swi-intents-index/1", "rule": "A5.13 (4)", "frozen_at": datetime.now(timezone.utc).isoformat(),
           "note": "A5.11 task list = included; the Stage-1 index.json is not rewritten",
           "gold_sha256": _sha(GOLD.read_bytes()), "stage1_index_sha256": _sha(STAGE1_INDEX.read_bytes()),
           "tasks": d["included"], "excluded": d["excluded"]}
    idx["index_digest"] = _sha(json.dumps({"tasks": {t: e["digest"] for t, e in d["included"].items()},
                                           "excluded": sorted(d["excluded"])}, sort_keys=True))
    A5_DIR.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(idx, indent=1, sort_keys=True) + "\n")
    return idx


def included_tasks() -> list[str]:
    return sorted(json.loads(INDEX.read_text())["tasks"]) if INDEX.exists() else []


# ── launch conditions ───────────────────────────────────────────────────────

def f_condition(rows_path: Path = F_ROWS, arms=None) -> dict:
    """A5.11 (7)(i): dspush F tasks whose 6 arms each have a row with ``reward_pre`` not null."""
    if arms is None:
        from ..swe_pushback import texts as T
        arms = T.ARMS
    by: dict[str, set] = {}
    try:
        lines = Path(rows_path).read_text().splitlines()
    except OSError:
        lines = []
    for ln in lines:
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("reward_pre") is not None:
            by.setdefault(r.get("task"), set()).add(r.get("arm"))
    done = sorted(t for t, s in by.items() if set(arms) <= s)
    return {"rows": str(rows_path), "n_tasks_all_arms": len(done), "need": F_MIN_TASKS, "ok": len(done) >= F_MIN_TASKS,
            "tasks": done}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def tree_condition() -> dict:
    head = _git("rev-parse", "HEAD").stdout.strip()
    dirty = bool(_git("status", "--porcelain", "fsa/redo", "configs/intents/swe-interact").stdout.strip())
    pushed = _git("merge-base", "--is-ancestor", "HEAD", "origin/main").returncode == 0
    return {"head": head, "dirty": dirty, "on_origin_main": pushed, "ok": bool(head) and not dirty and pushed}


def conditions(*, smoke: bool = False, tasks: list[str] | None = None) -> dict:
    from . import patchset
    out: dict = {}
    out["i_dspush_F"] = f_condition() if not smoke else {"ok": True, "skipped": "smoke"}
    inc = included_tasks()
    gold = json.loads(GOLD.read_text()) if GOLD.exists() else {"tasks": {}}
    want = tasks or inc
    out["ii_gold"] = {"ok": bool(want) and all((gold["tasks"].get(t) or {}).get("pass") for t in want),
                      "n_pass": gold.get("n_pass"), "n_checked": len(gold["tasks"])}
    out["iii_index"] = {"ok": INDEX.exists() and bool(inc) and all(t in inc for t in want), "n_included": len(inc),
                        "excluded": sorted(json.loads(INDEX.read_text())["excluded"]) if INDEX.exists() else None}
    try:
        v = patchset.verify()
        out["iv_patch"] = {"ok": True, "layer1_sha256": v["layer1_sha256"]}
    except SystemExit as e:
        out["iv_patch"] = {"ok": False, "error": str(e)[:300]}
    ts = a5.texts_sha256()
    out["v_texts"] = {"ok": ts == REGISTERED_TEXTS_SHA256, "texts_sha256": ts}
    if not smoke:
        sm = json.loads(SMOKE_RECORD.read_text()) if SMOKE_RECORD.exists() else {}
        out["vi_smoke"] = {"ok": sm.get("pass") is True, "record": str(SMOKE_RECORD)}
        out["tree"] = tree_condition()
    out["ok"] = all(v.get("ok") for v in out.values() if isinstance(v, dict))
    return out


# ── lane runner ─────────────────────────────────────────────────────────────

def trial_order(tasks: list[str], arms) -> list[tuple[str, str]]:
    """Task by task: all arms of a task before the next task (A5.13 (6))."""
    return [(t, arm) for t in tasks for arm in arms]


def lane_record(cfg: dict, rung: str, tasks: list[str], arms, smoke: bool) -> dict:
    from .run import _gemini_record, base_manifest, image_record, judge_seats_record, sidecar_base_record
    r = cfg["rungs"][rung]
    man = base_manifest(cfg, "a5_11_smoke" if smoke else "a5_11_lane")
    man.update({
        "rung": rung, "smoke": smoke, "seed": 0, "arms": list(arms), "redo_arms": [a5.PREFIX + x for x in arms],
        "agent": {"model": r["model"], "thinking": r["thinking"], "harness": cfg["harness"],
                  "provider_record": _gemini_record(r["model"].split("/", 1)[1])},
        "user_sim": {**cfg["seats"]["user_sim"], "provider_record": _gemini_record(cfg["seats"]["user_sim"]["model"])},
        "judge_seats": judge_seats_record(), "harbor": cfg["harbor"], "sidecar_base": sidecar_base_record(),
        "tasks": {t: {"family": FAMILY, **image_record(t)} for t in tasks},
        "a5_11": {"texts_sha256": a5.texts_sha256(), "registered_texts_sha256": REGISTERED_TEXTS_SHA256,
                  "router_model": a5.ROUTER_MODEL, "gate_tries": a5.GATE_TRIES,
                  "index": {"path": str(INDEX.relative_to(ROOT)), "sha256": _sha(INDEX.read_bytes()),
                            "index_digest": json.loads(INDEX.read_text())["index_digest"],
                            "excluded": json.loads(INDEX.read_text())["excluded"]} if INDEX.exists() else None,
                  "gold": {"path": str(GOLD.relative_to(ROOT)), "sha256": _sha(GOLD.read_bytes())}
                  if GOLD.exists() else None,
                  "prereg_commit": _git("log", "-n", "1", "--format=%H", "--",
                                        "docs/results/REVIEWER_EXPS_PREREGISTRATION.md").stdout.strip(),
                  "module_sha256": {m: _sha((Path(__file__).parent / m).read_bytes())
                                    for m in ("a5.py", "a5_lane.py", "a5_score.py", "agent.py", "broker.py", "run.py",
                                              "overlay/server.py")},
                  "gate": "A5.14 (1): ask_user sidecar gate", "smoke_record": str(SMOKE_RECORD.relative_to(ROOT)),
                  "test_globs": {t: task_test_globs(t) for t in tasks},
                  "nonsource_globs": {t: task_nonsource_globs(t) for t in tasks},
                  "harness_files": list(a5.HARNESS_FILES)},
    })
    return man


def cmd_run(a) -> int:
    from .config import load_rungs
    from .run import ANON_DOCKER_CONFIG, run_one, write_manifest
    cfg = load_rungs()
    run_dir = Path(a.run_dir).resolve()
    arms = tuple(a.arms.split(",")) if a.arms else a5.ARMS
    bad_arms = [x for x in arms if x not in a5.ALL_ARMS]
    if bad_arms:
        raise SystemExit(f"unknown A5.11 arms {bad_arms}")
    if a.smoke:
        if not str(run_dir).startswith("/tmp/"):
            raise SystemExit("--smoke runs only under /tmp (A5.11 (7)(vi); smoke trials are never scored)")
        tasks = a.tasks.split(",") if a.tasks else []
        if len(tasks) != 1 or tuple(arms) != a5.ARMS:
            raise SystemExit("the smoke is exactly 1 task x the 4 primary arms")
    else:
        if a.tasks:
            raise SystemExit("the lane runs the frozen A5.11 task list (index.json); --tasks is for --smoke only")
        tasks = included_tasks()
        if set(arms) - set(a5.ARMS):
            raise SystemExit("secondary arms run only under a later decision, after the primary trials (A5.11 (4))")
        if str(run_dir).startswith("/tmp/"):
            raise SystemExit("the lane writes to the results tree, not /tmp")
    cond = conditions(smoke=a.smoke, tasks=tasks)
    print(json.dumps(cond, indent=1), flush=True)
    if not cond["ok"] and not a.dry_run:
        raise SystemExit("A5.11 launch conditions not met (see above); nothing launched")
    if a.dry_run:
        print(json.dumps({"dry_run": True, "trials": trial_order(tasks, arms)}, indent=1))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    ANON_DOCKER_CONFIG.mkdir(parents=True, exist_ok=True)
    (ANON_DOCKER_CONFIG / "config.json").write_text("{}")
    man = lane_record(cfg, a.rung, tasks, arms, a.smoke)
    man.update(conditions=cond, workers=a.workers)
    missing = [t for t, r in man["tasks"].items() if not r["present"]]
    if missing:
        raise SystemExit(f"task images missing or stale: {missing}")
    write_manifest(run_dir, man)
    order = trial_order(tasks, arms)
    print(f"A5.11 {'smoke' if a.smoke else 'lane'}: {len(order)} trials on {a.rung}, workers={a.workers}, {run_dir}",
          flush=True)

    def one(ta):
        t, arm = ta
        r = run_one(cfg, a.rung, t, run_dir / arm, arm=a5.PREFIX + arm)
        print(json.dumps({**r, "a5_arm": arm}), flush=True)
        return {**r, "a5_arm": arm}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(one, order))
    write_manifest(run_dir, {"stage": "a5_11_done", "finished_at": datetime.now(timezone.utc).isoformat(),
                             "results": results})
    return 0


# ── (vi) smoke check ────────────────────────────────────────────────────────

def latest_trial(arm_dir: Path, task: str, rung: str) -> Path | None:
    from .run import trial_dirs
    jobs = sorted((arm_dir / "jobs").glob(f"{task[:80]}__{rung}__a*"), key=lambda p: int(p.name.rsplit("__a", 1)[1]))
    for j in reversed(jobs):
        ds = trial_dirs(j)
        if ds:
            return ds[0]
    return None


def read_asks(trial: Path) -> list[dict]:
    out = []
    p = trial / "user-sim" / "asks.jsonl"
    for ln in (p.read_text().splitlines() if p.exists() else []):
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def _jload(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def smoke_arm(trial: Path | None, arm: str, task: str, *, complete: bool, run=None) -> dict:
    """A5.14 (2) checks for one arm's trial (the A5.13 (7) checks with the sidecar gate)."""
    if run is None:
        from .exec_judge import run_snapshot as run
    rec: dict = {"arm": arm, "trial": str(trial) if trial else None, "complete": complete, "fail": [], "info": []}
    if trial is None:
        rec["fail"].append("no_trial")
        return rec
    if not complete:
        rec["fail"].append("trial_not_complete")
    gate = _jload(trial / GATE_FILE)
    art_arm = a5.artifact_kind(arm) is not None
    gst = gate.get("status") or "never_opened"
    rec.update(gate_status=gst, gate_call=gate.get("call_index"), gate_step=gate.get("step"),
               gate_resolved_call=gate.get("resolved_call"), gate_tries=len(gate.get("tries") or []))
    expected = "ok" if art_arm else ("not_required", "never_opened")
    if (gst != expected) if art_arm else (gst not in expected):
        rec["fail"].append(f"gate_{gst}_expected_{expected if art_arm else 'not_required'}")
    asks = read_asks(trial)
    withheld = [a for a in asks if (a.get("redo") or {}).get("a5_withheld")]
    sims = [a for a in asks if not (a.get("redo") or {}).get("a5_withheld")]
    rec["n_withheld"] = len(withheld)
    for a in asks:
        redo = a.get("redo") or {}
        for hook in ("turn_end", "sim_reply"):
            if hook in redo and not (redo.get(hook) or {}).get("ok"):
                rec["fail"].append(f"broker_{hook}_failed@ask{a.get('call_index')}")
    if withheld and not art_arm:
        rec["fail"].append(f"reply_withheld_in_{arm}")
    pre, post = [], []
    for a in sims:
        redo = a.get("redo") or {}
        sr = redo.get("sim_reply") or {}
        te_gate = (redo.get("turn_end") or {}).get("a5_gate") or {}
        item = {"call_index": a.get("call_index"), "delivered_ts": redo.get("delivered_ts"), "phase": sr.get("a5_phase"),
                "gate_phase": te_gate.get("phase"), "step": te_gate.get("step"), "note": bool(sr.get("a5_note")),
                "delivered_eq_history": a.get("delivered") == a.get("history_reply"),
                "sim_question": a.get("sim_question")}
        (pre if item["phase"] == "pre_gate" else post).append(item)
    rec["n_pre"], rec["n_post"] = len(pre), len(post)
    for it in pre:   # planning Q&A only: in 01_plan, no note
        if a5.in_gate_steps(it["step"]) or it["note"] or not it["delivered_eq_history"]:
            rec["fail"].append(f"pre_gate_reply_at_step_ge_2_or_with_note@ask{it['call_index']}")
    for it in post:
        if it["phase"] != "post_gate" or not a5.in_gate_steps(it["step"]):
            rec["fail"].append(f"post_gate_reply_not_marked@ask{it['call_index']}")
        elif arm in NOTE_ARMS and (not it["note"] or it["delivered_eq_history"]):
            rec["fail"].append(f"post_gate_reply_without_note@ask{it['call_index']}")
        elif arm == "stock" and (it["note"] or not it["delivered_eq_history"]):
            rec["fail"].append(f"stock_reply_with_note@ask{it['call_index']}")
    # gate_before_first_reply
    first = post[0] if post else None
    gc = gate.get("call_index")
    if art_arm:
        if gc is None:
            rec["fail"].append("gate_before_first_reply:gate_never_opened")
        else:
            w_calls = [a.get("call_index") for a in withheld]
            demand = next((a for a in withheld if a.get("call_index") == gc), None)
            ok = (demand is not None and demand.get("delivered") == a5.gate_prompt(arm)
                  and first is not None and gate.get("resolved_call") == first["call_index"]
                  and first["sim_question"] == gate.get("question")
                  and (first["delivered_ts"] or 0) >= (gate.get("resolved_ts") or float("inf"))
                  and all(gc <= c < first["call_index"] for c in w_calls)
                  and not any(gc <= (it["call_index"] or 0) < first["call_index"] for it in pre + post))
            if not ok:
                rec["fail"].append("gate_before_first_reply")
            p = trial / "user-sim" / "sim_inputs" / f"ask-{first['call_index']}.json" if first else None
            tag = "[" + a5.T.ARTIFACT_TAG[a5.artifact_kind(arm)] + "]"
            if p is None or not p.exists() or tag in p.read_text(errors="replace"):
                rec["fail"].append("artifact_reached_simulator_or_no_sim_input")
    elif gc is not None and (first is None or first["call_index"] != gc):
        rec["fail"].append("gate_before_first_reply:first_reply_at_step_ge_2_not_the_gate_call")
    elif gc is None:
        rec["info"].append("no_ask_at_step_ge_2 (reported; no gate in this arm)")
    # pre_review_snapshot_graded
    if gc is not None:
        sp = trial / "user-sim" / "snapshots" / f"ask-{gc}.patch"
        rec["pre_source"] = f"gate_snapshot_ask_{gc}"
    else:
        sp = trial / "steps" / a5.GATE_STEP / "agent" / "redo" / "step-2-end.patch"
        rec["pre_source"] = "step2_end_fallback"
    if sp.exists() and not (gc is not None and _jload(sp.with_suffix(".json")).get("error")):
        g = run(task, sp.read_text(errors="surrogateescape"), label=f"a5_smoke_pre_{arm}")
        rec["pre_graded"] = {"status": g.get("status"), "key": g.get("key"), "n_tests": len(g.get("tests") or {})}
        if g.get("status") != "ok":
            rec["fail"].append(f"pre_review_snapshot_graded:{g.get('status')}")
    else:
        rec["fail"].append("pre_review_snapshot_graded:no_snapshot")
    rec["pass"] = not rec["fail"]
    return rec


def smoke_check(run_dir: Path, *, run=None) -> dict:
    from .run import trial_complete
    man = json.loads((run_dir / "manifest.json").read_text())
    entry = next(e for e in man["invocations"] if e.get("stage") == "a5_11_smoke")
    task, rung = next(iter(entry["tasks"])), entry["rung"]
    arms = {}
    for arm in a5.ARMS:
        tr = latest_trial(run_dir / arm, task, rung)
        arms[arm] = smoke_arm(tr, arm, task, complete=bool(tr) and trial_complete(tr.parent), run=run)
    return {"schema": "a5_14/swi-smoke/1", "rule": "A5.14 (2)", "run_dir": str(run_dir), "task": task, "rung": rung,
            "checked_at": datetime.now(timezone.utc).isoformat(), "arms": arms,
            "pass": all(r["pass"] for r in arms.values())}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    from .. import machine
    machine.apply_process_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gold")
    g.add_argument("--tasks", default=None)
    g.add_argument("--workers", type=int, default=3)
    sub.add_parser("freeze")
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--rung", default="F")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--workers", type=int, required=True)
    r.add_argument("--arms", default=None)
    r.add_argument("--tasks", default=None)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="check conditions and print the trial order only")
    s = sub.add_parser("smoke-check")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--out", default=str(SMOKE_RECORD))
    a = ap.parse_args(argv)
    if a.cmd == "gold":
        tasks = a.tasks.split(",") if a.tasks else a5_tasks()
        res: dict = {}

        def one(t):
            t0 = time.time()
            try:
                res[t] = gold_check(t)
            except Exception as e:  # noqa: BLE001 - recorded as a failed check
                res[t] = {"task": t, "pass": False, "reasons": [f"error: {type(e).__name__}: {e}"[:300]]}
            print(json.dumps({"task": t, "pass": res[t]["pass"], "reasons": res[t]["reasons"],
                              "sec": round(time.time() - t0, 1)}), flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(one, tasks))
        doc = write_gold(res)
        print(f"gold: {doc['n_pass']}/{len(doc['tasks'])} pass -> {GOLD}", flush=True)
    elif a.cmd == "freeze":
        idx = freeze()
        print(json.dumps({"included": sorted(idx["tasks"]), "excluded": idx["excluded"],
                          "index_digest": idx["index_digest"]}, indent=1))
    elif a.cmd == "status":
        print(json.dumps(conditions(), indent=1))
    elif a.cmd == "run":
        return cmd_run(a)
    else:
        rec = smoke_check(Path(a.run_dir).resolve())
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec, indent=1))
        return 0 if rec["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
