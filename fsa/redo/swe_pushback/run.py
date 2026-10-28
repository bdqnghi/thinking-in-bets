"""SWE-Pushback-Hidden runner (amendment A5).

    python -u -m fsa.redo.swe_pushback.run --model gemini-3.5-flash --tag F \
        --objections results/swe_pushback/datasets/objections_v1.jsonl \
        --out-dir results/swe_pushback/F_low --workers 4 --prereg-commit <sha>

Jobs: every (task, arm, seed) of the model in one list; task order = sha256(NS:order:task) (arm-independent, the
same for every model), arms shuffled within a task by sha256(NS:arm:model:task:arm), so the arms of a task run close
together in time (outages hit all arms; interim looks are paired). First attempt: a trial that produced a
result.json is never re-run, except status=infra_error *without* a pre-review verifier result (docker crash),
whose dir is moved to _infra/ and re-run (Harbor repair rule, CONFOUND_ARMS.md §3). LLM errors stay failures.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import protocol as P
from . import texts as T
from .agent import Config, Trial
from .docker_env import Task

RUNNER_VERSION = "dspush-v1"
ROOT = Path(__file__).resolve().parents[3]


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def job_order(task_ids: list[str], arms: list[str], model: str, seeds: list[int]) -> list[tuple[str, str, int]]:
    jobs = []
    for seed in seeds:
        for t in sorted(task_ids, key=lambda x: P.h("order", x, seed)):
            for a in sorted(arms, key=lambda x: P.h("arm", model, t, x, seed)):
                jobs.append((t, a, seed))
    return jobs


def trial_dir(out: Path, task: str, arm: str, seed: int) -> Path:
    return out / "trials" / f"{task}__{arm}__s{seed}"


def needs_run(d: Path) -> bool:
    rp = d / "result.json"
    if not d.exists():
        return True
    if not rp.exists():  # interrupted mid-trial (host restart): no outcome was observed
        dest = d.parent.parent / "_incomplete" / f"{d.name}.{int(time.time())}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), dest)
        return True
    r = json.loads(rp.read_text())
    if r.get("status") == "infra_error" and "reward_pre" not in r:
        dest = d.parent.parent / "_infra" / f"{d.name}.{int(time.time())}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), dest)
        return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default=str(Path.home() / "harbor_datasets/ds11/deep-swe-1-1"))
    ap.add_argument("--objections", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default=",".join(T.ARMS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tasks", default="", help="comma list or @file (smoke only)")
    ap.add_argument("--limit-tasks", type=int, default=0, help="first N tasks of the job order (smoke only)")
    ap.add_argument("--prereg-commit", default=None)
    ap.add_argument("--solve-steps", type=int, default=75)
    ap.add_argument("--push-steps", type=int, default=15)
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms = [x for x in a.arms.split(",") if x]
    for x in arms:
        if x not in T.ARMS:
            sys.exit(f"unknown arm {x}")
    seeds = [int(s) for s in a.seeds.split(",")]
    objs = {json.loads(ln)["task"]: json.loads(ln) for ln in Path(a.objections).read_text().splitlines() if ln.strip()}
    bad = [t for t, o in objs.items() if "error" in o]
    tasks = {d.name: Task(d) for d in sorted(Path(a.dataset).iterdir()) if (d / "task.toml").exists()}
    ids = [t for t in tasks if t in objs and t not in bad]
    if a.tasks:
        sel = (Path(a.tasks[1:]).read_text().split() if a.tasks.startswith("@") else a.tasks.split(","))
        ids = [t for t in ids if t in sel]
    jobs = job_order(ids, arms, a.model, seeds)
    if a.limit_tasks:
        keep = []
        for t, _, _ in jobs:
            if t not in keep:
                keep.append(t)
        keep = set(keep[:a.limit_tasks])
        jobs = [j for j in jobs if j[0] in keep]

    import hashlib
    cfg = Config(model=a.model, effort=a.effort or None, solve_steps=a.solve_steps, push_steps=a.push_steps)
    mp = out / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"launches": []}
    man["launches"].append(dict(
        at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runner_version=RUNNER_VERSION, code_commit=_git_head(),
        preregistration_commit=a.prereg_commit, texts_sha256=T.texts_sha256(),
        objections=a.objections, objections_sha256=hashlib.sha256(Path(a.objections).read_bytes()).hexdigest(),
        excluded_tasks_objection_error=bad, model=a.model, tag=a.tag, effort=a.effort, arms=arms, seeds=seeds,
        n_tasks=len({j[0] for j in jobs}), n_jobs=len(jobs), workers=a.workers, config=cfg.__dict__,
        dataset=a.dataset, argv=sys.argv))
    mp.write_text(json.dumps(man, indent=1))

    todo = [j for j in jobs if needs_run(trial_dir(out, *j))]
    print(f"[{a.tag}] {len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    lock = threading.Lock()
    n_done = [0]

    def one(job):
        t, arm, seed = job
        d = trial_dir(out, t, arm, seed)
        row = Trial(tasks[t], arm, objs[t], cfg, d, seed).run()
        with lock:
            with open(out / "rows.jsonl", "a") as f:
                f.write(json.dumps({k: v for k, v in row.items() if k != "tb"}) + "\n")
            n_done[0] += 1
            print(f"[{a.tag}] {n_done[0]}/{len(todo)} {t} {arm} status={row.get('status')} "
                  f"track={row.get('track')} pre={row.get('reward_pre')} final={row.get('reward_final')} "
                  f"cave={row.get('false_cave')} wall={row.get('wall_sec')}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n") if not a.tasks and \
        not a.limit_tasks else None
    print(f"[{a.tag}] complete", flush=True)


if __name__ == "__main__":
    main()
