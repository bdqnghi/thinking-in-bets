"""Stage 0 judge calibration (PLAN.md §5.4): the substitute gpt-5.5 judge must pass the
reference patch and fail the unmodified base repository.

    python -u -m fsa.redo.swe_together.calibrate_judge --out results/redo_v1/stage0/judge_calibration

For each task the benchmark's own Phase-2 judge (frozen ``canonical_goals.json``
rubric, ``judge_phase2_system.md`` prompt, fresh E2B sandbox of the task image,
mechanical ``sum(weight x met)`` score, pass bar 0.85) is run twice:

* ``reference`` - agent patch = ``reference_patch.json::patch``;
* ``base``      - agent patch = empty, workspace left unmodified
                  (``SWT_JUDGE_ALLOW_EMPTY_PATCH=1``, local patch; the stock batch
                  runner would score an empty patch 0.0 without calling the judge).

Tasks: the first 10 of the frozen subset, in its order, that have a non-empty
reference patch (rule fixed before any judging). Gate: reference >= 0.85 on >= 9/10
and base < 0.85 on >= 9/10. Thresholds are not tuned.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import CLONE, load_rungs, subset_tasks

PASS = 0.85


def calibration_tasks(n: int = 10) -> list[str]:
    out = []
    for t in subset_tasks():
        p = json.loads((CLONE / "tasks" / t / "reference_patch.json").read_text()).get("patch") or ""
        if p.strip():
            out.append(t)
        if len(out) == n:
            break
    return out


async def judge_one(task: str, arm: str, out_dir: Path, sem: asyncio.Semaphore) -> dict:
    from eval.correctness.run_batch import PHASE2_PROMPT, _derive_judge_score, _load_tests_files
    from eval.correctness.sandbox import JudgeInputs, judge_timeout_for_task, run_judge_in_e2b

    td = CLONE / "tasks" / task
    dest = out_dir / task / arm
    dest.mkdir(parents=True, exist_ok=True)
    vpath = dest / "judge_verdict.json"
    if vpath.exists():
        v = json.loads(vpath.read_text())
        if "error" not in v:
            return {"task": task, "arm": arm, "judge_score": v.get("judge_score"), "status": "cached"}
    ref = json.loads((td / "reference_patch.json").read_text())
    patch = (ref.get("patch") or "") if arm == "reference" else ""
    (dest / "agent.patch").write_text(patch)
    rubric_text = (td / "canonical_goals.json").read_text()
    rubric = json.loads(rubric_text)
    usp = td / "user_simulation_prompt.md"
    inputs = JudgeInputs(
        readme=(td / "README.md").read_text() if (td / "README.md").exists() else "",
        user_sim_prompt=usp.read_text() if usp.exists() else "",
        oracle_patch="", agent_patch=patch,
        test_sh=(td / "tests" / "test.sh").read_text(),
        system_prompt=PHASE2_PROMPT, tests_files=_load_tests_files(td),
        phase=2, canonical_goals_json=rubric_text,
    )
    async with sem:
        t0 = time.time()
        try:
            r = await run_judge_in_e2b(task_name=task, trial_id=f"calib-{arm}", inputs=inputs,
                                       oauth_token="", timeout_sec=judge_timeout_for_task(task), api_key=None)
        except Exception as e:  # noqa: BLE001
            rec = {"task": task, "arm": arm, "status": "sandbox_failed", "error": str(e)[:500]}
            (dest / "failure.json").write_text(json.dumps(rec, indent=2))
            return rec
        elapsed = round(time.time() - t0, 1)
    v = dict(r.verdict) if isinstance(r.verdict, dict) else {"error": "non-dict verdict"}
    if "goal_results" in v:
        score, bucket = _derive_judge_score(rubric, v["goal_results"])
        v["judge_reported_score"], v["judge_reported_verdict"] = v.get("judge_score"), v.get("verdict")
        v["judge_score"], v["verdict"] = score, bucket
    v.update(task=task, calibration_arm=arm, judge_elapsed_sec=elapsed, sandbox_id=r.sandbox_id,
             judge_exit_code=r.exit_code, judge_model=r.judge_model, judge_backend=r.judge_backend,
             rubric_n_goals=len(rubric.get("completeness_goals", [])))
    # keep failed attempts: never overwrite, rotate
    if vpath.exists():
        vpath.rename(dest / f"judge_verdict.failed-{int(time.time())}.json")
    vpath.write_text(json.dumps(v, indent=2, ensure_ascii=False))
    return {"task": task, "arm": arm, "judge_score": v.get("judge_score"), "status": "error" if "error" in v else "ok",
            "error": v.get("error"), "elapsed_sec": elapsed}


async def amain(args) -> int:
    from .run import base_manifest, judge_seat_active, openai_model_record, seat_env, write_manifest
    cfg = load_rungs()
    if not judge_seat_active(cfg):
        raise SystemExit("the codex/gpt-5.5 correctness judge seat is dropped (Gemini-only seats, 2026-09-27); "
                         "this Stage 0 calibration is historical")
    os.environ.update(seat_env(cfg, "e2b"))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    man = base_manifest(cfg, args, "judge_calibration")
    man.update({"judge": {**cfg["seats"]["judge"],
                          "provider_record": openai_model_record(cfg["seats"]["judge"]["model"].split("/", 1)[1])},
                "judge_sandbox": "e2b (official task image template)", "pass_bar": PASS,
                "task_rule": "first 10 frozen-subset tasks (subset order) with a non-empty reference patch"})
    write_manifest(out, man)
    sys.path[:0] = [str(CLONE), str(CLONE / "src")]
    os.chdir(CLONE)
    os.environ["SWT_JUDGE_ALLOW_EMPTY_PATCH"] = "1"
    from sandbox_config import load_dotenv
    load_dotenv(CLONE / ".env")
    import llm_config
    judge = llm_config.judge_model()
    problem = llm_config.check_judge_auth(judge)
    if problem:
        raise SystemExit(problem)
    tasks = args.tasks.split(",") if args.tasks else calibration_tasks()
    sem = asyncio.Semaphore(args.workers)
    jobs = [judge_one(t, arm, out, sem) for t in tasks for arm in ("reference", "base")]
    res = await asyncio.gather(*jobs)
    rows = {}
    for r in res:
        rows.setdefault(r["task"], {})[r["arm"]] = r
    table = []
    for t in tasks:
        ref, base = rows[t].get("reference", {}), rows[t].get("base", {})
        table.append({"task": t, "reference_score": ref.get("judge_score"), "base_score": base.get("judge_score"),
                      "reference_pass": (ref.get("judge_score") or 0) >= PASS if ref.get("judge_score") is not None else None,
                      "base_fail": (base.get("judge_score") or 0) < PASS if base.get("judge_score") is not None else None,
                      "reference_status": ref.get("status"), "base_status": base.get("status")})
    n_ref = sum(1 for r in table if r["reference_pass"])
    n_base = sum(1 for r in table if r["base_fail"])
    summary = {"finished_at": datetime.now(timezone.utc).isoformat(), "judge": judge.model,
               "judge_backend": judge.backend, "pass_bar": PASS, "tasks": tasks, "table": table,
               "reference_pass": f"{n_ref}/{len(tasks)}", "base_fail": f"{n_base}/{len(tasks)}",
               "gate": "PASS" if (n_ref >= 9 and n_base >= 9) else "FAIL",
               "incomplete": [r for r in res if r.get("status") not in ("ok", "cached")]}
    (out / "calibration_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("reference_pass", "base_fail", "gate")}))
    for r in table:
        print(r)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
