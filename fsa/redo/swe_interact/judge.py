"""Per-intent "satisfied at snapshot t" for SWE-Interact (redo_v1 Stage 1 ground truth).

Judges, per intent of the frozen intent set (``configs/intents/swe-interact/<task>.json``):

* ``judge == "exec"`` (primary; deepswe / swebenchpro intents with mapped hidden tests):
  ``exec_judge.run_snapshot`` runs the task's hidden tests on the snapshot in a fresh container;
  satisfied = every mapped test group ran and every test in it PASSED (a mapped test that did
  not run counts as failing, as for the verifier's fail_to_pass list);
* ``judge == "exec_rf"`` (rf ``behaviour_preserved``): the rf verifier's ``tests_reward`` = 1
  (no relevant pass->fail / missing->fail test in the master validator's comparison);
* ``judge == "llm"`` (secondary; intents no test maps): ``fsa.redo.intent_judge`` (gpt-5.5,
  repository diff + intent text, no dialogue; strict-reviewer prompt; ``not_addressed`` counts
  as unsatisfied, ``not_applicable`` is excluded);
* ``judge == "rubric"`` (rf rubric items): ``rubric_judge`` = the benchmark's rubric grader
  prompt per item on the snapshot.

Verdict labels: satisfied | not_satisfied | not_addressed | not_applicable | error | pending
(``pending`` only in-trial, when an executable run has not finished within the overlay's wait).
Every underlying result is cached (snapshot hash x intent), so offline judging after the
in-trial trigger is free for the snapshots the trigger already judged.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from ..intent_judge import DEFAULT_CACHE as IJ_CACHE
from ..intent_judge import Intent, IntentJudge, judge_config
from . import exec_judge
from .config import family
from .rubric_judge import MODEL as RUBRIC_MODEL
from .rubric_judge import SETTINGS as RUBRIC_SETTINGS
from .rubric_judge import RubricJudge

UNSATISFIED = frozenset({"not_satisfied", "not_addressed"})
_EXEC_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="swi-exec")  # host-wide cap per process
_RF_EXEC_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swi-exec-rf")


def config_record() -> dict:
    return {"exec": {"recipe_version": exec_judge.RECIPE_VERSION, "per_test_edits": exec_judge.PER_TEST_EDITS,
                     "rf_cpus": exec_judge.RF_CPUS},
            "llm": judge_config(), "rubric": {"model": RUBRIC_MODEL, **RUBRIC_SETTINGS,
                                              "prompts": "task's own rubrics_system_prompt.txt / rubrics_user_prompt_template.txt"}}


def exec_verdicts(intents: list[dict], run: dict) -> dict[str, dict]:
    fam = run.get("family")
    out = {}
    for it in intents:
        if run.get("status") != "ok":
            out[it["key"]] = {"verdict": "error", "judge": it["judge"], "evidence": f"exec {run.get('status')}"}
            continue
        if it["judge"] == "exec_rf":
            tr = run.get("tests_reward")
            out[it["key"]] = {"verdict": "satisfied" if tr == 1.0 else "not_satisfied", "judge": "exec_rf",
                              "evidence": f"tests_reward={tr} relevant={run.get('filtered_summary')}"}
            continue
        tests = run.get("tests") or {}
        groups = {}
        for tid, st in tests.items():
            groups.setdefault(exec_judge.test_group(tid, fam), []).append(st)
        missing = [g for g in it["tests"] if g not in groups]
        failing = [g for g in it["tests"] if g in groups and any(s != "PASSED" for s in groups[g])]
        ok = not missing and not failing
        out[it["key"]] = {"verdict": "satisfied" if ok else "not_satisfied", "judge": "exec",
                          "evidence": ("all mapped tests pass" if ok else
                                       f"missing {missing[:5]} failing {failing[:5]}")}
    return out


class SnapshotJudge:
    def __init__(self, llm_cache=IJ_CACHE, workers: int = 6):
        self.llm = IntentJudge(llm_cache, workers=workers)
        self.rubric = RubricJudge(workers=8)
        self._lock = threading.Lock()

    def exec_future(self, task: str, patch: str, label: str = "") -> Future:
        pool = _RF_EXEC_POOL if family(task) == "rf" else _EXEC_POOL
        return pool.submit(exec_judge.run_snapshot, task, patch, label=label)

    def verdicts(self, task: str, patch: str, intents: list[dict], *, exec_mode: str = "run",
                 exec_future: Future | None = None, exec_wait: float | None = None, label: str = "") -> dict:
        """key -> {verdict, judge, evidence}; plus ``_meta``. ``exec_mode``: run (block until the
        tests ran), cached (only what is cached; else pending), off (exec intents -> pending)."""
        exec_its = [i for i in intents if i["judge"] in ("exec", "exec_rf")]
        llm_its = [i for i in intents if i["judge"] == "llm"]
        rub_its = [i for i in intents if i["judge"] == "rubric"]
        out: dict = {}
        meta: dict = {}
        # LLM-side first (runs while a submitted exec future is working)
        if llm_its:
            try:
                v = self.llm.judge_snapshot(patch, [Intent(i["key"], 0, i["kind"], i["text"]) for i in llm_its])
                for i in llm_its:
                    out[i["key"]] = {"verdict": v[i["key"]]["verdict"], "judge": "llm",
                                     "evidence": v[i["key"]].get("evidence")}
            except Exception as e:  # noqa: BLE001
                for i in llm_its:
                    out[i["key"]] = {"verdict": "error", "judge": "llm", "evidence": f"{type(e).__name__}: {e}"[:300]}
        if rub_its:
            for k, r in self.rubric.judge(task, patch, rub_its).items():
                out[k] = {"verdict": r["verdict"], "judge": "rubric", "evidence": r.get("evidence")}
        if exec_its:
            run = None
            if exec_mode == "run":
                fut = exec_future or self.exec_future(task, patch, label)
                try:
                    run = fut.result(timeout=exec_wait)
                except Exception as e:  # noqa: BLE001 (TimeoutError -> pending)
                    meta["exec_wait_error"] = f"{type(e).__name__}: {e}"[:200]
            elif exec_mode == "cached":
                run = exec_judge.cached(task, patch)
            if run is None:
                for i in exec_its:
                    out[i["key"]] = {"verdict": "pending", "judge": i["judge"], "evidence": "executable run not available"}
            else:
                out.update(exec_verdicts(exec_its, run))
                meta["exec"] = {k: run.get(k) for k in ("key", "status", "wall_sec", "head", "log", "tests_reward")}
        out["_meta"] = meta
        return out
