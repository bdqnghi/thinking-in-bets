"""Collect Harbor job results (per-trial result.json + job result.json) into the tib repo and print summary tables.

Usage: python scripts/harbor_collect.py [--scoring first_attempt|retry_best] [job_name ...]
       (default: all tb4_* / ds11_* / ds11m_* jobs in ~/jobs; ``<job>_p<N>`` shards are merged into ``<job>``)

Scoring rules (all three are always computed and printed; ``--scoring`` picks which one is written to
``results/harbor/<job>/per_task.json``):

* ``first_attempt`` (DEFAULT, intent-to-treat): every attempt ever seen for a task is pooled -- live trials in
  ``~/jobs/<job>[_pN]``, trials moved to ``~/jobs/_dups_<job>`` or ``~/jobs/_attempts_<job>`` by the repair loop, and the archived copies in
  ``results/harbor/<job>/<trial>/result.json`` (which keep attempts that were later deleted from ``~/jobs`` and
  re-run). Attempts are ordered by ``started_at`` and the FIRST VALID attempt counts, whatever its outcome:
    - a clean attempt (no exception) with a verifier reward counts with that reward;
    - an ``AgentTimeoutError`` counts: with its verifier reward if the verifier scored it, otherwise as a FAILURE
      (running out of wall-clock is an outcome of agent + protocol, not an infrastructure fault);
    - any other exception (``NonZeroAgentExitCodeError``, ``ApiRateLimitError``, ``CancelledError``, network /
      verifier errors, ...) is an infrastructure crash: it is excused and the next attempt is used, EVEN IF the
      verifier graded the crashed attempt's partial work (those grades are ~always 0: 1 pass in ~150 such
      attempts, so counting them would score bridge health, not the method);
    - unfinished attempts (no ``finished_at``) are ignored.
  The choice of which attempt counts never depends on its reward, so there is no best-of-N selection.
  A task whose attempts are all excused crashes is reported as errored and excluded.
  Caveat: the archive only holds attempts that some earlier collect run saw; attempts created and deleted between
  two collect runs are invisible.
* ``first_scored`` (sensitivity, stricter): same pool and order, but ANY attempt with a verifier reward counts,
  including crashed (``NonZeroAgentExitCodeError`` ...) attempts that were graded; only crashes with no verifier
  reward are excused. This is ``itt_first_scored`` in ``scripts/analysis/log_mining.py``.
* ``retry_best`` (the previous rule, kept to reproduce old numbers): live trials only; when a task has several,
  keep a passing record first, then a non-errored one, then a scored timeout.

Writes ``results/harbor/summary_{first_attempt,first_scored,retry_best}.json`` (on a full collect only).
"""
import argparse, glob, json, os, re, shutil
from datetime import datetime, timezone

JOBS = os.path.expanduser("~/jobs")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "harbor")
MODELS = {"S": "3.8 Flash", "F": "3.5 Flash", "W": "3.5 Flash-Lite", "G": "4.0 Pro"}
N_TASKS = {"tb4": 66, "ds11": 113, "ds11m": 113, "ds11r2": 113}
PARTIAL_NOTE = "ds11m_G_* and tb4_G_tib may still be running: their numbers are partial snapshots."

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("jobs", nargs="*")
ap.add_argument("--scoring", choices=("first_attempt", "first_scored", "retry_best"), default="first_attempt",
                help="rule written to per_task.json (both are always printed / written to summary_*.json)")
args = ap.parse_args()

jobs = args.jobs or sorted(os.path.basename(p) for p in glob.glob(f"{JOBS}/tb4_*") + glob.glob(f"{JOBS}/ds11_*") + glob.glob(f"{JOBS}/ds11m_*") + glob.glob(f"{JOBS}/ds11r2_*") if os.path.isdir(p) and not re.search(r"_p\d+$", os.path.basename(p)))


def _iso(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _passed(rew):
    return rew is not None and float(rew) >= 0.999


def parse(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    task = (d.get("task_name") or "").split("/")[-1]
    if not task:
        return None
    rew = (d.get("verifier_result") or {}).get("rewards") or {}
    exc = d.get("exception_info")
    return {"task": task, "trial": os.path.basename(os.path.dirname(path)), "reward": rew.get("reward"),
            "partial": rew.get("partial"), "exc_type": exc["exception_type"] if exc else None,
            "started": _iso(d.get("started_at")), "finished": bool(d.get("finished_at")),
            "tokens_in": (d.get("agent_result") or {}).get("n_input_tokens")}


def retry_best(live):
    """The previous harbor_collect rule (best-of-N over live retries)."""
    per_task = {}
    for a in live:
        passed = _passed(a["reward"])
        # AgentTimeoutError where the verifier ran and produced a reward is a valid scored trial (timeout=True)
        scored_timeout = (a["exc_type"] == "AgentTimeoutError" and a["reward"] is not None)
        rec = {"reward": a["reward"], "partial": a["partial"],
               "exc": None if (passed or scored_timeout) else a["exc_type"],
               "timeout": bool(scored_timeout), "tokens_in": a["tokens_in"], "trial": a["trial"]}
        cur = per_task.get(a["task"])
        if (cur is None
                or (passed and not _passed(cur["reward"]))
                or (cur["exc"] and not rec["exc"])
                or (cur.get("timeout") and not rec.get("timeout") and not rec["exc"])):
            per_task[a["task"]] = rec
    return per_task


def first_attempt(attempts, excuse_scored_crashes=True):
    """First valid attempt per task (see module docstring). excuse_scored_crashes=False -> ``first_scored``."""
    by_task = {}
    for a in attempts:
        by_task.setdefault(a["task"], []).append(a)
    per_task = {}
    for task, v in by_task.items():
        v.sort(key=lambda a: a["started"] or 0)
        n_excused = 0
        rec = None
        for a in v:
            exc = a["exc_type"]
            if exc == "AgentTimeoutError":
                if a["reward"] is not None:
                    rec = {"reward": a["reward"], "partial": a["partial"], "exc": None, "timeout": True,
                           "scored_exc": exc}
                else:
                    rec = {"reward": 0.0, "partial": None, "exc": None, "timeout": True, "scored_exc": exc,
                           "unscored_timeout": True}
            elif exc and (excuse_scored_crashes or a["reward"] is None):
                n_excused += 1
                continue
            elif a["reward"] is not None:
                rec = {"reward": a["reward"], "partial": a["partial"], "exc": None, "timeout": False,
                       "scored_exc": exc}
            else:
                continue  # unfinished / in progress, or finished without any verifier output
            rec.update(tokens_in=a["tokens_in"], trial=a["trial"])
            break
        if rec is None:
            crashes = [a for a in v if a["exc_type"]]
            if not crashes:
                continue  # only in-progress attempts
            rec = {"reward": None, "partial": None, "exc": crashes[-1]["exc_type"], "timeout": False,
                   "tokens_in": None, "trial": crashes[-1]["trial"]}
        rec["n_attempts"] = len(v)
        rec["n_excused_crashes"] = n_excused
        per_task[task] = rec
    return per_task


rows = {"first_attempt": {}, "first_scored": {}, "retry_best": {}}
for job in jobs:
    src = os.path.join(JOBS, job)
    dst = os.path.join(OUT, job)
    os.makedirs(dst, exist_ok=True)
    for f in ("result.json", "config.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), dst)
    live = []
    for r in glob.glob(f"{src}/*__*/result.json") + glob.glob(f"{src}_p[0-9]*/*__*/result.json"):
        a = parse(r)
        if not a:
            continue
        live.append(a)
        tdst = os.path.join(dst, a["trial"])
        os.makedirs(tdst, exist_ok=True)
        shutil.copy2(r, tdst)  # archive: keeps attempts that the repair loop later deletes
    # every attempt ever seen: archive (now includes the live copies) + repair-loop duplicates
    allatt = {}
    for r in glob.glob(f"{dst}/*__*/result.json") + glob.glob(f"{JOBS}/_dups_{job}/*__*/result.json") + glob.glob(f"{JOBS}/_attempts_{job}/*__*/result.json"):
        a = parse(r)
        if a and a["trial"] not in allatt:
            allatt[a["trial"]] = a
    for a in live:
        allatt[a["trial"]] = a
    rows["retry_best"][job] = retry_best(live)
    rows["first_attempt"][job] = first_attempt(list(allatt.values()))
    rows["first_scored"][job] = first_attempt(list(allatt.values()), excuse_scored_crashes=False)
    json.dump(rows[args.scoring][job], open(os.path.join(dst, "per_task.json"), "w"), indent=1, sort_keys=True)


def _parts(job):
    p = job.split("_")
    return p[0], p[1], "_".join(p[2:])


LABELS = {"first_attempt": "  (DEFAULT: first valid attempt; infra crashes excused, timeouts count)",
          "first_scored": "  (sensitivity: first attempt with any verifier reward, graded crashes count as fails)",
          "retry_best": "  (OLD rule, prefers a passing retry; sensitivity only)"}


def summarize(scoring):
    jobs_out, paired = {}, []
    print(f"\n=== scoring = {scoring}{LABELS[scoring]} ===")
    print("job | model | arm | tasks scored | errored | pass@1 (of scored) | pass@1 (of all tasks)")
    for job, pt in rows[scoring].items():
        bench, m, arm = _parts(job)
        if bench not in N_TASKS or m not in MODELS:
            continue
        ok = {t: v for t, v in pt.items() if not v["exc"]}
        n_pass = sum(1 for v in ok.values() if _passed(v["reward"]))
        n_err = sum(1 for v in pt.values() if v["exc"])
        n_all = N_TASKS[bench]
        n_to = sum(1 for v in ok.values() if v.get("timeout"))
        jobs_out[job] = {"bench": bench, "model": MODELS[m], "arm": arm, "n_scored": len(ok), "n_tasks": n_all,
                         "n_errored": n_err, "n_pass": n_pass, "n_timeout_counted": n_to,
                         "pass_rate_scored": n_pass / len(ok) if ok else None, "pass_rate_all": n_pass / n_all,
                         "per_task": {t: (None if v["exc"] else int(_passed(v["reward"]))) for t, v in sorted(pt.items())}}
        print(f"{job} | {MODELS[m]} | {arm} | {len(ok)}/{n_all} | {n_err} | {n_pass}/{len(ok)} = {100*n_pass/max(1,len(ok)):.1f}% | {100*n_pass/n_all:.1f}%")
    print(f"Paired arm vs stock (tasks scored in both arms), scoring = {scoring}:")
    for job in rows[scoring]:
        bench, m, arm = _parts(job)
        if arm == "stock" or bench not in N_TASKS or m not in MODELS:
            continue
        s, t = rows[scoring].get(f"{bench}_{m}_stock"), rows[scoring][job]
        if not s or not t:
            continue
        common = sorted(k for k in s if k in t and not s[k]["exc"] and not t[k]["exc"])
        if not common:
            continue
        ps = sum(1 for k in common if _passed(s[k]["reward"]))
        ptb = sum(1 for k in common if _passed(t[k]["reward"]))
        w = sum(1 for k in common if _passed(t[k]["reward"]) and not _passed(s[k]["reward"]))
        l = sum(1 for k in common if _passed(s[k]["reward"]) and not _passed(t[k]["reward"]))
        n = len(common)
        paired.append({"bench": bench, "model": MODELS[m], "arm": arm, "n": n, "stock_pass": ps, "arm_pass": ptb,
                       "stock_rate": ps / n, "arm_rate": ptb / n, "delta_pts": 100 * (ptb - ps) / n,
                       "W": w, "T": n - w - l, "L": l})
        print(f"{bench} {MODELS[m]} {arm}: n={n} stock {ps}/{n}={100*ps/n:.1f}% -> {arm} {ptb}/{n}={100*ptb/n:.1f}% "
              f"(delta {100*(ptb-ps)/n:+.1f}) | W/T/L {w}/{n-w-l}/{l}")
    out = {"scoring": scoring, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "rule": (__doc__.split("Scoring rules")[1].split("Writes")[0]).strip(),
           "note": PARTIAL_NOTE + " W/T/L are on pass (reward >= 0.999).", "jobs": jobs_out, "paired": paired}
    if not args.jobs:  # only overwrite the summaries on a full collect
        json.dump(out, open(os.path.join(OUT, f"summary_{scoring}.json"), "w"), indent=1)


summarize("first_attempt")
summarize("first_scored")
summarize("retry_best")
print("\n" + PARTIAL_NOTE)
