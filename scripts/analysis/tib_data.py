"""Long-form loader for every benchmark with per-task paired outcomes (TIB vs stock and baselines).

Pure post-hoc analysis over result files already on disk; nothing here launches a model, docker or a
systemd unit. Used by ``difficulty_split.py``, ``capability_model.py`` and ``multiple_comparisons.py``.

Every record is one (benchmark, model rung, arm, task, replicate) outcome::

    {"bench": "ds11", "model": "G", "arm": "cb_draftfirst", "task": "...", "rep": 1,
     "y": 0/1 (binary pass), "score": float | None (continuous score where the benchmark has one)}

Benchmarks (``bench``) and sources
----------------------------------
* ``ds11``  DeepSWE 1.1 on opencode, Harbor jobs ``ds11_{W,F,S,G}_{stock,tib}`` -> ``results/harbor/<job>/per_task.json``
* ``ds11m`` DeepSWE 1.1 on mini-swe-agent, ``ds11m_G_{stock,tib}`` (still running: partial)
* ``tb4``   Terminal-Bench 4.0 on opencode, ``tb4_{W,F,S,G}_{stock,tib}`` (``tb4_G_tib`` still running)
  Harbor ``tib`` == ``cb_draftfirst``. Tasks whose trial raised an infra exception (``exc``) are dropped.
* ``hle_ss`` HLE single-shot, ``results/hle/single_shot_*.jsonl`` (stock / cb_draftfirst, y = judge correct)
* ``hle_mt`` HLE 2-turn pushback, ``results/hle/multi_turn_*.jsonl`` (stock / claims_after / claims_no_p /
  commit_before, y = turn-2 correct)
* ``fin_tools`` Vals Finance Agent, live 4-tool harness, ``results/finance_agent_tools/tools_*.jsonl``
  (y = passed, score = rubric score)
* ``fin_closed`` Vals Finance Agent closed-book, ``results/finance_agent/single_shot_*.jsonl``
* ``fin_mt`` Vals Finance Agent 2-turn pushback, ``results/finance_agent/multi_turn_*.jsonl`` (y = t2 passed)
* ``ale_t300`` / ``ale_s1`` ALE-Claw docker sweep (300-turn / 80-turn budget),
  ``results/redo_v1_ale/docker99_per_task_final.txt`` (score; y = score >= 0.999)
* ``ale_e2`` ALE-E2 stakeholder pushback, ``results/redo_v1_ale/ale_e2_*.json`` (y = passed_t2; unit = scenario)
* ``swi_mixed`` SWE-Interact mixed feedback (W/F/M/S ``mixed_r1``, S ``mixed_r2``), y = final_pass
* ``swi_overlay`` SWE-Interact pure mistaken pushback (S ``overlay_r1``, G ``overlay_r1``+``overlay_r2``)
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "analysis"
FIG = ROOT / "docs" / "figures" / "analysis"
SEED = 20261014

RANK = {"W": 1, "F": 2, "M": 3, "S": 4, "G": 5}
MODEL_LABEL = {"W": "3.5 Flash-Lite", "F": "3.5 Flash", "M": "3.6 Flash", "S": "3.8 Flash", "G": "4.0 Pro"}
FILE_RUNG = {"gemini_3_5_flash_lite": "W", "gemini_3_5_flash": "F", "gemini_3_6_flash": "M",
             "gemini_3_8_flash": "S", "gemini_4_0_argon": "G"}
LABEL_RUNG = {v: k for k, v in MODEL_LABEL.items()}
OURS = ("cb_draftfirst", "commit_before", "cb_gated")
BENCH_LABEL = {
    "ds11": "DeepSWE 1.1 (opencode)", "ds11m": "DeepSWE 1.1 (mini-swe-agent, partial)",
    "tb4": "Terminal-Bench 4.0", "hle_ss": "HLE single-shot", "hle_mt": "HLE 2-turn pushback",
    "fin_tools": "Finance Agent (tools)", "fin_closed": "Finance Agent (closed-book)",
    "fin_mt": "Finance Agent 2-turn pushback", "ale_t300": "ALE-Claw docker t300", "ale_s1": "ALE-Claw docker s1 (80 turns)",
    "ale_e2": "ALE-E2 pushback", "swi_mixed": "SWE-Interact mixed", "swi_overlay": "SWE-Interact pure pushback",
}
# Difficulty sources: the stock cells (of *other* models under leave-one-model-out) that define a task's
# difficulty for each benchmark. Never contains a TIB arm.
DIFF_SOURCE = {
    "ds11": ["ds11", "ds11m"], "ds11m": ["ds11", "ds11m"], "tb4": ["tb4"],
    "hle_ss": ["hle_ss"], "hle_mt": ["hle_ss"],
    "fin_tools": ["fin_tools"], "fin_closed": ["fin_closed"], "fin_mt": ["fin_closed"],
    "ale_t300": ["ale_t300", "ale_s1"], "ale_s1": ["ale_t300", "ale_s1"],
    "ale_e2": ["ale_e2"], "swi_mixed": ["swi_mixed", "swi_overlay"], "swi_overlay": ["swi_mixed", "swi_overlay"],
}
# Benchmarks whose difficulty uses the continuous score instead of binary pass.
SCORE_DIFF = {"ale_t300", "ale_s1"}
SCORE_BENCH = {"ale_t300", "ale_s1", "fin_tools", "fin_closed"}  # also report score deltas


def _rec(bench, model, arm, task, y, score=None, rep=1):
    return {"bench": bench, "model": model, "arm": arm, "task": str(task), "rep": rep,
            "y": int(bool(y)), "score": None if score is None else float(score)}


def load_harbor() -> list[dict]:
    out = []
    for d in sorted((RES / "harbor").iterdir()):
        m = re.match(r"(ds11m|ds11|tb4)_([WFMSG])_(stock|tib)$", d.name)
        if not m or not (d / "per_task.json").exists():
            continue
        bench, model, arm = m.groups()
        arm = "cb_draftfirst" if arm == "tib" else arm
        for task, v in json.loads((d / "per_task.json").read_text()).items():
            if v.get("exc"):
                continue
            r = float(v.get("reward") or 0.0)
            out.append(_rec(bench, model, arm, task, r >= 0.999, r))
    return out


def _jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _rung_of(path):
    stem = Path(path).stem
    for k in sorted(FILE_RUNG, key=len, reverse=True):
        if stem.endswith(k):
            return FILE_RUNG[k]
    raise KeyError(path)


def load_hle() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(RES / "hle" / "single_shot_*.jsonl"))):
        m = _rung_of(f)
        for r in _jsonl(f):
            conf = r.get("confidence")
            rec = _rec("hle_ss", m, r["arm"], r["id"], r.get("correct_bool"))
            rec["conf"] = None if conf is None else float(conf) / 100.0
            out.append(rec)
    for f in sorted(glob.glob(str(RES / "hle" / "multi_turn_*.jsonl"))):
        m = _rung_of(f)
        for r in _jsonl(f):
            rec = _rec("hle_mt", m, r["arm"], r["id"], r.get("t2_correct"))
            rec.update(push_type=r.get("push_type"), t1=int(bool(r.get("t1_correct"))), caved=int(bool(r.get("caved"))),
                       recovered=int(bool(r.get("recovered"))))
            out.append(rec)
    return out


def load_finance() -> list[dict]:
    out = []
    for bench, pat in (("fin_tools", "finance_agent_tools/tools_*.jsonl"),
                       ("fin_closed", "finance_agent/single_shot_*.jsonl"),
                       ("fin_mt", "finance_agent/multi_turn_*.jsonl")):
        for f in sorted(glob.glob(str(RES / pat))):
            m = _rung_of(f)
            for r in _jsonl(f):
                if bench == "fin_mt":
                    y = r.get("t2_passed", r.get("t2_correct"))
                    if y is None and isinstance(r.get("t2_grade"), dict):
                        y = r["t2_grade"].get("passed")
                    rec = _rec(bench, m, r["arm"], r["id"], y, r.get("t2_score"))
                    rec["push_type"] = r.get("push_type")
                else:
                    rec = _rec(bench, m, r["arm"], r["id"], r.get("passed"), r.get("score"))
                out.append(rec)
    return out


def load_ale_docker() -> list[dict]:
    out, cur = [], None
    for line in open(RES / "redo_v1_ale" / "docker99_per_task_final.txt"):
        h = re.match(r"== (.+?) (s1|s2|t300)\s*$", line)
        if h:
            cur = (LABEL_RUNG.get(h.group(1).strip()), h.group(2))
            continue
        t = re.match(r"\s+(\S+)\s+(\S+) -> (\S+)\s*$", line)
        if not (t and cur and cur[0] and cur[1] in ("s1", "t300")):
            continue
        bench = f"ale_{cur[1]}"
        task, a, b = t.groups()
        for arm, v in (("stock", a), ("cb_draftfirst", b)):
            if v != "-":
                out.append(_rec(bench, cur[0], arm, task, float(v) >= 0.999, float(v)))
    return out


def load_ale_e2() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(RES / "redo_v1_ale" / "ale_e2_*.json"))):
        m = _rung_of(f)
        for t in json.loads(open(f).read())["trials"]:
            rec = _rec("ale_e2", m, t["arm"], t["scenario_id"], t.get("passed_t2"), t.get("score_t2"))
            rec.update(category=t.get("category"), caved=int(bool(t.get("caved"))))
            out.append(rec)
    return out


def swi_trials() -> dict[str, list[dict]]:
    """Cached SWE-Interact per-trial rows grouped by setting (built by bootstrap_cis.py / deference_costs.py)."""
    A = OUT
    g = {}
    mixed = json.loads((A / "swi_mixed_r1_trials.json").read_text())
    s12 = json.loads((A / "swi_S_mixed_r1r2_trials.json").read_text())
    g["swi_mixed"] = [r for r in mixed] + [r for r in s12 if r["replicate"] == 2]
    ov = json.loads((A / "swi_g4_overlay_r1r2_trials.json").read_text())
    p = A / "swi_S_overlay_r1_trials.json"
    sov = json.loads(p.read_text()) if p.exists() else []
    g["swi_overlay"] = ov + sov
    return g


def load_swi() -> list[dict]:
    out = []
    for bench, rows in swi_trials().items():
        seen = set()
        for r in rows:
            key = (r["rung"], r["arm"], r["task"], r["replicate"])
            if key in seen or not r.get("judged"):
                continue
            seen.add(key)
            out.append(_rec(bench, r["rung"], r["arm"], r["task"], r["final_pass"], rep=r["replicate"]))
    return out


def load_all() -> list[dict]:
    return load_harbor() + load_hle() + load_finance() + load_ale_docker() + load_ale_e2() + load_swi()


# ── difficulty ──────────────────────────────────────────────────────────────

def stock_table(recs: list[dict]) -> dict:
    """{bench: {task: {model: mean stock outcome over replicates}}} (score for SCORE_DIFF benches)."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        if r["arm"] != "stock":
            continue
        v = r["score"] if r["bench"] in SCORE_DIFF and r["score"] is not None else r["y"]
        acc[r["bench"]][r["task"]][r["model"]].append(v)
    return {b: {t: {m: sum(v) / len(v) for m, v in mm.items()} for t, mm in tt.items()} for b, tt in acc.items()}


def difficulty(st: dict, bench: str, task: str, model: str, mode: str = "loo"):
    """Mean stock success of ``task`` over the stock cells in DIFF_SOURCE[bench].

    mode="loo": drop every cell of ``model`` (leave-one-model-out; guards regression to the mean);
    mode="all": keep all models, including the evaluated model's own stock outcome (biased; shown as contrast).
    Each (source bench, model) cell is one vote. Returns (d, n_cells) or (None, 0).
    """
    vals = []
    for sb in DIFF_SOURCE[bench]:
        for m, v in st.get(sb, {}).get(task, {}).items():
            if mode == "loo" and m == model:
                continue
            vals.append(v)
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def dbin(d) -> str | None:
    if d is None:
        return None
    if d == 0:
        return "hard"
    return "easy" if d > 0.5 else "medium"


def paired_cells(recs: list[dict], min_n: int = 10) -> dict:
    """{(bench, model, arm): {task: {"stock": [y...], "arm": [y...], "stock_s": [...], "arm_s": [...]}}} for every
    non-stock arm with a stock partner on the same (bench, model) and >= min_n common tasks."""
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[(r["bench"], r["model"])][r["arm"]].append(r)
    cells = {}
    for (b, m), arms in by.items():
        if "stock" not in arms:
            continue
        st = defaultdict(dict)
        for r in arms["stock"]:
            st[r["task"]][r["rep"]] = r
        for arm, rs in arms.items():
            if arm == "stock":
                continue
            tt = defaultdict(dict)
            for r in rs:
                tt[r["task"]][r["rep"]] = r
            common = sorted(set(tt) & set(st))
            if len(common) < min_n:
                continue
            cell = {}
            for t in common:
                cell[t] = {"stock": [x["y"] for x in st[t].values()], "arm": [x["y"] for x in tt[t].values()],
                           "stock_s": [x["score"] for x in st[t].values()], "arm_s": [x["score"] for x in tt[t].values()],
                           "reps_s": sorted(st[t]), "reps_a": sorted(tt[t]),
                           "pairs": [(st[t][k]["y"], tt[t][k]["y"]) for k in sorted(set(st[t]) & set(tt[t]))],
                           "pairs_s": [(st[t][k]["score"], tt[t][k]["score"]) for k in sorted(set(st[t]) & set(tt[t]))]}
            cells[(b, m, arm)] = cell
    return cells


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
