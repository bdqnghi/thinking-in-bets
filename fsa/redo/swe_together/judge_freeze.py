"""Freeze and validate the SWE-Together per-intent judge (v2) - docs/results/INTENT_JUDGE_VALIDATION_PLAN.md.

    python -u -m fsa.redo.swe_together.judge_freeze map-freeze        # build dir + review -> exec_map.json
    python -u -m fsa.redo.swe_together.judge_freeze validate          # C1 reference/base/cross-task, C2 goals
    python -u -m fsa.redo.swe_together.judge_freeze gold-sample       # C3: blinded 60-pair sheet + sealed key
    python -u -m fsa.redo.swe_together.judge_freeze gold-score        # C3: judge the pairs, compare with labels
    python -u -m fsa.redo.swe_together.judge_freeze freeze            # configs/intents/swe-together/JUDGE_FREEZE.json
    python -u -m fsa.redo.swe_together.judge_freeze swi-revalidate    # SWE-Interact ref/base on the new seat
    python -u -m fsa.redo.swe_together.judge_freeze swi-freeze        # configs/intents/swe-interact/JUDGE_FREEZE.json

Blinding (pilot data, ``results/redo_v1/stage1/``): the gold sampler pools every cell and writes only
opaque pair ids, task, intent and code to the sheet; the id -> source key is a separate file. Nothing
here reads a pilot trial's verdict, reward, rung, arm or condition, and nothing computes a per-arm,
regression, caving or pass quantity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from .. import intent_judge as v1
from .. import llm_seats
from . import exec_judge
from . import intent_judge_v2 as v2
from .config import CLONE, RESULTS, ROOT, subset_tasks
from .intent_map import BUILD_DIR, MAP_FILE, map_digest

WORK = RESULTS / "judge_freeze_work"
REVIEW_FILE = ROOT / "configs" / "intents" / "swe-together" / "exec_map_review.json"
VALIDATION_FILE = ROOT / "configs" / "intents" / "swe-together" / "JUDGE_VALIDATION.json"
GOLD_SHEET = WORK / "gold_sheet.md"
GOLD_PAIRS = WORK / "gold_pairs.json"          # what the labeller sees (no source, no verdict)
GOLD_KEY = WORK / "gold_key.sealed.json"        # pair id -> source (opened only by gold-score)
GOLD_LABELS = WORK / "gold_labels.json"         # the labeller's labels (written before gold-score)
C3_FILE = WORK / "c3_gold_score.json"
AUDIT_SHEET = ROOT / "docs" / "results" / "INTENT_JUDGE_AUDIT_SHEET.md"
AUDIT_KEY = WORK / "audit_key.json"
SWI_FREEZE = ROOT / "configs" / "intents" / "swe-interact" / "JUDGE_FREEZE.json"
SEED = 20260927
MAX_GOLD_DIFF = 60_000
LABELS4 = ("satisfied", "not_satisfied", "not_addressed", "not_applicable")


def _sha(b: str | bytes) -> str:
    return hashlib.sha256(b.encode("utf-8", "surrogateescape") if isinstance(b, str) else b).hexdigest()


def _kappa(pairs: list[tuple[bool, bool]]) -> dict:
    n = len(pairs)
    tp = sum(1 for a, b in pairs if a and b)
    tn = sum(1 for a, b in pairs if not a and not b)
    fp = sum(1 for a, b in pairs if a and not b)   # judge yes, gold no
    fn = sum(1 for a, b in pairs if not a and b)
    po = (tp + tn) / n if n else None
    pe = (((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)) if n else None
    k = (po - pe) / (1 - pe) if n and pe is not None and pe < 1 else None
    return {"n": n, "agree": tp + tn, "agreement": round(po, 4) if po is not None else None,
            "judge_sat_gold_sat": tp, "judge_unsat_gold_unsat": tn, "judge_sat_gold_unsat": fp,
            "judge_unsat_gold_sat": fn, "cohen_kappa": round(k, 4) if k is not None else None}


# ── map ─────────────────────────────────────────────────────────────────────

def cmd_map_freeze(args) -> int:
    """exec_map.json from the judge-seat proposals (``intent_map build``) and the recorded review."""
    review = json.loads(REVIEW_FILE.read_text()) if REVIEW_FILE.exists() else {"tasks": {}}
    tasks = {}
    notes = Counter()
    for task in subset_tasks():
        b = json.loads((BUILD_DIR / f"{task}.json").read_text())
        td = CLONE / "tasks" / task
        goals = json.loads((td / "canonical_goals.json").read_text())["completeness_goals"]
        gturn = {g["id"]: g["source_turn"] for g in b["proposal"]["goals"]}
        checks = {c["id"]: c for c in b["eligible_checks"]}
        rv = (review.get("tasks") or {}).get(task, {})
        intents = {}
        for x in b["proposal"]["intents"]:
            e = {k: x[k] for k in ("checks", "check_coverage", "check_note", "goals", "goal_relation", "goal_note")}
            e["checks"] = [c for c in e["checks"] if c in checks]
            e["goals"] = [g for g in e["goals"] if g != "-"]
            if e["check_coverage"] == "full":
                bad = [c for c in e["checks"] if checks[c]["reference"] != "pass"]
                if bad or not e["checks"]:
                    e["check_coverage"] = "partial"
                    e["auto_downgrade"] = f"checks not passing on the reference: {bad}" if bad else "no checks"
                    notes["auto_downgraded"] += 1
            r = rv.get(x["key"])
            if r:
                e["proposal"] = {k: x[k] for k in ("checks", "check_coverage", "goals", "goal_relation")}
                for k in ("checks", "check_coverage", "goals", "goal_relation", "superseded"):
                    if k in r:
                        e[k] = r[k]
                e["review"] = r.get("note", "")
                notes["reviewed_edits"] += 1
            e.setdefault("superseded", False)
            intents[x["key"]] = e
        tasks[task] = {
            "intents": intents,
            "goals": [{"id": g["id"], "goal": g.get("goal", ""), "tier": g.get("tier"), "weight": g.get("weight"),
                       "source_turn": gturn.get(g["id"])} for g in goals],
            "check_descriptions": {c: (checks[c].get("description") or checks[c].get("command", "")[:300])
                                   for c in checks},
            "check_reference": {c: checks[c]["reference"] for c in checks},
            "check_base": {c: checks[c]["base"] for c in checks},
            "proposal": {"model": b["model"], "effort": b["effort"], "resolved_model": b.get("resolved_model"),
                         "prompt_sha256": b["prompt_sha256"], "response_id": b.get("response_id")},
        }
    doc = {"schema": "redo_v1/swt-exec-map/1", "version": args.version, "tasks": tasks}
    doc["digest"] = map_digest(doc)
    doc["built"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "review_file": str(REVIEW_FILE.relative_to(ROOT)),
                    "review_sha256": _sha(REVIEW_FILE.read_bytes()) if REVIEW_FILE.exists() else None,
                    "counts": dict(notes)}
    MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    MAP_FILE.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    full = sum(1 for t in tasks.values() for e in t["intents"].values() if e["check_coverage"] == "full")
    sup = sum(1 for t in tasks.values() for e in t["intents"].values() if e.get("superseded"))
    n = sum(len(t["intents"]) for t in tasks.values())
    print(f"{MAP_FILE.relative_to(ROOT)}: {n} intents, {full} full, {sup} superseded, digest {doc['digest'][:16]}; "
          f"{dict(notes)}")
    return 0


# ── C1 / C2 ─────────────────────────────────────────────────────────────────

def dev_judge(**kw) -> v2.CombinedJudge:
    """The configuration to be frozen (validation runs before the freeze file exists)."""
    cfg = freeze_config()
    return v2.CombinedJudge(cfg, cache_dir=v2.DEFAULT_CACHE, **kw)


def freeze_config(version: str = "v2.0") -> dict:
    seat = llm_seats.judge_seat()
    return {"version": version, "model": seat["model"], "reasoning_effort": seat["reasoning_effort"],
            "combine": "exec", "batch": 8, "max_diff_chars": 100_000, "max_files_chars": 80_000,
            "max_file_chars": 40_000, "file_window_lines": 80, "output_subdir": "intent_judge_v2",
            "use_goals": True, "use_files": True, "use_checks": True,
            "prompt_file": str(v2.PROMPT_FILE.relative_to(ROOT)), "prompt_sha256": v2.prompt_sha(),
            "applicability_prompt_file": str(v1.APPLICABILITY_FILE.relative_to(ROOT)),
            "applicability_prompt_sha256": v1.applicability_sha(),
            "exec_map_file": str(MAP_FILE.relative_to(ROOT)), "exec_map_file_sha256": _sha(MAP_FILE.read_bytes()),
            "exec_map_digest": json.loads(MAP_FILE.read_text())["digest"],
            "exec_recipe": exec_judge.RECIPE_VERSION}


def _intents(task: str) -> list[v1.Intent]:
    return v1.load_task_intents(CLONE / "tasks" / task)


def _rate(labels: dict[str, str], keys) -> dict:
    keys = [k for k in keys if labels.get(k) not in (None, "not_applicable")]
    s = sum(1 for k in keys if labels[k] == "satisfied")
    return {"n": len(keys), "satisfied": s, "rate": round(s / len(keys), 4) if keys else None}


def c1(judge: v2.CombinedJudge) -> dict:
    tasks = subset_tasks()
    emap = judge.exec_map["tasks"]
    refbase = json.loads((RESULTS / "swt_exec_cache" / "refbase_summary.json").read_text())
    rows = []
    for i, task in enumerate(tasks):
        its = _intents(task)
        ref = exec_judge.reference_patch(task)
        other = next((tasks[(i + j) % len(tasks)] for j in range(1, len(tasks))
                      if exec_judge.reference_patch(tasks[(i + j) % len(tasks)]).strip()), None)
        row = {"task": task, "n_intents": len(its), "has_reference": bool(ref.strip()),
               "reference_reward": (refbase.get(task, {}).get("reference") or {}).get("reward"),
               "superseded": sorted(k for k, e in emap[task]["intents"].items() if e.get("superseded")),
               "route": {i_.key: ("exec" if judge.exec_checks(task, i_.key) else "llm") for i_ in its}}
        arms = [("base", "")]
        if ref.strip():
            arms.insert(0, ("reference", ref))
        if other:
            arms.append(("cross_task", exec_judge.reference_patch(other)))
            row["cross_task_source"] = other
        for lab, patch in arms:
            t0 = time.time()
            v = judge.judge_snapshot(task, patch, its)
            row[lab] = {"labels": {k: x["verdict"] for k, x in v.items()},
                        "sources": {k: x.get("source") for k, x in v.items()},
                        "evidence": {k: (x.get("evidence") or "")[:300] for k, x in v.items()},
                        "wall_sec": round(time.time() - t0, 1)}
        print(f"C1 {task}: " + " ".join(f"{lab}={_rate(row[lab]['labels'], row[lab]['labels'])['satisfied']}/"
                                        f"{_rate(row[lab]['labels'], row[lab]['labels'])['n']}"
                                        for lab, _ in arms), flush=True)
        rows.append(row)

    def pool(lab, keep):
        n = s = 0
        for r in rows:
            if lab not in r:
                continue
            for k, l in r[lab]["labels"].items():
                if l == "not_applicable" or not keep(r, k):
                    continue
                n += 1
                s += l == "satisfied"
        return {"n": n, "satisfied": s, "rate": round(s / n, 4) if n else None}
    prim = pool("reference", lambda r, k: k not in r["superseded"])
    base = pool("base", lambda r, k: True)
    cross = pool("cross_task", lambda r, k: True)
    out = {
        "primary_reference_non_superseded": prim,
        "reference_all_applicable": pool("reference", lambda r, k: True),
        "reference_tasks_reward_ge_0.85": pool("reference", lambda r, k: (r["reference_reward"] or 0) >= 0.85),
        "reference_exec_route": pool("reference", lambda r, k: r["route"][k] == "exec"),
        "reference_llm_route": pool("reference", lambda r, k: r["route"][k] == "llm"),
        "base_all_applicable": base, "cross_task_all_applicable": cross,
        "criteria": {"reference >= 0.90 (primary)": prim["rate"] is not None and prim["rate"] >= 0.90,
                     "base <= 0.10": base["rate"] is not None and base["rate"] <= 0.10,
                     "cross_task <= 0.10": cross["rate"] is not None and cross["rate"] <= 0.10},
        "rows": rows,
    }
    out["pass"] = all(out["criteria"].values())
    return out


def _goal_verdicts() -> list[dict]:
    """Non-pilot benchmark verdicts with goal_results: calibration (reference/base) + dev trials."""
    out = []
    calib = RESULTS / "stage0" / "judge_calibration"
    for vp in sorted(calib.glob("*/*/judge_verdict.json")):
        arm, task = vp.parent.name, vp.parent.parent.name
        pp = vp.parent / "agent.patch"
        out.append({"source": f"stage0/judge_calibration/{task}/{arm}", "task": task,
                    "patch": pp.read_text(errors="surrogateescape") if pp.exists() else "",
                    "verdict": json.loads(vp.read_text())})
    for root in ("stage0", "stage1_smoke"):
        for vp in sorted((RESULTS / root).rglob("trials/*/judge_verdict.json")):
            td = vp.parent
            fp = td / "agent" / "final.patch"
            if not fp.exists():
                continue
            try:
                task = v1.trial_task_dir(td).name
            except Exception:  # noqa: BLE001
                continue
            out.append({"source": str(td.relative_to(RESULTS)), "task": task,
                        "patch": fp.read_text(errors="surrogateescape"), "verdict": json.loads(vp.read_text())})
    return out


def c2(judge: v2.CombinedJudge) -> dict:
    emap = judge.exec_map["tasks"]
    rows = []
    for rec in _goal_verdicts():
        task = rec["task"]
        gr = {g["id"]: bool(g.get("met")) for g in (rec["verdict"].get("goal_results") or [])}
        if not gr or task not in emap:
            continue
        its = _intents(task)
        v = judge.judge_snapshot(task, rec["patch"], its)
        for i in its:
            e = emap[task]["intents"].get(i.key) or {}
            gs = [g for g in e.get("goals", []) if g in gr]
            if not gs or v[i.key]["verdict"] == "not_applicable":
                continue
            rows.append({"source": rec["source"], "task": task, "intent": i.key, "relation": e.get("goal_relation"),
                         "goals": gs, "bench_all_met": all(gr[g] for g in gs),
                         "judge": v[i.key]["verdict"], "judge_source": v[i.key].get("source")})
        print(f"C2 {rec['source']}: {len(its)} intents", flush=True)
    same = [(r["judge"] == "satisfied", r["bench_all_met"]) for r in rows if r["relation"] == "same"]
    sn = [(r["judge"] == "satisfied", r["bench_all_met"]) for r in rows if r["relation"] in ("same", "narrower")]
    ks = _kappa(same)
    out = {"n_verdict_files": len({r["source"] for r in rows}), "same": ks, "same_or_narrower": _kappa(sn),
           "all_related": _kappa([(r["judge"] == "satisfied", r["bench_all_met"]) for r in rows]),
           "criterion": "kappa >= 0.40 on `same` pairs (no criterion if n < 20)",
           "rows": rows}
    out["pass"] = None if ks["n"] < 20 else bool(ks["cohen_kappa"] is not None and ks["cohen_kappa"] >= 0.40)
    return out


def cmd_validate(args) -> int:
    judge = dev_judge(workers=args.workers)
    out = {"config": judge.config_record(), "plan": "docs/results/INTENT_JUDGE_VALIDATION_PLAN.md",
           "plan_sha256": _sha((ROOT / "docs/results/INTENT_JUDGE_VALIDATION_PLAN.md").read_bytes()),
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    prev = json.loads(VALIDATION_FILE.read_text()) if VALIDATION_FILE.exists() else {}
    out["c1"] = c1(judge) if "c1" in args.parts else prev.get("c1")
    out["c2"] = c2(judge) if "c2" in args.parts else prev.get("c2")
    out["c3"] = json.loads(C3_FILE.read_text()) if C3_FILE.exists() else prev.get("c3")
    out["stats"] = judge.stats
    VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_FILE.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    summ = {k: {kk: vv for kk, vv in (out[k] or {}).items() if kk != "rows"} for k in ("c1", "c2")}
    print(json.dumps(summ, indent=1))
    return 0


# ── C3: gold set ────────────────────────────────────────────────────────────

def _trial_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.rglob("trials/*__*") if d.is_dir() and "_failed" not in d.parts and "judge_staging" not in d.parts
                  and (d / "agent" / "snapshots").is_dir() and (d / "config.json").exists())


def _candidates(td: Path) -> list[tuple[int, str]]:
    snaps = td / "agent" / "snapshots"
    out = []
    for p in sorted(snaps.glob("turn-*.patch")):
        t = int(p.stem.split("-")[1])
        txt = p.read_text(errors="surrogateescape")
        n = v1.normalize_patch(txt)
        if n.strip() and len(n) <= MAX_GOLD_DIFF:
            out.append((t, txt))
    return out


def _revealed_at(td: Path, t: int, intents: list[v1.Intent]) -> list[v1.Intent]:
    nsim = v1.sim_messages_by_turn(v1.read_turns(td))
    if t == 999:
        return intents
    n = nsim.get(t, max((v for k, v in nsim.items() if k <= t), default=0))
    return v1.revealed(intents, n)


def cmd_gold_sample(args) -> int:
    if GOLD_KEY.exists() and not args.force:
        raise SystemExit(f"{GOLD_KEY} exists; the gold sample is drawn once")
    rng = random.Random(SEED)
    tasks = set(subset_tasks())
    pilot = [d for d in _trial_dirs(RESULTS / "stage1")]
    dev = [d for r in ("stage0", "stage1_smoke", "baseline_ports_smoke") for d in _trial_dirs(RESULTS / r)]
    pairs = []

    def draw(pool: list[Path], k: int, kind: str):
        order = list(pool)
        rng.shuffle(order)
        got = 0
        for td in order:
            if got >= k:
                break
            try:
                task = v1.trial_task_dir(td).name
            except Exception:  # noqa: BLE001
                continue
            if task not in tasks:
                continue
            cands = _candidates(td)
            if not cands:
                continue
            t, patch = rng.choice(cands)
            rev = _revealed_at(td, t, _intents(task))
            if not rev:
                continue
            it = rng.choice(rev)
            pairs.append({"kind": kind, "trial_dir": str(td.relative_to(RESULTS)), "turn": t, "task": task,
                          "intent": it.key, "patch": patch})
            got += 1
    draw(pilot, 45, "pilot")
    draw(dev, 12, "dev")
    # reference patches (development data too): 3 pairs
    for task in rng.sample(sorted(t for t in tasks if exec_judge.reference_patch(t).strip()
                                  and len(v1.normalize_patch(exec_judge.reference_patch(t))) <= MAX_GOLD_DIFF), 3):
        it = rng.choice(_intents(task))
        pairs.append({"kind": "reference", "trial_dir": None, "turn": None, "task": task, "intent": it.key,
                      "patch": exec_judge.reference_patch(task)})
    rng.shuffle(pairs)
    key, shown = {}, []
    for n, p in enumerate(pairs, 1):
        pid = f"G{n:02d}-{_sha(json.dumps([SEED, n, p['task'], p['intent'], _sha(p['patch'])]))[:6]}"
        key[pid] = {k: p[k] for k in ("kind", "trial_dir", "turn", "task", "intent")} | {"patch_sha256": _sha(p["patch"])}
        its = {i.key: i for i in _intents(p["task"])}
        i = its[p["intent"]]
        shown.append({"id": pid, "task": p["task"], "intent_key": i.key, "source_turn": i.source_turn,
                      "kind": i.kind, "intent": i.text, "patch": v1.normalize_patch(p["patch"])})
    WORK.mkdir(parents=True, exist_ok=True)
    GOLD_KEY.write_text(json.dumps({"seed": SEED, "sampled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                    "note": "sealed: opened only by gold-score, after gold_labels.json is final",
                                    "pairs": key}, indent=1))
    GOLD_PAIRS.write_text(json.dumps({"seed": SEED, "pairs": shown}, indent=1, ensure_ascii=False))
    lines = ["# Gold set - labelling sheet (blinded)", "",
             "Label each pair from the code only: satisfied | not_satisfied | not_addressed | not_applicable.", ""]
    for s in shown:
        lines += [f"## {s['id']} - {s['task']} - intent `{s['intent_key']}` (message {s['source_turn']}, {s['kind']})",
                  "", s["intent"], "", "```diff", s["patch"].rstrip(), "```", ""]
    GOLD_SHEET.write_text("\n".join(lines))
    print(f"{len(shown)} pairs ({Counter(p['kind'] for p in pairs)}); sheet {GOLD_SHEET}", flush=True)
    return 0


def cmd_gold_score(args) -> int:
    labels_doc = json.loads(GOLD_LABELS.read_text())
    labels = {k: v["label"] for k, v in labels_doc["labels"].items()}
    key = json.loads(GOLD_KEY.read_text())["pairs"]
    shown = {p["id"]: p for p in json.loads(GOLD_PAIRS.read_text())["pairs"]}
    missing = sorted(set(shown) - set(labels))
    if missing:
        raise SystemExit(f"unlabelled pairs: {missing}")
    judge = dev_judge(workers=args.workers)
    rows = []
    for pid, s in shown.items():
        k = key[pid]
        if k["trial_dir"]:
            patch = (RESULTS / k["trial_dir"] / "agent" / "snapshots" / f"turn-{k['turn']}.patch").read_text(
                errors="surrogateescape")
        else:
            patch = exec_judge.reference_patch(k["task"])
        assert _sha(patch) == k["patch_sha256"], pid
        it = next(i for i in _intents(k["task"]) if i.key == k["intent"])
        v = judge.judge_snapshot(k["task"], patch, [it])[it.key]
        rows.append({"id": pid, "task": k["task"], "intent": it.key, "pool": k["kind"], "gold": labels[pid],
                     "judge": v["verdict"], "judge_source": v.get("source"), "judge_evidence": v.get("evidence"),
                     "gold_note": labels_doc["labels"][pid].get("note")})
        print(f"{pid} gold={labels[pid]:14s} judge={v['verdict']:14s} ({v.get('source')})", flush=True)
    both_app = [r for r in rows if r["gold"] != "not_applicable" and r["judge"] != "not_applicable"]
    k2 = _kappa([(r["judge"] == "satisfied", r["gold"] == "satisfied") for r in both_app])
    one_na = [r for r in rows if (r["gold"] == "not_applicable") != (r["judge"] == "not_applicable")]
    exact = sum(1 for r in rows if r["gold"] == r["judge"])
    out = {"labeller": "Claude (the agent), acting as the human-audit stand-in; labels written before the judge ran "
                       "on the pairs", "labels_file_sha256": _sha(GOLD_LABELS.read_bytes()),
           "labels_written_at": labels_doc.get("written_at"), "n": len(rows),
           "binary_both_applicable": k2, "one_side_not_applicable": len(one_na), "exact_4way": exact,
           "binary_with_na_as_disagreement": {"n": len(rows), "agree": k2["agree"],
                                              "agreement": round(k2["agree"] / len(rows), 4)},
           "by_pool": {pool: _kappa([(r["judge"] == "satisfied", r["gold"] == "satisfied") for r in both_app
                                     if r["pool"] == pool]) for pool in ("pilot", "dev", "reference")},
           "by_route": {rt: _kappa([(r["judge"] == "satisfied", r["gold"] == "satisfied") for r in both_app
                                    if r["judge_source"] == rt]) for rt in ("exec", "llm", "empty")},
           "criterion": "binary agreement >= 0.85 and kappa >= 0.60 (both-applicable pairs)",
           "rows": rows}
    out["pass"] = bool(k2["agreement"] is not None and k2["agreement"] >= 0.85 and
                       k2["cohen_kappa"] is not None and k2["cohen_kappa"] >= 0.60)
    C3_FILE.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    if VALIDATION_FILE.exists():
        doc = json.loads(VALIDATION_FILE.read_text())
        doc["c3"] = out
        VALIDATION_FILE.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


def cmd_audit(args) -> int:
    """20 of the gold pairs for the user, unlabelled; the key (agent label, judge verdict) kept apart."""
    shown = json.loads(GOLD_PAIRS.read_text())["pairs"]
    rng = random.Random(SEED + 1)
    pick = sorted(rng.sample(range(len(shown)), 20))
    rows = {r["id"]: r for r in (json.loads(C3_FILE.read_text()).get("rows") or [])}
    key = {}
    lines = ["# Per-intent judge - audit sheet (20 pairs)", "",
             "For the user. Each pair is one repository snapshot (the cumulative diff an agent had produced at "
             "that point of a SWE-Together session) and one user intent from that session. Please label each "
             "pair **from the code alone**: does the repository, as the diff leaves it, fulfil the intent?", "",
             "Labels: `satisfied` (a careful reviewer would accept it for this intent), `not_satisfied` (work on "
             "it is there but incomplete, wrong or undone), `not_addressed` (nothing in the diff relates to it), "
             "`not_applicable` (the intent asks for no repository change, e.g. a question or a git action).", "",
             "The pairs were drawn at random (seed 20260928) from the 60-pair gold set of "
             "[the validation plan](INTENT_JUDGE_VALIDATION_PLAN.md). Pilot-trial pairs carry no rung, arm or "
             "condition. The agent's own labels and the judge's verdicts are **not** on this sheet; they are in "
             "`results/redo_v1/judge_freeze_work/audit_key.json` for comparison after you label.", "",
             "| # | pair | your label | note |", "|---|---|---|---|"]
    for n, j in enumerate(pick, 1):
        lines.append(f"| {n} | {shown[j]['id']} | | |")
    lines.append("")
    for n, j in enumerate(pick, 1):
        s = shown[j]
        r = rows.get(s["id"], {})
        key[s["id"]] = {"agent_label": r.get("gold"), "judge": r.get("judge"), "judge_source": r.get("judge_source")}
        lines += [f"## {n}. {s['id']} - `{s['task']}`", "",
                  f"**Intent** (`{s['intent_key']}`, from user message {s['source_turn']}, {s['kind']}):", "",
                  "> " + s["intent"].replace("\n", "\n> "), "", "```diff", s["patch"].rstrip(), "```", "",
                  "Your label: ______", ""]
    AUDIT_SHEET.write_text("\n".join(lines))
    AUDIT_KEY.write_text(json.dumps({"seed": SEED + 1, "pairs": key}, indent=1))
    print(AUDIT_SHEET)
    return 0


# ── freeze ──────────────────────────────────────────────────────────────────

def cmd_freeze(args) -> int:
    val = json.loads(VALIDATION_FILE.read_text())
    cfg = freeze_config(args.version)
    if cfg["prompt_sha256"] != val["config"]["prompt_sha256"] or \
            cfg["exec_map_file_sha256"] != val["config"].get("exec_map_file_sha256", cfg["exec_map_file_sha256"]):
        raise SystemExit("the configuration changed since validation")
    seat = llm_seats.judge_seat()
    doc = {"schema": "redo_v1/swt-judge-freeze/1", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "config": cfg,
           "seat": {"model": seat["model"], "reasoning_effort": seat["reasoning_effort"],
                    "provider_record": llm_seats.seat_record(seat["model"], seat["reasoning_effort"])["provider_record"],
                    "seats_file": str(llm_seats.SEATS_FILE.relative_to(ROOT)),
                    "seats_file_sha256": _sha(llm_seats.SEATS_FILE.read_bytes())},
           "overlay_trigger": {"module": "fsa/redo/intent_judge.py", "config": v1.judge_config(),
                               "module_sha256": _sha((ROOT / "fsa/redo/intent_judge.py").read_bytes()),
                               "note": "decides only when an overlay fires in new overlay trials; analysis uses config"},
           "validation": {"file": str(VALIDATION_FILE.relative_to(ROOT)), "sha256": _sha(VALIDATION_FILE.read_bytes()),
                          "plan": "docs/results/INTENT_JUDGE_VALIDATION_PLAN.md",
                          "c1_pass": (val.get("c1") or {}).get("pass"), "c2_pass": (val.get("c2") or {}).get("pass"),
                          "c3_pass": (val.get("c3") or {}).get("pass")},
           "final_pass_rule": FINAL_PASS_RULE,
           "note": args.note or ""}
    v2.FREEZE_FILE.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{v2.FREEZE_FILE.relative_to(ROOT)} sha256 {v2.freeze_digest()}")
    return 0


FINAL_PASS_RULE = (
    "SWE-Together final pass without the dropped codex correctness judge (2026-09-27): "
    "tests_ok = the trial's own verifier reward (tests/test.sh, result.json) >= the reference patch's reward on "
    "the same arm64 image (swt_exec_cache/refbase_summary.json), or >= 0.85 when the task has no reference; "
    "intents_ok = every durable applicable intent (reference-satisfied; all applicable intents when there is no "
    "reference) is satisfied at the final snapshot by the frozen judge; final_pass = tests_ok and intents_ok. "
    "Both parts are reported separately.")


# ── SWE-Interact ────────────────────────────────────────────────────────────

def cmd_swi_revalidate(args) -> int:
    """Reference/base re-check of the frozen SWE-Interact intent sets on the Gemini judge seats
    (read-only for the intent files; results in the work dir). With ``--out FILE`` the result goes to
    FILE instead of the committed ``JUDGE_REVALIDATION_GEMINI.json`` and records, per task, every label
    that differs from that committed record (e.g. the same check on another host architecture's images)."""
    from ..swe_interact import intents as I
    from ..swe_interact.config import subset_tasks as swi_tasks
    from ..swe_interact.judge import SnapshotJudge
    committed = ROOT / "configs" / "intents" / "swe-interact" / "JUDGE_REVALIDATION_GEMINI.json"
    prev = json.loads(committed.read_text()) if args.out and committed.exists() else None
    j = SnapshotJudge(workers=args.workers)
    out = {"seat": llm_seats.judge_seat(), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tasks": {}}
    if prev is not None:
        from .. import machine
        out["host_arch"] = machine.host_arch()
        out["compared_with"] = {"file": str(committed.relative_to(ROOT)), "sha256": _sha(committed.read_bytes()),
                                "at": prev.get("at")}
    for task in swi_tasks():
        doc = I.load(task)
        its = doc["intents"]
        old = doc.get("validation") or {}
        rec = {}
        for lab, patch in (("reference", I.reference_patch(task)), ("base", "")):
            v = j.verdicts(task, patch, its, label=lab)
            v.pop("_meta", None)
            labels = {k: x["verdict"] for k, x in v.items()}
            rec[lab] = {"labels": labels, "all": I._rate(labels, [i["key"] for i in its]),
                        "positive": I._rate(labels, [i["key"] for i in its if i["kind"] in I.POSITIVE])}
        changed = {k: [(old.get("reference") or {}).get("labels", {}).get(k), rec["reference"]["labels"].get(k)]
                   for k in rec["reference"]["labels"]
                   if (old.get("reference") or {}).get("labels", {}).get(k) != rec["reference"]["labels"].get(k)}
        r, b = rec["reference"]["all"]["rate"], rec["base"]["positive"]["rate"]
        rec["pass"] = bool(r is not None and r >= I.REF_MIN and b is not None and b <= I.BASE_MAX)
        rec["gpt55"] = {"reference": (old.get("reference") or {}).get("all"), "base": (old.get("base") or {}).get("positive"),
                        "pass": old.get("pass")}
        rec["reference_changed_vs_gpt55"] = changed
        if prev is not None:
            pt = (prev.get("tasks") or {}).get(task) or {}
            rec["changed_vs_committed"] = {lab: {k: [(pt.get(lab) or {}).get("labels", {}).get(k), v]
                                                 for k, v in rec[lab]["labels"].items()
                                                 if (pt.get(lab) or {}).get("labels", {}).get(k) != v}
                                           for lab in ("reference", "base")}
            rec["pass_committed"] = pt.get("pass")
        out["tasks"][task] = rec
        print(f"SWI {task}: ref {rec['reference']['all']} base+ {rec['base']['positive']} pass={rec['pass']} "
              f"(gpt-5.5 pass={old.get('pass')}; {len(changed)} reference labels changed"
              + (f"; vs committed: {sum(len(x) for x in rec['changed_vs_committed'].values())} labels differ"
                 if prev is not None else "") + ")", flush=True)
    tot = lambda lab, part, k: sum((t[lab][part][k] or 0) for t in out["tasks"].values())  # noqa: E731
    out["summary"] = {"reference": f"{tot('reference', 'all', 'satisfied')}/{tot('reference', 'all', 'n')}",
                      "base_positive": f"{tot('base', 'positive', 'satisfied')}/{tot('base', 'positive', 'n')}",
                      "tasks_pass": sum(1 for t in out["tasks"].values() if t["pass"]),
                      "tasks_pass_gpt55": sum(1 for t in out["tasks"].values() if t["gpt55"]["pass"]),
                      "stats_llm": j.llm.stats, "stats_rubric": j.rubric.stats}
    if prev is not None:
        out["summary"]["committed"] = prev.get("summary", {}).get("reference"), prev.get("summary", {}).get("base_positive")
        out["summary"]["labels_differing_from_committed"] = sum(
            len(x) for t in out["tasks"].values() for x in t["changed_vs_committed"].values())
    p = Path(args.out) if args.out else committed
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out["summary"], indent=1))
    return 0


def cmd_swi_freeze(args) -> int:
    from ..swe_interact import exec_judge as sx
    from ..swe_interact import rubric_judge, tagger
    from ..swe_interact.intents import index_digest
    seat = llm_seats.judge_seat()
    reval = ROOT / "configs" / "intents" / "swe-interact" / "JUDGE_REVALIDATION_GEMINI.json"
    rv = json.loads(reval.read_text()) if reval.exists() else None
    doc = {"schema": "redo_v1/swi-judge-freeze/1", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "version": args.version,
           "seat": {"model": seat["model"], "reasoning_effort": seat["reasoning_effort"],
                    "provider_record": llm_seats.seat_record(seat["model"], seat["reasoning_effort"])["provider_record"],
                    "seats_file": str(llm_seats.SEATS_FILE.relative_to(ROOT)),
                    "seats_file_sha256": _sha(llm_seats.SEATS_FILE.read_bytes())},
           "intents_index_digest": index_digest(),
           "llm_judge": v1.judge_config(),
           "rubric_judge": {"model": rubric_judge.MODEL, **rubric_judge.SETTINGS,
                            "module_sha256": _sha((ROOT / "fsa/redo/swe_interact/rubric_judge.py").read_bytes()),
                            "prompts": "the task's own rubrics_system_prompt.txt / rubrics_user_prompt_template.txt"},
           "tagger": tagger.config_record(),
           "exec": {"recipe_version": sx.RECIPE_VERSION},
           "final_verifier_rubric_grader": {"EVAL_MODEL": seat["model"], "reasoning_effort": "high",
                                            "max_completion_tokens": 16384,
                                            "note": "rf_* final step, inside the task container (fsa/redo/swe_interact/rungs.json)"},
           "module_sha256": {m: _sha((ROOT / "fsa/redo/swe_interact" / m).read_bytes())
                             for m in ("judge.py", "exec_judge.py", "rubric_judge.py", "tagger.py", "metrics.py")},
           "revalidation": ({"file": str(reval.relative_to(ROOT)), "sha256": _sha(reval.read_bytes()),
                             "summary": rv["summary"]} if rv else None)}
    SWI_FREEZE.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{SWI_FREEZE.relative_to(ROOT)} sha256 {_sha(SWI_FREEZE.read_bytes())}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["map-freeze", "validate", "gold-sample", "gold-score", "audit", "freeze",
                                    "swi-revalidate", "swi-freeze"])
    ap.add_argument("--version", default="v2.0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parts", default="c1,c2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--note", default=None)
    ap.add_argument("--out", default=None,
                    help="swi-revalidate: write here (and compare with the committed revalidation) instead of "
                         "overwriting configs/intents/swe-interact/JUDGE_REVALIDATION_GEMINI.json")
    a = ap.parse_args(argv)
    return {"map-freeze": cmd_map_freeze, "validate": cmd_validate, "gold-sample": cmd_gold_sample,
            "gold-score": cmd_gold_score, "audit": cmd_audit, "freeze": cmd_freeze,
            "swi-revalidate": cmd_swi_revalidate, "swi-freeze": cmd_swi_freeze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
