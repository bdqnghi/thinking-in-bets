"""Summarise the ALE docker90 sweep: per model x arm x seed mean score, pass@1 (score==1), and paired stock vs draft-first.

Usage: python scripts/ale_docker90_table.py [--per-task]
"""
import glob, json, os, sys
from collections import defaultdict

import os
ROOT = os.path.join(os.environ.get("ALE_REPO", "../agents-last-exam"), ".logs/ale")
MODELS = {"W": "3.5 Flash-Lite", "F": "3.5 Flash", "M": "3.6 Flash", "S": "3.8 Flash", "G": "4.0 Pro"}
ARMS = {"stock": "stock", "cb_draftfirst": "draft-first"}
SEEDS = ["s1", "s2", "t300"]

rows = {}  # (model, arm, seed, task) -> score (None if failed/absent)
for m in MODELS:
    for arm in ARMS:
        for seed in SEEDS:
            exp = f"tib_docker90_{arm}_{m}_{seed}"
            for p in glob.glob(f"{ROOT}/{exp}/ale_claw/*/*/v0/*/eval_result.json"):
                d = os.path.dirname(p)
                task = p.split("/ale_claw/")[1].split("/")[1]
                ts = d.split("/")[-1]
                try:
                    r = json.load(open(p))
                except Exception:
                    continue
                run = json.load(open(d + "/run.json")) if os.path.exists(d + "/run.json") else {}
                key = (m, arm, seed, task)
                if key in rows and rows[key]["ts"] > ts:
                    continue
                sc = r.get("score")
                if run.get("status") == "failed" and sc in (None, 0.0):
                    sc = None
                rows[key] = {"score": sc, "ts": ts}

tasks_all = sorted({k[3] for k in rows})
print(f"tasks seen: {len(tasks_all)}\n")
print("model | arm | seed | n evaluated | mean score | pass@1 (score=1.0) | partial>0")
summary = {}
for m, mname in MODELS.items():
    for arm, aname in ARMS.items():
        for seed in SEEDS:
            scs = [v["score"] for k, v in rows.items() if k[:3] == (m, arm, seed) and v["score"] is not None]
            if not scs:
                continue
            n = len(scs)
            summary[(m, arm, seed)] = scs
            print(f"{mname} | {aname} | {seed} | {n} | {sum(scs)/n:.3f} | {sum(1 for s in scs if s >= 0.999)/n:.1%} | {sum(1 for s in scs if s > 0)/n:.1%}")

print("\nPaired comparison (same model, same seed, tasks evaluated in BOTH arms):")
print("model | seed | n paired | stock mean | draft-first mean | delta | draft-first wins / ties / losses")
for m, mname in MODELS.items():
    for seed in SEEDS:
        st = {k[3]: v["score"] for k, v in rows.items() if k[:3] == (m, "stock", seed) and v["score"] is not None}
        cb = {k[3]: v["score"] for k, v in rows.items() if k[:3] == (m, "cb_draftfirst", seed) and v["score"] is not None}
        common = sorted(set(st) & set(cb))
        if not common:
            continue
        a = sum(st[t] for t in common) / len(common)
        b = sum(cb[t] for t in common) / len(common)
        w = sum(1 for t in common if cb[t] > st[t] + 1e-9)
        l = sum(1 for t in common if cb[t] < st[t] - 1e-9)
        print(f"{mname} | {seed} | {len(common)} | {a:.3f} | {b:.3f} | {b-a:+.3f} | {w} / {len(common)-w-l} / {l}")

if "--per-task" in sys.argv:
    print("\nPer-task scores (model/seed: stock -> draft-first):")
    for m, mname in MODELS.items():
        for seed in SEEDS:
            st = {k[3]: v["score"] for k, v in rows.items() if k[:3] == (m, "stock", seed)}
            cb = {k[3]: v["score"] for k, v in rows.items() if k[:3] == (m, "cb_draftfirst", seed)}
            if not st and not cb:
                continue
            print(f"\n== {mname} {seed}")
            for t in sorted(set(st) | set(cb)):
                f = lambda x: "-" if x is None else f"{x:.2f}"
                print(f"  {t[:60]:60s} {f(st.get(t))} -> {f(cb.get(t))}")
