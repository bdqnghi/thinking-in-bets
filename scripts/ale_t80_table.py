"""Summarise ALE-Claw t80 wave-2 results: per model x task x arm score, status, steps."""
import json, os, glob, sys
import os
ROOT = os.path.join(os.environ.get("ALE_REPO", "../agents-last-exam"), ".logs/ale")
EXPS = {"stock": "tib_core5_stock_t80", "cb": "tib_core5_commit_before_t80", "cb_df": "tib_core5_cb_draftfirst_t80", "stock_w90": "tib_core5_stock_t80_S_w90", "cb_w90": "tib_core5_commit_before_t80_S_w90", "cb_df_w90": "tib_core5_cb_draftfirst_t80_S_w90"}
MODELS = ["3-5-flash-lite", "3-5-flash", "3-6-flash", "3-8-flash"]
rows = {}
for arm, exp in EXPS.items():
    for p in glob.glob(f"{ROOT}/{exp}/ale_claw/*/*/v0/*/eval_result.json"):
        d = os.path.dirname(p)
        model = p.split("/ale_claw/")[1].split("/")[0].replace("openai-gemini-", "")
        task = p.split("/ale_claw/")[1].split("/")[1]
        r = json.load(open(p))
        run = json.load(open(d + "/run.json")) if os.path.exists(d + "/run.json") else {}
        try:
            n = len(json.load(open(d + "/trajectory.json"))["steps"])
        except Exception:
            n = None
        ts = d.split("/")[-1]
        key = (model, task, arm)
        if key in rows and rows[key]["ts"] > ts:
            continue
        rows[key] = {"score": r.get("score"), "status": run.get("status"), "steps": n, "ts": ts}
tasks = sorted({k[1] for k in rows})
short = {t: t.split("__")[1][:26] for t in tasks}
arms = [a for a in EXPS if any(k[2] == a for k in rows)]
print("model | task | " + " | ".join(arms))
for m in MODELS:
    for t in tasks:
        cells = []
        for a in arms:
            x = rows.get((m, t, a))
            if not x:
                cells.append("-")
            else:
                s = x["score"]
                s = "fail" if x["status"] == "failed" and s in (None, 0.0) else (f"{s:.2f}" if isinstance(s, (int, float)) else str(s))
                cells.append(f"{s} ({x['steps']}{'*' if (x['steps'] or 0) >= 80 else ''})")
        print(f"{m} | {short[t]} | " + " | ".join(cells))
    tot = {a: sum((rows[(m, t, a)]["score"] or 0) for t in tasks if (m, t, a) in rows) for a in arms}
    print(f"{m} | MEAN | " + " | ".join(f"{tot[a]/len(tasks):.3f}" for a in arms))
