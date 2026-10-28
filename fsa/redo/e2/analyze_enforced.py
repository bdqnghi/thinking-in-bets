"""Analysis of the pre-registered ``cb_enforced`` E2 arm (docs/PREREG_ENFORCED_BETS.md).

Compares ``cb_enforced`` (``bet -> enf_real`` / ``bet -> enf_unjustified``, results/redo_v1/e2/enforced/<run>/<model>)
with ``commit_before`` (``bet -> bet_real`` / ``bet -> bet_unjustified``) and ``stock`` (standard check-after,
``none -> real`` / ``none -> unjustified``) on the same items. Paired cluster bootstrap over benchmark/task
(all models' rows of a task resampled together), 10,000 resamples, seed 20261015.

Usage: python -m fsa.redo.e2.analyze_enforced [--reps 10000] [--out results/redo_v1/e2/enforced/analysis.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .run import _read

ROOT = Path(__file__).resolve().parents[3]
E2 = ROOT / "results" / "redo_v1" / "e2"
RUNS = [  # (run, items file, model, label)
    ("swi_W", "items_swi_W.json", "gemini-3.5-flash-lite", "3.5 Flash-Lite"),
    ("swt_W", "items_swt_W.json", "gemini-3.5-flash-lite", "3.5 Flash-Lite"),
    ("swi_F", "items_swi_S.json", "gemini-3.5-flash", "3.5 Flash"),
    ("swt_F", "items_swt_S.json", "gemini-3.5-flash", "3.5 Flash"),
    ("swi_M", "items_swi_S.json", "gemini-3.6-flash", "3.6 Flash"),
    ("swt_M", "items_swt_S.json", "gemini-3.6-flash", "3.6 Flash"),
    ("swi_S", "items_swi_S.json", "gemini-3.8-flash", "3.8 Flash"),
    ("swt_S", "items_swt_S.json", "gemini-3.8-flash", "3.8 Flash"),
    ("swi_S_g4pro", "items_swi_S.json", "gemini-4.0-argon", "4.0 Pro"),
    ("swt_S_g4pro", "items_swt_S.json", "gemini-4.0-argon", "4.0 Pro"),
]
LABELS = ["3.5 Flash-Lite", "3.5 Flash", "3.6 Flash", "3.8 Flash", "4.0 Pro"]
THRESHOLDS = ("0.6", "0.75", "0.9")
SEED = 20261015
# arm -> (pre condition, after generic unjustified, after real reply)
ARMS = {"stock": ("none", "unjustified", "real"), "commit_before": ("bet", "bet_unjustified", "bet_real"),
        **{f"cb_enforced@{t}": ("bet", f"enf_unjustified@{t}", f"enf_real@{t}") for t in THRESHOLDS},
        # exploratory: the same harness but an automatic cave (no re-examination) on challenged low-c claims
        **{f"rule_cave@{t}": ("bet", f"cave_unjustified@{t}", f"cave_real@{t}") for t in THRESHOLDS}}
PRIMARY = "cb_enforced@0.75"


def _v(d):
    return None if d is None else dict(p=d["p_satisfied"], sat=d["satisfied"], decision=d.get("decision"))


def load(include_partial: bool = False) -> tuple[list[dict], dict]:
    rows, status = [], {}
    for run, itf, model, label in RUNS:
        enf_p = E2 / "enforced" / run / model / "ratings.jsonl"
        if not enf_p.exists():
            status[run] = "missing"
            continue
        items = json.loads((E2 / itf).read_text())["items"]
        base = {(r["cond"], r["item"]): r for r in _read(E2 / run / model / "ratings.jsonl")}
        enf = {(r["cond"], r["item"]): r for r in _read(enf_p)}
        n_exp = 2 * len(items)
        status[run] = dict(enforced_records=len(enf), expected=n_exp,
                           parsed=sum(1 for r in enf.values() if r["parsed"] is not None),
                           model_called=sum(1 for r in enf.values() if r.get("model_called")))
        if len(enf) < n_exp and not include_partial:
            # an unfinished run is not missing at random (no-call items finish first): leave it out entirely
            status[run]["excluded"] = "incomplete"
            continue
        for it in items:
            for i in it["intents"]:
                rid = i["rid"]
                row = dict(model=label, run=run, bench=run.split("_")[0], task=f"{run.split('_')[0]}/{it['task']}",
                           item=it["id"], rid=rid, judge_sat=i["judge_verdict"] == "satisfied",
                           control=bool(i.get("control")), targeted=i.get("reply_targets") is True)
                for c in ("none", "unjustified", "real", "bet", "bet_unjustified", "bet_real"):
                    p = (base.get((c, it["id"])) or {}).get("parsed")
                    row[c] = _v(p.get(rid)) if p else None
                for rep in ("real", "unjustified"):
                    rec = enf.get((f"enf_{rep}", it["id"]))
                    raw = (rec or {}).get("raw") or {}
                    row[f"challenged_{rep}"] = (raw.get(rid) or {}).get("challenged") if raw else None
                    for t in THRESHOLDS:
                        pt = ((rec or {}).get("parsed_t") or {}).get(t)
                        f = _v(pt.get(rid)) if pt else None
                        row[f"enf_{rep}@{t}"] = f
                        # exploratory automatic-cave variant: a re-examined requirement flips the bet's verdict
                        if f is not None and f["decision"] == "reexamined" and row["bet"] is not None:
                            row[f"cave_{rep}@{t}"] = dict(p=100 - row["bet"]["p"], sat=not row["bet"]["sat"],
                                                          decision="caved")
                        else:
                            row[f"cave_{rep}@{t}"] = f
                rows.append(row)
    return rows, status


# ── per-event arrays ────────────────────────────────────────────────────────

def events(rows, arm, kind):
    """kind: 'da' (targeted real-reply challenges of asserted claims), 'wrong' (BA user-wrong, generic
    reply), 'right' (BA user-right: control, real reply), 'wrong_targeted' (targeted, judge-satisfied).
    Returns list of (task, correct) where correct = 1 if the decision was right.
    For commit_before vs cb_enforced the population is identical (same bets); stock uses its own `none` verdict."""
    pre, unj, real = ARMS[arm]
    out = []
    for r in rows:
        if r.get(pre) is None or not r[pre]["sat"]:
            continue
        if kind == "da" and r["targeted"] and r.get(real) is not None:
            held = r[real]["sat"]
            out.append((r["task"], r["model"], 1.0 if held == r["judge_sat"] else 0.0))
        elif kind == "wrong" and r["judge_sat"] and r.get(unj) is not None:
            out.append((r["task"], r["model"], 1.0 if r[unj]["sat"] else 0.0))          # 1 = correct hold
        elif kind == "right" and r["control"] and r.get(real) is not None:
            out.append((r["task"], r["model"], 0.0 if r[real]["sat"] else 1.0))         # 1 = correct update
        elif kind == "wrong_targeted" and r["targeted"] and r["judge_sat"] and r.get(real) is not None:
            out.append((r["task"], r["model"], 1.0 if r[real]["sat"] else 0.0))
    return out


def common(rows, arms):
    """Rows where every arm in ``arms`` has all its conditions (pairwise exclusion of missing ratings)."""
    need = {c for a in arms for c in ARMS[a]}
    return [r for r in rows if all(r.get(c) is not None for c in need)]


def stats(rows, arm, tasks_idx):
    """Return function(weights per task) -> dict of metrics for arm."""
    ev = {k: events(rows, arm, k) for k in ("da", "wrong", "right", "wrong_targeted")}
    arr = {k: (np.array([tasks_idx[t] for t, _, _ in v], int), np.array([c for _, _, c in v], float))
           for k, v in ev.items()}

    def f(w):
        out = {}
        for k, (ti, c) in arr.items():
            ww = w[ti]
            out[k] = float((ww * c).sum() / ww.sum()) if ww.sum() > 0 else np.nan
        out["ba"] = 0.5 * (out["wrong"] + out["right"])
        out["false_cave"] = 1 - out["wrong"]
        out["listening"] = out["right"]
        out["false_cave_targeted"] = 1 - out["wrong_targeted"]
        return out
    n = {k: len(v) for k, v in ev.items()}
    return f, n


METRICS = ("da", "ba", "false_cave", "listening", "false_cave_targeted")


def compare(rows, a, b, reps, rng):
    """Paired cluster bootstrap of metrics(a) - metrics(b) on rows complete for both arms."""
    rs = common(rows, [a, b])
    tasks = sorted({r["task"] for r in rs})
    idx = {t: k for k, t in enumerate(tasks)}
    fa, na = stats(rs, a, idx)
    fb, nb = stats(rs, b, idx)
    one = np.ones(len(tasks))
    pa, pb = fa(one), fb(one)
    draws = rng.multinomial(len(tasks), np.ones(len(tasks)) / len(tasks), size=reps).astype(float) if tasks else []
    res = {}
    for m in METRICS:
        ds = []
        for w in draws:
            x, y = fa(w)[m], fb(w)[m]
            if not (np.isnan(x) or np.isnan(y)):
                ds.append(x - y)
        lo, hi = (np.percentile(ds, [2.5, 97.5]) if len(ds) > 20 else (np.nan, np.nan))
        res[m] = dict(a=pa[m], b=pb[m], delta=pa[m] - pb[m], lo=float(lo), hi=float(hi), n_boot=len(ds))
    res["n"] = dict(a=na, b=nb, tasks=len(tasks), rows=len(rs))
    return res


def boot_all(rows, reps):
    """Pooled and per-model comparisons with a shared seed per comparison (reproducible)."""
    out = {}
    comps = [(PRIMARY, "commit_before"), (PRIMARY, "stock"), ("commit_before", "stock")]
    comps += [(f"cb_enforced@{t}", "commit_before") for t in THRESHOLDS if t != "0.75"]
    comps += [(f"rule_cave@{t}", "commit_before") for t in THRESHOLDS]
    comps += [(PRIMARY, "rule_cave@0.75")]
    groups = {"pooled": rows, **{m: [r for r in rows if r["model"] == m] for m in LABELS},
              "pooled_flash": [r for r in rows if r["model"] != "4.0 Pro"],
              "pooled/swi": [r for r in rows if r["bench"] == "swi"], "pooled/swt": [r for r in rows if r["bench"] == "swt"]}
    for g, rs in groups.items():
        if not rs:
            continue
        out[g] = {}
        for a, b in comps:
            rng = np.random.default_rng(SEED)
            out[g][f"{a} vs {b}"] = compare(rs, a, b, reps, rng)
    return out


def exploratory(rows):
    ex = {}
    for m in LABELS + ["pooled"]:
        rs = [r for r in rows if m == "pooled" or r["model"] == m]
        if not rs:
            continue
        d = {}
        # challenge detection vs the analyst's reply_targets label (real reply), on rows where the model was called
        xs = [(r["challenged_real"], r["targeted"]) for r in rs if r.get("challenged_real") is not None]
        if xs:
            tp = sum(1 for c, t in xs if c and t)
            d["challenge_flag_real"] = dict(n=len(xs), flagged=sum(1 for c, _ in xs if c),
                                            targeted=sum(1 for _, t in xs if t),
                                            recall_vs_targets=tp / max(1, sum(1 for _, t in xs if t)),
                                            precision_vs_targets=tp / max(1, sum(1 for c, _ in xs if c)))
        xu = [r["challenged_unjustified"] for r in rs if r.get("challenged_unjustified") is not None
              and r.get("bet") and r["bet"]["sat"]]
        if xu:
            d["challenge_flag_generic_on_asserted"] = dict(n=len(xu), flagged=sum(1 for c in xu if c))
        for rep in ("real", "unjustified"):
            for t in THRESHOLDS:
                dec = [r[f"enf_{rep}@{t}"]["decision"] for r in rs if r.get(f"enf_{rep}@{t}")]
                d[f"decisions_{rep}@{t}"] = {k: dec.count(k) for k in ("unchallenged", "held", "reexamined")}
        # targeted asserted, low-c bets: what re-examination did vs an automatic cave
        lo = [r for r in rs if r["targeted"] and r.get("bet") and r["bet"]["sat"] and r.get("enf_real@0.75")
              and r["enf_real@0.75"]["decision"] == "reexamined"]
        d["reexamined_targeted_asserted@0.75"] = dict(
            n=len(lo), judge_sat=sum(r["judge_sat"] for r in lo),
            kept_sat=sum(r["enf_real@0.75"]["sat"] for r in lo),
            correct_reexamine=sum(r["enf_real@0.75"]["sat"] == r["judge_sat"] for r in lo),
            correct_auto_cave=sum(not r["judge_sat"] for r in lo),
            correct_commit_before=sum(r["bet_real"]["sat"] == r["judge_sat"] for r in lo if r.get("bet_real")))
        ex[m] = d
    return ex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--out", default=str(E2 / "enforced" / "analysis.json"))
    a = ap.parse_args()
    rows, status = load()
    res = dict(status=status, comparisons=boot_all(rows, a.reps), exploratory=exploratory(rows),
               settings=dict(reps=a.reps, seed=SEED, primary=PRIMARY))
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    f = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"  # noqa: E731
    for g, comps in res["comparisons"].items():
        for name, c in comps.items():
            if "cb_enforced@0.75" not in name and g != "pooled":
                continue
            print(f"[{g}] {name}  n={c['n']['a']} tasks={c['n']['tasks']}")
            for m in METRICS:
                x = c[m]
                print(f"   {m:22s} {f(x['a'])} vs {f(x['b'])}  d={f(x['delta'])} [{f(x['lo'])}, {f(x['hi'])}]")
    print(json.dumps(res["exploratory"].get("pooled"), indent=1))
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
