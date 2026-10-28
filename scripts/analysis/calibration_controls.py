"""#7 (L-O): calibration controls for the HLE single-shot Brier/ECE result (W4).

Question: is ``cb_draftfirst``'s calibration gain more than what a cheap post-hoc recalibration of ``stock`` gets?
For each model, on the items both arms answered:

* split items 50/50 (stratified by HLE category), fit on the ``stock`` half A, evaluate on half B, swap, average;
  repeated over ``N_SPLITS`` random splits (split 0 is the pre-specified headline split, seed 0);
* stock recalibrators: temperature scaling on logit(conf) (p' = sigmoid(logit(p)/T), T fit by Brier on half A),
  a constant shift (p' = clip(p + b)), and Platt scaling (p' = sigmoid(a*logit(p) + c)), all fit by minimising Brier;
* compare on the held-out halves: Brier, ECE (10 equal-width bins, L1), the official HLE RMS calibration error
  (beta=25, as in hle_benchmark.summarize_single_shot), and AURC (risk-coverage, from hle_benchmark.compute_aurc),
  versus ``cb_draftfirst`` raw. Note: monotone recalibration cannot change AURC (ranking is unchanged), so AURC is
  the "ranking quality" part a recalibrator cannot buy.

Output: ``results/analysis/calibration_controls.json`` + markdown on stdout. Re-run on the #1 draft-first arms
(``self_refine``, ``cove``, ``sc_matched``) once their single-shot JSONLs exist (``--glob``).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from hle_benchmark import compute_aurc, official_hle_calib_err  # noqa: E402

N_SPLITS = 200
EPS = 1e-4


def ece10(p: np.ndarray, y: np.ndarray) -> float:
    bins = np.minimum((p * 10).astype(int), 9)
    e = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_temperature(p, y):
    z = logit(p)
    grid = np.exp(np.linspace(math.log(0.2), math.log(20), 400))
    briers = [np.mean((sigmoid(z / t) - y) ** 2) for t in grid]
    return float(grid[int(np.argmin(briers))])


def fit_shift(p, y):
    grid = np.linspace(-1, 1, 801)
    briers = [np.mean((np.clip(p + b, 0, 1) - y) ** 2) for b in grid]
    return float(grid[int(np.argmin(briers))])


def fit_platt(p, y, iters: int = 500, lr: float = 0.5):
    z = logit(p)
    a, c = 1.0, 0.0
    for _ in range(iters):  # gradient descent on Brier
        q = sigmoid(a * z + c)
        g = 2 * (q - y) * q * (1 - q)
        a -= lr * float(np.mean(g * z))
        c -= lr * float(np.mean(g))
    return a, c


def metrics(p, y) -> dict:
    return dict(brier=float(np.mean((p - y) ** 2)), ece10=ece10(p, y),
                rms_ce_b25=official_hle_calib_err(p.copy(), y.copy(), p="2", beta=25),
                aurc=compute_aurc(p, y), mean_conf=float(p.mean()), acc=float(y.mean()))


def load(path: str) -> dict[str, dict[str, dict]]:
    by: dict[str, dict[str, dict]] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by.setdefault(r["arm"], {})[r["id"]] = r  # last row per (id, arm) = what the summaries used
    return by


def split_ids(ids: list[str], cats: dict[str, str], rng: np.random.Generator) -> tuple[list[str], list[str]]:
    a, b = [], []
    by: dict[str, list[str]] = {}
    for i in ids:
        by.setdefault(cats[i], []).append(i)
    for c in sorted(by):
        xs = list(by[c])
        rng.shuffle(xs)
        h = len(xs) // 2 + (rng.random() < 0.5 if len(xs) % 2 else 0)
        a += xs[:h]
        b += xs[h:]
    return a, b


def analyse(path: str, other_arms: list[str]) -> dict | None:
    by = load(path)
    if "stock" not in by:
        return None
    res = {}
    for arm in ["cb_draftfirst"] + other_arms:
        if arm not in by:
            continue
        ids = sorted(set(by["stock"]) & set(by[arm]))
        cats = {i: by["stock"][i]["category"] for i in ids}
        ps = np.array([by["stock"][i]["confidence"] / 100 for i in ids], float)
        ys = np.array([1.0 if by["stock"][i]["correct_bool"] else 0.0 for i in ids])
        pc = np.array([by[arm][i]["confidence"] / 100 for i in ids], float)
        yc = np.array([1.0 if by[arm][i]["correct_bool"] else 0.0 for i in ids])
        idx = {i: k for k, i in enumerate(ids)}
        per_split = []
        for s in range(N_SPLITS):
            rng = np.random.default_rng(s)
            A, B = split_ids(ids, cats, rng)
            acc: dict[str, list[dict]] = {}
            for fit_half, ev_half in ((A, B), (B, A)):
                fi = np.array([idx[i] for i in fit_half])
                ei = np.array([idx[i] for i in ev_half])
                T = fit_temperature(ps[fi], ys[fi])
                b = fit_shift(ps[fi], ys[fi])
                pa, pcc = fit_platt(ps[fi], ys[fi])
                cands = {
                    "stock_raw": ps[ei],
                    "stock_temp": sigmoid(logit(ps[ei]) / T),
                    "stock_shift": np.clip(ps[ei] + b, 0, 1),
                    "stock_platt": sigmoid(pa * logit(ps[ei]) + pcc),
                    f"{arm}_raw": pc[ei],
                }
                for k, v in cands.items():
                    yy = yc[ei] if k.startswith(arm) else ys[ei]
                    acc.setdefault(k, []).append(metrics(v, yy))
                acc.setdefault("_params", []).append(dict(T=T, shift=b, platt_a=pa, platt_c=pcc))
            per_split.append({k: ({m: float(np.mean([d[m] for d in v])) for m in v[0]} if k != "_params" else v)
                              for k, v in acc.items()})
        summ = {}
        for k in per_split[0]:
            if k == "_params":
                continue
            summ[k] = {}
            for m in per_split[0][k]:
                xs = np.array([sp[k][m] for sp in per_split])
                summ[k][m] = dict(split0=round(float(xs[0]), 4), mean=round(float(xs.mean()), 4),
                                  lo=round(float(np.percentile(xs, 2.5)), 4), hi=round(float(np.percentile(xs, 97.5)), 4))
        # key comparison: best stock recalibrator vs arm raw, Brier, per split
        best = {}
        for rc in ("stock_temp", "stock_shift", "stock_platt"):
            d = np.array([sp[f"{arm}_raw"]["brier"] - sp[rc]["brier"] for sp in per_split])
            best[rc] = dict(delta_brier_arm_minus_recal_mean=round(float(d.mean()), 4),
                            share_splits_arm_better=round(float((d < 0).mean()), 3))
        res[arm] = dict(n_items=len(ids), summary=summ, arm_vs_recalibrated_stock=best,
                        params_split0=per_split[0]["_params"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "results/hle/single_shot_*.jsonl"))
    ap.add_argument("--arms", default="self_refine,cove,sc_matched,review_only")
    a = ap.parse_args()
    out = {"n_splits": N_SPLITS, "files": {}}
    md = ["| Model file | Arm | n | Brier stock raw | stock temp | stock shift | stock Platt | arm raw | "
          "ECE10 stock raw | stock temp | arm raw | AURC stock | AURC arm | arm better than best recal (share of splits) |",
          "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"]
    for f in sorted(glob.glob(a.glob)):
        r = analyse(f, [x for x in a.arms.split(",") if x])
        if not r:
            continue
        out["files"][Path(f).name] = r
        for arm, v in r.items():
            s = v["summary"]
            g = lambda k, m: s[k][m]["mean"]
            bestshare = max(v["arm_vs_recalibrated_stock"].values(), key=lambda d: -d["share_splits_arm_better"])
            md.append(f"| {Path(f).stem.replace('single_shot_', '')} | {arm} | {v['n_items']} | "
                      f"{g('stock_raw', 'brier'):.4f} | {g('stock_temp', 'brier'):.4f} | {g('stock_shift', 'brier'):.4f} | "
                      f"{g('stock_platt', 'brier'):.4f} | **{g(arm + '_raw', 'brier'):.4f}** | "
                      f"{g('stock_raw', 'ece10'):.3f} | {g('stock_temp', 'ece10'):.3f} | {g(arm + '_raw', 'ece10'):.3f} | "
                      f"{g('stock_raw', 'aurc'):.3f} | {g(arm + '_raw', 'aurc'):.3f} | "
                      f"{bestshare['share_splits_arm_better']:.2f} |")
    dst = ROOT / "results/analysis/calibration_controls.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print("\n".join(md))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
