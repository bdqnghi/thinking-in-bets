"""Live progress & paired comparison table for Humanity's Last Exam (`results/hle`)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts.hle_benchmark import summarize_multi_turn, summarize_single_shot


def main(out_dir: str = "results/hle") -> None:
    p = Path(out_dir)
    if not p.exists():
        print(f"No directory {p}")
        return

    print("=== Setting 1: Single-Shot Official HLE Leaderboard (Accuracy % + Calibration Error %) ===")
    print(
        "Model | Arm | N | Accuracy (%) | Mean Conf (%) | Overconf Gap | Official RMS-CE (b100 / b25) | Brier | Acc@Conf>=80%"
    )
    for ss_file in sorted(p.glob("single_shot_*.jsonl")):
        rows = [json.loads(line) for line in ss_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        model = rows[0]["model"]
        by_arm: dict[str, dict[str, dict]] = {}
        for r in rows:
            by_arm.setdefault(r["arm"], {})[r["id"]] = r
        for arm in ["stock", "cb_draftfirst"]:
            if arm not in by_arm:
                continue
            s = summarize_single_shot(list(by_arm[arm].values()))
            print(
                f"{model} | {arm:13s} | {s['n']:3d} | {s['accuracy_pct']:5.1f}% ± {s['ci95_pct']:4.1f}% | "
                f"{s['mean_confidence_pct']:5.1f}% | {s['overconfidence_gap_pct']:+5.1f}% | "
                f"{s['rms_calibration_error_beta100_pct']:5.1f}% / {s['rms_calibration_error_beta25_pct']:5.1f}% | "
                f"{s['brier_score']:.3f} | {s['acc_when_conf_ge_80_pct']}%"
            )
        if "stock" in by_arm and "cb_draftfirst" in by_arm:
            common = sorted(set(by_arm["stock"]) & set(by_arm["cb_draftfirst"]))
            if common:
                s_st = summarize_single_shot([by_arm["stock"][i] for i in common])
                s_cb = summarize_single_shot([by_arm["cb_draftfirst"][i] for i in common])
                wins = sum(
                    1
                    for i in common
                    if by_arm["cb_draftfirst"][i]["correct_bool"] and not by_arm["stock"][i]["correct_bool"]
                )
                losses = sum(
                    1
                    for i in common
                    if by_arm["stock"][i]["correct_bool"] and not by_arm["cb_draftfirst"][i]["correct_bool"]
                )
                ties = len(common) - wins - losses
                print(
                    f"  -> PAIRED ({len(common)} q): Acc {s_st['accuracy_pct']:.1f}% -> {s_cb['accuracy_pct']:.1f}% "
                    f"({s_cb['accuracy_pct'] - s_st['accuracy_pct']:+.1f}%), "
                    f"RMS-CE(b25) {s_st['rms_calibration_error_beta25_pct']:.1f}% -> {s_cb['rms_calibration_error_beta25_pct']:.1f}% "
                    f"({s_cb['rms_calibration_error_beta25_pct'] - s_st['rms_calibration_error_beta25_pct']:+.1f}%), "
                    f"Brier {s_st['brier_score']:.3f} -> {s_cb['brier_score']:.3f} | W/T/L = {wins}/{ties}/{losses}"
                )

    print("\n=== Setting 2: Multi-Turn HLE Reviewer Pushback & Recovery ===")
    print("Model | Arm | N | T1 Acc -> T2 Acc (Net) | False-Pushback Caving Rate | Constructive-Hint Recovery Rate")
    for mt_file in sorted(p.glob("multi_turn_*.jsonl")):
        rows = [json.loads(line) for line in mt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        model = rows[0]["model"]
        by_arm_mt: dict[str, dict[str, dict]] = {}
        for r in rows:
            by_arm_mt.setdefault(r["arm"], {})[r["id"]] = r
        for arm in ["stock", "claims_after", "claims_no_p", "commit_before"]:
            if arm not in by_arm_mt:
                continue
            s = summarize_multi_turn(list(by_arm_mt[arm].values()))
            print(
                f"{model} | {arm:13s} | {s['n_total']:3d} | "
                f"{s['t1_accuracy_pct']:5.1f}% -> {s['t2_accuracy_pct']:5.1f}% ({s['net_acc_gain_pct']:+5.1f}%) | "
                f"{s['caving_rate_pct']:5.1f}% ({s['caving_count']}) | "
                f"{s['recovery_rate_pct']:5.1f}% ({s['recovery_count']})"
            )


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "results/hle"
    main(d)
