"""Unit tests for the Agents' Last Exam (ALE) deterministic-verifier pilot (fsa/redo/ale_pilot.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsa.redo.ale_pilot import (
    ALE_DATA,
    ALE_TASKS,
    AMBER_3S_FLAWED,
    AMBER_3S_GOLD,
    AMBER_MIN_FLAWED,
    AMBER_MIN_GOLD,
    FIN_FLAWED,
    FIN_GOLD,
    K8S_FLAWED,
    K8S_GOLD,
    LEGAL_FLAWED,
    LEGAL_GOLD,
    SCORERS,
    build_scenarios,
    score_amber_3s,
    score_amber_min,
    score_american_option,
    score_k8s_rca,
    score_legal_fees,
)

pytestmark = pytest.mark.skipif(
    not ALE_TASKS.exists() or not ALE_DATA.exists(),
    reason="External agents-last-exam repository not present on this machine",
)


def test_gold_and_flawed_turn1_scores() -> None:
    s_k8s_g, p_k8s_g, _ = score_k8s_rca(K8S_GOLD)
    assert p_k8s_g is True and s_k8s_g == pytest.approx(1.0)
    s_k8s_f, p_k8s_f, _ = score_k8s_rca(K8S_FLAWED)
    assert p_k8s_f is False and 0.5 < s_k8s_f < 0.9

    s_am_g, p_am_g, _ = score_amber_min(AMBER_MIN_GOLD)
    assert p_am_g is True and s_am_g == pytest.approx(1.0)
    s_am_f, p_am_f, _ = score_amber_min(AMBER_MIN_FLAWED)
    assert p_am_f is False and s_am_f == pytest.approx(0.0)

    s_a3_g, p_a3_g, _ = score_amber_3s(AMBER_3S_GOLD)
    assert p_a3_g is True and s_a3_g == pytest.approx(1.0)
    s_a3_f, p_a3_f, _ = score_amber_3s(AMBER_3S_FLAWED)
    assert p_a3_f is False and s_a3_f == pytest.approx(0.0)

    s_lg_g, p_lg_g, _ = score_legal_fees(LEGAL_GOLD)
    assert p_lg_g is True and s_lg_g == pytest.approx(1.0)
    s_lg_f, p_lg_f, _ = score_legal_fees(LEGAL_FLAWED)
    assert p_lg_f is False and s_lg_f == pytest.approx(0.6)

    s_fn_g, p_fn_g, _ = score_american_option(FIN_GOLD)
    assert p_fn_g is True and s_fn_g == pytest.approx(1.0)
    s_fn_f, p_fn_f, _ = score_american_option(FIN_FLAWED)
    assert p_fn_f is False and s_fn_f == pytest.approx(0.5)


def test_scenarios_and_saved_results() -> None:
    scenarios = build_scenarios()
    assert len(scenarios) == 20
    cats = [s.category for s in scenarios]
    assert cats.count("unjustified") == 8
    assert cats.count("misaimed") == 7
    assert cats.count("cooperative") == 5

    for sc in scenarios:
        scorer = SCORERS[sc.task_id]
        score_t1, passed_t1, _ = scorer(sc.turn1_deliverable)
        if sc.category in ("unjustified", "misaimed"):
            assert passed_t1 is True and score_t1 == pytest.approx(1.0)
        else:
            assert passed_t1 is False and score_t1 < 1.0

    for fname in ("ale_e2_gemini_3_5_flash_lite.json", "ale_e2_gemini_3_6_flash.json"):
        p = Path("results/redo_v1_ale") / fname
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            assert len(data["trials"]) == 80
            arms = data["summary"]["arms"]
            assert arms["commit_before"]["final_pass_rate"] > arms["stock"]["final_pass_rate"]
            assert arms["commit_before"]["caving_rate_all_pushback"] < arms["stock"]["caving_rate_all_pushback"]

