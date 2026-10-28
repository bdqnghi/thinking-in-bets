"""E1 metrics on synthetic ratings with known answers."""
import math
import random

from fsa.redo.e1 import metrics as M


def _main(fn, n=40):
    return {(f"S{i}", e, o): fn(i, e, o) for i in range(n) for e in ("good", "bad") for o in ("lucky", "unlucky")}, [f"S{i}" for i in range(n)]


def test_no_outcome_bias_gives_zero_d():
    main, sids = _main(lambda i, e, o: (10 if e == "good" else -10) + (i % 5))
    assert M.cohen_d(main, sids) == 0
    assert M.luck_reversal(main, sids) == 0
    assert M.within_outcome_auc(main, sids) == 1


def test_known_d():
    rng = random.Random(0)
    noise = {i: rng.gauss(0, 1) for i in range(2000)}
    main, sids = _main(lambda i, e, o: noise[i] + (0.5 if o == "lucky" else 0), n=2000)
    # pooled SD ~ 1, shift 0.5 -> d ~ 0.5; paired diff has zero variance -> d_z infinite
    assert abs(M.cohen_d(main, sids) - 0.5) < 0.05
    assert M.d_z(main, sids) > 1e6
    assert abs(M.mean_luck_gap(main, sids) - 0.5) < 1e-9


def test_pure_outcome_judge():
    main, sids = _main(lambda i, e, o: 20 if o == "lucky" else -20)
    assert math.isinf(M.cohen_d(main, sids))            # zero within-cell variance
    assert M.luck_reversal(main, sids) == 1
    assert M.within_outcome_auc(main, sids) == 0.5


def test_missing_excluded():
    main, sids = _main(lambda i, e, o: (i % 3) + (1 if o == "lucky" else 0))
    main[("S0", "good", "lucky")] = None
    d = M.cohen_d(main, sids)
    assert d > 0 and not math.isnan(d)


def test_auc():
    assert M.auc([3, 4], [1, 2]) == 1
    assert M.auc([1], [1]) == 0.5
    assert M.auc([1, 3], [2]) == 0.5


def test_correct_update():
    ctrl = {"a": {"control": 0, "twin": 10}, "b": {"control": 8, "twin": 10}, "c": {"control": None, "twin": 3},
            "d": {"control": 5, "twin": 5}}
    assert M.correct_update_rate(ctrl, list(ctrl)) == 1 / 3
    ref = {"a": 12, "b": 12, "c": 12, "d": 5}
    assert M.drop_vs_reference(ctrl, ref, list(ctrl), "control") == 1 / 3   # a (12->0): yes; b (12->8): no; d: no


def test_bootstrap_ci_covers_and_is_deterministic():
    main, sids = _main(lambda i, e, o: (i % 7) + (3 if o == "lucky" else 0) + (1 if e == "good" else 0))
    b1 = M.bootstrap(lambda s: M.mean_luck_gap(main, s), sids, reps=500)
    b2 = M.bootstrap(lambda s: M.mean_luck_gap(main, s), sids, reps=500)
    assert b1 == b2 and b1["lo"] <= b1["point"] <= b1["hi"] and b1["point"] == 3
