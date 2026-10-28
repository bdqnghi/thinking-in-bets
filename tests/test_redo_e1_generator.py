"""E1 generator: EV ground truth, cell balance, determinism, prompt parsing."""
from collections import Counter
from fractions import Fraction

import pytest

from fsa.redo.e1 import generator as G
from fsa.redo.e1 import prompts as P


@pytest.fixture(scope="module")
def body():
    return G.generate()


def _role_params(it):
    chosen, alt = it["params"]["chosen"], it["params"]["alt"]
    return (chosen, alt) if it["factors"]["chosen_role"] == "risky" else (alt, chosen)


def test_counts_and_cells(body):
    items = body["items"]
    kinds = Counter(i["kind"] for i in items)
    assert kinds == {"main": 240, "control": 300, "twin": 300}
    cells = Counter((i["ev_cell"], i["outcome"]) for i in items if i["kind"] == "main")
    assert set(cells.values()) == {60}
    per_sid = Counter(i["scenario_id"] for i in items if i["kind"] == "main")
    assert len(per_sid) == 60 and set(per_sid.values()) == {4}
    assert Counter(i["domain"] for i in items if i["kind"] == "main") == {d: 60 for d in ("medical", "finance", "engineering", "agent")}
    firsts = [i for i in items if i["kind"] == "main" and i["ev_cell"] == "good" and i["outcome"] == "lucky"]
    for f in ("chosen_role", "mode", "stakes", "frame", "chosen_first"):
        assert set(Counter(i["factors"][f] for i in firsts).values()) == {30}, f
    ctrl = [i for i in items if i["kind"] == "control"]
    assert set(Counter(i["factors"]["chosen_role"] for i in ctrl).values()) == {150}


def test_ev_ground_truth(body):
    for it in body["items"]:
        c, a = it["params"]["chosen"], it["params"]["alt"]
        r, s = _role_params(it)
        assert G.ordered(r, s), it["id"]                                  # no dominance
        e_c, e_a = G.ev(**c), G.ev(**a)
        assert Fraction(it["ev"]["chosen"]) == e_c and Fraction(it["ev"]["alt"]) == e_a
        gap = (e_c - e_a) / (r["good"] - r["bad"])
        assert G.P_CHOSEN[0] <= c["p"] <= G.P_CHOSEN[1]                  # both outcomes plausible
        if it["kind"] == "main":
            want = 1 if it["ev_cell"] == "good" else -1
            assert G.REL_GAP[0] <= want * gap <= G.REL_GAP[1], it["id"]    # clear but not trivial
        else:
            assert G.REL_GAP[0] <= gap <= G.REL_GAP[1]                    # EV-good as stated
            true_c = dict(c, p=it["params"]["p_true"])
            tg = (G.ev(**true_c) - e_a) / (r["good"] - r["bad"])
            assert G.CONTROL_TRUE_GAP[0] <= -tg <= G.CONTROL_TRUE_GAP[1]  # EV-bad given the hidden fact


def test_ev_by_hand():
    assert G.ev(70, 100, 0) == 70
    assert G.ev(35, 90, -40) == Fraction(35 * 90 - 65 * 40, 100)


def test_render_values_match_params(body):
    it = next(i for i in body["items"] if i["id"] == "S01-good-lucky")
    c = it["params"]["chosen"]
    assert f"{c['p']}%" in it["decision_text"]
    assert "chose Option" in it["decision_text"]
    lucky = next(i for i in body["items"] if i["id"] == "S01-good-unlucky")
    assert lucky["decision_text"] == it["decision_text"] and lucky["decision_key"] == it["decision_key"]
    assert lucky["outcome_text"] != it["outcome_text"]


def test_control_twin_share_decision_and_hide_fact(body):
    items = {i["id"]: i for i in body["items"]}
    for pid in {i["pair_id"] for i in body["items"] if i["kind"] != "main"}:
        c, t = items[f"{pid}-control"], items[f"{pid}-twin"]
        assert c["decision_text"] == t["decision_text"]
        assert c["fact"][0] in c["outcome_text"] and c["fact"][0] not in c["decision_text"]
        assert "could cheaply" not in c["decision_text"]
        assert t["cause"] in t["outcome_text"]


def test_value_formatting():
    t = next(t for t in G.TEMPLATES if t["unit"] == "money_gain")
    assert G.fmt_value(t, "low", 50) == "net gain $50k"
    assert G.fmt_value(t, "high", 60) == "net gain $1.2M"
    assert G.fmt_value(t, "low", -40) == "net loss $40k"
    t = next(t for t in G.TEMPLATES if t["unit"] == "downtime_hours")
    assert G.fmt_value(t, "low", 75) == "2.5 hours of downtime"          # cost = 100 - base


def test_deterministic_and_hash(body):
    again = G.generate()
    assert again["items_sha256"] == body["items_sha256"] == G.items_hash(body["items"])
    assert again["pilot_ids"] == body["pilot_ids"]
    assert G.generate(seed=1)["items_sha256"] != body["items_sha256"]


def test_pilot_is_small(body):
    assert len(body["pilot_ids"]) <= 0.05 * len(body["items"])


def test_parse():
    assert P.parse('{"reason": "x", "rating": 12}', "standard")["rating"] == 12
    assert P.parse('```json\n{"reason": "x", "rating": "-7"}\n```', "standard")["rating"] == -7
    assert P.parse('Sure: {"rating": 5}', "standard")["rating"] == 5
    assert P.parse('{"rating": 45}', "standard") is None
    assert P.parse('{"rating": 2.5}', "standard") is None
    assert P.parse("no json", "standard") is None
    assert P.parse('{"rating": 3, "p_good_outcome": 70}', "bet_first", 1)["p_good_outcome"] == 0.7
    assert P.parse('{"rating": 3}', "bet_first", 1) is None
    assert P.parse('{"rating": 3, "revise": "false"}', "bet_first", 2)["revise"] is False
    assert P.parse('{"rating": 3}', "bet_first", 2) is None


def test_prompts_hide_outcome_when_blind(body):
    it = next(i for i in body["items"] if i["kind"] == "control")
    for arm in ("outcome_blind",):
        text = P.build(arm, it)[1]["content"]
        assert it["outcome_text"] not in text
    assert it["outcome_text"] not in P.build("bet_first", it, 1)[1]["content"]
    s2 = P.build("bet_first", it, 2, {"rating": 10, "p_good_outcome": 0.7, "reason": "r"})[1]["content"]
    assert it["outcome_text"] in s2 and "rating: 10" in s2
    for arm in ("standard", "ignore_outcome", "counterfactual"):
        assert it["outcome_text"] in P.build(arm, it)[1]["content"]
