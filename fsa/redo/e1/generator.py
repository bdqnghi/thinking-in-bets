"""E1 item generator: templated decisions with programmatic expected-value ground truth.

Main set: 60 scenarios (20 templates x 3 instances) x {EV-good, EV-bad} x
{lucky, unlucky} = 240 items. Control set: 300 control items whose bad outcome
reveals a cheaply checkable decision-time fact that turns the choice EV-bad,
each paired with a luck twin whose bad outcome has a cause nothing available at
decision time could reveal (300 twins). Twin and control share the decision
text, so an outcome-blind judge sees them identically.

All values live on an integer base scale (step 5) and all probabilities are
integer percentages (step 5), so expected values are exact rationals. The
normative criterion is stated in every item: risk-neutral expected value.

Usage: python -m fsa.redo.e1.generator [--seed 20260926] [--out results/redo_v1/e1/items.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path

from .templates import TEMPLATES, UNITS

GENERATOR_VERSION = "e1-gen-1"
DEFAULT_SEED = 20260926
REL_GAP = (Fraction(8, 100), Fraction(22, 100))      # |EV difference| / outcome range for main items
CONTROL_TRUE_GAP = (Fraction(10, 100), Fraction(40, 100))
P_CHOSEN = (30, 85)        # chosen option's P(good) in every cell: both outcomes plausible
P_RISKY = (35, 80)
P_SAFE = (55, 90)
MIN_P_SHIFT = 10           # "prob" mode: EV-good and EV-bad cells differ by >= 10 points of P(good)


def ev(p: int, good: int, bad: int) -> Fraction:
    return Fraction(p * good + (100 - p) * bad, 100)


def _grid(lo: int, hi: int, step: int = 5) -> list[int]:
    return list(range(lo, hi + 1, step))


def ordered(risky: dict, safe: dict) -> bool:
    """Neither option dominates: the risky option has the best and the worst branch."""
    return risky["bad"] + 10 <= safe["bad"] and safe["bad"] + 10 <= safe["good"] and safe["good"] + 10 <= risky["good"]


def outcome_range(risky: dict) -> int:
    return risky["good"] - risky["bad"]


def rel_gap(chosen: dict, alt: dict, risky: dict) -> Fraction:
    """Signed: positive when the chosen option has the higher expected value."""
    return (ev(**chosen) - ev(**alt)) / outcome_range(risky)


def _in(x, lo_hi) -> bool:
    return lo_hi[0] <= x <= lo_hi[1]


def _sample_option(rng: random.Random, role: str, t: dict) -> dict:
    lo, hi = t["lo"], t["hi"]
    if role == "risky":
        return dict(p=rng.choice(_grid(*P_RISKY)), good=rng.choice(_grid(hi - 25, hi)), bad=rng.choice(_grid(lo, lo + 30)))
    return dict(p=rng.choice(_grid(*P_SAFE)), good=rng.choice(_grid(lo + 65, hi - 15)), bad=rng.choice(_grid(lo + 20, lo + 60)))


def _pair(chosen_role: str, chosen: dict, alt: dict) -> tuple[dict, dict]:
    return (chosen, alt) if chosen_role == "risky" else (alt, chosen)


def sample_main(rng: random.Random, t: dict, chosen_role: str, mode: str) -> dict:
    """Parameters for the EV-good and EV-bad cells of one scenario."""
    alt_role = "safe" if chosen_role == "risky" else "risky"
    for _ in range(100000):
        chosen = _sample_option(rng, chosen_role, t)
        if not _in(chosen["p"], P_CHOSEN):
            continue
        if mode == "alt":
            cells = {}
            for cell, sign in (("good", 1), ("bad", -1)):
                for _ in range(400):
                    alt = _sample_option(rng, alt_role, t)
                    r, s = _pair(chosen_role, chosen, alt)
                    if ordered(r, s) and _in(sign * rel_gap(chosen, alt, r), REL_GAP):
                        cells[cell] = dict(chosen=dict(chosen), alt=alt)
                        break
            if len(cells) == 2:
                return cells
        else:
            alt = _sample_option(rng, alt_role, t)
            r0, s0 = _pair(chosen_role, chosen, alt)
            if not ordered(r0, s0):
                continue
            ok = {}
            for p in _grid(*P_CHOSEN):
                c = dict(chosen, p=p)
                r, _ = _pair(chosen_role, c, alt)
                g = rel_gap(c, alt, r)
                if _in(g, REL_GAP):
                    ok.setdefault("good", []).append(p)
                if _in(-g, REL_GAP):
                    ok.setdefault("bad", []).append(p)
            pairs = [(pg, pb) for pg in ok.get("good", []) for pb in ok.get("bad", []) if abs(pg - pb) >= MIN_P_SHIFT]
            if pairs:
                pg, pb = rng.choice(pairs)
                return {"good": dict(chosen=dict(chosen, p=pg), alt=dict(alt)),
                        "bad": dict(chosen=dict(chosen, p=pb), alt=dict(alt))}
    raise RuntimeError(f"could not sample parameters for {t['key']} {chosen_role} {mode}")


def sample_control(rng: random.Random, t: dict, chosen_role: str) -> dict:
    """Stated parameters make the choice EV-good; the hidden fact's true P(good) makes it EV-bad."""
    alt_role = "safe" if chosen_role == "risky" else "risky"
    for _ in range(100000):
        chosen, alt = _sample_option(rng, chosen_role, t), _sample_option(rng, alt_role, t)
        r, s = _pair(chosen_role, chosen, alt)
        if not (_in(chosen["p"], P_CHOSEN) and ordered(r, s) and _in(rel_gap(chosen, alt, r), REL_GAP)):
            continue
        truths = [p for p in _grid(5, chosen["p"] - 15)
                  if _in(-rel_gap(dict(chosen, p=p), alt, r), CONTROL_TRUE_GAP)]
        if truths:
            return dict(chosen=chosen, alt=alt, p_true=rng.choice(truths))
    raise RuntimeError(f"could not sample control parameters for {t['key']}")


# ------------------------------------------------------------------ rendering
def _num(x: float) -> str:
    return f"{x:,.1f}".rstrip("0").rstrip(".") if x != int(x) else f"{int(x):,}"


def fmt_value(t: dict, stakes: str, base: int) -> str:
    kind, lo_mult, hi_mult, _ = UNITS[t["unit"]]
    mult = lo_mult if stakes == "low" else hi_mult
    unit = t["unit"]
    shown = round((base if kind == "gain" else 100 - base) * mult, 6)
    if unit == "qaly":
        return f"{_num(shown)} QALYs"
    if unit == "points":
        return f"{'+' if shown > 0 else ''}{_num(shown)} points"
    if unit == "downtime_hours":
        return f"{_num(shown)} hours of downtime"

    def money(v):
        return f"${_num(v / 1000)}M" if abs(v) >= 1000 else f"${_num(v)}k"
    if unit == "money_cost":
        return f"total cost {money(shown)}"
    if shown > 0:
        return f"net gain {money(shown)}"
    if shown < 0:
        return f"net loss {money(-shown)}"
    return "break even ($0)"


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def render_option(t: dict, text: dict, prm: dict, stakes: str, frame: str) -> str:
    g, b = fmt_value(t, stakes, prm["good"]), fmt_value(t, stakes, prm["bad"])
    if frame == "success":
        return f"{prm['p']}% chance that {text['good']} ({g}); otherwise ({100 - prm['p']}%) {text['bad']} ({b})"
    return f"{100 - prm['p']}% chance that {text['bad']} ({b}); otherwise ({prm['p']}%) {text['good']} ({g})"


def render_decision(t: dict, chosen_role: str, chosen: dict, alt: dict, stakes: str, frame: str, chosen_first: bool) -> str:
    alt_role = "safe" if chosen_role == "risky" else "risky"
    entries = [(chosen_role, chosen), (alt_role, alt)]
    if not chosen_first:
        entries.reverse()
    letter = "A" if chosen_first else "B"
    lines = [f"Option {'AB'[i]} ({t[role]['label']}): {render_option(t, t[role], prm, stakes, frame)}."
             for i, (role, prm) in enumerate(entries)]
    objective = UNITS[t["unit"]][3]
    role = _cap(t["role"])
    return (f"{t['context']} The objective is to {objective}; "
            f"only expected value matters (the decision should be risk-neutral).\n\n" + "\n".join(lines) +
            f"\n\n{role} chose Option {letter}: to {t[chosen_role]['act']}.")


def render_outcome(t: dict, chosen_role: str, chosen: dict, stakes: str, lucky: bool) -> str:
    text = t[chosen_role]
    base = chosen["good"] if lucky else chosen["bad"]
    return f"{text['past_good' if lucky else 'past_bad']}. The result was: {fmt_value(t, stakes, base)}."


def render_control_outcome(t: dict, chosen_role: str, chosen: dict, stakes: str, fact: tuple, p_true: int) -> str:
    fact_text, source, time = fact
    return (render_outcome(t, chosen_role, chosen, stakes, lucky=False) +
            f" A review afterwards found that {fact_text}. This was recorded in {source}, which was available to "
            f"{t['role']} before the decision; checking it would have taken {'' if time.startswith('a ') else 'about '}{time}. Given this fact, the real chance "
            f"of the good outcome for the chosen option was only about {p_true}%, not the {chosen['p']}% that was assumed.")


def render_twin_outcome(t: dict, chosen_role: str, chosen: dict, stakes: str, cause: str) -> str:
    return (render_outcome(t, chosen_role, chosen, stakes, lucky=False) +
            f" A review afterwards found that {cause}. Nothing available to {t['role']} at the time of the decision could "
            f"have revealed this; the {chosen['p']}% estimate was accurate given everything that could be known.")


def _balanced(rng: random.Random, n: int, values: list) -> list:
    out = [values[i % len(values)] for i in range(n)]
    rng.shuffle(out)
    return out


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _ev_record(chosen: dict, alt: dict, chosen_role: str) -> dict:
    r, _ = _pair(chosen_role, chosen, alt)
    return dict(chosen=str(ev(**chosen)), alt=str(ev(**alt)), rel_gap=str(rel_gap(chosen, alt, r)))


def generate(seed: int = DEFAULT_SEED, instances: int = 3, controls_per_template: int = 15) -> dict:
    rng = random.Random(seed)
    items: list[dict] = []
    n_main = len(TEMPLATES) * instances
    modes, stakes_l = _balanced(rng, n_main, ["alt", "prob"]), _balanced(rng, n_main, ["low", "high"])
    frames, firsts = _balanced(rng, n_main, ["success", "failure"]), _balanced(rng, n_main, [True, False])
    k = 0
    for ti, t in enumerate(TEMPLATES):
        for inst in range(instances):
            sid = f"S{k + 1:02d}"
            chosen_role = "risky" if (ti + inst) % 2 == 0 else "safe"
            factors = dict(chosen_role=chosen_role, mode=modes[k], stakes=stakes_l[k], frame=frames[k], chosen_first=firsts[k])
            cells = sample_main(rng, t, chosen_role, modes[k])
            for ev_cell in ("good", "bad"):
                c = cells[ev_cell]
                decision = render_decision(t, chosen_role, c["chosen"], c["alt"], stakes_l[k], frames[k], firsts[k])
                for outcome in ("lucky", "unlucky"):
                    items.append(dict(
                        id=f"{sid}-{ev_cell}-{outcome}", kind="main", scenario_id=sid, template=t["key"], domain=t["domain"],
                        subtype=t["subtype"], ev_cell=ev_cell, outcome=outcome, factors=factors,
                        params=dict(chosen=c["chosen"], alt=c["alt"]), ev=_ev_record(c["chosen"], c["alt"], chosen_role),
                        decision_text=decision, decision_key=_key(decision),
                        outcome_text=render_outcome(t, chosen_role, c["chosen"], stakes_l[k], outcome == "lucky")))
            k += 1
    n_ctrl = len(TEMPLATES) * controls_per_template
    stakes_c, frames_c = _balanced(rng, n_ctrl, ["low", "high"]), _balanced(rng, n_ctrl, ["success", "failure"])
    firsts_c = _balanced(rng, n_ctrl, [True, False])
    k = 0
    for ti, t in enumerate(TEMPLATES):
        for inst in range(controls_per_template):
            pid = f"C{k + 1:03d}"
            chosen_role = "risky" if (ti + inst) % 2 == 0 else "safe"
            prm = sample_control(rng, t, chosen_role)
            facts, causes = t[chosen_role]["facts"], t[chosen_role]["causes"]
            fact, cause = facts[(inst // 2) % len(facts)], causes[(inst // 2 + inst) % len(causes)]
            factors = dict(chosen_role=chosen_role, stakes=stakes_c[k], frame=frames_c[k], chosen_first=firsts_c[k])
            decision = render_decision(t, chosen_role, prm["chosen"], prm["alt"], stakes_c[k], frames_c[k], firsts_c[k])
            common = dict(pair_id=pid, template=t["key"], domain=t["domain"], subtype=t["subtype"], ev_cell="good_as_stated",
                          outcome="unlucky", factors=factors, params=prm, ev=_ev_record(prm["chosen"], prm["alt"], chosen_role),
                          ev_true=_ev_record(dict(prm["chosen"], p=prm["p_true"]), prm["alt"], chosen_role),
                          decision_text=decision, decision_key=_key(decision))
            items.append(dict(id=f"{pid}-control", kind="control", **common, fact=list(fact),
                              outcome_text=render_control_outcome(t, chosen_role, prm["chosen"], stakes_c[k], fact, prm["p_true"])))
            items.append(dict(id=f"{pid}-twin", kind="twin", **common, cause=cause,
                              outcome_text=render_twin_outcome(t, chosen_role, prm["chosen"], stakes_c[k], cause)))
            k += 1
    body = dict(generator_version=GENERATOR_VERSION, seed=seed, n_items=len(items), items=items)
    body["items_sha256"] = items_hash(items)
    body["pilot_ids"] = pilot_ids(items, seed)
    return body


def items_hash(items: list[dict]) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pilot_ids(items: list[dict], seed: int) -> list[str]:
    """<= 5 % of items: 3 whole main scenarios (12 items) + 15 control/twin pairs (30 items) = 42 of 840."""
    rng = random.Random(seed + 1)
    sids = sorted({i["scenario_id"] for i in items if i["kind"] == "main"})
    pids = sorted({i["pair_id"] for i in items if i["kind"] != "main"})
    s, p = set(rng.sample(sids, 3)), set(rng.sample(pids, 15))
    return sorted(i["id"] for i in items if i.get("scenario_id") in s or i.get("pair_id") in p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default="results/redo_v1/e1/items.json")
    a = ap.parse_args()
    body = generate(a.seed)
    out = Path(a.out)
    if out.exists():
        old = json.loads(out.read_text())
        if old["items_sha256"] != body["items_sha256"]:
            raise SystemExit(f"refusing to overwrite frozen item set {out} (hash {old['items_sha256']})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1, sort_keys=True))
    kinds = {}
    for i in body["items"]:
        kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
    print(json.dumps(dict(out=str(out), seed=a.seed, items_sha256=body["items_sha256"], counts=kinds,
                          pilot=len(body["pilot_ids"])), indent=1))


if __name__ == "__main__":
    main()
