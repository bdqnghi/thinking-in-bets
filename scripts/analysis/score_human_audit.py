"""Score the human audit of SWE-Interact pushback turns against the automated judges.

Reads the rater's labels from ``results/analysis/human_audit_sheet.csv`` (column ``human_label``; ids as in
``docs/results/HUMAN_AUDIT_SHEET.md``) and the hidden key ``results/analysis/human_audit_key.json``
(``scripts/analysis/build_human_audit.py``), and reports human-vs-judge agreement (raw %, Cohen's kappa) for
the primary judge (``gemini-3.5-flash``) and the second judge (``gemini-3.8-flash``): all labels, the
user-wrong subset (held vs caved), the user-right subset, a collapsed "agent updated vs held" view, and per
arm, plus the share of judge-disagreement turns each judge got right.

Label mapping: the human label ``already_met`` corresponds to the judges' ``justified_invalid``; ``unclear``
(or a blank) is excluded and counted.

    python3 scripts/analysis/score_human_audit.py [--csv ...] [--key ...] [--out results/analysis/human_audit_scores.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "analysis" / "human_audit_sheet.csv"
KEY = ROOT / "results" / "analysis" / "human_audit_key.json"
OUT = ROOT / "results" / "analysis" / "human_audit_scores.json"
VALID = ("held", "caved", "correct_update", "stubborn", "already_met")
JUDGE_TO_HUMAN = {"justified_invalid": "already_met"}
UPDATED = {"caved": "updated", "correct_update": "updated", "held": "held", "stubborn": "held", "already_met": "held"}


def kappa(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    if not n:
        return {"n": 0, "agree": 0, "raw": None, "kappa": None}
    agree = sum(a == b for a, b in pairs)
    po = agree / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return {"n": n, "agree": agree, "raw": round(po, 4), "kappa": None if pe == 1 else round((po - pe) / (1 - pe), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--key", default=str(KEY))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    key = json.loads(Path(a.key).read_text())["items"]
    human, bad, blank = {}, [], 0
    with open(a.csv, newline="") as f:
        for r in csv.DictReader(f):
            lab = (r.get("human_label") or "").strip().lower()
            if not lab or lab == "unclear":
                blank += 1
                continue
            if lab not in VALID:
                bad.append((r["id"], lab))
                continue
            if r["id"] not in key:
                bad.append((r["id"], "unknown id"))
                continue
            human[r["id"]] = lab
    if bad:
        print(f"unrecognised labels / ids (ignored): {bad}", file=sys.stderr)
    if not human:
        print(f"no usable human labels yet ({blank} blank / unclear rows); fill `human_label` in {a.csv}")
        return 1
    out = {"n_labelled": len(human), "n_blank_or_unclear": blank, "n_invalid": len(bad), "judges": {}}
    for judge in ("primary", "second"):
        def jl(i):
            v = key[i].get(f"label_{judge}")
            return JUDGE_TO_HUMAN.get(v, v)
        ids = [i for i in human if jl(i) not in (None, "error")]
        pr = lambda xs: [(human[i], jl(i)) for i in xs]  # noqa: E731
        wrong = [i for i in ids if key[i]["kind"] != "justified"]
        right = [i for i in ids if key[i]["kind"] == "justified"]
        res = {"all": kappa(pr(ids)),
               "user_wrong_turns": kappa(pr(wrong)),
               "user_right_turns": kappa(pr(right)),
               "updated_vs_held": kappa([(UPDATED[h], UPDATED[j]) for h, j in pr(ids)]),
               "llm_dependent_turns": kappa(pr([i for i in ids if key[i].get("llm_dependent")])),
               "per_arm": {arm: kappa(pr([i for i in ids if key[i]["arm"] == arm]))
                           for arm in ("stock", "claims_after", "commit_before", "cb_gated")},
               "confusion_human_to_judge": dict(Counter(f"{h}->{j}" for h, j in pr(ids)))}
        out["judges"][judge] = res
    dis = [i for i in human if key[i].get("judges_disagree")]
    out["judge_disagreement_turns"] = {
        "n": len(dis),
        "human_sides_with_primary": sum(human[i] == JUDGE_TO_HUMAN.get(key[i]["label_primary"], key[i]["label_primary"])
                                        for i in dis),
        "human_sides_with_second": sum(human[i] == JUDGE_TO_HUMAN.get(key[i]["label_second"], key[i]["label_second"])
                                       for i in dis)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for judge, r in out["judges"].items():
        print(f"{judge:8s} all: {r['all']['agree']}/{r['all']['n']} = {r['all']['raw']}, kappa {r['all']['kappa']}; "
              f"updated-vs-held kappa {r['updated_vs_held']['kappa']}")
    d = out["judge_disagreement_turns"]
    print(f"judge-disagreement turns: {d['n']}; human sides with primary {d['human_sides_with_primary']}, "
          f"with second {d['human_sides_with_second']}")
    print(f"wrote {Path(a.out).relative_to(ROOT) if Path(a.out).is_relative_to(ROOT) else a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
