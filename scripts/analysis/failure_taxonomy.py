"""#11 (L-O, no model calls): heuristic failure taxonomy for the TIB loss cells (W5, W6).

Loss cells (from the published tables): HLE 2-turn constructive-hint (``commit_before`` vs ``stock`` recovery),
Finance closed-book 2-turn constructive-hint (W and S worst), and Terminal-Bench ``tb4_G`` (``tib`` and ``tibv2``).
For every *paired loss* (stock succeeds, the TIB arm fails on the same item) the TIB row gets one primary label,
assigned by fixed rules in this order:

HLE / Finance 2-turn (text available for HLE; Finance rows keep only scores + usage):
  malformed       no final ``Answer:`` line / empty reply (HLE)
  truncated       completion tokens at the call's cap (HLE 3072; Finance 4096)
  lock_in         final answer unchanged from turn 1 (HLE: normalised answer string equal; Finance: t2 score
                  <= t1 score, i.e. no uptake of the hint)
  stand_firm_cue  (secondary flag) the reply cites its claim book / bets / "stand firm"
  switched_wrong  changed the answer but still wrong (HLE) / improved less than stock (Finance: partial uptake)

Harbor tb4 (one trial per task, outcome-blind ``rule="first"``):
  timeout / infra_error / length_finish / no_claim_book (non-compliance) /  [tb4 has no model.patch artifact]
  edits_after_claim_book (the audit changed code, then failed) / wrong_fix (none of the above)

A judge pass on L-J (lock-in vs refusal-on-valid-hint vs broken edits, from the text) is the planned second step;
this script only does what can be read off the logs. Output: ``results/analysis/failure_taxonomy.json``.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

MODELS = {"gemini_3_5_flash_lite": "W", "gemini_3_5_flash": "F", "gemini_3_8_flash": "S",
          "gemini_4_0_argon": "G"}
CUE_RE = re.compile(r"claim[_ ]?book|my bets?\b|stand(?:ing)? firm|locked|pre-?commit", re.I)


def norm(s: str | None) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s$`*\\{}]", "", s)
    return s.rstrip(".")


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def hle_cells() -> dict:
    out = {}
    for stem, tag in MODELS.items():
        rs = rows(ROOT / f"results/hle/multi_turn_{stem}.jsonl")
        by = {}
        for r in rs:
            by.setdefault(r["arm"], {})[r["id"]] = r
        if "stock" not in by:
            continue
        for arm in ("commit_before", "claims_no_p", "claims_after"):
            if arm not in by:
                continue
            labels, examples = Counter(), []
            cues = 0
            n_hint = 0
            for i, s in by["stock"].items():
                a = by[arm].get(i)
                if not a or s["push_type"] != "constructive_hint":
                    continue
                n_hint += 1
                if not (s["recovered"] and not a["recovered"]):
                    continue
                txt = a.get("t2_response") or ""
                ct = int((a.get("usage") or {}).get("completion_tokens") or 0)
                if not txt.strip() or not re.search(r"Answer\s*:", txt, re.I):
                    lab = "malformed"
                elif ct >= 3000:
                    lab = "truncated"
                elif norm(a.get("t2_answer")) == norm(a.get("t1_answer")):
                    lab = "lock_in"
                else:
                    lab = "switched_wrong"
                cue = bool(CUE_RE.search(txt))
                cues += cue
                labels[lab] += 1
                if len(examples) < 5:
                    examples.append(dict(id=i, label=lab, stand_firm_cue=cue, t1=a.get("t1_answer"),
                                         t2=a.get("t2_answer"), tail=txt[-300:]))
            out[f"hle_{tag}_{arm}"] = dict(n_hint_items=n_hint, n_paired_losses=sum(labels.values()),
                                          labels=dict(labels), stand_firm_cue=cues, examples=examples)
    return out


def finance_cells() -> dict:
    out = {}
    for stem, tag in MODELS.items():
        rs = rows(ROOT / f"results/finance_agent/multi_turn_{stem}.jsonl")
        by = {}
        for r in rs:
            by.setdefault(r["arm"], {})[r["id"]] = r
        if "stock" not in by:
            continue
        for arm in ("commit_before", "claims_no_p", "claims_after"):
            if arm not in by:
                continue
            labels = Counter()
            n_hint = 0
            for i, s in by["stock"].items():
                a = by[arm].get(i)
                if not a or s["push_type"] != "constructive_hint":
                    continue
                n_hint += 1
                s_gain = float(s["t2_score"]) - float(s["t1_score"])
                a_gain = float(a["t2_score"]) - float(a["t1_score"])
                if not (s["recovered"] and not a["recovered"]) and not (s_gain > a_gain + 0.25):
                    continue
                ct = int((a.get("usage") or {}).get("completion_tokens") or 0)
                if ct >= 4000:
                    lab = "truncated"
                elif float(a["t2_score"]) < float(a["t1_score"]) - 1e-6:
                    lab = "regressed"
                elif a_gain <= 1e-6:
                    lab = "lock_in"
                else:
                    lab = "partial_uptake"
                labels[lab] += 1
            out[f"finance2turn_{tag}_{arm}"] = dict(n_hint_items=n_hint, n_paired_losses=sum(labels.values()),
                                                   labels=dict(labels),
                                                   note="no reply text stored; lock_in = no score gain")
    return out


def harbor_cells() -> dict:
    import harbor_trials as H
    H.JOB_RE = re.compile(r"^(tb4|ds11m|ds11r2|ds11)_([WFMSG])_([a-z0-9]+?)(?:_p(\d+))?$")
    H.HARNESS = dict(H.HARNESS, ds11r2="opencode")
    trials = []
    for jd in sorted(H.JOBS.iterdir()):
        if jd.is_dir() and jd.name.startswith("tb4_G_") and H.JOB_RE.match(jd.name):
            for d in sorted(jd.glob("*__*")):
                if d.is_dir():
                    r = H.trial_row(d, jd.name)
                    if r:
                        trials.append(r)
    sel = H.select_per_task(trials, rule="first")
    out = {}
    stock = sel.get("tb4_G_stock", {})
    for bj, A in sel.items():
        if bj == "tb4_G_stock":
            continue
        labels, ex = Counter(), []
        for t, s in stock.items():
            a = A.get(t)
            if not a or not (s["passed"] and not a["passed"]):
                continue
            if a["timeout"]:
                lab = "timeout"
            elif a["exc_type"]:
                lab = "infra_error"
            elif a.get("oc_length_finishes"):
                lab = "length_finish"
            elif not a.get("claim_book"):
                lab = "no_claim_book"
            elif a.get("edits_after_claim_book"):
                lab = "edits_after_claim_book"
            else:
                lab = "wrong_fix"
            labels[lab] += 1
            ex.append(dict(task=t, label=lab, exc=a["exc_type"], claim_book=a.get("claim_book"),
                           edits_after=a.get("edits_after_claim_book"), agent_sec=a.get("agent_sec")))
        out[f"harbor_{bj}"] = dict(n_tasks=len(set(stock) & set(A)), n_paired_losses=sum(labels.values()),
                                   labels=dict(labels), losses=ex)
    return out


def main() -> None:
    res = dict(hle=hle_cells(), finance_2turn=finance_cells(), harbor_tb4=harbor_cells())
    dst = ROOT / "results/analysis/failure_taxonomy.json"
    dst.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("| Cell | n items | paired losses | labels |")
    print("| :--- | ---: | ---: | :--- |")
    for grp in res.values():
        for k, v in grp.items():
            n = v.get("n_hint_items", v.get("n_tasks"))
            extra = f" (stand-firm cue in {v['stand_firm_cue']})" if "stand_firm_cue" in v else ""
            print(f"| {k} | {n} | {v['n_paired_losses']} | {v['labels']}{extra} |")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
