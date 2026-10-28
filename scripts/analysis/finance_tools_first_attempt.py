"""#13b (L-O): Finance Agent 4-tool first-attempt re-analysis from git history.

``finance_agent_tools_benchmark.py --repair-paired`` re-ran ``cb_draftfirst`` (and synthesized empty ``stock`` answers)
only on tasks selected by outcome (cb lost to / tied below / looked worse than stock), then rewrote the JSONL with
one row per (id, arm). This script reconstructs, per model:

* the first-attempt rows (the commit that first added the complete run: a4896954 for S/F/W, fc857758 for G),
* the published rows (011a1413 for S/F/W, 2defeada for G; HEAD is checked to be identical),
* every (id, arm) whose row changed between the two, classified as: in the surviving repair log, an empty-stock
  synthesis, or "changed, not in the surviving log" (the log file is overwritten on each pass, so earlier passes
  leave no record),
* paired W/T/L and mean rubric score (and pass rate) for: first-attempt, published, and "published with every
  changed cb_draftfirst row reverted to its first-attempt row" (removes the outcome-dependent repair but keeps any
  stock-side change).

Output: ``results/analysis/finance_tools_first_attempt.json`` and a markdown table on stdout.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-6
MODELS = {
    # model tag: (file stem, first-attempt commit, published commit, repair log)
    "S": ("tools_gemini_3_8_flash", "a4896954", "011a1413", "repair_paired.log"),
    "F": ("tools_gemini_3_5_flash", "a4896954", "011a1413", "repair_paired.log"),
    "W": ("tools_gemini_3_5_flash_lite", "a4896954", "011a1413", "repair_paired.log"),
    "G": ("tools_gemini_4_0_argon", "fc857758", "2defeada", "run_g4pro.log"),
}
FULL = {"S": "gemini-3.8-flash", "F": "gemini-3.5-flash", "W": "gemini-3.5-flash-lite",
        "G": "gemini-4.0-argon"}


def git_show(commit: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def load_rows(text: str | None, first_wins: bool) -> tuple[dict, int]:
    rows: dict[tuple[str, str], dict] = {}
    n = 0
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        n += 1
        r = json.loads(line)
        k = (r["id"], r["arm"])
        if first_wins and k in rows:
            continue
        rows[k] = r
    return rows, n


def score(r: dict | None) -> float | None:
    if r is None:
        return None
    try:
        return float(r.get("score"))
    except (TypeError, ValueError):
        return None


def paired(rows: dict) -> dict:
    ids = sorted({i for i, a in rows if a == "stock"} & {i for i, a in rows if a == "cb_draftfirst"})
    w = t = l = 0
    ss, cs, sp, cp = [], [], [], []
    for i in ids:
        s, c = score(rows[(i, "stock")]), score(rows[(i, "cb_draftfirst")])
        s = 0.0 if s is None else s
        c = 0.0 if c is None else c
        ss.append(s)
        cs.append(c)
        sp.append(bool(rows[(i, "stock")].get("passed")))
        cp.append(bool(rows[(i, "cb_draftfirst")].get("passed")))
        if c > s + EPS:
            w += 1
        elif c < s - EPS:
            l += 1
        else:
            t += 1
    n = len(ids)
    return dict(n=n, W=w, T=t, L=l,
                stock_mean=round(100 * sum(ss) / n, 2) if n else None,
                cb_mean=round(100 * sum(cs) / n, 2) if n else None,
                delta=round(100 * (sum(cs) - sum(ss)) / n, 2) if n else None,
                stock_pass=round(100 * sum(sp) / n, 1) if n else None,
                cb_pass=round(100 * sum(cp) / n, 1) if n else None)


def repair_log(text: str, model: str) -> dict:
    out = {"audited": {}, "synthesized": {}}
    for line in text.splitlines():
        m = re.search(r"\[cb_draftfirst paired\] (\S+) (fin_\d+): stock=([\d.]+) -> cb=([\d.]+)", line)
        if m and m.group(1) == model:
            out["audited"][m.group(2)] = dict(stock=float(m.group(3)), cb=float(m.group(4)))
        m = re.search(r"\[stock synthesized\] (\S+) (fin_\d+) score=([\d.]+)", line)
        if m and m.group(1) == model:
            out["synthesized"][m.group(2)] = float(m.group(3))
    return out


def flag_reason(s: dict | None, c: dict | None) -> list[str]:
    """The --repair-paired selection rule (011a1413, lines ~1105-1123), re-applied to first-attempt rows."""
    why = []
    if not s or not (s.get("response") or "").strip():
        return ["stock_empty"]
    if not c:
        return ["cb_missing"]
    c_resp, s_resp = (c.get("response") or "").strip(), (s.get("response") or "").strip()
    sc, cc = score(s) or 0.0, score(c) or 0.0
    if c_resp and (cc >= 0.999 or cc > sc + EPS):
        return []  # never re-run: cb already won or was perfect
    if not c_resp:
        why.append("cb_empty")
    if "CLAIM_BOOK" in c_resp:
        why.append("claim_book_leftover")
    if len(c_resp) < int(0.75 * len(s_resp)):
        why.append("cb_short")
    gc, gs = c.get("grade") or {}, s.get("grade") or {}
    if isinstance(gc, dict) and isinstance(gs, dict) and gc.get("has_contradiction") and not gs.get("has_contradiction"):
        why.append("cb_contradiction")
    if cc < sc - EPS:
        why.append("cb_lost")
    return why


def main() -> None:
    out: dict = {"models": {}, "notes": []}
    md = ["| Model | Version | n | W / T / L | stock mean | cb mean | Δ | stock pass | cb pass |",
          "| :--- | :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |"]
    tot = {k: dict(W=0, T=0, L=0) for k in ("first", "published", "reverted")}
    for tag, (stem, c0, c1, logname) in MODELS.items():
        path = f"results/finance_agent_tools/{stem}.jsonl"
        first, n0 = load_rows(git_show(c0, path), first_wins=True)
        pub, n1 = load_rows(git_show(c1, path), first_wins=False)
        head, _ = load_rows(git_show("HEAD", path), first_wins=False)
        head_same = {k: score(v) for k, v in head.items()} == {k: score(v) for k, v in pub.items()}
        logtxt = (ROOT / "results/finance_agent_tools" / logname).read_text()
        log = repair_log(logtxt, FULL[tag])
        changed = []
        for k in sorted(set(first) | set(pub)):
            a, b = first.get(k), pub.get(k)
            if a is None or b is None or (a.get("response") != b.get("response")) or score(a) != score(b):
                tid, arm = k
                cls = ("repair_log_audit" if arm == "cb_draftfirst" and tid in log["audited"] else
                       "repair_log_stock_synth" if arm == "stock" and tid in log["synthesized"] else
                       "changed_not_in_log")
                changed.append(dict(id=tid, arm=arm, cls=cls, first_score=score(a), published_score=score(b),
                                    first_len=len((a or {}).get("response") or ""),
                                    published_len=len((b or {}).get("response") or ""),
                                    flag_on_first=flag_reason(first.get((tid, "stock")),
                                                              first.get((tid, "cb_draftfirst")))))
        reverted = dict(pub)
        for ch in changed:
            if ch["arm"] == "cb_draftfirst" and (ch["id"], "cb_draftfirst") in first:
                reverted[(ch["id"], "cb_draftfirst")] = first[(ch["id"], "cb_draftfirst")]
        # tasks the selection rule would flag on the first-attempt rows
        ids = sorted({i for i, _ in first})
        flagged = {i: flag_reason(first.get((i, "stock")), first.get((i, "cb_draftfirst"))) for i in ids}
        flagged = {i: v for i, v in flagged.items() if v}
        p_first, p_pub, p_rev = paired(first), paired(pub), paired(reverted)
        for k, p in (("first", p_first), ("published", p_pub), ("reverted", p_rev)):
            for x in ("W", "T", "L"):
                tot[k][x] += p[x]
        cb_changed = [c for c in changed if c["arm"] == "cb_draftfirst"]
        out["models"][tag] = dict(
            model=FULL[tag], file=path, first_commit=c0, published_commit=c1, head_identical_to_published=head_same,
            lines_first=n0, lines_published=n1, first_attempt=p_first, published=p_pub,
            published_cb_reverted=p_rev, n_changed_rows=len(changed),
            n_changed_by_class={c: sum(1 for x in changed if x["cls"] == c) for c in
                                ("repair_log_audit", "repair_log_stock_synth", "changed_not_in_log")},
            n_cb_changed=len(cb_changed),
            cb_changed_that_were_flagged=sum(1 for c in cb_changed if c["flag_on_first"]),
            cb_changed_that_were_losses_on_first=sum(
                1 for c in cb_changed
                if (c["first_score"] or 0) < (score(first.get((c["id"], "stock"))) or 0) - EPS),
            flagged_on_first=flagged, repair_log=log, changed=changed)
        for lab, p in (("first attempt (" + c0 + ")", p_first), ("published (" + c1 + ")", p_pub),
                       ("published, changed cb rows reverted", p_rev)):
            md.append(f"| {tag} | {lab} | {p['n']} | {p['W']} / {p['T']} / {p['L']} | {p['stock_mean']} | "
                      f"{p['cb_mean']} | {p['delta']:+.2f} | {p['stock_pass']} | {p['cb_pass']} |")
    out["totals_SFW_G"] = tot
    out["notes"] = [
        "first attempt = first row per (id, arm) in the commit that first contains the full run",
        "the repair log is overwritten on each --repair-paired pass; rows changed but absent from it may come "
        "from earlier passes or from non-repair reruns, and cannot be told apart from git alone",
        "published_cb_reverted keeps any stock-side change (e.g. synthesized empty stock answers)",
    ]
    dst = ROOT / "results/analysis/finance_tools_first_attempt.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\n".join(md))
    print()
    for k, v in tot.items():
        print(f"all 4 models, {k}: W/T/L = {v['W']}/{v['T']}/{v['L']}")
    for tag, m in out["models"].items():
        print(f"{tag}: changed rows {m['n_changed_rows']} {m['n_changed_by_class']}; cb changed {m['n_cb_changed']}, "
              f"of which flagged-on-first {m['cb_changed_that_were_flagged']}, losses-on-first "
              f"{m['cb_changed_that_were_losses_on_first']}; head==published {m['head_identical_to_published']}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
