"""P0.6 prompt-asymmetry audit (PARALLEL_REVIEWER_EXPERIMENTS.md §3).

Renders every arm's final pushback-turn prompt for one fixed item per benchmark/runner and diffs it against
``stock``. For each arm: extra characters / approximate tokens (chars/4) versus stock, whether an adjudication rule
is present (and which: p-rule vs p-free), whether a claim book is shown, the number of extra model calls before the
pushback turn, and the sha256 of the arm-specific text. The pushback (user) text itself is checked to be identical
across arms.

Covers: ``scripts/sycophancy_benchmarks.py`` (SycoBench, SycEval) and ``scripts/hle_benchmark.py`` multi-turn
(the published arms, rendered from the source with placeholder turn-1 / claim-book text).

Output: ``docs/results/P0_6_PROMPT_ASYMMETRY.md`` (table) + ``results/analysis/prompt_asymmetry.json``.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import sycophancy_benchmarks as SB  # noqa: E402

PLACE_T1 = "<TURN-1 ANSWER>"
PLACE_BOOK = "C1: <claim> | evidence: <...> | p=0.90"


def _flat(msgs: list[dict]) -> str:
    return "\n".join(f"[{m['role']}]\n{m['content']}" for m in msgs)


def _rule_kind(text: str) -> str:
    if SB.RULE_P in text or "p >= 0.85" in text or "`p >= 0.85`" in text:
        return "p-rule"
    if SB.RULE_NOP in text:
        return "p-free rule"
    if "STAND FIRM" in text or "stand firm" in text:
        return "rule (other)"
    return "none"


def audit_syco(bench: str) -> list[dict]:
    if bench == "sycobench":
        item = SB.load_sycobench()[0]
        qp = SB.sb_question_prompt(item)
        push = SB.sb_followup("wrong_suggest", 0, item["correct"])
    else:
        qp = "Factor the following quadratic: $5 x^2-5 x-1050$"
        item = dict(dataset="amps", regressive=dict(valid=True, components=dict(
            proposed_answer="<WRONG ANSWER>", justification="<JUSTIFICATION>", citation="<CITATION>",
            abstract="<ABSTRACT>")))
        push = SB.se_rebuttal(item, "regressive", "citation")
    stock = _flat(SB.turn2_messages("stock", qp, PLACE_T1, push, None))
    out = []
    for arm in SB.ARMS:
        spec = SB.ARM_SPEC[arm]
        msgs = SB.turn2_messages(arm, qp, PLACE_T1, push, PLACE_BOOK if spec["book"] else None)
        flat = _flat(msgs)
        extra = SB.arm_system_text(arm, PLACE_BOOK if spec["book"] else None) or ""
        user_push = [m for m in msgs if m["role"] == "user"][-1]["content"]
        pre_calls = (1 if spec["gate"] else 0) + (1 if spec["book"] else 0)
        out.append(dict(
            runner="sycophancy_benchmarks.py", bench=bench, arm=arm,
            extra_chars=len(flat) - len(stock), extra_tokens_approx=round((len(flat) - len(stock)) / 4),
            arm_text_chars=len(extra), rule=_rule_kind(extra), claim_book=bool(spec["book"]),
            claim_book_kind=spec["book"], precommit_calls=pre_calls if not spec["gate"] else "1 gate + 1 book if p<0.9",
            pushback_identical=(user_push == push), arm_text_sha256=hashlib.sha256(extra.encode()).hexdigest()[:12],
        ))
    return out


def audit_hle() -> list[dict]:
    """All HLE multi-turn arms (published + P0.2 ladder), rendered exactly by stubbing ``call_chat`` (no model calls)."""
    import hle_benchmark as H
    q = dict(id="x", category="c", question="<QUESTION>", answer="X", answer_type="exactMatch",
             rationale="<RATIONALE>")
    t1 = dict(response=PLACE_T1, confidence=80, correct_bool=True, judge_response=dict(model_answer="X"))
    captured: dict[str, list] = {}

    def fake_call_chat(base_url, token, model, messages, max_tokens=4096, reasoning_effort="low", **kw):
        captured.setdefault("calls", []).append(messages)
        txt = messages[-1]["content"]
        if "Output ONLY the `[CLAIM_BOOK]`" in txt:
            return PLACE_BOOK, {}
        if "[correct_answer]" in txt:  # judge
            return '{"extracted_final_answer": "X", "reasoning": "", "correct": "yes", "confidence": 80}', {}
        return "Explanation: e\nAnswer: X\nConfidence: 80%", {}

    orig = H.call_chat
    H.call_chat = fake_call_chat
    try:
        rendered = {}
        for arm in H.MT_ARMS_ALL:
            captured.clear()
            H.run_multi_turn_item(q, t1, arm, "m", "j", "u", "t")
            calls = captured["calls"]
            t2 = [c for c in calls if c[0]["content"] == H.OFFICIAL_HLE_SYSTEM_PROMPT and "[correct_answer]"
                  not in c[-1]["content"]][-1]
            rendered[arm] = (t2[-1]["content"], len(calls) - 2)  # minus turn-2 and judge
    finally:
        H.call_chat = orig
    stock = rendered["stock"][0]
    _, reviewer = H.build_pushback_message(q, t1)
    out = []
    for arm, (txt, pre) in rendered.items():
        extra = len(txt) - len(stock)
        out.append(dict(runner="hle_benchmark.py multi_turn", bench="hle_2turn", arm=arm,
                        extra_chars=extra, extra_tokens_approx=round(extra / 4),
                        rule="p-rule" if "p >= 0.85" in txt else ("p-free rule" if "STAND FIRM" in txt else "none"),
                        claim_book=arm in H.MT_ARM_BOOK,
                        claim_book_kind=H.MT_ARM_BOOK.get(arm),
                        precommit_calls=pre, pushback_identical=reviewer in txt,
                        arm_text_sha256=hashlib.sha256(txt.encode()).hexdigest()[:12]))
    return out


def _kind(text: str, rules: dict[str, str]) -> str:
    for name, r in rules.items():
        if r and r in text:
            return name
    return "none"


def audit_e2() -> list[dict]:
    """E2 Wave-2 conditions (fsa/redo/e2/ladder.py) on one fixed W SWE-Together item, vs ``unjustified`` (stock)."""
    from fsa.redo import arm_texts as AT
    from fsa.redo.e2 import ladder as L
    from fsa.redo.e2 import prompts as P
    item = json.loads((ROOT / "results/redo_v1/e2/items_swt_W.json").read_text())["items"][0]
    rids = [i["rid"] for i in item["intents"]]
    book_parsed = {r: dict(satisfied=True, p_satisfied=90, reason="<reason>") for r in rids}
    stock = _flat(P.build("unjustified", item))
    rules = {"p-rule": L.RULE_P, "p-free rule": L.RULE_NOP, "bet rule (published)": L._rule_paragraph(),
             "placebo": L.PLACEBO, **{s: L.texts()[s] for s in AT.STUB_ARMS}, "eval-vs-book": L.EVAL_BOOK}
    out = []
    conds = [f"{a}:unjustified" for a in L.LADDER_SPEC] + [f"{a}:unjustified" for a in L.H2_ARMS] + \
        [f"adv_{k}:{a}" for a in L.ADV_ARMS for k in L.ADV_KINDS if not k.startswith("round") or k == "round1"]
    for c in ["unjustified"] + conds:
        if c == "unjustified":
            msgs, arm = P.build(c, item), "stock"
        else:
            arm = L._arm_of(c)
            spec = L.LADDER_SPEC.get(arm, {})
            book = dict(kind=spec.get("book"), parsed=book_parsed, gated_off=False) if L.book_arm(c) else None
            try:
                msgs = L.build(c, item, book, book_parsed)
            except ValueError:
                continue
        flat = _flat(msgs)
        reply = L.reply_text(c, item) if c != "unjustified" else item["unjustified_reply"]
        pre = 0
        if L.book_arm(c):
            pre = "1 gate + 1 book if p<90" if L.LADDER_SPEC[L.book_arm(c)].get("gate") else 1
        elif arm in L.H2_ARMS:
            pre = "shared stage-1 bet"
        out.append(dict(runner="fsa/redo/e2", bench="e2 (" + ("pushback" if not c.startswith("adv_") else "#12") + ")",
                        arm=c, extra_chars=len(flat) - len(stock), extra_tokens_approx=round((len(flat) - len(stock)) / 4),
                        rule=_kind(flat, rules), claim_book_kind=(L.LADDER_SPEC.get(L.book_arm(c) or "", {}) or {}).get("book"),
                        precommit_calls=pre, pushback_identical=(reply == item["unjustified_reply"]) if not c.startswith("adv_") else
                        "n/a (#12 varies the pushback)", arm_text_sha256=hashlib.sha256(flat.encode()).hexdigest()[:12]))
    return out


def audit_ale() -> list[dict]:
    import fsa.redo.ale_pilot as A
    sc = A.build_scenarios()[0]
    stock = _flat(A.t2_messages(sc, "stock"))
    rules = {"p-rule": A.ALE_RULE_P, "p-free rule": A.ALE_RULE_NOP, "placebo": A.ALE_PLACEBO}
    out = []
    for arm in A.ALE_ARM_SPEC:
        spec = A.ALE_ARM_SPEC[arm]
        flat = _flat(A.t2_messages(sc, arm, PLACE_BOOK if spec["book"] else ""))
        out.append(dict(runner="fsa/redo/ale_pilot.py", bench="ale_e2", arm=arm, extra_chars=len(flat) - len(stock),
                        extra_tokens_approx=round((len(flat) - len(stock)) / 4),
                        rule=_kind(flat, rules) if arm not in A.ALE_ARMS_STUB else arm,
                        claim_book_kind=spec["book"],
                        precommit_calls=("1 gate + 1 book if p<0.9" if spec.get("gate") else int(bool(spec["book"]))),
                        pushback_identical=sc.turn2_user_message in flat,
                        arm_text_sha256=hashlib.sha256(flat.encode()).hexdigest()[:12]))
    return out


def audit_hle_validity() -> list[dict]:
    import hle_benchmark as H
    q = dict(id="x", question="<QUESTION>", answer="X", answer_type="exactMatch", rationale="<RATIONALE>")
    t1 = dict(response=PLACE_T1, judge_response=dict(model_answer="<T1 ANSWER>"))
    base = f"Question:\n{q['question']}\n\nYour Turn-1 Response:\n{t1['response']}"
    out = []
    for push in ("bare", "hint"):
        _, msg = H.validity_message(push, q, t1)
        stock = H.mt_t2_user("stock", base, msg, "")
        for arm in H.MT_ARMS_ALL:
            txt = H.mt_t2_user(arm, base, msg, PLACE_BOOK if arm in H.MT_ARM_BOOK else "")
            out.append(dict(runner="hle_benchmark.py validity", bench=f"hle_pool ({push})", arm=arm,
                            extra_chars=len(txt) - len(stock), extra_tokens_approx=round((len(txt) - len(stock)) / 4),
                            rule="p-rule" if "p >= 0.85" in txt else ("p-free rule" if "STAND FIRM" in txt else
                                                                      ("stub" if arm in H.STUB_ARMS else "none")),
                            claim_book_kind=H.MT_ARM_BOOK.get(arm), precommit_calls=int(arm in H.MT_ARM_BOOK),
                            pushback_identical="yes (same template for valid and invalid turn 1)" if msg in txt else False,
                            arm_text_sha256=hashlib.sha256(txt.encode()).hexdigest()[:12]))
    return out


def audit_draftfirst() -> list[dict]:
    """Draft-first arms have no pushback turn: report the instruction text each adds after the shared draft."""
    import hle_benchmark as H
    sys.path.insert(0, str(ROOT))
    import scripts.finance_agent_tools_benchmark as FT
    out = []
    hle = {"stock": ("", 0), "cb_draftfirst": (H.TIB_DRAFTFIRST_SYSTEM_PROMPT, 0),
           "review_only": (H.DF_REVIEW, 1), "self_refine": (H.DF_FEEDBACK + H.DF_REFINE, 2),
           "cove": (H.COVE_PLAN + H.COVE_EXEC_SYSTEM + H.COVE_FINAL, 3),
           "sc_matched": ("", "k-1 samples (k: " + ", ".join(f"{m.split('-')[1]}{'-lite' if 'lite' in m else ''}={k}"
                                                              for m, k in H.SC_K.items()) + ")")}
    for arm, (txt, calls) in hle.items():
        out.append(dict(runner="hle_benchmark.py single_shot", bench="hle_draftfirst", arm=arm, extra_chars=len(txt),
                        extra_tokens_approx=round(len(txt) / 4), rule="p (joint-bet)" if "p=" in txt else "none",
                        claim_book_kind="ev_p (in-call)" if arm == "cb_draftfirst" else None,
                        precommit_calls=calls, pushback_identical="n/a (no pushback)",
                        arm_text_sha256=hashlib.sha256(txt.encode()).hexdigest()[:12]))
    for arm in FT.FT_ARMS_ALL:
        txt = FT.DF_GATES.get(arm, "") + (FT.SELF_REFINE_STEP2 if arm == "self_refine" else "")
        out.append(dict(runner="finance_agent_tools_benchmark.py", bench="finance_4tool_draftfirst", arm=arm,
                        extra_chars=len(txt), extra_tokens_approx=round(len(txt) / 4),
                        rule="p-rule" if "p < 0.85" in txt else "none",
                        claim_book_kind="ev_p (gate)" if arm == "cb_draftfirst" else None,
                        precommit_calls=("k-1 = 1 extra agent run + 1 selection call" if arm == "sc_matched" else
                                         ("gate turn" + ("s (2)" if arm == "self_refine" else "") if txt else 0)),
                        pushback_identical="n/a (no pushback)",
                        arm_text_sha256=hashlib.sha256(txt.encode()).hexdigest()[:12]))
    return out


def audit_hle_adv() -> list[dict]:
    import hle_benchmark as H
    q = dict(id="x", question="<QUESTION>", answer="X", answer_type="exactMatch", rationale="<RATIONALE>")
    t1 = dict(response=PLACE_T1, judge_response=dict(model_answer="<T1 ANSWER>"))
    _, bare = H.build_pushback_message(q, dict(t1, correct_bool=True))
    out = []
    for kind in H.ADV_KINDS:
        if kind in ("round2", "round3"):
            continue
        _, msg = H.adv_message(kind, q, t1)
        out.append(dict(runner="hle_benchmark.py adversarial", bench="hle_#12", arm=f"adv_{kind}",
                        extra_chars=len(msg) - len(bare), extra_tokens_approx=round((len(msg) - len(bare)) / 4),
                        rule="per arm (stock / rule_only / commit_before)", claim_book_kind=None,
                        precommit_calls="commit_before: 1 book per item, shared", pushback_identical=
                        "identical across arms (varies by condition)", arm_text_sha256=hashlib.sha256(msg.encode()).hexdigest()[:12]))
    return out


def wave2_hashes() -> dict[str, str]:
    import hle_benchmark as H
    import fsa.redo.ale_pilot as A
    from fsa.redo.e2 import ladder as L
    from fsa.redo import arm_texts as AT
    import finance_agent_benchmark as F
    sys.path.insert(0, str(ROOT))
    import scripts.finance_agent_tools_benchmark as FT
    import sycon_bench as SY
    return {"e2 ladder_sha256": L.ladder_sha256(), "ale_prompts_sha256": A.ale_prompts_sha256(),
            "hle_prompts_sha256": H.hle_prompts_sha256(), "hle_wave2_sha256": H.hle_wave2_sha256(),
            "fin_prompts_sha256 (2-turn)": F.fin_prompts_sha256(), "ft_prompts_sha256 (4-tool)": FT.ft_prompts_sha256(),
            "arm_texts_sha256": AT.texts_sha256(), "sycon_prompts_sha256 (#4)": SY.prompts_sha256()}


def main() -> None:
    rows = audit_syco("sycobench") + audit_syco("syceval") + audit_hle() + audit_e2() + audit_ale() + \
        audit_hle_validity() + audit_draftfirst() + audit_hle_adv()
    known = {
        "placebo": "length-matched neutral text; controls for 'any system text'",
        "rule_only": "the p-free rule = the variable under test (step 2->3)",
        "claims_no_p": "continuity arm; has an 'evaluate against it' instruction but no rule",
        "claims_no_p_rule": "adds claim book (claims + evidence) to rule_only; same rule text (step 3->4)",
        "claims_p_noevid": "changes the book to p-without-evidence AND the rule to the p-rule (step 4->5: "
                           "'probabilities' = p in book + p in rule, by design)",
        "commit_before": "adds evidence to claims_p_noevid; identical rule text (step 5->6)",
        "cb_gated": "= commit_before when the gate p < 0.9, = stock otherwise; +1 gate call",
        "claims_after": "HLE only: claim book written after the pushback, in the same call; no rule",
        "stub_0..stub_3": "#3 stubbornness ladder: one 'Deference policy' sentence (fsa/redo/arm_texts.py); lengths "
                          "differ across the four rungs (that is the treatment); no claim book",
        "E2 bet_hidden_*": "#2: says an assessment was recorded but does not show it (one fixed sentence); the bet "
                           "call really happened",
        "E2 bet_shown_*/bet_hidden_*_rule": "#2: the rule is the published bet_stage2.md revision paragraph verbatim; "
                                            "schema = rate.md (no revise field, no copy instruction, no pinning)",
        "ALE placebo/rule_only/claims_no_p_rule": "end with the published commit_before final line ('Provide brief "
                                                  "reasoning ...'); stock keeps its published line ('Please respond to "
                                                  "the reviewer ...'), so stock -> placebo also changes that line",
        "HLE validity hint": "second factor: the published hint template (rationale snippet) shown for right AND wrong turn 1",
        "draft-first": "no pushback turn; arms share the stock draft (HLE) or intercept the first submission (Finance)",
        "#12": "pushback text varies by condition (that is the treatment) and is identical across the 3 arms",
    }
    md = [
        "# P0.6 prompt-asymmetry audit",
        "",
        "Generated by `scripts/analysis/prompt_asymmetry_audit.py` (no model calls). One fixed item per runner;",
        "turn-1 answers and claim books are placeholders. `extra` = characters added to the pushback-turn prompt",
        "relative to `stock` (tokens ≈ chars/4). The user's pushback text is identical across arms in every runner.",
        "",
        f"`prompts_sha256` (sycophancy runner): `{SB.prompts_sha256()}`",
        "",
        "Wave-2 hashes (A3): " + ", ".join(f"{k} `{v}`" for k, v in wave2_hashes().items()),
        "",
        "| Runner | Bench | Arm | Extra chars | ≈ tokens | Rule | Claim book | Pre-commit calls | Pushback identical |",
        "| :--- | :--- | :--- | ---: | ---: | :--- | :--- | :--- | :---: |",
    ]
    for r in rows:
        md.append(f"| {r['runner']} | {r['bench']} | `{r['arm']}` | {r['extra_chars']} | {r['extra_tokens_approx']} | "
                  f"{r['rule']} | {r['claim_book_kind'] or '—'} | {r['precommit_calls']} | "
                  f"{'yes' if r['pushback_identical'] else '**NO**'} |")
    md += ["", "## Arm-specific text that is not the variable under test (known confounds)", ""]
    for k, v in known.items():
        md.append(f"- `{k}`: {v}")
    md += ["",
           "- **HLE published arms (confound, as in §0 of the plan):** `commit_before` gets the p-phrased adjudication",
           "  rule; `claims_no_p`, `claims_after` and `stock` get no rule. The published commit_before − claims_no_p",
           "  gap therefore mixes probabilities with the rule. The new ladder (sycophancy runner; HLE/E2/ALE/Finance",
           "  #1 cells) holds the rule fixed within each step. HLE ladder arms: `rule_only` and `claims_no_p_rule`",
           "  share HLE_RULE_NOP; `claims_p_noevid` and `commit_before` share the published p-rule byte-for-byte.",
           "- **Rule lengths:** p-rule " + str(len(SB.RULE_P)) + " chars, p-free rule " + str(len(SB.RULE_NOP))
           + " chars, placebo " + str(len(SB.PLACEBO)) + " chars.",
           ""]
    dst = ROOT / "docs/results/P0_6_PROMPT_ASYMMETRY.md"
    dst.write_text("\n".join(md))
    j = ROOT / "results/analysis/prompt_asymmetry.json"
    j.parent.mkdir(parents=True, exist_ok=True)
    j.write_text(json.dumps(dict(prompts_sha256=SB.prompts_sha256(), wave2=wave2_hashes(), rows=rows), indent=1,
                            default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
