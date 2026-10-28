"""E2 item builder: self-judgement items from SWE-Together and SWE-Interact trial directories.

An **item** is one agent turn t of one trial whose end is followed by a message of the
benchmark's user. It carries what the rating prompt shows (never the judge's labels):

* ``diff``          - the cumulative repository diff after turn t (the per-turn snapshot, normalised
                      as the per-intent judge normalises it);
* ``agent_message`` - the agent's own last message of turn t (SWE-Together: last text event of
                      ``opencode.txt.turn-t``; SWE-Interact: the ``ask_user`` question);
* ``intents``       - the intents revealed by t (the frozen per-intent judge's ``revealed`` set of
                      snapshot t) whose judge verdict is satisfied / not_satisfied / not_addressed
                      (``not_applicable`` intents are dropped), shown as R1..Rn in task order;
* ``real_reply``    - the simulator's reply to turn t. On overlay trials where the overlay replaced
                      that reply, the simulator's own (replaced) reply is used: it is what the
                      benchmark's user said about the same snapshot;
* ``unjustified_reply`` - one of the pushback overlay's unjustified templates, chosen by a
                      seeded RNG per item (the same templates the C1 overlay injects).

and, for analysis only, the ground truth of each intent (the frozen per-intent judge's verdict at
snapshot t, which never saw the dialogue), whether the reference solution satisfies it
(``durable``), and the revelation tagger's reading of the real reply (its act and, per intent,
explicit / implied / no). A *control* intent is one the judge marks unsatisfied at t while the
real reply is a revision request that names it (tag explicit or implied): the reply points at a
real defect, and a good judge should update on it. ``control_loose`` (secondary) drops the
targeting condition: unsatisfied at t and the reply is a revision request (tagger act, or
SWE-Together's own simulator act ``redirect``).

Turn and reply alignment:

* SWE-Together (``agent/turns.jsonl``): the reply to snapshot t is the message that opened turn
  t+1 when ``opened_by`` is ``sim`` (``message``) or ``overlay`` (``redo.sim_message_replaced``).
  ``noop_continue``, ``session_end`` and the end-of-run capture (turn-999) are not user replies.
* SWE-Interact (``intent_judge/<job>.json`` turn index, ``user-sim/asks.jsonl``): a turn that
  ended with an ``ask_user`` call; the reply is that ask's ``reply`` (the simulator's decision,
  also on overlay trials); turns that end at a step boundary have no user reply.

Exclusions (counted in ``exclusions``, never silent): no intent-judge file; no user reply after
the turn; a last-resort (fallback) simulator reply; empty diff; diff longer than
``MAX_DIFF_CHARS`` (so the model and the judge see the same unclipped diff); no rateable intent;
the same (trial, diff, revealed set) as an earlier item of that trial (stalled turns).

Pilot data guard: trial directories under ``results/redo_v1/stage1/`` and
``results/redo_v1/stage2_swe_interact/`` are refused unless ``--allow-pilot`` is given. That flag
is for the main run only, after the per-intent judge is frozen and Gate 1 has been decided
(docs/results/E2_PREREGISTRATION.md).

CLI:

    python -u -m fsa.redo.e2.items build --out results/redo_v1/e2_smoke/items.json \\
        --rungs W --max-items 40 --per-trial-cap 3 <run_dir|root> [...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from .. import metrics as SWT
from ..intent_judge import normalize_patch

ROOT = Path(__file__).resolve().parents[3]
BUILDER_VERSION = "e2-items-1"
MAX_DIFF_CHARS = 60_000
AGENT_MESSAGE_CHARS = 4_000
REPLY_CHARS = 6_000
SEED = 20260926
RATEABLE = ("satisfied", "not_satisfied", "not_addressed")
RUNG_MODEL = {"W": "gemini-3.5-flash-lite", "M": "gemini-3.6-flash", "S": "gemini-3.8-flash"}
PILOT_DIRS = ("results/redo_v1/stage1/", "results/redo_v1/stage2_swe_interact/")
TEMPLATES = ROOT / "fsa" / "redo" / "swe_interact" / "prompts" / "redo" / "overlay_templates.json"


def unjustified_templates() -> list[str]:
    return list(json.loads(TEMPLATES.read_text())["unjustified"])


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def items_hash(items: list[dict]) -> str:
    return _sha(json.dumps(items, sort_keys=True, ensure_ascii=False))


def is_pilot_path(p: Path) -> bool:
    rp = str(Path(p).resolve()) + "/"
    return any(rp.startswith(str((ROOT / d).resolve()) + "/") for d in PILOT_DIRS)


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _clip(s: str | None, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "\n[... truncated ...]"


class Excl(Counter):
    def add(self, reason: str, n: int = 1) -> None:
        self[reason] += n


# ── SWE-Together ────────────────────────────────────────────────────────────

def swt_items(run_dir: Path, cell: dict, excl: Excl) -> list[dict]:
    out = []
    for trial in sorted((run_dir / "trials").iterdir()):
        if not (trial.is_dir() and "__" in trial.name and (trial / "agent" / "turns.jsonl").exists()):
            continue
        sub, digest = SWT.frozen_judge_files()  # the frozen per-intent judge (JUDGE_FREEZE.json)
        jp = run_dir / sub / f"{trial.name}.json"
        if not jp.exists():
            excl.add("no_judge_doc")
            continue
        jd = json.loads(jp.read_text())
        if digest is not None and (jd.get("freeze") or {}).get("sha256") != digest:
            excl.add("judge_doc_other_freeze")
            continue
        intents = {i["key"]: i for i in jd["intents"]}
        order = [i["key"] for i in jd["intents"]]
        durable = set(jd.get("durable") or []) if jd.get("durable") is not None else None
        turns = {r["turn"]: r for r in SWT._jsonl(trial / "agent" / "turns.jsonl")
                 if r.get("executed", True)}
        seen = set()
        for t in sorted(k for k in turns if k != 999):
            nxt = turns.get(t + 1)
            if nxt is None or nxt.get("opened_by") not in ("sim", "overlay"):
                excl.add("no_user_reply")
                continue
            if nxt.get("opened_by") == "overlay":
                reply, src = (nxt.get("redo") or {}).get("sim_message_replaced"), "sim_replaced_by_overlay"
            else:
                reply, src = nxt.get("message"), "sim"
            if not (reply or "").strip():
                excl.add("empty_reply")
                continue
            snap = jd["snapshots"].get(str(t))
            p = trial / "agent" / "snapshots" / f"turn-{t}.patch"
            if snap is None or not p.exists():
                excl.add("no_snapshot_verdicts")
                continue
            ag = trial / "agent" / f"opencode.txt.turn-{t}"
            agent_msg = SWT._last_text(ag.read_text(errors="replace")) if ag.exists() else ""
            base = dict(benchmark="swe_together", task=jd["task"], trial=trial.name, turn=t, judge_doc=_rel(jp),
                        real_reply=_clip(reply, REPLY_CHARS), real_reply_source=src,
                        sim_action=nxt.get("sim_action"), agent_message=_clip(agent_msg, AGENT_MESSAGE_CHARS),
                        judge=dict(prompt_sha256=jd["judge"]["prompt_sha256"], model=jd["judge"]["model"],
                                   applicability_prompt_sha256=jd["judge"].get("applicability_prompt_sha256")),
                        intent_digest=_sha(json.dumps(jd["intents"], sort_keys=True)))
            it = _finish(base, p.read_text(errors="replace"), snap, intents, order, durable, cell, seen, excl,
                         reply_tag=None)
            if it:
                out.append(it)
    return out


# ── SWE-Interact ────────────────────────────────────────────────────────────

def swi_items(run_dir: Path, cell: dict, excl: Excl) -> list[dict]:
    from ..swe_interact import metrics as SWI
    from ..swe_interact.intents import load as load_intents
    out = []
    for trial in SWI.trials_of(run_dir):
        jp = run_dir / "intent_judge" / f"{trial.parent.name}.json"
        if not jp.exists():
            excl.add("no_judge_doc")
            continue
        jd = json.loads(jp.read_text())
        doc = load_intents(jd["task"])
        intents = {i["key"]: i for i in doc["intents"]}
        order = [i["key"] for i in doc["intents"]]
        durable = set(jd.get("durable") or []) if jd.get("durable") is not None else None
        asks = {a["call_index"]: a for a in SWI._jsonl(trial / "user-sim" / "asks.jsonl")}
        rows = {r["turn"]: r for r in jd["turns"]}
        seen = set()
        for t in sorted(rows):
            r = rows[t]
            if r.get("ended_by") != "ask_user" or r.get("ask_call_index") is None:
                excl.add("no_user_reply")
                continue
            a = asks.get(r["ask_call_index"]) or {}
            if a.get("last_resort"):
                excl.add("last_resort_reply")
                continue
            reply = a.get("reply")
            if not (reply or "").strip():
                excl.add("empty_reply")
                continue
            overlay = ((a.get("redo") or {}).get("opened_by_next") == "overlay")
            nxt = rows.get(t + 1) or {}
            # the judge doc tags only messages the benchmark's user delivered; a reply the overlay
            # replaced has no tag there (tagged later by ``tag_replies``)
            reply_tag = None if overlay or nxt.get("opened_by") != "sim" else \
                (dict(act=nxt.get("act"), levels=nxt.get("tag")) if nxt.get("tag") else None)
            snap = jd["snapshots"].get(str(t))
            p = trial / r["snapshot"]
            if snap is None or not p.exists():
                excl.add("no_snapshot_verdicts")
                continue
            base = dict(benchmark="swe_interact", task=jd["task"], trial=trial.name, turn=t, step=r.get("step"),
                        judge_doc=_rel(jp),
                        real_reply=_clip(reply, REPLY_CHARS),
                        real_reply_source="sim_replaced_by_overlay" if overlay else "sim", sim_action=None,
                        agent_message=_clip(a.get("question"), AGENT_MESSAGE_CHARS),
                        judge=dict(prompt_sha256=jd["judge"]["llm"]["prompt_sha256"], model=jd["judge"]["llm"]["model"],
                                   exec_recipe=jd["judge"]["exec"].get("recipe_version"),
                                   tagger_prompt_sha256=jd["tagger"]["prompt_sha256"]),
                        intent_digest=jd["intent_digest"])
            it = _finish(base, p.read_text(errors="surrogateescape"), snap, intents, order, durable, cell, seen,
                         excl, reply_tag=reply_tag)
            if it:
                out.append(it)
    return out


# ── shared ──────────────────────────────────────────────────────────────────

def _finish(base: dict, patch: str, snap: dict, intents: dict, order: list[str], durable, cell: dict,
            seen: set, excl: Excl, reply_tag: dict | None) -> dict | None:
    diff = normalize_patch(patch)
    if not diff.strip():
        excl.add("empty_diff")
        return None
    if len(diff) > MAX_DIFF_CHARS:
        excl.add("diff_too_long")
        return None
    verdicts = snap.get("verdicts") or {}
    rev = [k for k in order if k in set(snap.get("revealed") or []) and verdicts.get(k) in RATEABLE]
    if not rev:
        excl.add("no_rateable_intent")
        return None
    key = (_sha(diff), tuple(rev))
    if key in seen:
        excl.add("duplicate_state")
        return None
    seen.add(key)
    ints = []
    for n, k in enumerate(rev, 1):
        i = intents[k]
        ints.append(dict(key=k, rid=f"R{n}", text=i["text"].strip(), kind=i.get("kind"),
                         judge_verdict=verdicts[k], durable=None if durable is None else (k in durable)))
    item = dict(base, id=None, run=cell["run"], rung=cell.get("rung"), producer_model=RUNG_MODEL.get(cell.get("rung")),
                arm=cell.get("arm"), overlay_trial=bool(cell.get("overlay")), replicate=cell.get("replicate"),
                diff=diff, diff_chars=len(diff), diff_sha256=_sha(diff), intents=ints, reply_tag=reply_tag)
    item["id"] = f"{base['benchmark'][:3]}:{base['trial']}:t{base['turn']}"
    return item


def tag_replies(items: list[dict], log: callable = print) -> None:
    """Fill ``reply_tag`` with the SWE-Interact revelation tagger (judge seat: gemini-3.5-flash since
    2026-09-27, gpt-5.5 before; sees only the intent list
    and the reply; cached). SWE-Together has no tagger of its own; the same tagger is used there
    with the task's intent list, so control items are defined the same way on both benchmarks."""
    from ..swe_interact import tagger
    from ..swe_interact.intents import load as load_intents
    failed = None
    for it in items:
        if it.get("reply_tag"):
            continue
        if it["benchmark"] == "swe_interact":
            doc = load_intents(it["task"])
        else:
            jd = _swt_judge_doc(it)
            doc = dict(task=it["task"], digest=it["intent_digest"],
                       intents=[dict(key=i["key"], text=i["text"]) for i in jd["intents"]])
        if failed:
            it["reply_tag_error"] = failed
            continue
        try:
            tg = tagger.tag_message(it["real_reply"], doc)
        except Exception as e:  # noqa: BLE001  (seat unavailable: leave untagged, stop calling)
            failed = f"{type(e).__name__}: {e}"[:300]
            it["reply_tag_error"] = failed
            log(f"  tagger unavailable, remaining items left untagged: {failed}")
            continue
        it["reply_tag"] = dict(act=tg["act"], levels=tg["levels"], tagged_by="e2.tag_replies",
                               cached=tg.get("cached"))
        log(f"  tagged {it['id']}: {tg['act']}")


def _swt_judge_doc(it: dict) -> dict:
    return json.loads((ROOT / it["judge_doc"]).read_text())


def annotate_controls(items: list[dict]) -> None:
    """Per intent: ``reply_targets`` (the real reply tags it explicit/implied) and ``control``
    (judge says unsatisfied at t, the reply is a revision request and targets it)."""
    for it in items:
        tag = it.get("reply_tag") or {}
        levels = tag.get("levels") or {}
        revision = tag.get("act") == "revision_request"
        # loose (secondary): a revision request by the tagger's act or SWE-Together's own simulator
        # act ``redirect``; no per-intent targeting
        loose_rev = revision or it.get("sim_action") == "redirect"
        for i in it["intents"]:
            i["control_loose"] = bool(loose_rev and i["judge_verdict"] != "satisfied")
            lvl = levels.get(i["key"])
            i["reply_level"] = lvl
            i["reply_targets"] = None if not tag else lvl in ("explicit", "implied")
            i["control"] = None if not tag else bool(revision and i["reply_targets"]
                                                     and i["judge_verdict"] != "satisfied")


def assign_unjustified(items: list[dict], seed: int = SEED) -> None:
    tpl = unjustified_templates()
    for it in items:
        rng = random.Random(f"{seed}:{it['id']}")
        k = rng.randrange(len(tpl))
        it["unjustified_index"], it["unjustified_reply"] = k, tpl[k]


def select(items: list[dict], max_items: int | None, per_trial_cap: int | None, seed: int = SEED) -> list[dict]:
    """Seeded selection: shuffle, keep at most ``per_trial_cap`` items per trial, then the first
    ``max_items``; returned in a stable (benchmark, trial, turn) order."""
    rng = random.Random(seed)
    pool = sorted(items, key=lambda i: i["id"])
    rng.shuffle(pool)
    per, out = Counter(), []
    for it in pool:
        if per_trial_cap and per[it["trial"]] >= per_trial_cap:
            continue
        per[it["trial"]] += 1
        out.append(it)
        if max_items and len(out) >= max_items:
            break
    return sorted(out, key=lambda i: (i["benchmark"], i["trial"], i["turn"]))


def discover(paths: list[str]) -> list[tuple[str, Path]]:
    from ..swe_interact import metrics as SWI
    runs = []
    for rd in SWT.find_run_dirs(paths):
        runs.append(("swe_together", rd))
    for rd in SWI.find_run_dirs(paths):
        runs.append(("swe_interact", rd))
    return runs


def build(paths: list[str], *, rungs: set[str] | None, allow_pilot: bool, max_items: int | None,
          per_trial_cap: int | None, tag: bool, seed: int = SEED, expect_judge_sha: str | None = None) -> dict:
    excl = Excl()
    items = []
    if not allow_pilot:
        for p in paths:
            rp = str(Path(p).resolve()) + "/"
            if is_pilot_path(Path(p)) or any(str((ROOT / d).resolve()).startswith(rp) for d in PILOT_DIRS):
                raise SystemExit(f"refusing {p}: it is or contains a pilot directory {PILOT_DIRS}")
    for bench, rd in discover(paths):
        if is_pilot_path(rd) and not allow_pilot:
            raise SystemExit(f"refusing pilot run dir {rd} (E2 reads pilot trials only after the judge freeze and Gate 1)")
        cell = SWT.cell_of(rd)
        if rungs and cell.get("rung") not in rungs:
            excl.add(f"rung_{cell.get('rung')}_not_selected")
            continue
        items += (swt_items if bench == "swe_together" else swi_items)(rd, cell, excl)
    shas = sorted({i["judge"]["prompt_sha256"] for i in items})
    if expect_judge_sha and shas != [expect_judge_sha]:
        raise SystemExit(f"judge prompt sha {shas} != frozen {expect_judge_sha}")
    eligible = len(items)
    chosen = select(items, max_items, per_trial_cap, seed)
    assign_unjustified(chosen, seed)
    if tag:
        tag_replies(chosen)
    annotate_controls(chosen)
    n_int = sum(len(i["intents"]) for i in chosen)
    summary = dict(
        eligible_items=eligible, selected_items=len(chosen), intent_ratings=n_int,
        by_benchmark=dict(Counter(i["benchmark"] for i in chosen)),
        judge_verdicts=dict(Counter(x["judge_verdict"] for i in chosen for x in i["intents"])),
        control_intents=sum(1 for i in chosen for x in i["intents"] if x.get("control")),
        control_loose_intents=sum(1 for i in chosen for x in i["intents"] if x.get("control_loose")),
        untagged_items=sum(1 for i in chosen if not i.get("reply_tag")),
        satisfied_intents=sum(1 for i in chosen for x in i["intents"] if x["judge_verdict"] == "satisfied"),
        reply_acts=dict(Counter((i.get("reply_tag") or {}).get("act") for i in chosen)),
        tasks=len({i["task"] for i in chosen}), trials=len({i["trial"] for i in chosen}))
    return dict(schema="redo_v1/e2-items/1", builder_version=BUILDER_VERSION, seed=seed, sources=paths,
                rungs=sorted(rungs) if rungs else None, max_items=max_items, per_trial_cap=per_trial_cap,
                max_diff_chars=MAX_DIFF_CHARS, judge_prompt_sha256=shas, exclusions=dict(excl),
                summary=summary, items_sha256=items_hash(chosen), items=chosen)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E2 item builder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("paths", nargs="+")
    b.add_argument("--out", required=True)
    b.add_argument("--rungs", default=None, help="comma list, e.g. W")
    b.add_argument("--max-items", type=int, default=None)
    b.add_argument("--per-trial-cap", type=int, default=None)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--no-tag", action="store_true", help="skip the revelation tagger (no control items)")
    b.add_argument("--allow-pilot", action="store_true")
    b.add_argument("--expect-judge-sha", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    body = build(a.paths, rungs=set(a.rungs.split(",")) if a.rungs else None, allow_pilot=a.allow_pilot,
                 max_items=a.max_items, per_trial_cap=a.per_trial_cap, tag=not a.no_tag, seed=a.seed,
                 expect_judge_sha=a.expect_judge_sha)
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("items_sha256") != body["items_sha256"]:
            raise SystemExit(f"{out} exists with a different item set ({old.get('items_sha256')}); use a new path")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1, ensure_ascii=False))
    print(json.dumps(dict(out=str(out), items_sha256=body["items_sha256"], exclusions=body["exclusions"],
                          **body["summary"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
