"""Run the E2 self-judgement conditions for one model over a frozen item set.

As in E1 (``fsa/redo/e1/run.py``): every provider attempt (429/5xx retries included) goes to
``events.jsonl`` through the client's event callback; every judge attempt (unparseable replies
included) to ``calls.jsonl``; the final per-(condition, item) record to ``ratings.jsonl``
(``parsed`` null when still unparseable after ``PARSE_ATTEMPTS``). Bet-first stage-1 bets are
appended and fsync'ed to ``bets.jsonl`` before any stage-2 call for that item. The run resumes
from ``ratings.jsonl``; a run directory is reused only with the identical identity.

Stage-2 enforcement: a requirement with ``revise: false`` keeps its recorded bet whatever values
the reply returned (counted as ``enforced``). ``--no-enforce`` (#2) switches this off: the model's
own values are kept (``raw`` is always recorded).

Wave-2 conditions (``fsa/redo/e2/ladder.py``: #1 ladder, #2 no-copy 2x2, #3 stubs, #12 adversarial) are
added through ``--conditions`` or ``--preset``. When any of them is requested, all (condition, item) jobs go
into one seeded, shuffled pool (P0.7) and pre-commitments (claim books, the ``cb_gated`` gate, the stage-1
bet) are made lazily, once per (item, arm), before the first rating that needs them (``books.jsonl``,
``bets.jsonl``). A legacy-only condition list keeps the published two-stage order. Every model call holds a
shared per-model quota slot (``fsa/redo/quota.py``, P0.3).

Usage: python -u -m fsa.redo.e2.run --model gemini-3.5-flash-lite --items results/redo_v1/e2_smoke/items.json \\
           --out results/redo_v1/e2_smoke --workers 8 [--preset wave2] [--effort low] [--no-enforce]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fsa.llm.client import LLM
from fsa.redo import quota

from . import ladder as L
from . import prompts
from .items import BUILDER_VERSION, items_hash

ROOT = Path(__file__).resolve().parents[3]
SETTINGS = {
    # the E1 settings (docs/results/E1_PREREGISTRATION.md): "low" is the one effort both models accept
    "gemini-3.5-flash-lite": dict(reasoning_effort="low", temperature=1.0, max_tokens=8192),
    "gemini-3.5-flash": dict(reasoning_effort="low", temperature=1.0, max_tokens=8192),
    "gemini-3.6-flash": dict(reasoning_effort="low", temperature=1.0, max_tokens=8192),
    "gemini-3.8-flash": dict(reasoning_effort="low", temperature=1.0, max_tokens=8192),
    # Exploratory frontier rung (Gemini 4.0 Argon):
    "gemini-4.0-argon": dict(reasoning_effort="low", temperature=1.0, max_tokens=8192),
}
MAX_CONCURRENCY = {
    "gemini-3.5-flash-lite": 12,
    "gemini-3.5-flash": 8,
    "gemini-3.6-flash": 8,
    "gemini-3.8-flash": 4,
    "gemini-4.0-argon": 4,
}
EFFORTS = ("low", "medium", "high")
PARSE_ATTEMPTS = 3


class Writer:
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()

    def write(self, rec: dict, sync: bool = False) -> None:
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
            if sync:
                f.flush()
                os.fsync(f.fileno())


def _read(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for l in path.read_text().splitlines():
        if l.strip():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:  # a torn last line from a kill
                continue
    return out


def git_commit() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "fsa/redo/e2", "fsa/redo/arm_texts.py"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return dict(commit=head, e2_code_dirty=bool(dirty))


def _ask(llm: LLM, cfg: dict, msgs: list[dict], parse, calls: Writer, tag: dict) -> dict:
    model = cfg.get("_model") or getattr(llm, "model", "unknown")
    """Up to PARSE_ATTEMPTS calls until ``parse(text)`` is not None. Returns parsed / attempts / text."""
    for attempt in range(PARSE_ATTEMPTS):
        t0 = time.time()
        err, text, resp = None, None, None
        try:
            with quota.slot(model):
                resp = llm.chat(msgs, temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                                sample_id=attempt, response_format={"type": "json_object"})
            text = resp.text
        except Exception as e:  # patient budget exhausted or a non-retryable error
            err = f"{type(e).__name__}: {e}"[:500]
        parsed = parse(text or "") if err is None else None
        calls.write(dict(tag, attempt=attempt, text=text, error=err, parsed=parsed is not None,
                         finish=getattr(resp, "raw_finish", None), cached=getattr(resp, "cached", None),
                         model_version=getattr(resp, "model_version", None),
                         reported_usage=getattr(resp, "reported_usage", None), seconds=round(time.time() - t0, 2)))
        if parsed is not None:
            return dict(parsed=parsed, attempts=attempt + 1, text=text)
    return dict(parsed=None, attempts=PARSE_ATTEMPTS, text=None)


def judge(llm: LLM, cfg: dict, cond: str, item: dict, bet: dict | None, calls: Writer) -> dict:
    rids = [i["rid"] for i in item["intents"]]
    res = _ask(llm, cfg, prompts.build(cond, item, bet), lambda t: prompts.parse(t, cond, rids), calls,
               dict(item=item["id"], cond=cond))
    return dict(parsed=res["parsed"], attempts=res["attempts"])


def run(model: str, items_path: Path, out_root: Path, workers: int, conds: list[str], phase: str,
        effort: str = "low", enforce: bool = True, limit: int | None = None) -> Path:
    body = json.loads(items_path.read_text())
    assert items_hash(body["items"]) == body["items_sha256"], "item set hash mismatch"
    assert body["builder_version"] == BUILDER_VERSION
    items = body["items"][:limit] if limit else body["items"]
    bad = [c for c in conds if c not in prompts.CONDITIONS and not L.is_new(c)]
    if bad:
        raise SystemExit(f"unknown conditions {bad}")
    cfg = dict(SETTINGS[model], reasoning_effort=effort)
    slot_cfg = dict(cfg, _model=model)  # quota slot key; not part of the identity
    workers = min(workers, MAX_CONCURRENCY[model])
    out = out_root / model
    out.mkdir(parents=True, exist_ok=True)
    ident = dict(model=model, phase=phase, settings=cfg, prompt_version=prompts.PROMPT_VERSION,
                 prompts_sha256=prompts.prompts_sha256(), builder_version=BUILDER_VERSION,
                 items_sha256=body["items_sha256"], n_items=len(items), parse_attempts=PARSE_ATTEMPTS,
                 cache_namespace=f"e2-{phase}-{prompts.PROMPT_VERSION}", ladder_sha256=L.ladder_sha256(),
                 enforce=enforce)
    cfg = slot_cfg
    mp = out / "manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text())
        if manifest["identity"] != ident:
            raise SystemExit(f"run directory {out} has a different identity; use a new --out / phase")
    else:
        manifest = dict(identity=ident, code=git_commit(), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), resumes=[])
    manifest["resumes"].append(dict(at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code=git_commit(), workers=workers,
                                    conditions=conds))
    mp.write_text(json.dumps(manifest, indent=1))

    events, calls, ratings, bets, books = (Writer(out / n) for n in (
        "events.jsonl", "calls.jsonl", "ratings.jsonl", "bets.jsonl", "books.jsonl"))
    llm = LLM(model, cache_path=str(out / "llm_cache.sqlite"), reasoning_effort=cfg["reasoning_effort"], thinking="on",
              cache_namespace=ident["cache_namespace"], event_callback=lambda e: events.write(dict(e, t=time.time())))
    done = {(r["cond"], r["item"]) for r in _read(out / "ratings.jsonl")}
    bet_map = {r["item"]: r for r in _read(out / "bets.jsonl")}
    book_map = {(r["item"], r["arm"]): r for r in _read(out / "books.jsonl")}
    locks: dict[tuple, threading.Lock] = {}
    glock = threading.Lock()

    def lock_for(key: tuple) -> threading.Lock:
        with glock:
            return locks.setdefault(key, threading.Lock())

    def stage1(cond: str, item: dict) -> None:
        res = judge(llm, cfg, cond, item, None, calls)
        rec = dict(cond=cond, item=item["id"], parsed=res["parsed"], attempts=res["attempts"])
        if cond == "bet":
            bets.write(rec, sync=True)      # persisted before any stage-2 call of this item
            bet_map[item["id"]] = rec
        ratings.write(rec)
        done.add((cond, item["id"]))

    def get_bet(item: dict) -> dict:
        with lock_for(("bet", item["id"])):
            if item["id"] not in bet_map:
                stage1("bet", item)
            return bet_map[item["id"]]

    def stage2(cond: str, item: dict) -> None:
        bet = bet_map.get(item["id"])
        if bet is None or bet["parsed"] is None:
            ratings.write(dict(cond=cond, item=item["id"], parsed=None, attempts=0, missing_reason="bet_missing"))
            return
        res = judge(llm, cfg, cond, item, bet["parsed"], calls)
        p = res["parsed"]
        final, enforced = None, []
        if p is not None:
            final = {}
            for rid, r in p.items():
                b = bet["parsed"][rid]
                if r["revise"] or not enforce:
                    final[rid] = dict(satisfied=r["satisfied"], p_satisfied=r["p_satisfied"], revise=r["revise"],
                                      new_information=r.get("new_information"))
                else:
                    final[rid] = dict(satisfied=b["satisfied"], p_satisfied=b["p_satisfied"], revise=False,
                                      new_information=r.get("new_information"))
                    if (r["satisfied"], r["p_satisfied"]) != (b["satisfied"], b["p_satisfied"]):
                        enforced.append(rid)
        ratings.write(dict(cond=cond, item=item["id"], parsed=final, raw=p, attempts=res["attempts"], enforced=enforced))

    # -- Wave-2 conditions ---------------------------------------------------------------------------
    def get_book(item: dict, arm: str) -> dict:
        key = (item["id"], arm)
        with lock_for(("book",) + key):
            if key in book_map:
                return book_map[key]
            spec = L.LADDER_SPEC[arm]
            rids = [i["rid"] for i in item["intents"]]
            rec = dict(item=item["id"], arm=arm, kind=spec["book"], gate_p=None, gated_off=False, parsed=None,
                       attempts=0)
            if spec.get("gate"):
                g = _ask(llm, cfg, L.gate_messages(item), L.parse_gate, calls,
                         dict(item=item["id"], cond=f"gate:{arm}"))
                rec["gate_p"] = g["parsed"]
                # an unparseable / failed gate counts as "uncertain" -> full claim book (fail towards treatment)
                rec["gated_off"] = g["parsed"] is not None and g["parsed"] >= L.GATE_TAU
            if not rec["gated_off"]:
                b = _ask(llm, cfg, L.book_messages(spec["book"], item),
                         lambda t, k=spec["book"]: L.parse_book(t, k, rids), calls,
                         dict(item=item["id"], cond=f"book:{arm}"))
                rec["parsed"], rec["attempts"] = b["parsed"], b["attempts"]
            books.write(rec, sync=True)
            book_map[key] = rec
            return rec

    def rate_new(cond: str, item: dict, msgs: list[dict], extra: dict) -> dict:
        rids = [i["rid"] for i in item["intents"]]
        res = _ask(llm, cfg, msgs, lambda t: prompts.parse(t, "real", rids), calls, dict(item=item["id"], cond=cond))
        rec = dict(cond=cond, item=item["id"], parsed=res["parsed"], attempts=res["attempts"], **extra)
        ratings.write(rec)
        done.add((cond, item["id"]))
        return dict(rec, text=res["text"])

    def new_cond(cond: str, item: dict) -> None:
        if L.reply_text(cond, item) is None:
            return  # #12 fab / fabreal on an item without a target requirement: not applicable, no record
        book, bet = None, None
        arm = L.book_arm(cond)
        if arm is not None:
            book = get_book(item, arm)
            if not book["gated_off"] and book["parsed"] is None:
                ratings.write(dict(cond=cond, item=item["id"], parsed=None, attempts=0,
                                   missing_reason="precommit_failed"))
                done.add((cond, item["id"]))
                return
        if L.needs_bet(cond):
            b = get_bet(item)
            if b["parsed"] is None:
                ratings.write(dict(cond=cond, item=item["id"], parsed=None, attempts=0, missing_reason="bet_missing"))
                done.add((cond, item["id"]))
                return
            bet = b["parsed"]
        rate_new(cond, item, L.build(cond, item, book, bet),
                 dict(gated_off=bool(book and book.get("gated_off"))) if book else {})

    # #12 repeated pushback: rounds 1-3 in one conversation; a resumed chain continues after its last
    # written round (the reply texts are stored as ``_text:<cond>`` records next to the ratings).
    prior = _read(out / "ratings.jsonl")
    prior_text = {(r["cond"][6:], r["item"]): r.get("text") for r in prior if r["cond"].startswith("_text:")}
    prior_rounds: dict[tuple, dict] = {}
    for r in prior:
        if r["cond"].startswith("adv_round"):
            arm_ = r["cond"].split(":", 1)[1]
            prior_rounds.setdefault((arm_, r["item"]), {})[r["cond"]] = dict(
                r, text=prior_text.get((r["cond"], r["item"])))

    def adv_rounds_job(arm: str, item: dict) -> None:
        _run_rounds(arm, item, dict(prior_rounds.get((arm, item["id"]), {})))

    def _run_rounds(arm: str, item: dict, recs: dict) -> None:
        book = None
        if arm == "commit_before":
            book = get_book(item, arm)
            if book["parsed"] is None:
                for k in (1, 2, 3):
                    if f"adv_round{k}:{arm}" not in recs:
                        ratings.write(dict(cond=f"adv_round{k}:{arm}", item=item["id"], parsed=None, attempts=0,
                                           missing_reason="precommit_failed"))
                return
        msgs = L.build(f"adv_round1:{arm}", item, book, None)
        for k in (1, 2, 3):
            c = f"adv_round{k}:{arm}"
            if k > 1:
                msgs = msgs + [{"role": "user", "content": L.round_followup(k)}]
            if c in recs:
                if recs[c].get("parsed") is None or not recs[c].get("text"):
                    return  # an earlier round failed (or its text is lost): the chain stops (first attempt)
                msgs = msgs + [{"role": "assistant", "content": recs[c]["text"]}]
                continue
            r = rate_new(c, item, msgs, {})
            if r["parsed"] is None:
                for k2 in range(k + 1, 4):
                    ratings.write(dict(cond=f"adv_round{k2}:{arm}", item=item["id"], parsed=None, attempts=0,
                                       missing_reason="earlier_round_failed"))
                    done.add((f"adv_round{k2}:{arm}", item["id"]))
                return
            ratings.write(dict(cond=f"_text:{c}", item=item["id"], text=r["text"]))
            msgs = msgs + [{"role": "assistant", "content": r["text"]}]

    legacy = all(c in prompts.CONDITIONS for c in conds)
    t0 = time.time()

    def pool(jobs, label):
        print(f"[{model}] {len(jobs)} {label} jobs, workers={workers}", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(fn, *a) for fn, a in jobs]
            for n, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                except Exception as e:  # a bug, not a model failure: log and keep going
                    print(f"  [JOB ERROR] {type(e).__name__}: {e}", flush=True)
                if n % 50 == 0:
                    print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)

    if legacy:
        s1 = [c for c in conds if c in prompts.STAGE1]
        s2 = [c for c in conds if c in prompts.STAGE2]
        if s2 and "bet" not in s1 and not bet_map:
            raise SystemExit("bet-first stage 2 needs stage-1 bets (include 'bet')")
        pool([(stage1, (c, it)) for c in s1 for it in items if (c, it["id"]) not in done], "stage-1")
        pool([(stage2, (c, it)) for c in s2 for it in items if (c, it["id"]) not in done], "stage-2")
    else:
        if any(c in prompts.STAGE2 for c in conds) and "bet" not in conds and not bet_map:
            raise SystemExit("bet-first stage 2 needs stage-1 bets (include 'bet')")
        jobs = []
        chains = set()
        for c in conds:
            for it in items:
                if (c, it["id"]) in done:
                    continue
                if c.startswith("adv_round"):
                    arm = c.split(":", 1)[1]
                    if (arm, it["id"]) not in chains:
                        chains.add((arm, it["id"]))
                        jobs.append((adv_rounds_job, (arm, it)))
                elif c == "bet":
                    jobs.append((get_bet, (it,)))
                elif c in prompts.STAGE1:
                    jobs.append((stage1, (c, it)))
                elif c in prompts.STAGE2:
                    jobs.append((lambda c_, it_: (get_bet(it_), stage2(c_, it_)), (c, it)))
                else:
                    jobs.append((new_cond, (c, it)))
        random.Random(f"e2:{model}:{phase}:{effort}").shuffle(jobs)  # P0.7: arms x items interleaved
        pool(jobs, "interleaved")
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["usage"] = accounting(out)
    mp.write_text(json.dumps(manifest, indent=1))
    if not legacy:
        s = L.summarize(items, [r for r in _read(out / "ratings.jsonl") if not r["cond"].startswith("_text:")])
        s["identity"] = ident
        (out / "summary.json").write_text(json.dumps(s, indent=1))
        (out / "DONE").write_text(json.dumps(dict(finished=manifest["finished"]), indent=1))
    print(json.dumps(manifest["usage"], indent=1), flush=True)
    return out


def accounting(out: Path) -> dict:
    """Totals over every provider attempt in events.jsonl (retries and failed attempts included)."""
    tot = dict(requests_started=0, responses=0, cache_hits=0, request_errors=0, errors_by_status={},
               prompt_tokens=0, completion_tokens=0, total_tokens=0)
    for e in _read(out / "events.jsonl"):
        k = e.get("event")
        if k == "request_started":
            tot["requests_started"] += 1
        elif k == "cache_hit":
            tot["cache_hits"] += 1
        elif k == "request_error":
            tot["request_errors"] += 1
            s = str(e.get("status_code"))
            tot["errors_by_status"][s] = tot["errors_by_status"].get(s, 0) + 1
        elif k == "response":
            tot["responses"] += 1
            ru = e.get("reported_usage") or {}
            for f in ("prompt_tokens", "completion_tokens", "total_tokens"):
                tot[f] += int(ru.get(f, 0) or 0)
    cl = _read(out / "calls.jsonl")
    tot["judge_attempts"] = len(cl)
    tot["unparsed_attempts"] = sum(1 for c in cl if not c["parsed"])
    tot["finish_reasons"] = {}
    for c in cl:
        tot["finish_reasons"][str(c.get("finish"))] = tot["finish_reasons"].get(str(c.get("finish")), 0) + 1
    return tot


def main() -> None:
    from fsa.redo import machine
    machine.apply_process_env()  # this machine's endpoint profile, for this process and its children
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(SETTINGS))
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True, help="output root; the run dir is <out>/<model>")
    ap.add_argument("--phase", default="smoke", help="smoke | main | w2 (part of the cache namespace)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conditions", default=None, help=f"default: the published {','.join(prompts.CONDITIONS)}")
    ap.add_argument("--preset", default=None, choices=sorted(L.PRESETS), help="Wave-2 condition sets (ladder.py)")
    ap.add_argument("--effort", default="low", choices=EFFORTS, help="P0.1 (default low = the published runs)")
    ap.add_argument("--no-enforce", action="store_true", help="#2: keep the model's own stage-2 values (no pinning)")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N items")
    a = ap.parse_args()
    if a.preset and a.conditions:
        raise SystemExit("--preset and --conditions are exclusive")
    conds = list(L.PRESETS[a.preset]) if a.preset else (a.conditions.split(",") if a.conditions
                                                        else list(prompts.CONDITIONS))
    run(a.model, Path(a.items), Path(a.out), a.workers, conds, a.phase, a.effort, not a.no_enforce, a.limit)


if __name__ == "__main__":
    main()
