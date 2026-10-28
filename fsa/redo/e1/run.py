"""Run the E1 judge arms for one model over the frozen item set.

Every provider attempt (including 429/5xx retries) is logged by the client's
event callback to ``events.jsonl``; every judge attempt (including unparseable
replies) is logged to ``calls.jsonl``; the final per-(arm, unit) record goes to
``ratings.jsonl`` (``rating`` null when missing after the bounded re-asks).
Bet-first stage-1 bets are appended and fsync'ed to ``bets.jsonl`` before any
stage-2 call for that decision is made. The run resumes from ``ratings.jsonl``.

Call units: outcome_blind and bet_first stage 1 see only the decision text, so
they are called once per distinct decision (``decision_key``) and shared by the
items with that decision (lucky/unlucky cells; control/twin pairs). All other
arms are called once per item.

Usage: python -u -m fsa.redo.e1.run --model gemini-3.8-flash --phase main --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fsa.llm.client import LLM

from . import prompts
from .generator import GENERATOR_VERSION, items_hash

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results" / "redo_v1" / "e1"
SETTINGS = {
    # One explicit thinking setting per model, fixed for the whole study. gemini-3.8-flash rejects
    # thinking off and "minimal" (HTTP 400); gemini-3.5-flash-lite rejects "none" (HTTP 400). Both accept "low".
    "gemini-3.5-flash-lite": dict(reasoning_effort="low", temperature=1.0, max_tokens=4096),
    "gemini-3.8-flash": dict(reasoning_effort="low", temperature=1.0, max_tokens=4096),
}
MAX_CONCURRENCY = {"gemini-3.5-flash-lite": 12, "gemini-3.8-flash": 4}
PARSE_ATTEMPTS = 3


class Writer:
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()

    def write(self, rec: dict, sync: bool = False) -> None:
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            if sync:
                f.flush()
                os.fsync(f.fileno())


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def git_commit() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "fsa/redo/e1"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dict(commit=head, e1_code_dirty=bool(dirty))


def units(items: list[dict], arm: str) -> list[tuple[str, dict]]:
    """(unit_id, representative item) for single-stage calls of this arm (bet_first -> stage 1)."""
    if arm in ("outcome_blind", "bet_first"):
        seen: dict[str, dict] = {}
        for it in items:
            seen.setdefault(it["decision_key"], it)
        return [(f"d:{k}", it) for k, it in seen.items()]
    return [(f"i:{it['id']}", it) for it in items]


def judge(llm: LLM, cfg: dict, arm: str, item: dict, stage: int, bet: dict | None, calls: Writer, unit: str) -> dict:
    msgs = prompts.build(arm, item, stage, bet)
    for attempt in range(PARSE_ATTEMPTS):
        t0 = time.time()
        err, text, resp = None, None, None
        try:
            resp = llm.chat(msgs, temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                            sample_id=attempt, response_format={"type": "json_object"})
            text = resp.text
        except Exception as e:           # patient budget exhausted or non-retryable error
            err = f"{type(e).__name__}: {e}"[:500]
        parsed = prompts.parse(text or "", arm, stage) if err is None else None
        calls.write(dict(unit=unit, arm=arm, stage=stage, attempt=attempt, text=text, error=err, parsed=parsed is not None,
                         finish=getattr(resp, "raw_finish", None), cached=getattr(resp, "cached", None),
                         model_version=getattr(resp, "model_version", None),
                         reported_usage=getattr(resp, "reported_usage", None), seconds=round(time.time() - t0, 2)))
        if parsed is not None:
            return dict(parsed=parsed, attempts=attempt + 1)
    return dict(parsed=None, attempts=PARSE_ATTEMPTS)


def run(model: str, phase: str, items_path: Path, workers: int, arms: list[str]) -> Path:
    body = json.loads(items_path.read_text())
    assert items_hash(body["items"]) == body["items_sha256"], "item set hash mismatch"
    assert body["generator_version"] == GENERATOR_VERSION
    items = body["items"]
    if phase == "pilot":
        keep = set(body["pilot_ids"])
        items = [i for i in items if i["id"] in keep]
    cfg = SETTINGS[model]
    workers = min(workers, MAX_CONCURRENCY[model])
    out = BASE / phase / model
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    ident = dict(model=model, phase=phase, settings=cfg, prompt_version=prompts.PROMPT_VERSION,
                 generator_version=GENERATOR_VERSION, items_sha256=body["items_sha256"], seed=body["seed"],
                 n_items=len(items), parse_attempts=PARSE_ATTEMPTS, cache_namespace=f"e1-{phase}-{prompts.PROMPT_VERSION}")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["identity"] != ident:
            raise SystemExit(f"run directory {out} has a different identity; use a new phase name")
        manifest = old
    else:
        manifest = dict(identity=ident, code=git_commit(), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), resumes=[])
    manifest["resumes"].append(dict(at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code=git_commit(), workers=workers, arms=arms))
    manifest_path.write_text(json.dumps(manifest, indent=1))

    events, calls, ratings, bets = (Writer(out / n) for n in ("events.jsonl", "calls.jsonl", "ratings.jsonl", "bets.jsonl"))
    llm = LLM(model, cache_path=str(out / "llm_cache.sqlite"), reasoning_effort=cfg["reasoning_effort"], thinking="on",
              cache_namespace=ident["cache_namespace"], event_callback=lambda e: events.write(dict(e, t=time.time())))
    done = {(r["arm"], r["stage"], r["unit"]) for r in _read(out / "ratings.jsonl")}
    bet_map = {r["unit"]: r for r in _read(out / "bets.jsonl")}
    by_key = {}
    for it in items:
        by_key.setdefault(it["decision_key"], []).append(it)

    def single(arm, unit, item):
        res = judge(llm, cfg, arm, item, 1, None, calls, unit)
        p = res["parsed"]
        rec = dict(arm=arm, stage=1, unit=unit, rating=p["rating"] if p else None, parsed=p, attempts=res["attempts"],
                   item_ids=[i["id"] for i in by_key[item["decision_key"]]] if unit.startswith("d:") else [item["id"]])
        if arm == "bet_first":
            bets.write(rec, sync=True)          # persisted before any stage-2 call
            bet_map[unit] = rec
        ratings.write(rec)

    def stage2(item):
        unit = f"i:{item['id']}"
        bet = bet_map.get(f"d:{item['decision_key']}")
        if bet is None or bet["rating"] is None:
            ratings.write(dict(arm="bet_first", stage=2, unit=unit, rating=None, parsed=None, attempts=0,
                               item_ids=[item["id"]], missing_reason="stage1_missing"))
            return
        res = judge(llm, cfg, "bet_first", item, 2, bet["parsed"], calls, unit)
        p = res["parsed"]
        final = None if p is None else (p["rating"] if p["revise"] else bet["rating"])
        ratings.write(dict(arm="bet_first", stage=2, unit=unit, rating=final, parsed=p, attempts=res["attempts"],
                           bet_rating=bet["rating"], item_ids=[item["id"]],
                           enforced=bool(p) and not p["revise"] and p["rating"] != bet["rating"]))

    jobs = [(arm, u, it) for arm in arms for u, it in units(items, arm) if (arm, 1, u) not in done]
    t0 = time.time()
    print(f"[{model}/{phase}] {len(jobs)} stage-1 units, workers={workers}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(single, *j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            f.result()
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s usage={llm.usage.as_dict()}", flush=True)
        if "bet_first" in arms:
            s2 = [it for it in items if ("bet_first", 2, f"i:{it['id']}") not in done]
            print(f"[{model}/{phase}] {len(s2)} bet-first stage-2 units", flush=True)
            futs = [ex.submit(stage2, it) for it in s2]
            for n, f in enumerate(as_completed(futs), 1):
                f.result()
                if n % 100 == 0:
                    print(f"  {n}/{len(s2)} {time.time() - t0:.0f}s", flush=True)
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["usage"] = accounting(out)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["usage"], indent=1), flush=True)
    return out


def accounting(out: Path) -> dict:
    """Totals over every provider attempt in events.jsonl (retries included)."""
    ev = _read(out / "events.jsonl")
    tot = dict(requests_started=0, responses=0, cache_hits=0, request_errors=0, errors_by_status={},
               prompt_tokens=0, completion_tokens=0, total_tokens=0)
    for e in ev:
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
    calls = _read(out / "calls.jsonl")
    tot["judge_attempts"] = len(calls)
    tot["unparsed_attempts"] = sum(1 for c in calls if not c["parsed"])
    return tot


def main() -> None:
    from fsa.redo import machine
    machine.apply_process_env()  # this machine's endpoint profile, for this process and its children
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(SETTINGS))
    ap.add_argument("--phase", required=True, help="pilot | main (a new name for any changed protocol)")
    ap.add_argument("--items", default=str(BASE / "items.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arms", default=",".join(prompts.ARMS))
    a = ap.parse_args()
    run(a.model, a.phase, Path(a.items), a.workers, a.arms.split(","))


if __name__ == "__main__":
    main()
