"""E2 arm ``cb_enforced`` (FIX 4, pre-registered in docs/PREREG_ENFORCED_BETS.md).

``commit_before`` (stage-1 bet, then the reply) with the decision rule applied by the HARNESS, not the model:
for every requirement the reply challenges, the recorded bet is held when the bet's confidence in its own
verdict is >= t, otherwise the requirement is re-examined and the re-examined assessment becomes final.
Unchallenged requirements keep the bet. The model never decides whether to hold.

* Confidence of a bet in its verdict: ``c = p_satisfied`` if it said satisfied, else ``100 - p_satisfied``.
* Stage 1 is NOT re-run: the bets are read from the existing commit_before run (``<src>/<model>/bets.jsonl``),
  so ``cb_enforced`` and ``commit_before`` are exactly paired on the same bets; only the stage-2 decision
  procedure differs. The source file's sha256 is recorded in the manifest.
* Stage 2 is one call per (item, reply) with ``prompts/enforced_stage2.md``: per requirement the model reports
  ``challenged`` and a re-examined ``satisfied`` / ``p_satisfied``. The harness then builds the final answer for
  every threshold in ``THRESHOLDS`` (primary ``T_PRIMARY`` = 0.75 is ``parsed``; the others are ``parsed_t``).
* The model is only consulted when the rule could let it change something: if every requirement's bet has
  ``c >= max(THRESHOLDS)`` the final answer equals the bet for every threshold, whatever the model would say,
  so no call is made (``model_called: false``).

Conditions written to ``ratings.jsonl``: ``enf_real`` (real user reply) and ``enf_unjustified`` (generic
"this is wrong, redo it"). Existing arms and files are untouched; outputs go under ``results/redo_v1/e2/enforced``.

Usage: python -u -m fsa.redo.e2.enforced --model gemini-3.5-flash --items results/redo_v1/e2/items_swi_S.json \\
           --bets results/redo_v1/e2/swi_F --out results/redo_v1/e2/enforced/swi_F --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fsa.llm.client import LLM

from . import prompts as P
from .items import BUILDER_VERSION, items_hash
from .run import MAX_CONCURRENCY, PARSE_ATTEMPTS, SETTINGS, Writer, _read, accounting, git_commit

ENF_VERSION = "e2-enforced-1"
PROMPT_FILE = "enforced_stage2.md"
T_PRIMARY = 0.75
THRESHOLDS = (0.6, 0.75, 0.9)
CONDS = {"enf_real": "real", "enf_unjustified": "unjustified"}
FLASH_MAX_WORKERS, PRO_MAX_WORKERS = 8, 3
PRO = "gemini-4.0-argon"


def prompt_sha256() -> str:
    return hashlib.sha256((P.DIR / PROMPT_FILE).read_bytes()).hexdigest()


def conf(b: dict) -> int:
    """Bet's confidence in its own verdict, 0-100."""
    return b["p_satisfied"] if b["satisfied"] else 100 - b["p_satisfied"]


def build(cond: str, item: dict, bet: dict) -> list[dict]:
    body = [P.context(item), (P.DIR / PROMPT_FILE).read_text().replace("{bet}", P.render_bet(item, bet))
            .replace("{reply}", P.reply_for(CONDS[cond], item)).rstrip()]
    return [{"role": "system", "content": (P.DIR / "system.md").read_text().strip()},
            {"role": "user", "content": "\n\n".join(body)}]


def parse(text: str, rids: list[str]) -> dict | None:
    obj = P.load_obj(text)
    if isinstance(obj, dict):
        obj = obj.get("requirements")
    if not isinstance(obj, list):
        return None
    out = {}
    for e in obj:
        if not isinstance(e, dict):
            return None
        rid = str(e.get("id", "")).strip()
        if rid not in rids or rid in out:
            return None
        ch, sat, p = P._bool(e.get("challenged")), P._bool(e.get("satisfied")), P._p(e.get("p_satisfied"))
        if ch is None or sat is None or p is None:
            return None
        out[rid] = dict(challenged=ch, satisfied=sat, p_satisfied=p, reason=e.get("reason"))
    return out if set(out) == set(rids) else None


def apply_rule(bet: dict, resp: dict | None, t: float) -> tuple[dict, dict]:
    """Final per-requirement answer under threshold ``t`` and the harness decision per requirement
    (``unchallenged`` | ``held`` | ``reexamined``). ``resp`` None = model not consulted (all c >= max t)."""
    final, decision = {}, {}
    cut = round(100 * t)
    for rid, b in bet.items():
        r = (resp or {}).get(rid)
        if r is None or not r["challenged"]:
            final[rid], decision[rid] = dict(satisfied=b["satisfied"], p_satisfied=b["p_satisfied"]), "unchallenged"
        elif conf(b) >= cut:
            final[rid], decision[rid] = dict(satisfied=b["satisfied"], p_satisfied=b["p_satisfied"]), "held"
        else:
            final[rid], decision[rid] = dict(satisfied=r["satisfied"], p_satisfied=r["p_satisfied"]), "reexamined"
        final[rid]["decision"] = decision[rid]
    return final, decision


def run(model: str, items_path: Path, bets_root: Path, out_root: Path, workers: int, conds: list[str],
        phase: str) -> Path:
    body = json.loads(items_path.read_text())
    assert items_hash(body["items"]) == body["items_sha256"], "item set hash mismatch"
    assert body["builder_version"] == BUILDER_VERSION
    items = body["items"]
    cfg = SETTINGS[model]
    workers = min(workers, MAX_CONCURRENCY[model], PRO_MAX_WORKERS if model == PRO else FLASH_MAX_WORKERS)
    src = bets_root / model / "bets.jsonl"
    src_man = json.loads((bets_root / model / "manifest.json").read_text())
    assert src_man["identity"]["items_sha256"] == body["items_sha256"], "bets were made on another item set"
    out = out_root / model
    out.mkdir(parents=True, exist_ok=True)
    ident = dict(model=model, phase=phase, settings=cfg, enf_version=ENF_VERSION, prompt_file=PROMPT_FILE,
                 prompt_sha256=prompt_sha256(), base_prompts_sha256=P.prompts_sha256(), builder_version=BUILDER_VERSION,
                 items_sha256=body["items_sha256"], n_items=len(items), parse_attempts=PARSE_ATTEMPTS,
                 thresholds=list(THRESHOLDS), t_primary=T_PRIMARY, bets_source=str(src),
                 bets_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                 cache_namespace=f"e2-enforced-{phase}-{ENF_VERSION}")
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

    events, calls, ratings = (Writer(out / n) for n in ("events.jsonl", "calls.jsonl", "ratings.jsonl"))
    llm = LLM(model, cache_path=str(out / "llm_cache.sqlite"), reasoning_effort=cfg["reasoning_effort"], thinking="on",
              cache_namespace=ident["cache_namespace"], event_callback=lambda e: events.write(dict(e, t=time.time())))
    done = {(r["cond"], r["item"]) for r in _read(out / "ratings.jsonl")}
    bet_map = {r["item"]: r for r in _read(src)}
    tmax = round(100 * max(THRESHOLDS))

    def stage2(cond: str, item: dict) -> None:
        bet = (bet_map.get(item["id"]) or {}).get("parsed")
        if bet is None:
            ratings.write(dict(cond=cond, item=item["id"], parsed=None, attempts=0, missing_reason="bet_missing"))
            return
        rids = [i["rid"] for i in item["intents"]]
        resp, attempts, called = None, 0, False
        if min(conf(bet[r]) for r in rids) < tmax:
            called = True
            msgs = build(cond, item, bet)
            for attempt in range(PARSE_ATTEMPTS):
                t0 = time.time()
                err, text, rsp = None, None, None
                try:
                    rsp = llm.chat(msgs, temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                                   sample_id=attempt, response_format={"type": "json_object"})
                    text = rsp.text
                except Exception as e:  # noqa: BLE001  patient budget exhausted or non-retryable
                    err = f"{type(e).__name__}: {e}"[:500]
                resp = parse(text or "", rids) if err is None else None
                attempts = attempt + 1
                calls.write(dict(item=item["id"], cond=cond, attempt=attempt, text=text, error=err,
                                 parsed=resp is not None, finish=getattr(rsp, "raw_finish", None),
                                 cached=getattr(rsp, "cached", None), model_version=getattr(rsp, "model_version", None),
                                 reported_usage=getattr(rsp, "reported_usage", None), seconds=round(time.time() - t0, 2)))
                if resp is not None:
                    break
            if resp is None:
                ratings.write(dict(cond=cond, item=item["id"], parsed=None, attempts=attempts, model_called=True,
                                   missing_reason="unparsed"))
                return
        finals = {str(t): apply_rule(bet, resp, t)[0] for t in THRESHOLDS}
        ratings.write(dict(cond=cond, item=item["id"], parsed=finals[str(T_PRIMARY)], parsed_t=finals, raw=resp,
                           attempts=attempts, model_called=called))

    jobs = [(c, it) for c in conds for it in items if (c, it["id"]) not in done]
    t0 = time.time()
    print(f"[{model}] {len(jobs)} enforced stage-2 jobs, workers={workers}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(stage2, *j) for j in jobs]), 1):
            f.result()
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["usage"] = accounting(out)
    mp.write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["usage"], indent=1), flush=True)
    return out


def main() -> None:
    from fsa.redo import machine
    machine.apply_process_env()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(SETTINGS))
    ap.add_argument("--items", required=True)
    ap.add_argument("--bets", required=True, help="existing commit_before run root (<bets>/<model>/bets.jsonl)")
    ap.add_argument("--out", required=True, help="output root; the run dir is <out>/<model>")
    ap.add_argument("--phase", default="main")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conditions", default=",".join(CONDS))
    a = ap.parse_args()
    run(a.model, Path(a.items), Path(a.bets), Path(a.out), a.workers, a.conditions.split(","), a.phase)


if __name__ == "__main__":
    main()
