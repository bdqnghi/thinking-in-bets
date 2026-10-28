"""Per-intent judge without the dialogue (PLAN.md §6).

For a repository snapshot (the cumulative diff after an agent turn) and a set of
user intents, a no-dialogue LLM judge (the judge seat, ``configs/seats/redo_v1_seats.json``:
``gemini-3.5-flash`` since 2026-09-27; gpt-5.5 before) labels every intent

    satisfied | not_satisfied | not_addressed | not_applicable

from the diff and the intent text only. It never sees the conversation, the
agent's narration or the user's reaction, so a negative reply cannot bias it.

Intent set of a task (``load_task_intents``):

* ``open`` - the opening request (``instruction.md``), revealed at turn 0;
* ``<intent_id>`` - every entry of ``oracle_intents.json`` (``source_turn`` = index
  of the original user's message that carried it; the opening message is 0).

"Revealed by snapshot t" (``revealed_by_turn``): the simulator replays the original
user's messages in order and the benchmark advances its ground-truth cursor by one
per delivered message (``UserAgent.advance_original_index``). An intent with
``source_turn = k`` is revealed by snapshot *t* when at least *k* simulator messages
opened agent turns <= t (``turns.jsonl`` rows with ``opened_by == "sim"``; overlay,
no-op "continue" and cap-rescue turns do not count). This is the benchmark's own
cursor alignment; the simulator paraphrases, so it is an approximation.

Implementation choices (recorded in every result):

* model and reasoning effort from the judge seat (``fsa.redo.llm_seats.judge_seat``: the seat
  config, overridable by ``REDO_JUDGE_MODEL`` / ``REDO_JUDGE_EFFORT``), read when an ``IntentJudge``
  is built - so the in-trial overlay trigger (which builds one at trial start) follows the host
  config without any change to the SWE-Together clone. Until 2026-09-26: ``gpt-5.5``, effort
  ``medium``, Responses API. Since 2026-09-27: ``gemini-3.5-flash``, effort ``high``, Gemini's
  OpenAI-compatible chat endpoint; strict JSON schema; patient retry on 429/5xx (``llm_seats``);
  model and effort are part of every cache key, so verdicts of different seats never mix;
* prompt ``prompts/intent_judge_system.md`` = the *strict-reviewer* revision (adopted in
  Stage 1 smoke on 50 dev patches: agreement of the opening-request verdict with the
  benchmark pass rose from kappa 0.27 to 0.47; the first, lenient prompt is kept as
  ``intent_judge_system_lenient.md`` and marked 29/30 agent final patches satisfied);
* applicability (does the intent ask for a repository change at all?) is decided once per
  intent from its text (``prompts/intent_applicability_system.md``), so an intent is never
  ``not_applicable`` on one snapshot and judged on another; the opening request is always
  applicable;
* intents of one snapshot are judged in batches of up to 12 per call (the diff is the
  expensive part of the prompt); each verdict is cached individually under
  sha256(model, effort, prompt sha, normalised-patch sha, intent kind+text), so re-runs
  and identical snapshots are free;
* snapshot banners (``=== /repo (cumulative vs harbor-base) ===``) are stripped before
  hashing/judging, so an unchanged repository hashes like the empty base;
* every call, including failed attempts, is kept as one JSON file under
  ``<cache>/calls/``.

CLI:

    python -m fsa.redo.intent_judge trials <run_dir> [<run_dir> ...]   # -> <run_dir>/intent_judge/<trial>.json
    python -m fsa.redo.intent_judge validate --out <dir>              # reference vs empty base, 10 calibration tasks
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from . import llm_seats

ROOT = Path(__file__).resolve().parents[2]
CLONE = ROOT / "third_party" / "swe-together"
DEFAULT_CACHE = ROOT / "results" / "redo_v1" / "intent_judge_cache"
PROMPT_FILE = Path(__file__).with_name("prompts") / "intent_judge_system.md"
APPLICABILITY_FILE = Path(__file__).with_name("prompts") / "intent_applicability_system.md"

JUDGE_MODEL = None     # resolved per IntentJudge from the judge seat (llm_seats.judge_seat)
JUDGE_EFFORT = None
MAX_PATCH_CHARS = 150_000
OPEN_TEXT_CHARS = 6_000
BATCH = 12

LABELS = ("satisfied", "not_satisfied", "not_addressed", "not_applicable")
SNAPSHOT_LABELS = ("satisfied", "not_satisfied", "not_addressed")
APPLICABILITY_LABELS = ("applicable", "not_applicable")
UNSATISFIED = frozenset({"not_satisfied", "not_addressed"})

_BANNER = re.compile(r"^=== .* ===\s*$")


@dataclass(frozen=True)
class Intent:
    key: str            # "open" or str(intent_id)
    source_turn: int    # original user message index (0 = opening request)
    kind: str
    text: str


def prompt_text() -> str:
    return PROMPT_FILE.read_text()


def prompt_sha() -> str:
    return hashlib.sha256(prompt_text().encode()).hexdigest()


def applicability_sha() -> str:
    return hashlib.sha256(APPLICABILITY_FILE.read_bytes()).hexdigest()


def judge_config(model: str | None = None, effort: str | None = None) -> dict:
    seat = llm_seats.judge_seat()
    model = model or seat["model"]
    effort = effort or seat["reasoning_effort"]
    return {"model": model, "reasoning_effort": effort, "backend": llm_seats.backend(model),
            "seat_source": seat["source"], "prompt_file": str(PROMPT_FILE.relative_to(ROOT)),
            "prompt_sha256": prompt_sha(), "applicability_prompt_file": str(APPLICABILITY_FILE.relative_to(ROOT)),
            "applicability_prompt_sha256": applicability_sha(), "open_intent": "always applicable",
            "batch": BATCH, "max_patch_chars": MAX_PATCH_CHARS, "retry": "llm_seats.json_call (patient)"}


# ── task intents and revelation ─────────────────────────────────────────────

def load_task_intents(task_dir: Path) -> list[Intent]:
    task_dir = Path(task_dir)
    opening = (task_dir / "instruction.md").read_text()
    if len(opening) > OPEN_TEXT_CHARS:
        opening = opening[:OPEN_TEXT_CHARS] + "\n[... opening request truncated ...]"
    out = [Intent("open", 0, "request", opening.strip())]
    p = task_dir / "oracle_intents.json"
    if p.exists():
        for i in json.loads(p.read_text()).get("intents", []):
            out.append(Intent(str(i["intent_id"]), int(i["source_turn"]), i.get("intent_kind", "?"), i["text"].strip()))
    return out


def read_turns(trial_dir: Path) -> list[dict]:
    p = Path(trial_dir) / "agent" / "turns.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def sim_messages_by_turn(turns: list[dict]) -> dict[int, int]:
    """turn -> number of simulator messages that opened executed turns <= turn."""
    out, n = {}, 0
    for r in sorted((r for r in turns if r.get("executed", True) and r.get("opened_by") != "session_end"),
                    key=lambda r: r["turn"]):
        if r.get("opened_by") == "sim":
            n += 1
        out[int(r["turn"])] = n
    return out


def revealed(intents: list[Intent], n_sim_messages: int) -> list[Intent]:
    return [i for i in intents if i.source_turn <= n_sim_messages]


# ── patches ─────────────────────────────────────────────────────────────────

def normalize_patch(text: str) -> str:
    lines = [l.rstrip() for l in (text or "").splitlines() if not _BANNER.match(l)]
    return "\n".join(lines).strip() + ("\n" if any(l.strip() for l in lines) else "")


def patch_sha(text: str) -> str:
    return hashlib.sha256(normalize_patch(text).encode()).hexdigest()


def _clip_patch(norm: str) -> tuple[str, bool]:
    if len(norm) <= MAX_PATCH_CHARS:
        return norm, False
    head = norm[: MAX_PATCH_CHARS * 2 // 3]
    tail = norm[-MAX_PATCH_CHARS // 3:]
    return head + f"\n[... {len(norm) - len(head) - len(tail)} characters of diff truncated ...]\n" + tail, True


# ── judge ───────────────────────────────────────────────────────────────────

def _schema(keys: list[str], labels=SNAPSHOT_LABELS, why: str = "evidence") -> dict:
    return {
        "type": "object",
        "properties": {"verdicts": {"type": "array", "items": {
            "type": "object",
            "properties": {"key": {"type": "string", "enum": keys},
                           "verdict": {"type": "string", "enum": list(labels)},
                           why: {"type": "string"}},
            "required": ["key", "verdict", why], "additionalProperties": False}}},
        "required": ["verdicts"], "additionalProperties": False,
    }


class IntentJudge:
    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE, *, model: str | None = None,
                 effort: str | None = None, workers: int = 6, client=None):
        self.cache_dir = Path(cache_dir)
        (self.cache_dir / "calls").mkdir(parents=True, exist_ok=True)
        seat = llm_seats.judge_seat()
        self.model, self.effort, self.workers = model or seat["model"], effort or seat["reasoning_effort"], workers
        self._client = client
        self._psha = prompt_sha()
        self._asha = applicability_sha()
        self._sys = prompt_text()
        self._asys = APPLICABILITY_FILE.read_text()
        self._lock = threading.Lock()
        self.stats = {"cache_hits": 0, "judged": 0, "calls": 0, "failed_calls": 0,
                      "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    @property
    def client(self):
        if self._client is None:
            self._client = llm_seats.client_for(self.model, timeout=600)
        return self._client

    def config(self) -> dict:
        return judge_config(self.model, self.effort)

    # cache keys: verdicts by (patch, intent); applicability by intent alone
    def key(self, psha: str, intent: Intent) -> str:
        blob = json.dumps([self.model, self.effort, self._psha, psha, intent.kind, intent.text])
        return hashlib.sha256(blob.encode()).hexdigest()

    def akey(self, intent: Intent) -> str:
        blob = json.dumps(["applicability", self.model, self.effort, self._asha, intent.kind, intent.text])
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_path(self, k: str) -> Path:
        return self.cache_dir / k[:2] / f"{k}.json"

    def _get(self, k: str) -> dict | None:
        p = self._cache_path(k)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def _put(self, k: str, rec: dict) -> None:
        p = self._cache_path(k)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, p)

    def cached(self, psha: str, intent: Intent) -> dict | None:
        return self._get(self.key(psha, intent))

    def _log_call(self, rec: dict) -> None:
        name = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        (self.cache_dir / "calls" / name).write_text(json.dumps(rec, indent=1))

    def _call(self, system: str, user: str, keys: list[str], labels, why: str, meta: dict) -> dict[str, dict]:
        def validate(d):
            got = {v["key"] for v in d["verdicts"] if v.get("verdict") in labels}
            missing = [k for k in keys if k not in got]
            if missing:
                raise ValueError(f"judge output missing keys {missing}")

        def log(rec):
            rec["intent_keys"] = keys
            with self._lock:
                if rec.get("ok"):
                    u = rec.get("usage") or {}
                    self.stats["calls"] += 1
                    self.stats["input_tokens"] += u.get("input_tokens") or 0
                    self.stats["output_tokens"] += u.get("output_tokens") or 0
                    self.stats["reasoning_tokens"] += u.get("reasoning_tokens") or 0
                else:
                    self.stats["failed_calls"] += 1
            self._log_call(rec)
        r = llm_seats.json_call(system, user, _schema(keys, labels, why), name="intent_verdicts", model=self.model,
                                effort=self.effort, validate=validate, log=log, meta=meta, client=self.client)
        got = {v["key"]: v for v in r["output"]["verdicts"] if v.get("verdict") in labels}
        return {k: {"verdict": got[k]["verdict"], why: got[k][why], "resolved_model": r["resolved_model"],
                    "response_id": r["response_id"]} for k in keys}

    @staticmethod
    def _intent_lines(batch: list[Intent]) -> str:
        return "\n".join(json.dumps({"key": i.key, "kind": i.kind, "intent": i.text}, ensure_ascii=False)
                         for i in batch)

    def applicability(self, intents: list[Intent]) -> dict[str, bool]:
        """key -> applicable. Decided once per intent (text + kind), independent of any diff;
        the opening request is always applicable (the benchmark grades a code change)."""
        out, todo = {}, []
        for i in intents:
            if i.key == "open":
                out[i.key] = True
                continue
            c = self._get(self.akey(i))
            if c is not None:
                out[i.key] = c["verdict"] == "applicable"
            else:
                todo.append(i)
        for j in range(0, len(todo), 3 * BATCH):
            b = todo[j:j + 3 * BATCH]
            res = self._call(self._asys, f"## Intents ({len(b)})\n{self._intent_lines(b)}", [i.key for i in b],
                             APPLICABILITY_LABELS, "reason", {"kind": "applicability", "prompt_sha256": self._asha})
            for i in b:
                rec = {**res[i.key], "intent_key": i.key, "judged_at": time.time()}
                self._put(self.akey(i), rec)
                out[i.key] = rec["verdict"] == "applicable"
        return out

    def judge_snapshot(self, patch_text: str, intents: list[Intent]) -> dict[str, dict]:
        """key -> {verdict, evidence, cached, ...} for every intent (cache first);
        intents that no repository change can fulfil get ``not_applicable`` without a call."""
        norm = normalize_patch(patch_text)
        psha = hashlib.sha256(norm.encode()).hexdigest()
        appl = self.applicability(intents)
        out, todo = {}, []
        for i in intents:
            if not appl[i.key]:
                out[i.key] = {"verdict": "not_applicable", "evidence": "not a repository change (applicability)",
                              "cached": True}
                continue
            c = self.cached(psha, i)
            if c is not None:
                out[i.key] = {**c, "cached": True}
                with self._lock:
                    self.stats["cache_hits"] += 1
            else:
                todo.append(i)
        clipped, truncated = _clip_patch(norm)
        diff_block = (f"## Cumulative diff of the repository{' (truncated)' if truncated else ''}\n"
                      f"```diff\n{clipped if clipped.strip() else '(empty diff: the repository is unchanged)'}\n```")
        batches = [todo[j:j + BATCH] for j in range(0, len(todo), BATCH)]

        def run(b):
            user = f"## Intents ({len(b)})\n{self._intent_lines(b)}\n\n{diff_block}"
            return self._call(self._sys, user, [i.key for i in b], SNAPSHOT_LABELS, "evidence",
                              {"kind": "snapshot", "prompt_sha256": self._psha, "patch_sha256": psha,
                               "patch_chars": len(norm), "patch_truncated": truncated})
        if batches:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(batches))) as ex:
                for b, res in zip(batches, ex.map(run, batches)):
                    for i in b:
                        rec = {**res[i.key], "intent_key": i.key, "patch_sha256": psha,
                               "patch_truncated": truncated, "judged_at": time.time()}
                        self._put(self.key(psha, i), rec)
                        out[i.key] = {**rec, "cached": False}
                        with self._lock:
                            self.stats["judged"] += 1
        for v in out.values():
            v.setdefault("patch_sha256", psha)
        return out

    def reference_verdicts(self, task_dir: Path, intents: list[Intent] | None = None) -> dict[str, str] | None:
        """Verdicts on the task's reference patch (None when it has none). Intents the
        reference satisfies are the task's *durable* intents: an intent it does not
        satisfy was typically superseded later in the original session."""
        rp = Path(task_dir) / "reference_patch.json"
        ref = json.loads(rp.read_text()).get("patch") if rp.exists() else None
        if not (ref or "").strip():
            return None
        intents = intents or load_task_intents(task_dir)
        return {k: v["verdict"] for k, v in self.judge_snapshot(ref, intents).items()}


# ── trial-level judging ─────────────────────────────────────────────────────

def trial_task_dir(trial_dir: Path) -> Path:
    cfg = json.loads((Path(trial_dir) / "config.json").read_text())
    p = Path(cfg["task"]["path"])
    if not p.exists():  # moved clone: fall back to the task name
        p = CLONE / "tasks" / p.name
    return p


def executed_snapshots(trial_dir: Path) -> list[int]:
    snaps = Path(trial_dir) / "agent" / "snapshots"
    return sorted(int(p.stem.split("-")[1]) for p in snaps.glob("turn-*.patch")) if snaps.exists() else []


def judge_trial(trial_dir: Path, judge: IntentJudge) -> dict:
    """Judge every snapshot t on the intents revealed by t; the final snapshot
    (turn-999, the end-of-run capture) on every intent."""
    trial_dir = Path(trial_dir)
    task_dir = trial_task_dir(trial_dir)
    intents = load_task_intents(task_dir)
    ref = judge.reference_verdicts(task_dir, intents)
    turns = read_turns(trial_dir)
    nsim = sim_messages_by_turn(turns)
    snaps = executed_snapshots(trial_dir)
    per = {}
    for t in snaps:
        text = (trial_dir / "agent" / "snapshots" / f"turn-{t}.patch").read_text()
        if t == 999:
            rev = intents
            n = max(nsim.values(), default=0)
        else:
            n = nsim.get(t, max((v for k, v in nsim.items() if k <= t), default=0))
            rev = revealed(intents, n)
        v = judge.judge_snapshot(text, rev)
        per[str(t)] = {"n_sim_messages": n, "revealed": [i.key for i in rev],
                       "patch_sha256": patch_sha(text),
                       "verdicts": {k: x["verdict"] for k, x in v.items()},
                       "evidence": {k: x.get("evidence") for k, x in v.items()}}
    return {"schema": "redo_v1/intent-judge-trial/1", "trial": trial_dir.name,
            "task": task_dir.name, "judge": judge.config(),
            "intents": [asdict(i) for i in intents], "reference_verdicts": ref,
            "durable": sorted(k for k, v in ref.items() if v == "satisfied") if ref else None,
            "snapshots": per,
            "judged_at": time.time()}


def cmd_trials(args) -> int:
    judge = IntentJudge(args.cache, workers=args.workers)
    for rd in args.run_dirs:
        rd = Path(rd).resolve()
        out_dir = rd / "intent_judge"
        out_dir.mkdir(exist_ok=True)
        trials = sorted(d for d in (rd / "trials").iterdir() if d.is_dir() and "__" in d.name)
        for td in trials:
            if not (td / "agent" / "turns.jsonl").exists():
                print(f"skip {td.name}: no turns.jsonl", flush=True)
                continue
            try:
                res = judge_trial(td, judge)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {td.name}: {e}", flush=True)
                continue
            (out_dir / f"{td.name}.json").write_text(json.dumps(res, indent=1))
            print(f"{rd.name}/{td.name}: {len(res['snapshots'])} snapshots; stats {judge.stats}", flush=True)
    return 0


# ── validation on the judge-calibration tasks ───────────────────────────────

def _agreement(rows: list[dict], pred: str, gold: str = "benchmark_pass") -> dict:
    tp = sum(1 for r in rows if r[pred] and r[gold])
    tn = sum(1 for r in rows if not r[pred] and not r[gold])
    fp = sum(1 for r in rows if r[pred] and not r[gold])
    fn = sum(1 for r in rows if not r[pred] and r[gold])
    n = len(rows)
    po = (tp + tn) / n if n else None
    pe = (((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)) if n else None
    kappa = (po - pe) / (1 - pe) if n and pe != 1 else None
    return {"n": n, "agree": tp + tn, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "cohen_kappa": round(kappa, 3) if kappa is not None else None}


def summarize_patch(labels: dict[str, str]) -> dict:
    app = [k for k, l in labels.items() if l != "not_applicable"]
    sat = [k for k in app if labels[k] == "satisfied"]
    return {"n_applicable": len(app), "n_satisfied": len(sat),
            "all_applicable_satisfied": bool(app) and len(sat) == len(app),
            "open_satisfied": labels.get("open") == "satisfied"}


def cmd_validate(args) -> int:
    """Reference patch vs empty base on the 10 Stage 0 judge-calibration tasks, and
    agreement with the benchmark's frozen-goal judge (score >= 0.85)."""
    calib = ROOT / "results" / "redo_v1" / "stage0" / "judge_calibration"
    summary = json.loads((calib / "calibration_summary.json").read_text())
    judge = IntentJudge(args.cache, workers=args.workers)
    rows = []
    for rec in summary["table"]:
        task = rec["task"]
        td = CLONE / "tasks" / task
        intents = load_task_intents(td)
        ref = json.loads((td / "reference_patch.json").read_text()).get("patch") or ""
        for arm, patch, bench_score in (("reference", ref, rec["reference_score"]), ("base", "", rec["base_score"])):
            v = judge.judge_snapshot(patch, intents)
            labels = {k: x["verdict"] for k, x in v.items()}
            rows.append({"task": task, "arm": arm, "n_intents": len(intents), **summarize_patch(labels),
                         "benchmark_judge_score": bench_score, "benchmark_pass": (bench_score or 0) >= 0.85,
                         "labels": labels, "evidence": {k: x.get("evidence") for k, x in v.items()}})
            r = rows[-1]
            print(f"{task:34s} {arm:9s} applicable {r['n_applicable']:2d} satisfied {r['n_satisfied']:2d} "
                  f"open={labels.get('open')} bench={bench_score}", flush=True)

    def tot(arm, key):
        return sum(r[key] for r in rows if r["arm"] == arm)

    out = {
        "judge": judge.config(), "tasks": [r["task"] for r in summary["table"]],
        "reference_satisfied_intents": f"{tot('reference', 'n_satisfied')}/{tot('reference', 'n_applicable')}",
        "base_satisfied_intents": f"{tot('base', 'n_satisfied')}/{tot('base', 'n_applicable')}",
        "reference_open_satisfied": f"{tot('reference', 'open_satisfied')}/{len(summary['table'])}",
        "base_open_satisfied": f"{tot('base', 'open_satisfied')}/{len(summary['table'])}",
        "reference_all_applicable_satisfied": f"{tot('reference', 'all_applicable_satisfied')}/{len(summary['table'])}",
        "base_all_applicable_satisfied": f"{tot('base', 'all_applicable_satisfied')}/{len(summary['table'])}",
        "agreement_open_satisfied_vs_benchmark_pass": _agreement(rows, "open_satisfied"),
        "agreement_all_applicable_satisfied_vs_benchmark_pass": _agreement(rows, "all_applicable_satisfied"),
        "stats": judge.stats, "rows": rows,
    }
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "tasks")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--workers", type=int, default=6)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trials")
    t.add_argument("run_dirs", nargs="+")
    v = sub.add_parser("validate")
    v.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_trials(a) if a.cmd == "trials" else cmd_validate(a)


if __name__ == "__main__":
    sys.exit(main())
