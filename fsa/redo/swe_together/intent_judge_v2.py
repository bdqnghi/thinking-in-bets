"""Per-intent judge v2 for SWE-Together: executable checks first, LLM judge with context otherwise.

"Is intent i satisfied at snapshot t" (PLAN.md §6; regression, recurrence triggering, caving and
the overlay trigger all depend on it) is decided without the dialogue:

1. **Executable** - intents whose frozen map entry (``configs/intents/swe-together/exec_map.json``,
   ``fsa.redo.swe_together.intent_map``) has ``check_coverage == "full"``: the snapshot is applied
   in a fresh container of the task's arm64 image and the task's own verifier is run
   (``exec_judge``); ``satisfied`` iff every mapped check passes, else ``not_satisfied``.
2. **LLM** - every other applicable intent: the judge seat (``gemini-3.5-flash`` since 2026-09-27; fixed
   effort, strict JSON, no dialogue,
   ``prompts/intent_judge_v2_system.md``) sees the cumulative diff, the post-change text of the
   touched files (from the same container run), the intent with its source message index, the
   benchmark's frozen goals related to it (``canonical_goals.json`` via the map) and the results
   of the related (``partial``) checks on this very snapshot. Labels:
   ``satisfied | not_satisfied | not_addressed``.
3. **Applicability** is the v1 decision (``fsa.redo.intent_judge``: once per intent from its text,
   same prompt and cache); the opening request is always applicable. An empty snapshot marks
   every applicable intent ``not_addressed`` without any call. A snapshot that fails to apply
   falls back to the LLM on the diff alone (recorded as ``apply_failed``).

The configuration in force is ``configs/intents/swe-together/JUDGE_FREEZE.json`` (``load_freeze``):
model, effort, prompt sha, map file + digest, the combination rule and the output directory
(``<run_dir>/<output_subdir>/<trial>.json``, the schema of ``intent_judge.judge_trial`` plus
per-intent ``sources``). ``fsa.redo.metrics`` and the pilot's ``intents`` stage read the frozen
configuration by default. Changing any judged input requires a new version in the freeze file.

The in-trial overlay trigger is the v1 judge (``fsa.redo.intent_judge``, diff + intent only; the
SWE-Together clone imports it from this checkout) on the same judge seat; it only decides *when* an
overlay fires. Every analysis quantity (regression, caving, final intents) uses this frozen v2 judge.

CLI:

    python -u -m fsa.redo.swe_together.intent_judge_v2 trials <run_dir> [...]
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
from dataclasses import asdict
from pathlib import Path

from .. import intent_judge as v1
from .. import llm_seats
from ..swe_interact.llm import structured_call
from . import exec_judge
from .config import RESULTS, ROOT

FREEZE_FILE = ROOT / "configs" / "intents" / "swe-together" / "JUDGE_FREEZE.json"
PROMPT_FILE = Path(__file__).with_name("prompts") / "intent_judge_v2_system.md"
DEFAULT_CACHE = RESULTS / "intent_judge_v2_cache"
LABELS = ("satisfied", "not_satisfied", "not_addressed")
DEFAULTS = {"version": "dev", "model": llm_seats.judge_seat()["model"],
            "reasoning_effort": llm_seats.judge_seat()["reasoning_effort"], "combine": "exec",
            "batch": 8, "max_diff_chars": 100_000, "max_files_chars": 80_000, "max_file_chars": 40_000,
            "file_window_lines": 80, "output_subdir": "intent_judge_v2", "use_goals": True, "use_files": True,
            "use_checks": True}
_EXEC_SEM = threading.BoundedSemaphore(int(os.environ.get("SWT_EXEC_WORKERS", "3")))


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogateescape") if isinstance(s, str) else s).hexdigest()


def prompt_sha(path: Path = PROMPT_FILE) -> str:
    return _sha(Path(path).read_bytes())


def load_freeze(path: Path = FREEZE_FILE, verify: bool = True) -> dict:
    """The frozen configuration; verifies the prompt and map files still match their digests."""
    doc = json.loads(Path(path).read_text())
    cfg = doc["config"]
    if verify:
        p = ROOT / cfg["prompt_file"]
        if prompt_sha(p) != cfg["prompt_sha256"]:
            raise SystemExit(f"{p} changed since the freeze ({cfg['prompt_sha256'][:12]}); record a new judge version")
        m = ROOT / cfg["exec_map_file"]
        if _sha(m.read_bytes()) != cfg["exec_map_file_sha256"]:
            raise SystemExit(f"{m} changed since the freeze; record a new judge version")
        ap = ROOT / cfg["applicability_prompt_file"]
        if _sha(ap.read_bytes()) != cfg["applicability_prompt_sha256"]:
            raise SystemExit(f"{ap} changed since the freeze; record a new judge version")
    return cfg


def freeze_digest(path: Path = FREEZE_FILE) -> str:
    return _sha(Path(path).read_bytes())


# ── context for the LLM ─────────────────────────────────────────────────────

def _hunk_ranges(patch: str) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    cur = None
    for line in patch.splitlines():
        m = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if m:
            cur = m.group(2)
            continue
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if m and cur:
            s, n = int(m.group(1)), int(m.group(2) or 1)
            out.setdefault(cur, []).append((s, s + max(n, 1) - 1))
    return out


def files_block(patch: str, files: dict[str, str], missing: list[str], cfg: dict) -> str:
    """Post-change text of the touched files: whole files when small, else windows around the
    changed lines; total capped."""
    if not files and not missing:
        return ""
    ranges = _hunk_ranges(v1.normalize_patch(patch))
    parts, total = [], 0
    for path in sorted(files, key=lambda p: len(files[p])):
        text = files[path]
        if len(text) > cfg["max_file_chars"]:
            lines = text.splitlines()
            w = cfg["file_window_lines"]
            spans = sorted((max(1, a - w), min(len(lines), b + w)) for a, b in ranges.get(path, []))
            merged: list[list[int]] = []
            for a, b in spans:
                if merged and a <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            chunks = [f"[lines {a}-{b} of {len(lines)}]\n" + "\n".join(lines[a - 1:b]) for a, b in merged]
            text = "\n...\n".join(chunks) if chunks else text[: cfg["max_file_chars"]] + "\n[... truncated ...]"
            if len(text) > cfg["max_file_chars"]:
                text = text[: cfg["max_file_chars"]] + "\n[... truncated ...]"
        if total + len(text) > cfg["max_files_chars"]:
            parts.append(f"### {path}\n(omitted: file context budget exhausted)")
            continue
        total += len(text)
        parts.append(f"### {path}\n```\n{text}\n```")
    for path in missing:
        parts.append(f"### {path}\n(deleted by the change)")
    return "## Files after the change\n" + "\n\n".join(parts)


class CombinedJudge:
    def __init__(self, cfg: dict | None = None, *, cache_dir: Path | str = DEFAULT_CACHE, workers: int = 4,
                 exec_map: dict | None = None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.prompt_file = ROOT / self.cfg.get("prompt_file", str(PROMPT_FILE.relative_to(ROOT)))
        self.system = self.prompt_file.read_text()
        self.psha = _sha(self.system)
        mp = ROOT / self.cfg.get("exec_map_file", "configs/intents/swe-together/exec_map.json")
        self.exec_map = exec_map if exec_map is not None else json.loads(mp.read_text())
        self.map_sha = _sha(json.dumps(self.exec_map, sort_keys=True))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        # applicability only; own cache, seeded read-only from the v1 cache (never written: running
        # overlay trials use that cache)
        self.v1 = v1.IntentJudge(self.cache_dir / "applicability", workers=workers, model=self.cfg["model"],
                                 effort=self.cfg.get("applicability_effort") or self.cfg["reasoning_effort"])
        self._lock = threading.Lock()
        self.stats = {"llm_calls": 0, "llm_cache_hits": 0, "exec_runs": 0, "exec_decided": 0, "llm_decided": 0,
                      "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "apply_failed": 0}

    # ── map lookups ──
    def entry(self, task: str, key: str) -> dict:
        return ((self.exec_map.get("tasks") or {}).get(task, {}).get("intents") or {}).get(key) or {}

    def goals(self, task: str) -> dict[str, dict]:
        return {g["id"]: g for g in (self.exec_map.get("tasks") or {}).get(task, {}).get("goals", [])}

    def exec_checks(self, task: str, key: str) -> list[str] | None:
        e = self.entry(task, key)
        if e.get("check_coverage") == "full" and e.get("checks"):
            return list(e["checks"])
        return None

    # ── cache ──
    def _key(self, task: str, psha: str, intent: v1.Intent, ctx: str) -> str:
        blob = json.dumps([self.cfg["model"], self.cfg["reasoning_effort"], self.psha, task, psha, intent.kind,
                           intent.text, intent.source_turn, _sha(ctx),
                           {k: self.cfg[k] for k in ("use_goals", "use_files", "use_checks", "max_diff_chars",
                                                     "max_files_chars", "max_file_chars", "file_window_lines")}])
        return _sha(blob)

    def _get(self, k: str) -> dict | None:
        p = self.cache_dir / k[:2] / f"{k}.json"
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except json.JSONDecodeError:
            return None

    def _put(self, k: str, rec: dict) -> None:
        p = self.cache_dir / k[:2] / f"{k}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        os.replace(tmp, p)

    def applicability(self, intents: list[v1.Intent]) -> dict[str, bool]:
        for i in intents:
            if i.key == "open":
                continue
            k = self.v1.akey(i)
            if self.v1._get(k) is None:
                src = v1.DEFAULT_CACHE / k[:2] / f"{k}.json"
                if src.exists():
                    self.v1._put(k, json.loads(src.read_text()))
        return self.v1.applicability(intents)

    # ── pieces ──
    def run_exec(self, task: str, patch: str) -> dict:
        c = exec_judge.cached(task, patch)
        if c is not None:
            return c
        with _EXEC_SEM:
            r = exec_judge.run_snapshot(task, patch)
        with self._lock:
            self.stats["exec_runs"] += 1
        return r

    def intent_context(self, task: str, intent: v1.Intent, ex: dict | None) -> dict:
        d = {"key": intent.key, "source_turn": intent.source_turn, "kind": intent.kind, "intent": intent.text}
        e = self.entry(task, intent.key)
        if self.cfg["use_goals"] and e.get("goals"):
            gs = self.goals(task)
            d["related_goals"] = [{"id": g, "from_message": gs.get(g, {}).get("source_turn"),
                                   "goal": gs.get(g, {}).get("goal", "")} for g in e["goals"] if g in gs]
        if self.cfg["use_checks"] and e.get("checks") and ex and ex.get("status") == "ok":
            chk = []
            for c in e["checks"]:
                r = (ex.get("checks") or {}).get(c)
                if r is None or r.get("bridged"):
                    continue
                desc = (self.exec_map["tasks"][task].get("check_descriptions") or {}).get(c, "")
                chk.append({"id": c, "description": desc, "result": "pass" if r.get("passed") else "fail"})
            if chk:
                d["related_checks"] = chk
        return d

    def _llm(self, task: str, patch: str, psha: str, batch: list[v1.Intent], ctxs: dict[str, dict],
             ex: dict | None) -> dict[str, dict]:
        norm = v1.normalize_patch(patch)
        clip = norm
        if len(norm) > self.cfg["max_diff_chars"]:
            h = self.cfg["max_diff_chars"] * 2 // 3
            clip = norm[:h] + f"\n[... {len(norm) - self.cfg['max_diff_chars']} characters of diff truncated ...]\n" \
                + norm[-(self.cfg["max_diff_chars"] - h):]
        fb = ""
        if self.cfg["use_files"] and ex and ex.get("status") == "ok":
            fb = files_block(patch, ex.get("files") or {}, ex.get("files_missing") or [], self.cfg)
        user = (f"## Intents ({len(batch)})\n"
                + "\n".join(json.dumps(ctxs[i.key], ensure_ascii=False) for i in batch)
                + f"\n\n## Cumulative diff of the repository\n```diff\n{clip}\n```\n\n{fb}")
        keys = [i.key for i in batch]
        schema = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {
            "type": "object", "properties": {"key": {"type": "string", "enum": keys},
                                             "verdict": {"type": "string", "enum": list(LABELS)},
                                             "evidence": {"type": "string"}},
            "required": ["key", "verdict", "evidence"], "additionalProperties": False}}},
            "required": ["verdicts"], "additionalProperties": False}

        def validate(out):
            got = {v["key"] for v in out["verdicts"]}
            if not set(keys) <= got:
                raise ValueError(f"missing keys {sorted(set(keys) - got)}")
        r = structured_call(self.system, user, schema, name="swt_intent_verdicts", log_dir=self.cache_dir,
                            effort=self.cfg["reasoning_effort"], model=self.cfg["model"], validate=validate,
                            meta={"kind": "snapshot", "task": task, "patch_sha256": psha, "prompt_sha256": self.psha,
                                  "intent_keys": keys, "user_chars": len(user)})
        u = r["usage"]
        with self._lock:
            self.stats["llm_calls"] += 1
            self.stats["input_tokens"] += u.get("input_tokens") or 0
            self.stats["output_tokens"] += u.get("output_tokens") or 0
            self.stats["reasoning_tokens"] += u.get("reasoning_tokens") or 0
        got = {v["key"]: v for v in r["output"]["verdicts"]}
        return {k: {"verdict": got[k]["verdict"], "evidence": got[k]["evidence"], "response_id": r["response_id"],
                    "resolved_model": r["resolved_model"]} for k in keys}

    # ── main entry ──
    def judge_snapshot(self, task: str, patch: str, intents: list[v1.Intent]) -> dict[str, dict]:
        norm = v1.normalize_patch(patch)
        psha = _sha(norm)
        appl = self.applicability(intents)
        out: dict[str, dict] = {}
        live = []
        for i in intents:
            if not appl[i.key]:
                out[i.key] = {"verdict": "not_applicable", "source": "applicability",
                              "evidence": "not a repository change (applicability)"}
            elif not norm.strip():
                out[i.key] = {"verdict": "not_addressed", "source": "empty", "evidence": "empty diff"}
            else:
                live.append(i)
        if not live:
            return out
        ex = self.run_exec(task, patch)
        ok = ex.get("status") == "ok"
        if not ok:
            with self._lock:
                self.stats["apply_failed"] += ex.get("status") == "apply_failed"
        todo = []
        for i in live:
            chk = self.exec_checks(task, i.key)
            if chk and ok:
                res = {c: (ex.get("checks") or {}).get(c) for c in chk}
                if all(r is not None and not r.get("bridged") for r in res.values()):
                    sat = all(r.get("passed") for r in res.values())
                    out[i.key] = {"verdict": "satisfied" if sat else "not_satisfied", "source": "exec",
                                  "evidence": "; ".join(f"{c}={'pass' if r.get('passed') else 'fail'}"
                                                        for c, r in res.items()),
                                  "exec_key": ex.get("key")}
                    with self._lock:
                        self.stats["exec_decided"] += 1
                    if self.cfg["combine"] == "exec":
                        continue
                    if not sat:  # exec_and_llm: a failing mapped check decides; a pass needs the LLM too
                        continue
            todo.append(i)
        ctxs = {i.key: self.intent_context(task, i, ex) for i in todo}
        pending = []
        for i in todo:
            k = self._key(task, psha, i, json.dumps(ctxs[i.key], sort_keys=True) + ("|files" if ok else "|nofiles"))
            c = self._get(k)
            if c is not None:
                with self._lock:
                    self.stats["llm_cache_hits"] += 1
                self._merge(out, i, c)
            else:
                pending.append((i, k))
        b = self.cfg["batch"]
        batches = [pending[j:j + b] for j in range(0, len(pending), b)]

        def run(bt):
            return self._llm(task, patch, psha, [i for i, _ in bt], ctxs, ex if ok else None)
        if batches:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(batches))) as pool:
                for bt, res in zip(batches, pool.map(run, batches)):
                    for i, k in bt:
                        rec = {**res[i.key], "intent_key": i.key, "task": task, "patch_sha256": psha,
                               "context": ctxs[i.key], "exec_status": ex.get("status"), "judged_at": time.time()}
                        self._put(k, rec)
                        self._merge(out, i, rec)
        for v in out.values():
            v.setdefault("patch_sha256", psha)
        return out

    def _merge(self, out: dict, i: v1.Intent, rec: dict) -> None:
        with self._lock:
            self.stats["llm_decided"] += 1
        prev = out.get(i.key)
        llm = {"verdict": rec["verdict"], "source": "llm", "evidence": rec.get("evidence")}
        if prev and prev.get("source") == "exec":  # exec_and_llm with passing checks
            llm = {"verdict": rec["verdict"], "source": "exec+llm",
                   "evidence": f"{prev['evidence']} | llm: {rec.get('evidence')}"}
        out[i.key] = llm

    def reference_verdicts(self, task: str, intents: list[v1.Intent]) -> dict[str, str] | None:
        ref = exec_judge.reference_patch(task)
        if not ref.strip():
            return None
        return {k: v["verdict"] for k, v in self.judge_snapshot(task, ref, intents).items()}

    def config_record(self) -> dict:
        return {**self.cfg, "prompt_sha256": self.psha, "exec_map_sha256_canonical": self.map_sha,
                "exec_recipe": exec_judge.RECIPE_VERSION}


# ── trials ──────────────────────────────────────────────────────────────────

def judge_trial(trial_dir: Path, judge: CombinedJudge, freeze_meta: dict | None = None) -> dict:
    """Same schema as ``intent_judge.judge_trial`` (+ ``sources``): every snapshot t on the
    intents revealed by t; the final capture (turn-999) on every intent."""
    trial_dir = Path(trial_dir)
    task_dir = v1.trial_task_dir(trial_dir)
    task = task_dir.name
    intents = v1.load_task_intents(task_dir)
    ref = judge.reference_verdicts(task, intents)
    nsim = v1.sim_messages_by_turn(v1.read_turns(trial_dir))
    per = {}
    for t in v1.executed_snapshots(trial_dir):
        text = (trial_dir / "agent" / "snapshots" / f"turn-{t}.patch").read_text(errors="surrogateescape")
        if t == 999:
            rev, n = intents, max(nsim.values(), default=0)
        else:
            n = nsim.get(t, max((v for k, v in nsim.items() if k <= t), default=0))
            rev = v1.revealed(intents, n)
        v = judge.judge_snapshot(task, text, rev)
        per[str(t)] = {"n_sim_messages": n, "revealed": [i.key for i in rev], "patch_sha256": v1.patch_sha(text),
                       "verdicts": {k: x["verdict"] for k, x in v.items()},
                       "sources": {k: x.get("source") for k, x in v.items()},
                       "evidence": {k: x.get("evidence") for k, x in v.items()}}
    return {"schema": "redo_v1/intent-judge-trial/2", "trial": trial_dir.name, "task": task,
            "judge": judge.config_record(), "freeze": freeze_meta,
            "intents": [asdict(i) for i in intents], "reference_verdicts": ref,
            "durable": sorted(k for k, x in ref.items() if x == "satisfied") if ref else None,
            "snapshots": per, "judged_at": time.time()}


def frozen_judge(**kw) -> tuple[CombinedJudge, dict]:
    cfg = load_freeze()
    meta = {"file": str(FREEZE_FILE.relative_to(ROOT)), "sha256": freeze_digest(), "version": cfg["version"]}
    return CombinedJudge(cfg, **kw), meta


def cmd_trials(args) -> int:
    judge, meta = frozen_judge(workers=args.workers)
    sub = judge.cfg["output_subdir"]
    jobs = []
    for rd in args.run_dirs:
        rd = Path(rd).resolve()
        (rd / sub).mkdir(exist_ok=True)
        for td in sorted(d for d in (rd / "trials").iterdir() if d.is_dir() and "__" in d.name):
            if not (td / "agent" / "turns.jsonl").exists():
                continue
            if args.completed_only and not (td / "result.json").exists():
                continue
            outp = rd / sub / f"{td.name}.json"
            if outp.exists() and not args.force:
                try:
                    if json.loads(outp.read_text()).get("freeze", {}).get("sha256") == meta["sha256"]:
                        continue
                except json.JSONDecodeError:
                    pass
            jobs.append((td, outp))

    def one(job):
        td, outp = job
        try:
            res = judge_trial(td, judge, meta)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {td}: {type(e).__name__}: {e}", flush=True)
            return
        if outp.exists():  # made under another freeze or --force: keep it aside, never overwrite
            (outp.parent / "_superseded").mkdir(exist_ok=True)
            outp.rename(outp.parent / "_superseded" / f"{outp.stem}.{int(time.time())}.json")
        outp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"{outp.relative_to(ROOT)}: {len(res['snapshots'])} snapshots; {judge.stats}", flush=True)
    with ThreadPoolExecutor(args.trial_workers) as pool:
        list(pool.map(one, jobs))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trials")
    t.add_argument("run_dirs", nargs="+")
    t.add_argument("--workers", type=int, default=4, help="LLM batches in parallel per snapshot")
    t.add_argument("--trial-workers", type=int, default=3, help="trials in parallel")
    t.add_argument("--completed-only", action="store_true", help="skip trials without result.json (still running)")
    t.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return cmd_trials(a)


if __name__ == "__main__":
    sys.exit(main())
