"""Live demo server: runs the real SWE-Pushback-Hidden harness (standard vs TIB arm) on a chosen task and streams
each agent step to the browser as it happens. Also serves the page and the recorded replays.

    .venv/bin/python demo/live_server.py --port 8791
    # then open http://localhost:8791/

Nothing here changes the harness: ``LiveTrial`` only observes ``Trial`` (messages appended, verifier calls) and
forwards them. Runs are written to results/demo_live/<run-id>/ like any other trial.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

import steps as ST  # noqa: E402
from fsa.redo.swe_pushback import docker_env as D  # noqa: E402
from fsa.redo.swe_pushback.agent import Config, Trial  # noqa: E402

DEMO = ROOT / "demo"
DATASET = Path.home() / "harbor_datasets/ds11/deep-swe-1-1"
OBJECTIONS = ROOT / "results/swe_pushback/datasets/objections_v1.jsonl"
OUT = ROOT / "results/demo_live"
ARMS = {"L": "stock", "R": "commit_before"}
FEATURED = ["cattrs-partial-structuring-recovery", "go-critic-doc-link-checker", "abs-stepped-slices"]
MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"]


# ------------------------------------------------------------------------------------------------ run state
class Run:
    def __init__(self, task_id: str, model: str):
        self.id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        self.task_id, self.model = task_id, model
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.done = {"L": False, "R": False}

    def emit(self, side: str, ev: dict):
        with self.cond:
            self.events.append({"side": side, "t_wall": round(time.time(), 2), **ev})
            self.cond.notify_all()

    def finish(self, side: str):
        self.done[side] = True
        self.emit(side, {"t": "done"})


RUNS: dict[str, Run] = {}


class LiveTrial(Trial):
    """Trial that forwards display steps to a Run as they happen."""

    def __init__(self, *a, run: Run, side: str, **kw):
        super().__init__(*a, **kw)
        self.live, self.side = run, side
        self.conv = ST.Converter()
        self.seen = 0

    def _flush(self):
        while self.seen < len(self.msgs):
            for st in self.conv.feed(self.msgs[self.seen]):
                self.live.emit(self.side, st)
            self.seen += 1

    def _ask(self, user, phase):
        self._flush()
        self.live.emit(self.side, {"t": "status", "text": "thinking…"})
        r = super()._ask(user, phase)
        self._flush()
        return r

    def _act(self, reply, phase):
        self.live.emit(self.side, {"t": "status", "text": "running command…"})
        r = super()._act(reply, phase)
        self._flush()
        return r

    def _verify(self, patch, stage):
        self._flush()
        self.live.emit(self.side, {"t": "status", "text": "running hidden tests…"})
        v = super()._verify(patch, stage)
        label = {"pre": "Hidden tests after first submission", "first_submit": "Hidden tests on first submission"}.get(
            stage, f"Hidden tests after review round {stage[1:]}")
        if stage != "first_submit":
            self.live.emit(self.side, ST.verdict(label, v["reward"], v.get("statuses")))
        return v

    def run_live(self):
        row = Trial.run(self)
        self._flush()
        if row.get("status") == "ok":
            fin = self.out / (sorted(d.name for d in self.out.glob("verify_r*")) or ["verify_pre"])[-1] / "verify.json"
            statuses = json.loads(fin.read_text()).get("statuses") if fin.exists() else None
            self.live.emit(self.side, ST.verdict("Final hidden tests", row["reward_final"], statuses, final=True))
        else:
            self.live.emit(self.side, {"t": "harness", "text": f"run ended with {row.get('status')}: {row.get('error', '')[:300]}"})
        self.live.finish(self.side)



def start_run(task_id: str, model: str) -> Run:
    objs = {json.loads(l)["task"]: json.loads(l) for l in OBJECTIONS.read_text().splitlines() if l.strip()}
    if task_id not in objs or "error" in objs[task_id]:
        raise ValueError(f"no frozen objection for {task_id}")
    task = D.Task(DATASET / task_id)
    run = Run(task_id, model)
    RUNS[run.id] = run
    cfg = Config(model=model, effort="low")
    for side, arm in ARMS.items():
        trial = LiveTrial(task, arm, objs[task_id], cfg, OUT / run.id / f"{task_id}__{arm}", run=run, side=side)
        threading.Thread(target=trial.run_live, name=f"{run.id}-{arm}", daemon=True).start()
    return run


def list_tasks() -> list[dict]:
    objs = {json.loads(l)["task"]: json.loads(l) for l in OBJECTIONS.read_text().splitlines() if l.strip()}
    ids = [d.name for d in sorted(DATASET.iterdir()) if (d / "task.toml").exists() and d.name in objs and "error" not in objs[d.name]]
    ids = FEATURED + [i for i in ids if i not in FEATURED]
    out = []
    for i in ids:
        t = D.Task(DATASET / i)
        out.append({"id": i, "featured": i in FEATURED, "kind": objs[i]["kind"],
                    "task": t.instruction.split("\n\nRecommended workflow:", 1)[0], "tests": len(t.f2p) + len(t.p2p)})
    return out


# ------------------------------------------------------------------------------------------------ http
class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter
        if "/api/stream" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _file(self, name, ctype):
        p = DEMO / name
        if not p.exists():
            self.send_error(404); return
        b = p.read_bytes()
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._file("index.html", "text/html; charset=utf-8")
        if u.path == "/cases.js":
            return self._file("cases.js", "application/javascript")
        if u.path == "/api/tasks":
            return self._json({"tasks": list_tasks(), "models": MODELS})
        if u.path == "/api/stream":
            run = RUNS.get(q.get("run", [""])[0])
            if not run:
                return self.send_error(404)
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            i = int(q.get("from", ["0"])[0])
            try:
                while True:
                    with run.cond:
                        if i >= len(run.events):
                            if all(run.done.values()):
                                self.wfile.write(b"event: end\ndata: {}\n\n"); self.wfile.flush(); return
                            run.cond.wait(timeout=15)
                            if i >= len(run.events):
                                self.wfile.write(b": ping\n\n"); self.wfile.flush(); continue
                        ev = run.events[i]
                    self.wfile.write(f"id: {i}\ndata: {json.dumps(ev)}\n\n".encode()); self.wfile.flush(); i += 1
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/api/run":
            try:
                run = start_run(q["task"][0], q.get("model", [MODELS[0]])[0])
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 400)
            return self._json({"run": run.id})
        self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--bind", default="127.0.0.1")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((a.bind, a.port), H)
    print(f"serving http://{a.bind}:{a.port}/  (live runs -> {OUT})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
