"""Build demo/cases.js (recorded replays) and the single-file demo/tib_demo.html.

Usage:  .venv/bin/python demo/build.py      (reads results/ read-only)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
import steps as ST  # noqa: E402

DEMO = ROOT / "demo"
SWE_SAMPLES = [  # lane, task, short title
    ("F_low", "cattrs-partial-structuring-recovery", "cattrs: partial_structure (Python)"),
    ("F_low", "go-critic-doc-link-checker", "go-critic: doc link checker (Go)"),
    ("F_low", "abs-stepped-slices", "abs: stepped slices"),
]


def tests(trial_dir: Path, name: str):
    p = trial_dir / name / "verify.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("statuses")


def swe_side(trial_dir: Path):
    traj = json.load(open(trial_dir / "trajectory.json"))
    res = json.loads((trial_dir / "result.json").read_text())
    conv = ST.Converter()
    for m in traj["messages"]:
        conv.feed(m)
    steps = conv.finish()
    rounds = sorted(d.name for d in trial_dir.glob("verify_r*"))
    first_user = next((i for i, s in enumerate(steps) if s["t"] == "user"), len(steps))
    steps.insert(first_user, ST.verdict("Hidden tests after first submission", res["reward_pre"], tests(trial_dir, "verify_pre")))
    steps.append(ST.verdict("Final hidden tests", res["reward_final"], tests(trial_dir, rounds[-1] if rounds else "verify_pre"), final=True))
    return steps


def swe_sample(lane, task, title):
    tdir = ROOT / "results" / "swe_pushback" / lane / "trials"
    traj = json.load(open(tdir / f"{task}__stock__s0" / "trajectory.json"))
    task_text = next(m["content"] for m in traj["messages"] if m["role"] == "user")
    task_text = task_text.split("Please solve this task:\n\n", 1)[-1].split("\n\nRecommended workflow:", 1)[0]
    model = json.loads((tdir / f"{task}__stock__s0" / "result.json").read_text())["model"]
    return {"id": task, "title": title, "model": model, "task": task_text,
            "left": swe_side(tdir / f"{task}__stock__s0"), "right": swe_side(tdir / f"{task}__commit_before__s0")}


def main():
    samples = [swe_sample(*s) for s in SWE_SAMPLES]
    (DEMO / "cases.js").write_text("window.SAMPLES = " + json.dumps(samples, ensure_ascii=False) + ";\n")
    html = (DEMO / "index.html").read_text()
    html = html.replace('<script src="cases.js"></script>', "<script>\n" + (DEMO / "cases.js").read_text() + "\n</script>")
    (DEMO / "tib_demo.html").write_text(html)
    for s in samples:
        print(s["id"], "left steps", len(s["left"]), "right steps", len(s["right"]))
    print("wrote demo/cases.js and demo/tib_demo.html")


if __name__ == "__main__":
    main()
