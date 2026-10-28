"""Convert Agents' Last Exam command-line tasks into Harbor tasks so fsa/tb_agent.py can run them.

Scope: ALE tasks whose grader is a self-contained script under scripts/ (no cua_bench) that takes
--output-dir/--reference-dir (optionally --input-dir) and prints JSON with "score" (or "normalized_score").
That is 37 of the 165 public tasks (list: `python -m fsa.ale_convert --list`). Everything else needs the
official ale_run harness with the CUA desktop image.

Requires the gated task data extracted at <task_data>/<domain>/<task>/<variant>/{input,software,reference}
(scripts/fetch_task_data.sh in third_party/agents-last-exam). Usage:
  python -m fsa.ale_convert --task-data third_party/agents-last-exam/task-data --out third_party/ale-cli
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil

ALE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "agents-last-exam")
GUI = ("Blender", "Chrome", "Edge", "GIMP", "LibreOffice", "QGIS", "Inkscape", "After", "Unreal", "Rhino", "KiCad", "FreeCAD",
       "Audacity", "OBS", "VLC", "Firefox", "Civil 3D", "PowerMill", "Moldex", "EnergyPlus", "OpenROAD", "Docker", "Minikube", "GAP", "Sabaki", "Flowable")


def grader_of(task_dir: str):
    for pat in ("verify*.py", "grade*.py", "evaluate*.py"):
        for f in sorted(glob.glob(os.path.join(task_dir, "scripts", pat))):
            src = open(f).read()
            if "cua_bench" in src:
                continue
            args = set(re.findall(r'add_argument\(\s*"(--[a-z\-]+)"', src))
            if {"--output-dir", "--reference-dir"} <= args:
                return f, args
    return None, set()


def eligible():
    out = []
    for tc in sorted(glob.glob(os.path.join(ALE, "tasks", "*", "*", "task_card.json"))):
        d = os.path.dirname(tc)
        card = json.load(open(tc))
        if card.get("category") == "demo":
            continue
        if any(g in " ".join(card.get("software", [])) for g in GUI):
            continue
        g, args = grader_of(d)
        if g:
            out.append((card["taskId"], d, g, args, card))
    return out


def convert(task_data: str, out: str, only: set[str] | None = None) -> list[str]:
    made = []
    for tid, d, g, args, card in eligible():
        if only and tid not in only:
            continue
        variant = "base"
        src = os.path.join(task_data, tid, variant)
        if not os.path.isdir(os.path.join(src, "input")) or not os.path.isdir(os.path.join(src, "reference")):
            continue
        name = tid.replace("/", "__")
        td = os.path.join(out, name)
        shutil.rmtree(td, ignore_errors=True)
        os.makedirs(os.path.join(td, "environment"), exist_ok=True)
        os.makedirs(os.path.join(td, "tests"), exist_ok=True)
        shutil.copytree(os.path.join(src, "input"), os.path.join(td, "environment", "input"))
        if os.path.isdir(os.path.join(src, "software")):
            shutil.copytree(os.path.join(src, "software"), os.path.join(td, "environment", "software"))
        shutil.copytree(os.path.join(src, "reference"), os.path.join(td, "tests", "reference"))
        shutil.copy(g, os.path.join(td, "tests", "grader.py"))
        open(os.path.join(td, "instruction.md"), "w").write(card["taskPrompt"].replace("`base/", "`/app/base/") + "\n\nAll paths are under /app (input: /app/base/input, output: /app/base/output).\n")
        open(os.path.join(td, "environment", "Dockerfile"), "w").write(
            "FROM python:3.12-slim\nRUN apt-get update && apt-get install -y --no-install-recommends curl git build-essential && rm -rf /var/lib/apt/lists/* \\\n"
            " && pip install --no-cache-dir uv numpy scipy pandas pyarrow\nWORKDIR /app\nCOPY input /app/base/input\n"
            + ("COPY software /app/base/software\n" if os.path.isdir(os.path.join(td, "environment", "software")) else "")
            + "RUN mkdir -p /app/base/output\n")
        extra = " --input-dir /app/base/input" if "--input-dir" in args else ""
        open(os.path.join(td, "tests", "test.sh"), "w").write(
            "#!/usr/bin/env bash\ncd /app\npython3 /tests/grader.py --output-dir /app/base/output --reference-dir /tests/reference" + extra +
            " > /tmp/grade.json 2>/tmp/grade.err; python3 - <<'PY'\nimport json\ntry:\n    d=json.load(open('/tmp/grade.json'))\n    s=d.get('normalized_score', d.get('score', 0.0))\nexcept Exception:\n    s=0.0\nopen('/logs/verifier/reward.txt','w').write(str(float(s)))\nprint('reward', s)\nPY\n")
        os.chmod(os.path.join(td, "tests", "test.sh"), 0o755)
        open(os.path.join(td, "task.toml"), "w").write(
            f'schema_version = "1.1"\n[task]\nname = "ale-cli/{name}"\ndescription = {json.dumps(card.get("summary", "")[:300])}\n'
            f'[verifier]\ntimeout_sec = 600.0\n[agent]\ntimeout_sec = 1800.0\n[environment]\nbuild_timeout_sec = 900.0\ncpus = 2\n')
        made.append(name)
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-data", default=os.path.join(ALE, "task-data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(ALE), "ale-cli"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for tid, d, g, args, card in eligible():
            print(tid, "|", os.path.basename(g), "|", sorted(args))
    else:
        made = convert(a.task_data, a.out)
        print(f"converted {len(made)} tasks into {a.out}")
