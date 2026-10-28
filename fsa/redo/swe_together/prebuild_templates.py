"""Build the E2B templates the judge needs, exactly as Harbor's trial path does.

    python -u -m fsa.redo.swe_together.prebuild_templates task1,task2 [--workers 3]

The judge (`eval/correctness/sandbox.py`) only looks templates up; they are
normally created by Harbor's E2B environment when a trial first runs
(`Template().from_image(<task.toml docker_image>)`, alias
`tb-<task>__<dirhash8(environment/)>`, cpus/memory from task.toml). Tasks judged
before any E2B trial (judge calibration, docker-only trials) need them built
first; this uses the identical recipe so trial and judge sandboxes match.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tomllib

from .config import CLONE


async def build(task: str, sem: asyncio.Semaphore) -> tuple[str, str]:
    from dirhash import dirhash
    from e2b import AsyncTemplate, Template

    td = CLONE / "tasks" / task
    env_cfg = tomllib.loads((td / "task.toml").read_text())["environment"]
    alias = f"tb-{task}__{dirhash(str(td / 'environment'), 'sha256')[:8]}".replace(".", "-")
    if await AsyncTemplate.alias_exists(alias):
        return task, f"exists {alias}"
    mem = str(env_cfg.get("memory", "4G")).upper()
    mem_mb = int(float(mem[:-1]) * 1024) if mem.endswith("G") else int(float(mem.rstrip("M")))
    async with sem:
        await AsyncTemplate.build(template=Template().from_image(image=env_cfg["docker_image"]), alias=alias,
                                  cpu_count=int(env_cfg.get("cpus", 2)), memory_mb=mem_mb)
    return task, f"built {alias} from {env_cfg['docker_image']}"


async def amain(tasks: list[str], workers: int) -> int:
    sys.path.insert(0, str(CLONE / "src"))
    from sandbox_config import load_dotenv
    load_dotenv(CLONE / ".env")
    sem = asyncio.Semaphore(workers)
    for r in await asyncio.gather(*(build(t, sem) for t in tasks), return_exceptions=True):
        print(r, flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    os.environ.setdefault("HARBOR_TEAM_PREFIX", "tb")
    return asyncio.run(amain(a.tasks.split(","), a.workers))


if __name__ == "__main__":
    sys.exit(main())
