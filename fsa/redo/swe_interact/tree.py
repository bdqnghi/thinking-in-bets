"""Apply the redo_v1 overlay to the SWE-Interact clone (after the arm64 patch).

    python -m fsa.redo.swe_interact.tree apply   # idempotent; then `patchset write`

Per multi-turn task (all 75, so the pilot can scale without a new patch):

* ``environment/user-server/server.py``  <- ``overlay/server.py`` (instrumentation:
  per-ask_user repository snapshot, usage records, optional patient retry;
  upstream server.py is byte-identical across the 75 tasks);
* ``environment/user-server/Dockerfile``: ``fastmcp>=3.0 litellm>=1.80.8`` pinned to
  the releases current at the benchmark commit (fastmcp 3.4.2, litellm 1.90.1);
* ``environment/docker-compose.yaml``: the user-server gets ``SWI_REDO_DIR=/redo``,
  ``SWI_SIM_RETRY``, the Stage 1 arm selection (``SWI_REDO_ARM``, ``SWI_REDO_OVERLAY``,
  ``SWI_REDO_RPC_TIMEOUT``; defaults = stock, no overlay) and a bind mount ``<trial>/user-sim -> /redo``; the project's default
  network gets an explicit subnet ``SWI_SUBNET`` (allocated by the runner), because the host's
  predefined docker address pools are exhausted by other jobs' networks;
* ``rf_*`` tasks: ``evaluate_rubrics.py`` (both copies) <- ``overlay/evaluate_rubrics.py``
  and the final step's verifier env passes ``EVAL_MAX_COMPLETION_TOKENS`` /
  ``EVAL_REASONING_EFFORT``.
"""
from __future__ import annotations

import argparse
import sys

from .config import OVERLAY, TASKS

PIP_OLD = 'RUN pip install --no-cache-dir "fastmcp>=3.0" "litellm>=1.80.8"'
PIP_NEW = ('# [tib redo_v1] pinned to the releases current at the benchmark commit (2026-06-30)\n'
           'RUN pip install --no-cache-dir "fastmcp==3.4.2" "litellm==1.90.1"')
ENV_ANCHOR = "      - SIM_USER_MODEL=${SIM_USER_MODEL:-}\n"
ENV_ADD = ("      - SWI_REDO_DIR=/redo\n"
           "      - SWI_SIM_RETRY=${SWI_SIM_RETRY:-0}\n"
           "    volumes:\n"
           "      # [tib redo_v1] <trial>/user-sim on the host (local docker only)\n"
           "      - ${HOST_AGENT_LOGS_PATH:?tib_redo_v1_needs_the_docker_agent_log_mount}/../user-sim:/redo\n")
NET_ADD = ("\n# [tib redo_v1] explicit per-trial subnet (runner-allocated 10.213.N.0/24): the host's predefined\n"
           "# docker address pools are exhausted by other jobs' compose networks\n"
           "networks:\n"
           "  default:\n"
           "    ipam:\n"
           "      config:\n"
           "        - subnet: ${SWI_SUBNET:?tib_redo_v1_needs_a_runner_allocated_SWI_SUBNET}\n")
RETRY_LINE = "      - SWI_SIM_RETRY=${SWI_SIM_RETRY:-0}\n"
ARM_ADD = ("      - SWI_REDO_ARM=${SWI_REDO_ARM:-stock}\n"
           "      - SWI_REDO_OVERLAY=${SWI_REDO_OVERLAY:-0}\n"
           "      - SWI_REDO_RPC_TIMEOUT=${SWI_REDO_RPC_TIMEOUT:-3000}\n")
TOML_ANCHOR = 'EVAL_MODEL = "${EVAL_MODEL:-anthropic/claude-opus-4-5-20251101}"\n'
TOML_ADD = ('EVAL_MAX_COMPLETION_TOKENS = "${EVAL_MAX_COMPLETION_TOKENS:-}"\n'
            'EVAL_REASONING_EFFORT = "${EVAL_REASONING_EFFORT:-}"\n')


def apply() -> dict:
    server = (OVERLAY / "server.py").read_text()
    rubric = (OVERLAY / "evaluate_rubrics.py").read_text()
    n = {"tasks": 0, "rf": 0}
    for td in sorted(p for p in TASKS.iterdir() if (p / "task.toml").exists()):
        us = td / "environment" / "user-server"
        (us / "server.py").write_text(server)
        df = (us / "Dockerfile").read_text()
        if PIP_OLD in df:
            (us / "Dockerfile").write_text(df.replace(PIP_OLD, PIP_NEW))
        elif PIP_NEW not in df:
            raise SystemExit(f"{td.name}: unexpected user-server Dockerfile")
        dc = td / "environment" / "docker-compose.yaml"
        text = dc.read_text()
        if "SWI_REDO_DIR" not in text:
            if text.count(ENV_ANCHOR) != 1:
                raise SystemExit(f"{td.name}: compose anchor not found")
            text = text.replace(ENV_ANCHOR, ENV_ANCHOR + ENV_ADD)
        if "SWI_REDO_ARM" not in text:  # Stage 1: arm / overlay selection for the sidecar hooks
            if text.count(RETRY_LINE) != 1:
                raise SystemExit(f"{td.name}: compose SWI_SIM_RETRY line not found")
            text = text.replace(RETRY_LINE, RETRY_LINE + ARM_ADD)
        if "SWI_SUBNET" not in text:
            text = text.rstrip("\n") + "\n" + NET_ADD
        dc.write_text(text)
        if td.name.startswith("rf_"):
            for p in (td / "tests" / "evaluate_rubrics.py", td / "steps" / "05_test_handoff" / "tests" / "evaluate_rubrics.py"):
                if p.exists():
                    p.write_text(rubric)
            tt = td / "task.toml"
            t = tt.read_text()
            if "EVAL_MAX_COMPLETION_TOKENS" not in t:
                if t.count(TOML_ANCHOR) != 1:
                    raise SystemExit(f"{td.name}: task.toml EVAL_MODEL anchor not found")
                tt.write_text(t.replace(TOML_ANCHOR, TOML_ANCHOR + TOML_ADD))
            n["rf"] += 1
        n["tasks"] += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["apply"])
    ap.parse_args(argv)
    print(apply())
    return 0


if __name__ == "__main__":
    sys.exit(main())
