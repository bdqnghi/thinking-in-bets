# External Benchmark Harnesses & Patches

Interactive container benchmarks are pinned to their official upstream repositories, with deterministic patches under `third_party/patches/` that wire in the Thinking-in-Bets schedules (`commit_before`, `cb_gated`, `claims_after`, `claims_no_p`) and comparison baselines (`verify_then_respond`, `conf_gate`, `sc_vote`, Situation Tracker, Intent Mediator).

| Local directory | Upstream repository | Pinned commit | Role in paper |
| :--- | :--- | :--- | :--- |
| `third_party/swe-together/` | [Togetherbench/SWE-Together](https://github.com/Togetherbench/SWE-Together) | `891d19e` | Multi-turn interactive coding & single-turn patch review (`fsa/redo/swe_together/`, `third_party/patches/swe-together/`) |
| `third_party/swe-interact/` | [scaleapi/SWE-Interact](https://github.com/scaleapi/SWE-Interact) | `b32f98c` | Multi-turn interactive coding with mixed/mistaken feedback (`fsa/redo/swe_interact/`, `third_party/patches/swe-interact/`) |
| `../agents-last-exam/` | [rdi-berkeley/agents-last-exam](https://github.com/rdi-berkeley/agents-last-exam) | latest release | Deliverable review (`fsa/redo/ale_pilot.py`) & container tasks (`third_party/patches/ale_claw_tib.patch`, `configs/ale/`) |

## Setup for SWE-Together and SWE-Interact

```bash
# 1. Clone pinned upstream harnesses
git clone https://github.com/Togetherbench/SWE-Together.git third_party/swe-together
git -C third_party/swe-together checkout 891d19e

git clone https://github.com/scaleapi/SWE-Interact.git third_party/swe-interact
git -C third_party/swe-interact checkout b32f98c

# 2. Apply and verify the Thinking-in-Bets harness patches
python -m fsa.redo.swe_together.patchset apply
python -m fsa.redo.swe_interact.patchset apply
```

For `arm64` hosts, pre-built native `arm64` task images are available on Docker Hub under `bdqnghi/swe-together:<task>` and `bdqnghi/swe-interact:<task>` (build scripts at [bdqnghi/swe_benchmark_arm](https://github.com/bdqnghi/swe_benchmark_arm)).
