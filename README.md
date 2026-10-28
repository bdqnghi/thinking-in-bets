<div align="center">

# Thinking-in-Bets at Test Time Reduces Sycophantic Revisions in LLM Agents

[![Project Page](https://img.shields.io/badge/Project-Website-0f766e?style=flat-square)](https://thinking-in-bets.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2610.34654-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.34654)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-dc2626?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2610.34654)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-2563eb?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-475569?style=flat-square)](LICENSE)

[Overview](#overview) · [Method](#method) · [Quickstart](#quickstart) · [Interactive Demo](#interactive-demo) · [Documentation](#documentation) · [Citation](#citation)

</div>

---

<p align="center">
  <img src="assets/tib-motivation.png" alt="Why agents cave to mistaken user feedback — and how Thinking-in-Bets fixes it" width="96%"/>
</p>

## Overview

Interactive LLM agents often abandon working code, accurate reports, and valid derivations the moment a user pushes back—even when the complaint is vague or mistaken (**false reversals** / agentic sycophancy). Static prompt rules telling an agent to *"stand your ground"* swing to the opposite extreme, causing it to ignore genuine bug reports (**missed corrections**).

**Thinking-in-Bets (TiB)** resolves this trade-off **at test time without model training or weight access**. Before reading user feedback, the agent records a private **Claim Checklist** mapping each key decision or assumption to concrete evidence and a calibrated probability $p_i \in [0, 1]$, alongside ruled-out options. When feedback arrives, only the targeted claims are re-evaluated against their sealed evidence and bet strength—preventing hindsight rationalization while remaining responsive to real errors.

- **Eliminates sycophantic caving:** Cuts false reversals in single-turn code patch review from **12.2% to 0.02%** across five models without hurting bug-fix rates, and lifts post-review passing report drafts from **58% to 89%** (vs. **65%** when recording *after* reading the complaint).
- **Lightweight gating:** **On-Demand Betting** runs a quick confidence check ($\tau_{\text{gate}} = 0.9$) and writes a full checklist only when uncertain, cutting checklist tokens by **~85%** while fixing more real bugs in multi-turn coding.
- **Single-pass self-audit:** **Draft-First Betting** applies the same pre-commit checklist and weakest-link audit to non-interactive reasoning, finance, and software-engineering tasks, improving both accuracy and confidence calibration.

---

## Method

<p align="center">
  <img src="assets/tib-method-overview-revised.png" alt="Overview of Always-On, On-Demand, and Draft-First Betting" width="96%"/>
</p>

| Schedule | SDK Entry Point | Thresholds | Recommended Setting |
| :--- | :--- | :--- | :--- |
| **Always-On Betting** | `AlwaysOnSession` | $\tau = 0.5$ | Multi-turn review, specification refinement, or adversarial pushback. Seals a Claim Checklist before every user reply. |
| **On-Demand Betting** | `OnDemandSession` | $\tau_{\text{gate}} = 0.9,\; \tau = 0.5$ | Cost- and latency-sensitive interactive agents. Writes a full checklist only when overall confidence $p_{\text{gate}} < 0.9$. |
| **Draft-First Betting** | `draft_first_solve` | $\tau_{\text{audit}} = 0.85$ / $0.95$ | Single-pass tasks without user feedback. Drafts a solution, scores load-bearing claims, and audits $\arg\min_i p_i$ before submitting. |

See [`docs/PROMPTS.md`](docs/PROMPTS.md) for the Claim Checklist JSON format, the 2×2 revision rule, and drop-in prompt templates for custom agent harnesses.

---

## Quickstart

### Installation

```bash
git clone https://github.com/bdqnghi/thinking-in-bets.git
cd thinking-in-bets
pip install -e .
```

Set a Gemini API key (Google AI Studio's OpenAI-compatible endpoint is used automatically for `gemini-*` models) or any OpenAI-compatible endpoint:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
# Or: export OPENAI_API_KEY="..." OPENAI_BASE_URL="https://api.openai.com/v1"
```

### Multi-Turn Interactive Session (`AlwaysOnSession` / `OnDemandSession`)

```python
from tib import AlwaysOnSession, OnDemandSession

# Swap in OnDemandSession(model="gemini-2.5-flash", tau_gate=0.9) for gated checklist generation
session = AlwaysOnSession(model="gemini-2.5-flash", tau=0.5)

# Turn 1: Solve the task and privately seal the Claim Checklist before feedback arrives
reply_1 = session.start(
    "Write `dedup_preserve_order(items)` in Python to deduplicate an iterable "
    "while preserving first-seen order, including unhashable dicts."
)
print(session.checklist.render())

# Turn 2: Mistaken pushback -> adjudicated against sealed bets (holds ground & asks what failed)
reply_2 = session.respond("This breaks on repeated elements. Rewrite it using `list(set(items))`.")

# Turn 3: Specific valid requirement -> updates only the targeted claim
reply_3 = session.respond("Dicts with identical key-value pairs in different key orders should count as duplicates.")
```

### Single-Pass Reasoning (`draft_first_solve`)

```python
from tib import draft_first_solve

res = draft_first_solve(
    "A fair coin is flipped 10 times. Given at least 8 heads, what is the exact probability of 10 heads?",
    model="gemini-2.5-flash",
)
print(res["response"])
print("Weakest claim p:", res["min_claim_p"], "| Calibrated confidence:", res["confidence"])
```

---

## Interactive Demo

Compare a **Standard Agent** and a **Thinking-in-Bets Agent** side by side on repository coding tasks with hidden unit tests:

- **Browser replay (zero setup):** Open [`demo/tib_demo.html`](demo/tib_demo.html) directly in any browser.
- **Live Docker run:** `python demo/live_server.py --port 8791` and open `http://localhost:8791/` (details in [`demo/README.md`](demo/README.md)).

---

## Documentation

- **[`docs/PROMPTS.md`](docs/PROMPTS.md)** — Claim Checklist schema, 2×2 selective revision matrix, and copy-pasteable prompt templates for custom harnesses.
- **[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)** — Summary of paper results and commands to reproduce all 8 benchmarks (`SWE-Together`, `SWE-Interact`, `SWE-Pushback-Hidden` / `DeepSWE`, `Agents' Last Exam`, `Humanity's Last Exam`, `SycoBench-600`, `SycEval-1000`, `Vals Finance Agent`, and `Terminal-Bench`).
- **[`demo/README.md`](demo/README.md)** — Side-by-side trajectory replay and live Docker demo server.
- **[`third_party/README.md`](third_party/README.md)** — Upstream container harness pins and patch application guide.

### Repository Layout

```text
thinking-in-bets/
├── tib/            # Standalone Python SDK (ClaimChecklist, prompts, AlwaysOn/OnDemand/DraftFirst)
├── fsa/            # Paper experiment harnesses, sidecar gates, and container scorers
├── scripts/        # Standalone benchmark runners & analysis scripts
├── configs/        # Frozen task subsets, rubric intents, and Harbor/ALE configs
├── demo/           # Side-by-side browser replay & live Docker demo
├── third_party/    # Upstream harness pins & deterministic patches
├── docs/           # Prompt templates (PROMPTS.md) & benchmark guide (BENCHMARKS.md)
└── tests/          # Offline unit test suite (`pytest -q`)
```

---

## Citation

```bibtex
@article{bui2026thinkinginbets,
  title   = {Thinking-in-Bets at Test Time Reduces Sycophantic Revisions in LLM Agents},
  author  = {Bui, Nghi D. Q.},
  journal = {arXiv preprint arXiv:2610.34654},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.34654}
}
```

## License

Released under the [Apache License 2.0](LICENSE). External benchmarks referenced in [`third_party/`](third_party/README.md) remain subject to their respective upstream licenses and dataset terms.
