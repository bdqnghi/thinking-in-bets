# Benchmarks & Reproducing Paper Results

This guide summarizes the experimental results in *Thinking-in-Bets at Test Time Reduces Sycophantic Revisions in LLM Agents* ([arXiv:2610.34654](https://arxiv.org/abs/2610.34654)) and lists the commands to run each evaluation pipeline.

---

## Setup

Install the optional benchmark and test dependencies:

```bash
pip install -e ".[benchmarks,dev]"
export GEMINI_API_KEY="your-gemini-api-key"
```

Run the offline verification suite (289 unit tests covering checklist parsers, prompt hashes, gating state machines, and scorers; no API key or Docker required):

```bash
pytest -q
```

---

## Summary of Main Results (Paper Table 2)

### A. Single-Turn Review Under User Criticism (Base $\to$ TiB)

| Benchmark & Metric | 3.5 Flash-Lite | 3.5 Flash | 3.6 Flash | 3.8 Flash | 4.0 Argon |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SWE-Together Patch Review** — False reversals (%) $\downarrow$ | 28.41 $\to$ **0.00** | 22.60 $\to$ **0.23** | 13.20 $\to$ **0.00** | 12.67 $\to$ **0.00** | 8.89 $\to$ **0.00** |
| **SWE-Interact Patch Review** — False reversals (%) $\downarrow$ | 8.56 $\to$ **0.00** | 14.71 $\to$ **0.00** | 13.27 $\to$ **0.00** | 6.60 $\to$ **0.00** | 10.36 $\to$ **0.00** |
| **SycoBench-600 Multi-Round** — False reversals (%) $\downarrow$ | 3.73 $\to$ **0.20** | 1.06 $\to$ **0.04** | 0.96 $\to$ **0.02** | 0.76 $\to$ **0.00** | 3.52 $\to$ **1.70** |
| **SycEval-1000 Open-Ended** — False reversals (%) $\downarrow$ | 17.3 $\to$ **8.3** | 9.4 $\to$ **5.4** | 10.8 $\to$ **5.8** | 12.9 $\to$ **6.4** | 8.5 $\to$ **3.6** |
| **Agents' Last Exam Review** — Post-review pass (%) $\uparrow$ | 30.0 $\to$ **95.0** | 25.0 $\to$ **75.0** | 60.0 $\to$ **90.0** | 80.0 $\to$ **95.0** | 95.0 $\to$ 90.0 |
| **Humanity's Last Exam Review** — False reversals (%) $\downarrow$ | 70.2 $\to$ **14.9** | 66.2 $\to$ **22.5** | 35.0 $\to$ **12.5** | 20.0 $\to$ **6.2** | 7.5 $\to$ **6.2** |
| **HLE Validity Pool (476 items)** — Balanced accuracy (%) $\uparrow$ | 56.9 $\to$ **63.0** | 69.1 $\to$ **76.1** | 70.4 $\to$ **76.1** | 71.6 $\to$ **79.0** | — |

### B. Multi-Turn Interactive Coding in Docker Containers

- **Mistaken user criticism (`SWE-Together` & `SWE-Interact`):** Cuts unnecessary source-file edits by **38–60%** (`66.7% -> 27.8%` on `SWE-Together` and `80.0% -> 50.0%` on `SWE-Interact` with `3.8 Flash`; `57.9% -> 22.9%` on `SWE-Interact` with `4.0 Argon`) while raising task pass rates (`55.0% -> 70.0%` on `3.8 Flash`, `30.0% -> 35.0%` on `4.0 Argon`).
- **Hidden-test review (`SWE-Pushback-Hidden` on `DeepSWE`):** Drops false reversals (`17.0% -> 4.2%` on `3.5 Flash`, `16.7% -> 0.0%` on `3.8 Flash`) while lifting final task pass rates (`28.2% -> 38.5%` and `13.5% -> 24.5%`).
- **Mixed valid & mistaken feedback (`On-Demand Betting`):** Reduces checklist tokens by **~85%** while boosting successful bug corrections on `SWE-Interact` (`3.8 Flash`) from **52.2% to 73.9%** (balanced accuracy **0.735 $\to$ 0.837**).

### C. Non-Interactive Single-Pass Tasks (`Draft-First Betting`, Base $\to$ TiB)

| Benchmark & Metric | 3.5 Flash-Lite | 3.5 Flash | 3.8 Flash | 4.0 Argon |
| :--- | :---: | :---: | :---: | :---: |
| **Humanity's Last Exam** — Answer accuracy (%) $\uparrow$ | 9.4 $\to$ **14.4** | 30.8 $\to$ **35.0** | 33.0 $\to$ **37.6** | 45.6 $\to$ **46.6** |
| **Humanity's Last Exam** — Brier calibration error $\downarrow$ | 0.704 $\to$ **0.472** | 0.580 $\to$ **0.488** | 0.549 $\to$ **0.440** | 0.438 $\to$ **0.344** |
| **Vals Finance Agent** — Benchmark score (%) $\uparrow$ | 8.4 $\to$ **11.1** | 20.2 $\to$ **22.6** | 29.0 $\to$ **34.2** | 44.6 $\to$ **52.5** |
| **DeepSWE (OpenCode, 113 tasks)** — Task pass (%) $\uparrow$ | — | — | 52.2 $\to$ **54.9** | 58.4 $\to$ **66.4** |
| **DeepSWE (Mini-SWE-Agent, 112 tasks)** — Task pass (%) $\uparrow$ | — | — | — | 73.2 $\to$ **79.5** |

---

## Running the Benchmarks

### 1. Single-Turn Patch, Deliverable, & Sycophancy Suites (Paper §4.2–§4.3)

```bash
# Single-turn patch review on SWE-Together & SWE-Interact snapshots
python -m fsa.redo.e2.run \
  --model gemini-3.5-flash \
  --items results/redo_v1/e2_items.json \
  --out results/e2/gemini_3_5_flash \
  --preset wave2 \
  --workers 8

# Agents' Last Exam deliverable review (requires ../agents-last-exam checkout; see third_party/README.md)
python -m fsa.redo.ale_pilot \
  --model gemini-3.5-flash \
  --effort low \
  --out-dir results/ale_review

# SycoBench-600 & SycEval-1000
python scripts/sycophancy_benchmarks.py run \
  --bench sycobench \
  --model gemini-3.5-flash \
  --out-dir results/sycophancy/wave1 \
  --effort low

python scripts/sycophancy_benchmarks.py run \
  --bench syceval \
  --model gemini-3.5-flash \
  --out-dir results/sycophancy/wave1 \
  --effort low
```

### 2. Multi-Turn Interactive Coding in Docker (Paper §4.4–§4.5)

Clone and patch the pinned upstream harnesses (see [`third_party/README.md`](../third_party/README.md)):

```bash
git clone https://github.com/Togetherbench/SWE-Together.git third_party/swe-together
git -C third_party/swe-together checkout 891d19e
python -m fsa.redo.swe_together.patchset apply

git clone https://github.com/scaleapi/SWE-Interact.git third_party/swe-interact
git -C third_party/swe-interact checkout b32f98c
python -m fsa.redo.swe_interact.patchset apply
```

Launch interactive multi-turn runs:

```bash
# SWE-Together
python -m fsa.redo.swe_together.run --rung F --arm commit_before --sandbox docker --workers 4

# SWE-Interact
python -m fsa.redo.swe_interact.run --rung F --arm commit_before --workers 4

# SWE-Pushback-Hidden (110 DeepSWE tasks with frozen valid & invalid reviewer objections)
python -m fsa.redo.swe_pushback.run \
  --model gemini-3.5-flash \
  --tag F \
  --objections results/swe_pushback/datasets/objections_v1.jsonl \
  --out-dir results/swe_pushback/F_low \
  --workers 4
```

### 3. Single-Pass Reasoning, Finance, & Container Agents (`Draft-First`, Paper §4.6)

```bash
# Humanity's Last Exam — Draft-First single-shot accuracy & calibration + 2-turn peer pushback
python scripts/hle_benchmark.py \
  --mode single_shot \
  --model gemini-3.8-flash \
  --arms stock,cb_draftfirst,self_refine,cove,sc_matched \
  --out-dir results/hle_runs

# Vals Finance Agent — 4-tool live SEC EDGAR retrieval with Draft-First pre-submission gate
python scripts/finance_agent_tools_benchmark.py \
  --model gemini-3.8-flash \
  --arms stock,cb_draftfirst,self_refine,cove \
  --out-dir results/finance_tools

# Harbor container task collector (DeepSWE / Terminal-Bench / ALE container tasks)
python scripts/harbor_collect.py --help
```

> **Dataset Terms Note:** Raw *Humanity's Last Exam* (`cais/hle`) items and model outputs are never redistributed in this repository in accordance with the HLE non-redistribution policy; `scripts/hle_benchmark.py` downloads `cais/hle` directly from Hugging Face using your authenticated `HF_TOKEN`.
