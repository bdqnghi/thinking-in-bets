# SWE-Together local patch (redo_v1)

`redo_v1.patch` applies to https://github.com/Togetherbench/SWE-Together at
`891d19eb4b3a64a47c3d49bbd066a311e0133254` (clone: `third_party/swe-together`, git-ignored).
Write/verify with `python -m fsa.redo.swe_together.patchset write|verify|apply`; every run
manifest records the patch sha256 and refuses to start if the clone differs from it.

## Changes, by reason

| File | Change | Why |
|---|---|---|
| `src/llm_config.py` | registry entries `gemini-3.5-flash-lite`, `gemini-3.6-flash`; `to_opencode_model("gemini/X") -> "google/X"`; `codex_auth_path()` honours `SWT_CODEX_AUTH_JSON` | native Gemini agent seat under opencode (Harbor rejects provider `gemini`); API-key auth for the codex judge instead of the host's ChatGPT OAuth file |
| `src/user_agent/user_agent.py` | patient retry (10 s … 300 s, 9 retries) on 429/503/overload for the simulator call; per-call usage/latency/attempt records | a throttled simulator call used to become a no-op, and 4 no-ops end the session |
| `src/user_agent/agents/user_enabled_opencode.py` | opencode binary chosen by sandbox arch (`uname -m`); Gemini throttle markers (`resource_exhausted`, `unavailable`, `high demand`); `turns.jsonl` turn index; `user_sim_calls.jsonl` | native arm64 docker images; wrapper-level turn re-issue on Gemini backpressure; Stage 1 instrumentation |
| `src/user_agent/exec_helpers.py` | per-exec cap: backend-raised docker timeout -> `(_TimeoutResult, True)` (Stage 1.1) | a turn > 1800 s on docker ended the trial with an uncaught RuntimeError instead of cap-rescue |
| `src/user_agent/turn_snapshots.py` (new), `src/user_agent/repo_diff.py` | `snapshots/turn-N.{patch,json}` written on every `capture_git_diff` (always, even when empty) | per-turn repository snapshot hook (PLAN.md §6) |
| `eval/correctness/sandbox.py` | (unused since 2026-09-27: the codex seat is dropped) codex judge: auth file from `SWT_CODEX_AUTH_JSON`, `-c model_reasoning_effort=$SWT_CODEX_REASONING_EFFORT`, no claude install for codex; empty-patch phase-2 judging only when `SWT_JUDGE_ALLOW_EMPTY_PATCH=1` (calibration); keep judge stdout/stderr tails (codex "tokens used") | gpt-5.5 judge seat (recorded deviation from Claude Opus), calibration of the unmodified base |
| `external/harbor/.../e2b.py` | sandbox lifetime from `SWT_E2B_SANDBOX_TIMEOUT_SEC` (default 5400) | E2B hobby tier refuses > 1 h |
| `external/harbor/.../docker/docker.py` | `SWT_DOCKER_IMAGE_TEMPLATE` (e.g. `bdqnghi/swe-together:{task}`); `SWT_DOCKER_EXEC_USER=root`; teardown never `--rmi`s a prebuilt image | arm64 images as a config switch; E2B parity (execs as root); `compose down --rmi all` deleted an unpushed arm64 image during Stage 0 |

All switches default to upstream behaviour when their variable is unset.

## Revisions

| Revision | sha256 | Adds |
|---|---|---|
| Stage 0 (commit dbafb2d) | `b249f802d7ac…` | seats, patient retry, arm64 docker, snapshot hook, judge switches |
| Stage 1 (`136231a0` smoke) | `136231a0fcd6…` | arms, pushback overlay (below); the smoke root `results/redo_v1/stage1_smoke/W/` ran on the intermediate `4cd76229c9b0…` (no claim-book closing request: empty claim books) |
| Stage 1.1 (`redo_v1.patch`; pilot) | `b0f1d24c8de4…` (full hash in every run manifest, `patch_sha256`) | `src/user_agent/exec_helpers.py`: the docker backend's per-exec timeout (`RuntimeError("Command timed out after N seconds")`, Harbor `docker.py`) is treated as a cap like E2B's, so the wrapper's cap-rescue path runs instead of the trial ending with an exception; other RuntimeErrors still raise. Arms, overlay and judge unchanged |
| Baseline ports (`redo_v1_ports.patch`, 2026-09-26; not yet applied to the clone) | `90521169c6c0…` | Stage 1.1 plus the `tracker` and `mediator` arms (controlled ports of IDSS and Mediator-Assistant, below). Developed in a separate clone (`results/redo_v1/baseline_ports_work/swe-together`) so the running pilot's clone and `redo_v1.patch` stay untouched; select it with `SWT_CLONE_DIR` / `SWT_PATCH_FILE` (`fsa/redo/swe_together/config.py`). Stock and the Stage 1 arms behave exactly as before |
| Stage 1.2 frozen trigger (`redo_v1_trigger.patch`, 2026-09-28; overlay cells) | `b04264b7d34e…` | Stage 1.1 plus `SWT_REDO_TRIGGER=agentic_v3` in `src/user_agent/redo_arms.py` (below). Separate copy `results/redo_v1/trigger_v3_work/swe-together` (git-ignored; own venv, `harbor.pth` re-pointed to the copy). The pilot launcher runs overlay cells from it (`config.TRIGGER_V3`, `pilot --overlay-clone-dir/--overlay-patch-file`) and natural cells from the pinned pair; each launch verifies the revision its cell uses |
| Stage 1.3 endpoint switch (x86 replicate, 2026-09-28; all three files) | `redo_v1.patch` `239387b3a0dd…`, `redo_v1_trigger.patch` `e203e1c9d439…`, `redo_v1_ports.patch` `c2efc0062ee2…` | the previous revision of each file plus one switch in `src/user_agent/agents/user_enabled_opencode.py` (`render_opencode_config_command` / host-side config builder): `SWT_GOOGLE_BASE_URL` sets `provider.google.options.baseURL` of the agent's opencode config (only when the agent is already on the `google` provider), so the agent can reach a local Gemini-REST-compatible endpoint (`fsa/redo/machine.py` profiles). Unset (the public-API profile) the behaviour is identical to the previous revision. First used by the x86 replicate ([notes](../../../docs/results/REDO_X86_REPLICATE.md)); the arm64 runs used the earlier revisions |
| Baseline ports + frozen trigger (`redo_v1_ports_trigger.patch`, 2026-10-02; overlay cells of §8 arms) | `ab75be5360a5…` | `redo_v1_ports.patch` (`c2efc0062ee2…`) plus the 19-line `SWT_REDO_TRIGGER=agentic_v3` addition from `redo_v1_trigger.patch` in `src/user_agent/redo_arms.py`. Separate copy `results/redo_v1/ports_trigger_v3_work/swe-together` (git-ignored; own venv, `harbor.pth` re-pointed to the copy), passed via `--overlay-clone-dir` / `--overlay-patch-file` for §8 overlay cells |

## Gemini-only seats (2026-09-27): no new revision

The OpenAI key has no credit (`results/redo_v1/OPENAI_CREDIT_STOP_20260926.json`); the user decided to
run on Gemini only. **The clone and `redo_v1.patch` are unchanged** (the running pilot re-verifies the
patch hash before every trial launch):

* **Overlay trigger.** The clone's `Overlay._load_judge` builds `fsa.redo.intent_judge.IntentJudge(cache,
  workers=4)` from `SWT_REDO_TIB_ROOT` with no model argument. The model and reasoning setting now come from
  the tib host config `configs/seats/redo_v1_seats.json` (judge seat: `gemini-3.5-flash`, `high`;
  `REDO_JUDGE_MODEL` / `REDO_JUDGE_EFFORT` override), read when the judge is built at trial start, and the
  call goes to Gemini's OpenAI-compatible endpoint with `GEMINI_API_KEY` (already in the trial
  environment). `overlay_judge/turn-N.json` records the model (`judge.model`). Model and effort are part
  of every cache key, so gemini verdicts never mix with the gpt-5.5 ones in the shared cache. Verified in
  the smoke `results/redo_v1/gemini_seats_smoke/swe_together/W_stock_overlay/` (the trigger judged with
  `gemini-3.5-flash` and fired twice).
* **Correctness judge (codex/gpt-5.5) dropped.** `rungs.json` marks the seat `dropped`; `run judge` passes
  the benchmark's own `--skip-correctness` (tagger and intent coverage still run on Gemini), and the trial
  environment no longer carries `OPENAI_API_KEY` or the codex switches. The patch's codex switches in
  `eval/correctness/sandbox.py` stay in the clone, unused. Final pass: `fsa.redo.metrics` (verifier reward
  + frozen per-intent judge).
* Every new manifest records `judge_seats` (seat file sha256, provider record, `JUDGE_FREEZE.json` digest)
  and the dropped judge seat.

## Stage 1.2: the frozen overlay trigger (2026-09-28)

`redo_v1_trigger.patch` = `redo_v1.patch` plus one change in `src/user_agent/redo_arms.py`:

* `TRIGGERS = ("v1", "agentic_v3")`; `RedoConfig.trigger` from `SWT_REDO_TRIGGER` (default `v1`, unknown
  values refused).
* `Overlay._load_judge`: with `agentic_v3`, `self._judge = self._ij = OverlayTrigger(self.task_dir)`, imported
  from the tib checkout (`fsa.redo.swe_together.agentic_judge`), which loads
  `configs/intents/swe-together/OVERLAY_TRIGGER_FREEZE.json` (and refuses unless it matches the judge freeze).
  Units are the frozen judge's goals; "revealed" follows each goal's turn of origin; each consultation runs
  the cheap frozen configuration (`gemini-3.8-flash`, `low`, 16 steps, 600 s) in one task container, taken
  from the host-wide v3 judge slots (`results/redo_v1/agentic_judge_cache/.slots`, 3).
* `overlay_judge/turn-N.json` records `judge.trigger_freeze_sha256` and the consultation's spend
  (`judge_stats_delta`: calls, failed attempts, input / output / reasoning / cached tokens, containers,
  wall time), which `fsa.redo.metrics` counts as the in-trial judge seat.

Everything else is byte-identical to Stage 1.1; with `SWT_REDO_TRIGGER` unset the copy behaves as the pinned
clone. `fsa.redo.swe_together.run` refuses an `agentic_v3` overlay on a revision without `TRIGGERS`, so an
overlay cell cannot silently fall back to the v1 trigger.

## Stage 1: arms and the pushback overlay

New: `src/user_agent/redo_arms.py`, prompts in `src/user_agent/prompts/redo/`; hooks in
`src/user_agent/agents/user_enabled_opencode.py` (`_redo_step`, called from the turn loop in
place of the plain simulator consultation when an arm or the overlay is selected). Selection
is by environment (`fsa.redo.swe_together.run.redo_env`); with nothing set (stock) the turn
loop, the resume messages and `turns.jsonl` are exactly the Stage 0 ones.

| Arm (`SWT_REDO_ARM`) | What the agent receives with a real simulator reply | Extra model call |
|---|---|---|
| `stock` | the reply | none |
| `commit_before` (ours, PLAN §7) | reply + claim book + rules (`rules_p.md`, tau = 0.5) | claim book (`claim_book_system.md`) **before** the simulator is consulted; `claims/turn-N.json` exists before the reply is produced |
| `claims_after` | same | identical call made **after** the reply is known; the reply is appended to the call's input (the only prompt difference) |
| `claims_no_p` | reply + claim book without p + `rules_no_p.md` | before, `claim_book_system_no_p.md` |
| `recap` | reply + every user requirement so far (`recap.md`) | none |
| `minimal_edit` | reply + `minimal_edit.md` | none |

On a silent (no-op) turn every arm sends the benchmark's plain `continue`; claim-book arms
still write the book (in `claims_after` the call is then identical to `commit_before`'s).
The claim-book call is one host-side LiteLLM call to the agent's own model
(`gemini/<rung model>`, `reasoning_effort=high` = thinkingLevel high, JSON output) over a
transcript parsed from `opencode.txt.turn-*`, the current cumulative diff and the previous
book; patient retry (10 s ... 300 s), every attempt in `agent/claim_calls.jsonl` (its own
token seat). Nothing carries across tasks. The simulator never sees the book: it is
consulted with the benchmark's own inputs, and only the delivered agent message carries it
(`turns.jsonl` `message` = the reply, `redo.agent_message` = what the agent received).

**Pushback overlay** (`SWT_REDO_OVERLAY=1`, seed `SWT_REDO_OVERLAY_SEED`): in parallel with
each simulator consultation, the per-intent judge (`fsa/redo/intent_judge.py`, imported from
`SWT_REDO_TIB_ROOT`, cache shared with offline judging) labels snapshot N on the intents
revealed by N (`overlay_judge/turn-N.json`). At the first consultation where every revealed
applicable intent is satisfied the simulator's reply is replaced by an unjustified
correction (`overlay_templates.json`); at a later such consultation by a mis-aimed one
naming a satisfied intent. At most two per session. The overlay message replaces the
simulator's last reply in its own history (and un-consumes its ground-truth cursor), is
logged as `opened_by="overlay"` in `turns.jsonl` and `episode-N/overlay.json`, and does not
count toward the no-op streak (an overlay can therefore pre-empt a session end).

`turns.jsonl` gains a `redo` object on arm/overlay trials: `sim_consult_started_ts`,
`sim_decided_ts`, `claims_call_started_ts`, `claims_written_ts`, `claims_phase`,
`overlay_*`, `arm_block_delivered`.

## Baseline ports (`redo_v1_ports.patch`)

New: `src/user_agent/redo_ports.py` (byte-identical copy of tib's `fsa/redo/ports.py`) and its prompts
`src/user_agent/prompts/ports/` (byte-identical to `fsa/redo/prompts/ports/`, incl. the frozen
Refiner experiences `mediator_experiences.json`); `redo_arms.py`: arms `tracker`, `mediator`,
`RedoArms.port_update`; `user_enabled_opencode.py::_redo_step`: the port call after the simulator
decided, for real replies only, then `compose`. Faithfulness and deviations:
[docs/results/REDO_BASELINE_PORTS.md](../../../docs/results/REDO_BASELINE_PORTS.md).

| Arm | What the agent receives with a real simulator reply | Extra model call (after the reply is known) |
|---|---|---|
| `tracker` (controlled port of IDSS, arXiv 2608.15755) | reply + situation state (fact layer + state layer) + IDSS behavioural guidelines (`tracker_block.md`) | state-layer update (`tracker_system.md`, JSON); fact layer parsed deterministically from `opencode.txt.turn-*` and the cumulative diff; constraint propagation deterministic; `ports/turn-N.json` |
| `mediator` (controlled port of Mediator-Assistant, arXiv 2602.07338) | the mediator's rewrite **instead of** the reply | the paper's Mediator prompt (`mediator_prompt.md`) with the rung's frozen Refiner experiences; `ports/turn-N.json` |

Both calls: host LiteLLM, the agent's own model (`gemini/<rung model>`, `reasoning_effort=high`),
patient retry, every attempt in `agent/port_calls.jsonl` (own seat). A failed call passes the reply
through unchanged. `turns.jsonl` `redo` gains `port_doc`, `port_ok`, `port_phase`,
`port_call_started_ts`, `port_written_ts`. "continue" (no-op) turns make no call.

To adopt it after the running pilot has finished: `git -C third_party/swe-together checkout .` and
remove the untracked Stage 1 files, then `git apply redo_v1_ports.patch` (or copy the work clone's
files), and point `PATCH_FILE` at the new file (or rename it to `redo_v1.patch` in a commit).

## Integration points (opencode harness)

Turn loop: `src/user_agent/agents/user_enabled_opencode.py::UserEnabledOpenCode.run`.
The agent turn ends at `_capture_git_diff` (turn 0 in the `finally` after the initial run,
resume turns in the loop's `finally`); the simulator is consulted at the top of the next
loop iteration (`_consult_user` -> `UserAgent.process`, `src/user_agent/user_agent.py`); the
reply is appended through `_build_resume_command` (`opencode run --session`). Snapshot *t*
= state after agent turn *t*, before the simulator sees it (`snapshots/turn-t.patch`).
