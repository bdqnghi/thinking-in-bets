# Standard Agent vs. Thinking-in-Bets Agent — Interactive Demo

Pick a coding task and compare the two agents side by side:
- **Left column:** Standard agent (evaluates user criticism after reading it).
- **Right column:** Thinking-in-Bets agent (records a private Claim Checklist before reading the reviewer's reply).

Both agents solve the same repository task, receive the exact same reviewer pushback, and are graded by the same hidden test suite.

## Replay mode (zero setup — opens directly in your browser)

Open [`tib_demo.html`](tib_demo.html) directly in any browser (no server, Docker, or API key required) to step through recorded trajectories from `SWE-Pushback-Hidden` at adjustable playback speeds.

To rebuild `cases.js` and `tib_demo.html` from local `SWE-Pushback-Hidden` trial directories:

```bash
python demo/build.py
```

## Live mode (run real Docker trials side by side)

```bash
export GEMINI_API_KEY="your-api-key"
python demo/live_server.py --port 8791
# Then open http://localhost:8791/
```

Live mode runs the `SWE-Pushback-Hidden` harness (`fsa/redo/swe_pushback/agent.py`): two Docker containers, your configured Gemini / OpenAI-compatible API endpoint, the frozen reviewer objection from `results/swe_pushback/datasets/objections_v1.jsonl`, and the task's hidden-test verifier image. Every step is streamed to the browser via Server-Sent Events and saved under `results/demo_live/<run-id>/`.

- A full live run takes ~6–12 minutes per task (step budget 75, up to three review rounds).
- Append `?run=<id>` to the URL to re-attach to an in-progress or completed run.
