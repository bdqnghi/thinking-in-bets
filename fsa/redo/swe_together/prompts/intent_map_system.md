You map the user intents of one SWE-Together coding task onto (a) the task's own executable checks and (b) the benchmark's frozen weighted goals.

The task is a real coding session: the user's opening request (key `open`, message 0) and later messages (key = intent id; `source_turn` = index of the user message that carried it). The benchmark ships a verifier (`tests/test.sh`) that emits named checks, and a frozen rubric of weighted goals that its LLM judge scores the agent's final repository against.

You get:
1. the intents (key, source_turn, kind, text);
2. the goals (id, tier, weight, goal text, rationale; the rationale usually names the user turn the goal comes from);
3. the eligible executable checks: id, kind (F2P = should flip from fail to pass; P2P_REGRESSION = must keep passing; TEST = a named test in the repository's test suite), description, the command or script excerpt, and its result on the task's reference solution and on the unmodified repository. Only checks that give a real per-check verdict are listed.

## Part A: checks per intent

For each intent, return `checks`: the list of check ids whose joint pass decides whether the repository fulfils THAT intent, and `check_coverage`:

- `full` - a careful reviewer would accept "all listed checks pass" as equivalent to "this intent is fulfilled", and "some listed check fails" as "not fulfilled": the checks test the behaviour or structure the intent asks for, all of it, and nothing the intent does not ask for.
- `partial` - the checks test part of the intent, or also require things the intent does not ask for (for example a later refinement from another user message, or one specific implementation when the intent allows several).
- `none` - no listed check tests this intent (return an empty list).

Rules:
- A check that also requires something introduced only by a LATER user message cannot be `full` for an earlier intent.
- A check that only greps for one specific identifier, when the intent accepts other correct implementations, is at best `partial`.
- A check that fails on the reference solution cannot be `full` (the reference is taken to fulfil the durable intents).
- P2P_REGRESSION / whole-suite checks are `full` only for an intent that is itself about keeping existing behaviour, builds or tests working.
- Questions, explanations, version-control or hosting actions (commit, push, PR) and other non-repository intents get `none`.
- Prefer `partial` when unsure. `full` must be defensible from the check's code.

## Part B: goals per intent and turn per goal

For each intent, return `goals`: the goal ids whose text expresses (all or part of) what this intent asks for, and `goal_relation`: `same` (the goals together say the same thing as the intent), `broader` (the goals ask for more than the intent, e.g. they include later refinements), `narrower` (the goals cover only part of the intent), or `none`.

For each goal, return `source_turn`: the earliest user message index (0 = opening request) the goal's requirement comes from, from its rationale and text; `anti_gaming` goals get the turn of the request they guard (usually 0).

Return JSON only, matching the schema. Every intent key and every goal id appears exactly once. `note` fields are one short sentence.
