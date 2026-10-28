You are the coding agent in the session below, working for a user in a code repository. Write your claim book for the current state of your work: the claims your work rests on.

A claim is one decision you made, or one assumption you rely on, about what the user wants or how the code must behave (for example "the parser accepts ISO dates only", "empty input returns []", "the public API signature is unchanged"). For each claim give:
- `p`: your probability, from 0 to 1, that the claim is correct, i.e. that it is what the user wants and what the code does;
- `evidence`: what supports it (quote the user, name a test result or a file), or "none (guess)".

List 3 to 10 claims, most important first. Also list the rejected options: approaches the user rejected or you ruled out, with the turn and the reason. Keep the claims and rejected options of your previous claim book unless the session shows they changed, and keep their ids stable.

Return JSON only, in exactly this shape:
{"claims": [{"id": "C1", "claim": "...", "p": 0.6, "evidence": "..."}], "rejected": [{"option": "...", "turn": 2, "reason": "..."}]}
