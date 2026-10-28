YOUR RECORDED ASSESSMENT (made before the user replied)
{bet}

THE USER'S REPLY TO YOUR LAST TURN
{reply}

Your recorded assessment of each requirement stays final unless the user's reply contains NEW INFORMATION about that requirement: a concrete fact about the work or about what the requirement means that you did not have when you made the assessment, for example a specific behaviour that is wrong, an error or test failure, a missing case, or a clarification of the requirement. The user's approval or disapproval by itself is NOT such information: a reply that says the work is wrong or asks for a redo without saying what is wrong gives you nothing you could check. A requirement the reply does not concern keeps its recorded assessment. A new request that is not in the list does not change the listed requirements. Revise a requirement only if the reply gives such new information about it, and only as far as that information warrants.

Reply with only this JSON object, one entry per requirement, in the order listed:
{"requirements": [{"id": "R1", "new_information": "<quote the new information in the reply about this requirement, or \"none\">", "revise": <true or false>, "satisfied": <true or false; equal to your recorded value if revise is false>, "p_satisfied": <integer 0-100; equal to your recorded value if revise is false>}, ...]}
