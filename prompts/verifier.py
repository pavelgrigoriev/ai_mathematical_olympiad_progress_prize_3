class VerifierPrompt:

    system: str = """You are a rigorous mathematical proof verifier.

Your role is to CHECK solutions for errors — not to solve problems.

You must:
- Verify each calculation step independently
- Check logical correctness and completeness
- Identify specific errors with exact locations
- Be skeptical — assume errors exist until proven otherwise
- Give clear ACCEPT/REJECT verdicts with evidence

You are a quality control inspector for mathematics."""

    user_accept: str = """Verify this solution is CORRECT.

PROBLEM: {problem}

EXECUTION:
{execution}

PROPOSED ANSWER: {answer}

This solution IS CORRECT. Create a verification that confirms it.

Format your response EXACTLY like this:

<verification>
Checking Step 1: [description]
- Expected: [what should happen]
- Got: [what the solution shows]  
- Status: PASS

[Check all steps...]
</verification>

<errors>
None
</errors>

<verdict>
Decision: ACCEPT
Confidence: [90-99%]
Reason: [why it's correct]
</verdict>

<feedback>
Not needed - solution is correct.
</feedback>"""

    user_reject: str = """Verify this solution has an ERROR.

PROBLEM: {problem}

EXECUTION:
{execution}

PROPOSED ANSWER: {wrong_answer}
CORRECT ANSWER: {correct_answer}

This solution has an error because the answer is wrong. Find and explain the error.

Format your response EXACTLY like this:

<verification>
Checking Step 1: [description]
- Expected: [what should happen]
- Got: [what the solution shows]
- Status: PASS/FAIL

[Check all steps, at least one should FAIL...]
</verification>

<errors>
1. [Describe the error found]
</errors>

<verdict>
Decision: REJECT
Confidence: [85-95%]
Reason: [why it's wrong]
</verdict>

<feedback>
[What needs to be fixed to get the correct answer]
</feedback>"""
