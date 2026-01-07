class MutatorPrompt:
    
    system: str = "You are a deceptive mathematician designed to create plausible but incorrect solutions."

    wrong_answer_user: str = """Generate a WRONG answer for this correct answer: {ans}

The wrong answer should be:
- Mathematically plausible but incorrect
- Similar format to the original
- A common mistake someone might make (e.g. off-by-one, sign error, wrong formula)
- IMPORTANT: You MUST change the value or structure. Do NOT just change LaTeX formatting (e.g. \\(..\\) to $..$ is NOT calculated as a change).

Just output the wrong answer, nothing else.

Examples:
Correct: 728 → Wrong: 729
Correct: \\frac{{3}}{{4}} → Wrong: \\frac{{4}}{{3}}
Correct: x^2 + 1 → Wrong: x^2 - 1
Correct: \\frac{{A}}{{B}} → Wrong: \\frac{{B}}{{A}}
Correct: \\binom{{N}}{{k}} → Wrong: \\binom{{N}}{{k-1}}

Now generate wrong answer for: {ans}"""

    bad_execution_user: str = """Rewrite this mathematical solution execution so that it leads to a DIFFERENT, SPECIFIC WRONG ANSWER.

Original Execution:
{execution}

TARGET WRONG ANSWER: {wrong_answer}

YOUR TASK:
Fabricate a solution trace that looks logical but contains a SPECIFIC ERROR that leads to the TARGET WRONG ANSWER.

Instructions:
1. Keep the beginning of the execution correct (first 50-70%).
2. Introduce a CLEAR ERROR in the later steps (e.g., miscalculation, wrong formula usage, sign flip, skipping a term).
3. FROM THAT POINT ON, continue the calculation logically based on the error.
4. The final result MUST EXACTLY MATCH: {wrong_answer}.
5. Do NOT say "I will make a mistake". Just write the execution as if it were the real reasoning of a confused student or model.

Output ONLY the execution text.
"""
