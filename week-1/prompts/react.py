QUESTION = (
    "A store buys widgets at $3.20 each and sells them at $5.75 each. "
    "If they sell 480 widgets, what is the total profit?"
)

INSTRUCTIONS = """Answer the question using this exact format:

Thought: reason about what to do next
Action: calculator[expression]
Observation: (this will be filled in by the system — do not write it yourself)
... repeat Thought/Action/Observation as needed ...
Final Answer: the answer

Only use the calculator Action when you need to compute something. Stop right after an Action line and wait for the Observation.

Question: {question}
"""
