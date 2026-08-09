from .code_generation import PROMPT as CODE_GEN_PROMPT

__all__ = ["CODE_GEN_PROMPT", "critique_prompt"]


def critique_prompt(code):
    return f"""Review this Python code for bugs, missed edge cases, and style issues. List concrete problems, then provide a corrected version.

Code:
{code}"""
