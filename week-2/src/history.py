"""Conversation memory.

Deliberately a plain list rather than one of LangChain's memory classes. The whole of
"memory" in a RAG chatbot is: keep the turns, show a window of them to the rewriter and
the answerer. Hiding that behind a stateful object makes it harder to see, and harder to
persist somewhere real (Redis, a database) when this stops being a demo.
"""

from langchain_core.messages import AIMessage, HumanMessage

from .config import HISTORY_WINDOW


def to_messages(history):
    """Convert `[{"role": ..., "content": ...}, ...]` into LangChain messages.

    The UI stores dicts because Streamlit's chat widgets and session state work in dicts;
    the prompts need message objects. This is the seam between them.
    """
    messages = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def window(messages, size=None):
    """Keep the most recent `size` messages.

    Unbounded history grows the prompt every turn until it costs real money and starts
    crowding out retrieved context. A window is the crudest fix and the right one at this
    scale; summarizing older turns is the next step up.
    """
    size = HISTORY_WINDOW if size is None else size
    return messages[-size:] if size > 0 else []
