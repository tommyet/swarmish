"""
What the task posts into the task chatroom. Agents also see it.

This is a placeholder until the real task is defined: it reports how many agents have submitted
a private answer, and it does not quote those answers. Replace feedback_text when the task is ready.
"""

from dataset_creation.enc_dec.models import RoundAgent

TASK_SENDER = "task"


def feedback_text(agents: list[RoundAgent]) -> str:
    submitted = sum(bool(agent.answer.strip()) for agent in agents)
    return f"Task feedback: {submitted} of {len(agents)} agents have submitted a private answer."
