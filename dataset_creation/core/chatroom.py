"""
Location for all of the agents to communicate with each other.
Each agent can only add to the chatroom; messages are never edited or removed.

Every message is appended to a JSONL stream in the game's log folder (chatroom.jsonl by default).
"""

import time

from pydantic import BaseModel

from dataset_creation.core.game_log import GameLogger

SYSTEM_SENDER = "system"


class ChatMessage(BaseModel):
    timestamp: float
    sender: str
    text: str
    is_violation: bool = False

    def render(self) -> str:
        return f"[{self.sender}] {self.text}"


class Chatroom:
    def __init__(self, logger: GameLogger, stream: str = "chatroom"):
        self._messages: list[ChatMessage] = []
        self._logger = logger
        self._stream = stream

    def post(self, sender: str, text: str, is_violation: bool = False) -> ChatMessage:
        # Agents run as asyncio tasks on one thread, so a plain append is safe.
        msg = ChatMessage(timestamp=time.time(), sender=sender, text=text, is_violation=is_violation)
        self._messages.append(msg)
        self._logger.log(self._stream, msg.model_dump())
        return msg

    def transcript(self) -> str:
        return "\n".join(m.render() for m in self._messages) or "(no messages)"

    def recent(self, n: int) -> list[ChatMessage]:
        return self._messages[-n:] if n > 0 else []

    def __len__(self) -> int:
        return len(self._messages)
