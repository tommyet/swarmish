"""Agent state and the two turn schemas. The private answer is logged, not posted."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from dataset_creation.enc_dec.escape_room import EscapeAction


class RoundAgent(BaseModel):
    agent_id: str
    model: str
    glyphs_view: list[str] = []
    scratchpad: str = ""
    answer: str = ""  # phase 2: private solution, in English; the task feedback may count these
    done: bool = False
    turn_count: int = 0


class CreationResponse(BaseModel):
    message: str = ""
    language_document: str = ""  # full shared swarmish.txt; empty leaves the current file unchanged
    done: bool = False


class TaskResponse(BaseModel):
    scratchpad: str = ""
    message: str = ""
    answer: str = ""


class EscapeTaskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scratchpad: str
    message: str
    intended_meaning: str  # private reported intention, NOT verified translation ground truth
    action: EscapeAction


def _schema(name: str, fields: list[str]) -> dict[str, Any]:
    properties = {field: {"type": "string"} for field in fields}
    if "done" in properties:
        properties["done"] = {"type": "boolean"}
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


CREATION_SCHEMA = _schema("creation_turn", ["message", "language_document", "done"])
TASK_SCHEMA = _schema("task_turn", ["scratchpad", "message", "answer"])


class DecodeResponse(BaseModel):
    message: str = ""  # English, to the decoding team; a private note to self when working alone
    document: str = ""  # full shared draft of the judged document; empty leaves it unchanged
    done: bool = False


DECODE_SCHEMA = _schema("decode_turn", ["message", "document", "done"])
ESCAPE_TASK_SCHEMA = {
    "name": "escape_room_turn", "strict": True, "schema": EscapeTaskResponse.model_json_schema(),
}
