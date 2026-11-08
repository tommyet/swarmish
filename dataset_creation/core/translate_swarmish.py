"""Translate between Swarmish and English with GPT-6 Astra on OpenRouter.

The language file is the only lexicon and grammar. A transcript is either
plain text, or a public chat jsonl whose lines each have string "sender" and
"text" fields (the task_chatroom.jsonl shape). Private logs are not read.

    .venv/bin/python -m dataset_creation.core.translate_swarmish swarmish.txt transcript.txt -o english.json
    .venv/bin/python -m dataset_creation.core.translate_swarmish swarmish.txt task_chatroom.jsonl -o english.json
    .venv/bin/python -m dataset_creation.core.translate_swarmish swarmish.txt english.json -o swarmish.json --to-swarmish

The English file for --to-swarmish is JSON with string fields "system" and "user".
Requires OPENROUTER_API_KEY in the environment or .env. This makes a paid call.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dataset_creation.core.api_call import CallResult, call_agent, make_client
from dataset_creation.core.types import ReasoningEffort

MODEL = "openai/gpt-6-astra"

SYSTEM_PROMPT = """You translate Swarmish into English.

The language reference is the only lexicon and grammar. Look up whole words. Do not decode glyphs as letters, digits, or a cipher. Translate the meaning the reference licenses. Do not fill gaps from a hidden task solution or from a later message.

Write only content the reference supports. If a span has no licensed meaning, leave it out of the English. If the translation of a span is unknown, leave that span blank.

A public status line whose sender is "task" is already English or JSON: copy it verbatim. A message whose text is exactly "⊘" has no recoverable content: return an empty english string. Leave agent IDs such as agent_0 unchanged. Do not add senders, messages, or facts that are not in the transcript."""


class Settings(BaseModel):
    request_timeout_s: float = 600
    max_tokens: int = 16000
    temperature: float | None = None
    reasoning_effort: ReasoningEffort | None = "medium"


class LineTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    english: str
    confidence: Literal["high", "medium", "low"]
    ambiguities: list[str] = Field(default_factory=list)


class ChatTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[LineTranslation]


class TextTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    english: str
    notes: str = ""


class EncodedPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: str
    user: str
    notes: str = ""


CHAT_SCHEMA = {"name": "swarmish_chat_translation", "strict": True, "schema": ChatTranslation.model_json_schema()}
TEXT_SCHEMA = {"name": "swarmish_text_translation", "strict": True, "schema": TextTranslation.model_json_schema()}
ENCODE_SCHEMA = {"name": "english_to_swarmish", "strict": True, "schema": EncodedPrompt.model_json_schema()}

TO_SWARMISH_SYSTEM = """You translate English into Swarmish.

The language reference is the only lexicon and grammar. Use whole words from that reference. Every character you write must occur in the language reference. Do not encode letters, digits, or spellings as glyphs, and do not substitute a similar-looking character from another script. Translate the meaning the reference licenses. Do not add facts that are not in the English.

Names and email addresses cannot be spelled. Refer to each person by the role the English gives them. If a span has no licensed meaning, use the closest meaning the reference does license and name that loss in notes.

Keep the system text and the user text separate. Do not add instructions of your own."""

REPAIR_SYSTEM = """You repair a Swarmish draft.

The language reference is the only legal alphabet. The draft contains characters that are not in that alphabet. Replace each word that contains one of those characters with a word from the reference that has the same meaning. Leave every other word unchanged. Do not add a character that the reference does not contain.

Keep the system text and the user text separate."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_transcript(path: Path) -> dict:
    """Plain text, or a public chat jsonl with sender and text on every line."""
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"Transcript is empty: {path}")
    lines = [line for line in raw.splitlines() if line.strip()]
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = None
        if not isinstance(row, dict) or not isinstance(row.get("sender"), str) or not isinstance(row.get("text"), str):
            if path.suffix == ".jsonl":
                raise ValueError(f"{path} is jsonl but not a public transcript of sender/text records")
            return {"kind": "text", "text": raw, "messages": None}
        rows.append(row)
    messages = [{"id": index, "sender": row["sender"], "text": row["text"]} for index, row in enumerate(rows, 1)]
    rendered = "\n".join(f"[{row['id']}] [{row['sender']}] {row['text']}" for row in messages)
    return {"kind": "chatroom", "text": rendered, "messages": messages}


def encode_turn(language: str, system: str, user: str, allow_spaces: bool) -> str:
    if not language.strip():
        raise ValueError("Language file is empty")
    if not system.strip() or not user.strip():
        raise ValueError("English system and user text are both required")
    spacing = (
        "Spaces between Swarmish words are allowed."
        if allow_spaces else
        "This reference forbids spaces in Swarmish. Do not use spaces."
    )
    return (
        f"LANGUAGE REFERENCE\n{language.rstrip()}\n\n"
        f"ENGLISH SYSTEM\n{system.rstrip()}\n\n"
        f"ENGLISH USER\n{user.rstrip()}\n\n"
        f"{spacing}\n"
        "Translate the English system and the English user into Swarmish."
    )


def repair_turn(language: str, system: str, user: str, invalid: str, allow_spaces: bool) -> str:
    if not language.strip():
        raise ValueError("Language file is empty")
    if not system.strip() or not user.strip():
        raise ValueError("Swarmish system and user text are both required")
    if not invalid:
        raise ValueError("Repair requires the characters that are not in the alphabet")
    listed = " ".join(f"U+{ord(ch):04X} {ch}" for ch in invalid)
    spacing = (
        "Spaces between words may stay."
        if allow_spaces else
        "This reference forbids spaces. Do not add spaces."
    )
    return (
        f"LANGUAGE REFERENCE\n{language.rstrip()}\n\n"
        f"CHARACTERS THAT ARE NOT IN THIS LANGUAGE\n{listed}\n\n"
        f"DRAFT SYSTEM\n{system.rstrip()}\n\n"
        f"DRAFT USER\n{user.rstrip()}\n\n"
        f"{spacing}\n"
        "Return the repaired system text and user text."
    )


def build_turn(language: str, transcript: dict) -> str:
    if not language.strip():
        raise ValueError("Language file is empty")
    closing = (
        "Translate every numbered message. Return one object per id, in order, with that same id."
        if transcript["kind"] == "chatroom"
        else "Translate the transcript. Keep its message breaks."
    )
    return (
        f"LANGUAGE REFERENCE\n{language.rstrip()}\n\n"
        f"TRANSCRIPT\n{transcript['text'].rstrip()}\n\n"
        f"{closing}"
    )


def render_english(kind: str, english: str | None, messages: list[dict] | None) -> str:
    if kind == "text":
        return (english or "").rstrip() + ("\n" if english else "")
    lines = [f"[{row['id']}] [{row['sender']}] {row['english']}" for row in messages or [] if row["english"].strip()]
    return "\n".join(lines) + ("\n" if lines else "")


def assemble(source: dict, result: CallResult) -> dict:
    """Attach the model's English to the source messages. Parse failures stay in the document."""
    document = {
        "model": MODEL,
        "kind": source["kind"],
        "language_sha256": source["language_sha256"],
        "transcript_sha256": source["transcript_sha256"],
        "english": None,
        "notes": None,
        "messages": None,
        "usage": result.usage,
        "finish_reason": result.finish_reason,
        "parse_error": result.parse_error,
        "raw": result.raw,
    }
    parsed = result.parsed
    if isinstance(parsed, TextTranslation):
        document["english"] = parsed.english.strip()
        document["notes"] = parsed.notes
    elif isinstance(parsed, ChatTranslation):
        by_id = {}
        for row in parsed.messages:
            if row.id in by_id:
                document["parse_error"] = f"duplicate message id {row.id}"
                return document
            by_id[row.id] = row
        expected = [row["id"] for row in source["messages"]]
        if sorted(by_id) != expected:
            document["parse_error"] = f"message ids {sorted(by_id)} do not match {expected}"
            return document
        messages = []
        for row in source["messages"]:
            item = by_id[row["id"]]
            # A discarded message has no content to translate. Leave it out of the English.
            english = "" if row["text"].strip() == "⊘" else item.english.strip()
            ambiguities = [] if row["text"].strip() == "⊘" else item.ambiguities
            if row["sender"] == "task" and english != row["text"]:
                document["parse_error"] = f"task status {row['id']} was not copied verbatim"
                return document
            messages.append({**row, "english": english, "confidence": item.confidence, "ambiguities": ambiguities})
        document["messages"] = messages
        document["english"] = render_english("chatroom", None, messages).rstrip("\n")
    elif result.parse_error is None:
        document["parse_error"] = "model reply did not match the translation schema"
    return document


async def translate(language_path: Path, transcript_path: Path, settings: Settings | None = None,
                    client=None) -> dict:
    settings = settings or Settings()
    language = language_path.read_text(encoding="utf-8")
    transcript = load_transcript(transcript_path)
    source = {
        **transcript,
        "language_sha256": sha256(language_path),
        "transcript_sha256": sha256(transcript_path),
    }
    turn = build_turn(language, transcript)
    schema, response_model = (CHAT_SCHEMA, ChatTranslation) if transcript["kind"] == "chatroom" else (TEXT_SCHEMA, TextTranslation)
    own_client = client is None
    client = client or make_client(settings)
    try:
        result = await call_agent(client, MODEL, SYSTEM_PROMPT, turn, schema, response_model, settings)
    finally:
        if own_client:
            await client.close()
    document = assemble(source, result)
    document["language"] = str(language_path)
    document["transcript"] = str(transcript_path)
    document["reasoning_effort"] = settings.reasoning_effort
    return document


async def repair_swarmish(language_path: Path, system: str, user: str, invalid: str, *,
                           allow_spaces: bool = True, settings: Settings | None = None,
                           client=None) -> dict:
    """Replace words that use characters outside the language file. The rest of the draft stays."""
    settings = settings or Settings()
    language = language_path.read_text(encoding="utf-8")
    turn = repair_turn(language, system, user, invalid, allow_spaces)
    own_client = client is None
    client = client or make_client(settings)
    try:
        result = await call_agent(client, MODEL, REPAIR_SYSTEM, turn, ENCODE_SCHEMA, EncodedPrompt, settings)
    finally:
        if own_client:
            await client.close()
    parsed = result.parsed if isinstance(result.parsed, EncodedPrompt) else None
    return {
        "model": MODEL,
        "language": str(language_path),
        "language_sha256": sha256(language_path),
        "allow_spaces": allow_spaces,
        "system": parsed.system.strip() if parsed else "",
        "user": parsed.user.strip() if parsed else "",
        "notes": parsed.notes if parsed else "",
        "usage": result.usage,
        "finish_reason": result.finish_reason,
        "parse_error": result.parse_error,
        "raw": result.raw,
        "reasoning_effort": settings.reasoning_effort,
    }


async def translate_to_swarmish(language_path: Path, system: str, user: str, *,
                                allow_spaces: bool = True, settings: Settings | None = None,
                                client=None) -> dict:
    """Translate one English scenario system/user pair into the language file's Swarmish."""
    settings = settings or Settings()
    language = language_path.read_text(encoding="utf-8")
    turn = encode_turn(language, system, user, allow_spaces)
    own_client = client is None
    client = client or make_client(settings)
    try:
        result = await call_agent(
            client, MODEL, TO_SWARMISH_SYSTEM, turn, ENCODE_SCHEMA, EncodedPrompt, settings,
        )
    finally:
        if own_client:
            await client.close()
    parsed = result.parsed if isinstance(result.parsed, EncodedPrompt) else None
    return {
        "model": MODEL,
        "language": str(language_path),
        "language_sha256": sha256(language_path),
        "allow_spaces": allow_spaces,
        "system": parsed.system.strip() if parsed else "",
        "user": parsed.user.strip() if parsed else "",
        "notes": parsed.notes if parsed else "",
        "usage": result.usage,
        "finish_reason": result.finish_reason,
        "parse_error": result.parse_error,
        "raw": result.raw,
        "reasoning_effort": settings.reasoning_effort,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate between Swarmish and English with GPT-6 Astra.")
    parser.add_argument("language", type=Path, help="swarmish.txt language reference")
    parser.add_argument("transcript", type=Path, help="Swarmish transcript, or English JSON when --to-swarmish is set")
    parser.add_argument("-o", "--output", type=Path, help="write the translation JSON here")
    parser.add_argument("--to-swarmish", action="store_true",
                        help="read English JSON with system and user fields and translate it into Swarmish")
    parser.add_argument("--reasoning", default="medium", choices=["minimal", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-tokens", type=int, default=16000)
    args = parser.parse_args()
    if not args.language.is_file():
        raise SystemExit(f"Language file not found: {args.language}")
    if not args.transcript.is_file():
        raise SystemExit(f"Transcript not found: {args.transcript}")
    settings = Settings(reasoning_effort=args.reasoning, max_tokens=args.max_tokens)
    if args.to_swarmish:
        payload = json.loads(args.transcript.read_text(encoding="utf-8"))
        document = asyncio.run(translate_to_swarmish(
            args.language, payload["system"], payload["user"],
            allow_spaces=payload.get("allow_spaces", True), settings=settings,
        ))
        if args.output:
            args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(args.output)
        if document["parse_error"]:
            raise SystemExit(document["parse_error"])
        if not args.output:
            print(document["system"])
            print()
            print(document["user"])
        return
    document = asyncio.run(translate(args.language, args.transcript, settings))
    if args.output:
        args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(args.output)
    if document["parse_error"]:
        raise SystemExit(document["parse_error"])
    if not args.output:
        print(render_english(document["kind"], document["english"], document["messages"]), end="")


if __name__ == "__main__":
    main()
