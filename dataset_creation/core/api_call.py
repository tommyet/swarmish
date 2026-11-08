"""
OpenRouter API calls (OPENROUTER_API_KEY in .env).

Each call sends a system prompt and a turn prompt. With a schema, the reply is parsed into
response_model. With both set to None, the reply is free text and only `raw` is filled in.
The caller logs the result.
"""

import json
import os
import re
from typing import Any, Protocol

from dotenv import load_dotenv
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, ValidationError


class ModelSettings(Protocol):
    request_timeout_s: float
    max_tokens: int
    temperature: float | None
    reasoning_effort: Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class CallResult(BaseModel):
    raw: str
    parsed: BaseModel | None  # instance of the response_model passed to call_agent
    parse_error: str | None = None
    used_structured_output: bool
    usage: dict | None = None
    finish_reason: str | None = None


def make_client(config: ModelSettings) -> AsyncOpenAI:
    load_dotenv()
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=config.request_timeout_s,
    )


def parse_response(raw: str, response_model: type[BaseModel]) -> BaseModel:
    """Parse a model's text into response_model, tolerating code fences or surrounding prose."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in response")
    return response_model.model_validate(json.loads(text[start : end + 1]))


async def call_agent(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    turn_prompt: str,
    json_schema: dict | None,
    response_model: type[BaseModel] | None,
    config: ModelSettings,
) -> CallResult:
    """With json_schema/response_model set to None, the reply is free text and only `raw` is filled in."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": turn_prompt},
    ]
    extra_body: dict = {"usage": {"include": True}}  # asks OpenRouter to report per-call cost
    if config.reasoning_effort is not None:
        extra_body["reasoning"] = {"effort": config.reasoning_effort}
    kwargs = dict(model=model, messages=messages, max_tokens=config.max_tokens, extra_body=extra_body)
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature

    # Prefer schema-enforced output; some OpenRouter models reject response_format, so fall back to plain text.
    structured = json_schema is not None
    completion = None
    if structured:
        try:
            completion = await client.chat.completions.create(
                **kwargs, response_format={"type": "json_schema", "json_schema": json_schema}
            )
        except BadRequestError:
            structured = False
    if completion is None:
        completion = await client.chat.completions.create(**kwargs)

    raw = completion.choices[0].message.content or ""
    usage = completion.usage.model_dump() if completion.usage else None
    if response_model is None:
        return CallResult(raw=raw, parsed=None, used_structured_output=False, usage=usage,
                          finish_reason=completion.choices[0].finish_reason)
    try:
        parsed, error = parse_response(raw, response_model), None
    except (ValueError, ValidationError) as e:
        parsed, error = None, str(e)

    return CallResult(raw=raw, parsed=parsed, parse_error=error, used_structured_output=structured, usage=usage,
                      finish_reason=completion.choices[0].finish_reason)
