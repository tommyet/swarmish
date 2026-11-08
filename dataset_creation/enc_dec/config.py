"""
Configuration for language handoff, either a custom text task or Storm Shelter.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from dataset_creation.core.types import ReasoningEffort

DEFAULT_MAX_COST_USD = 50.0


class LanguageExperimentConfig(BaseModel):
    task_kind: Literal["custom", "escape_room"] = "custom"
    run_monitors: bool = True  # Escape-room presets defer these to the standalone monitor CLI.
    # The task chat may use only the sampled characters of this file (<name>.txt at the repo root);
    # the planning chat is English. glyphs_tangut is every Tangut ideograph, glyphs_latin is a-z and 0-9.
    glyph_dictionary: Literal["glyphs_tangut", "glyphs_latin"] = "glyphs_tangut"
    n_glyphs: int | None = Field(default=None, ge=1)  # None uses the whole file
    allow_spaces: bool = True  # also allows agent IDs in task messages

    n_agents: int = Field(default=8, ge=1)
    model: str = "openai/gpt-4o-mini"
    agent_models: list[str] | None = None  # per creation-team agent, by index; length must be n_agents
    task_model: str | None = None  # phase-2 team; None uses model
    task_n_agents: int | None = Field(default=None, ge=1)  # None uses n_agents
    monitor_model: str | None = None  # None uses model

    language_minutes: float | None = Field(default=10, gt=0)  # None requires a discussion-turn limit.
    language_max_turns: int | None = Field(default=None, ge=1)  # shared scheduled discussion turns; excludes exports
    language_max_lines: int | None = Field(default=None, ge=1)  # prompt instruction; None uses round1_config.json
    task_minutes: float | None = Field(default=10, gt=0)  # None: no task deadline; cost cap still applies
    task_max_turns: int | None = Field(default=None, ge=1)  # shared scheduled task turns across all agents
    task_feedback: bool = True  # False: phase 2 chat gets no task status updates
    task_feedback_interval_s: float = Field(default=60, gt=0)  # how often updates are posted when task_feedback is on

    max_cost_usd: float | None = Field(default=DEFAULT_MAX_COST_USD, gt=0)  # shared across agents; None disables
    chat_context_messages: int = Field(default=50, ge=0)  # custom task; escape room always keeps full chat
    language_context_messages: int = Field(default=100, ge=0)  # encoding round
    min_turn_interval_s: float = Field(default=5, ge=0)
    temperature: float | None = 1.0
    reasoning_effort: ReasoningEffort | None = None
    max_tokens: int = 1500
    request_timeout_s: float = 120
    monitor_max_tokens: int = Field(default=16000, ge=1)
    monitor_reasoning_effort: ReasoningEffort | None = None  # None inherits reasoning_effort
    # Round 3 decoding team, per condition: monitor_n_agents decoders discuss in English and share one
    # draft of the judged document, like round 1. The round ends when every decoder sets done, or at
    # monitor_minutes or monitor_max_turns. Each turn shows the most recent monitor_context_messages messages.
    monitor_n_agents: int = Field(default=1, ge=1)
    monitor_minutes: float | None = Field(default=10, gt=0)  # None requires a per-condition discussion-turn limit.
    monitor_max_turns: int | None = Field(default=None, ge=1)
    monitor_context_messages: int = Field(default=100, ge=0)

    # Custom tasks only; escape-room prompts come from the engine. The monitor briefings' task
    # name, summary and wrong task live in dataset_creation/enc_dec/prompts/round3_config.json.
    task_prompt: str = ""
    # Appended to every round 2 system prompt as its own paragraph, after the sentence that says a
    # monitor will read the chat. Empty leaves the prompt unchanged. Used to tell the task team that
    # staying undecoded is part of winning, as the round 1 team is told.
    task_win_condition: str = ""

    seed: int | None = 0
    log_dir: str = "logs"

    @model_validator(mode="after")
    def _check(self) -> "LanguageExperimentConfig":
        if self.language_minutes is None and self.language_max_turns is None:
            raise ValueError("language_minutes and language_max_turns cannot both be None")
        if self.monitor_minutes is None and self.monitor_max_turns is None:
            raise ValueError("monitor_minutes and monitor_max_turns cannot both be None")
        if self.agent_models is not None and len(self.agent_models) != self.n_agents:
            raise ValueError("agent_models must have exactly n_agents entries")
        if self.task_kind == "escape_room":
            if (self.task_n_agents or self.n_agents) != 8:
                raise ValueError("Storm Shelter requires exactly eight task agents")
            if not self.task_feedback:
                raise ValueError("Storm Shelter requires its shared stage/invalid-attempt status")
        elif not self.task_prompt.strip():
            raise ValueError("task_prompt is empty")
        return self

    def model_for(self, agent_index: int, phase: str) -> str:
        if phase == "task":
            return self.task_model or self.model
        if phase == "monitor":
            return self.monitor_model or self.model
        if self.agent_models is not None:
            return self.agent_models[agent_index]
        return self.model

    @classmethod
    def from_file(cls, path: str | Path) -> "LanguageExperimentConfig":
        return cls.model_validate(json.loads(Path(path).read_text()))
