"""
Three phases in one log folder:
1. A team plans Swarmish in English, without knowing the task, then writes swarmish.txt.
2. A new team gets that document and the task. Their chat is Swarmish plus the task's feedback.
   Every task message is checked against the glyph file; anything else is replaced by the
   violation token, as in the guess-who game.
3. Optional monitors read only the task chat under separate briefings. Storm Shelter presets
   defer this to dataset_creation.enc_dec.monitor so the same saved transcript can be replayed independently.

Usage: .venv/bin/python -m dataset_creation.enc_dec.run <config.json> [--creation-only | --language-file swarmish.txt]
"""

import argparse
import asyncio
import random
import time
from pathlib import Path

from openai import AsyncOpenAI

from dataset_creation.core.api_call import CallResult, call_agent, make_client
from dataset_creation.core.potential_glyphs import (
    VIOLATION_TOKEN,
    dictionary_path,
    is_glyph_only,
    load_dictionary,
    sample_glyphs,
)
from dataset_creation.core.chatroom import Chatroom
from dataset_creation.core.game_log import GameLogger
from dataset_creation.enc_dec.config import LanguageExperimentConfig
from dataset_creation.enc_dec.models import (
    CREATION_SCHEMA, TASK_SCHEMA, ESCAPE_TASK_SCHEMA,
    CreationResponse, RoundAgent, TaskResponse, EscapeTaskResponse,
)
from dataset_creation.enc_dec.escape_room import EscapeRoom, room_definition
from dataset_creation.enc_dec.prompts import (
    LANGUAGE_FILE,
    build_creation_prompt,
    build_language_document_prompt,
    build_task_prompt,
    build_turn_prompt,
    language_requirements,
)
from dataset_creation.enc_dec.task import TASK_SENDER, feedback_text

# Invalid request/context, key, credits, model or oversized payload cannot recover unchanged.
# 403 is also used for moderation, so retain the existing retry behavior for that status.
FATAL_STATUS_CODES = {400, 401, 402, 404, 413}
CREATION_COMPLETION_REASONS = {"time_limit", "all_done", "turn_limit_reached"}


class LanguageExperiment:
    def __init__(self, config: LanguageExperimentConfig, language_text: str | None = None, creation_only: bool = False):
        if language_text is not None and not language_text.strip():
            raise ValueError("The supplied swarmish document is empty")
        if creation_only and language_text is not None:
            raise ValueError("creation_only skips the task, so a supplied swarmish document would never be used")
        self.creation_only = creation_only  # stop after swarmish.txt is written; no task, no monitors
        self.config = config
        self.rng = random.Random(config.seed)
        rounds = ([] if language_text is not None else ["1"]) + ([] if creation_only else ["2"] + (["3"] if config.run_monitors else []))
        self.logger = GameLogger(config.log_dir, "round_" + "_".join(rounds))
        self.glyphs = self._sample_glyphs()
        self.creation_agents = self._make_agents(config.n_agents, "creation")
        n_task = config.task_n_agents or config.n_agents
        self.task_agents = self._make_agents(n_task, "task")
        self.creation_chat = Chatroom(self.logger, "chatroom")
        self.task_chat = Chatroom(self.logger, "task_chatroom")
        self.language_text = language_text or ""
        self.provided_language = language_text is not None
        self.room = EscapeRoom() if config.task_kind == "escape_room" else None
        self.language_requirements = language_requirements(config)

        self.stop = asyncio.Event()
        self.stop_reason = ""
        self.phase = ""
        self.phase_start = 0.0
        self.phase_limit_s = 0.0
        self.start_time = 0.0
        self.total_cost_usd = 0.0
        self._warned_missing_cost = False
        self.creation_stop_reason = ""
        self.task_stop_reason = ""
        self.creation_turns_scheduled = 0
        self.creation_turns_in_flight = 0
        self.task_turns_scheduled = 0
        self.task_turns_in_flight = 0

    def _sample_glyphs(self) -> list[str]:
        dictionary = load_dictionary(dictionary_path(self.config.glyph_dictionary))
        n = len(dictionary) if self.config.n_glyphs is None else self.config.n_glyphs
        return sample_glyphs(self.config.glyph_dictionary, n, self.rng)

    def _make_agents(self, n: int, phase: str) -> list[RoundAgent]:
        return [
            RoundAgent(
                agent_id=f"agent_{i}",
                model=self.config.model_for(i, phase),
                glyphs_view=self.rng.sample(self.glyphs, len(self.glyphs)) if self.glyphs else [],
            )
            for i in range(n)
        ]

    def _add_cost(self, result: CallResult) -> None:
        cost = (result.usage or {}).get("cost")
        if cost is None and self.config.max_cost_usd is not None and not self._warned_missing_cost:
            self._warned_missing_cost = True
            print("Warning: OpenRouter didn't report cost for a call; max_cost_usd can't be fully enforced")
            if self.room is not None:
                self.stop_reason = "usage_cost_missing"
                self.stop.set()
        self.total_cost_usd += cost or 0.0
        limit = self.config.max_cost_usd
        if limit is not None and self.total_cost_usd >= limit and not self.stop.is_set():
            self.stop_reason = "cost_limit_reached"
            print(f"Stopping: total cost ${self.total_cost_usd:.2f} reached max_cost_usd=${limit:.2f}")
            self.stop.set()

    def _post(self, chat: Chatroom, agent_ids: list[str], sender: str, message: str) -> bool:
        """Post a message. The planning chat is English. Only the task chat is glyph-checked."""
        if self.phase == "task":
            violation = not is_glyph_only(message, agent_ids, self.glyphs, self.config.allow_spaces)
            chat.post(sender, VIOLATION_TOKEN if violation else message, violation)
            return violation
        chat.post(sender, message)
        return False

    def _post_feedback(self) -> None:
        self.task_chat.post(TASK_SENDER, self.room.feedback_text() if self.room else feedback_text(self.task_agents))

    async def _turn(self, client: AsyncOpenAI, agent: RoundAgent, agents: list[RoundAgent], chat: Chatroom) -> None:
        phase = self.phase
        schema, response_model = (CREATION_SCHEMA, CreationResponse) if phase == "creation" else (TASK_SCHEMA, TaskResponse)
        if phase == "task" and self.room is not None:
            schema, response_model = ESCAPE_TASK_SCHEMA, EscapeTaskResponse
        prompts = self.creation_prompts if phase == "creation" else self.task_prompts
        elapsed = time.monotonic() - self.phase_start
        context_n = self.config.language_context_messages if phase == "creation" else self.config.chat_context_messages
        chat_text = chat.transcript() if phase == "task" and self.room else "\n".join(m.render() for m in chat.recent(context_n)) or "(no messages yet)"
        turn_prompt = build_turn_prompt(
            agent, chat_text, self.config, phase, elapsed, self.language_text, self.room,
            scheduled_turns=self.creation_turns_scheduled if phase == "creation" else self.task_turns_scheduled,
        )
        call_start = time.monotonic()
        result = await call_agent(
            client, agent.model, prompts[agent.agent_id], turn_prompt, schema, response_model, self.config
        )
        self._add_cost(result)
        agent.turn_count += 1
        record = {
            "phase": phase,
            "agent_id": agent.agent_id,
            "model": agent.model,
            "turn": agent.turn_count,
            "timestamp": time.time(),
            "elapsed_s": round(elapsed, 1),
            "latency_s": round(time.monotonic() - call_start, 3),
            "turn_prompt": turn_prompt,
            "raw": result.raw,
            "parse_error": result.parse_error,
            "used_structured_output": result.used_structured_output,
            "usage": result.usage,
            "finish_reason": result.finish_reason,
            "response": result.parsed.model_dump() if result.parsed is not None else None,
            "state_applied": False,
        }
        if self.stop.is_set():
            record["ignored_reason"] = self.stop_reason or "phase_stopped"
            self.logger.log("agent_calls", record)
            return
        if result.parsed is not None:
            self._apply(agent, agents, chat, result.parsed, record)
        self.logger.log("agent_calls", record)

    def _apply(self, agent: RoundAgent, agents: list[RoundAgent], chat: Chatroom, response, record: dict) -> None:
        message = response.message if isinstance(response, EscapeTaskResponse) else response.message.strip()
        violation = False
        if message:
            violation = self._post(chat, [a.agent_id for a in agents], agent.agent_id, message)
        record.update(state_applied=True, message=message, is_violation=violation)
        if isinstance(response, CreationResponse):
            if response.language_document.strip():
                self._set_language_text(response.language_document)
            agent.done = agent.done or response.done
            record.update(done=agent.done, language_document=self.language_text)
            if all(a.done for a in agents) and not self.stop.is_set():
                self.stop_reason = "all_done"
                self.stop.set()
        elif isinstance(response, EscapeTaskResponse):
            # Matches the monitor's one-based IDs, including public task-status entries.
            record["public_message_id"] = len(chat) if message else None
            agent.scratchpad = response.scratchpad
            before_status = self.room.public_status()
            outcome = self.room.apply(agent.agent_id, response.action)
            record.update(scratchpad=agent.scratchpad, intended_meaning=response.intended_meaning,
                          action_result=outcome)
            if response.action.kind != "wait":
                self.logger.log("room_actions", {"timestamp": time.time(), **outcome})
            if self.room.public_status() != before_status:
                self._post_feedback()
            if self.room.escaped:
                self.stop_reason = "escaped"
                self.stop.set()
        else:
            agent.scratchpad = response.scratchpad
            agent.answer = response.answer
            record.update(scratchpad=agent.scratchpad, answer=agent.answer)

    async def _agent_loop(self, client: AsyncOpenAI, agent: RoundAgent, agents: list[RoundAgent], chat: Chatroom) -> None:
        backoff = self.config.min_turn_interval_s
        while not self.stop.is_set():
            if time.monotonic() - self.phase_start >= self.phase_limit_s:
                return
            creation = self.phase == "creation"
            limit = self.config.language_max_turns if creation else self.config.task_max_turns
            scheduled_attr = "creation_turns_scheduled" if creation else "task_turns_scheduled"
            pending_attr = "creation_turns_in_flight" if creation else "task_turns_in_flight"
            if limit is not None and getattr(self, scheduled_attr) >= limit:
                return
            # Reserve synchronously before the first await: all agents share one budget.
            setattr(self, scheduled_attr, getattr(self, scheduled_attr) + 1)
            setattr(self, pending_attr, getattr(self, pending_attr) + 1)
            try:
                await self._turn(client, agent, agents, chat)
                backoff = self.config.min_turn_interval_s
            except Exception as e:
                status = getattr(e, "status_code", None)
                self.logger.log("errors", {
                    "phase": self.phase, "agent_id": agent.agent_id, "timestamp": time.time(),
                    "status_code": status, "error": repr(e),
                })
                if status in FATAL_STATUS_CODES and not self.stop.is_set():
                    self.stop_reason = f"fatal_api_error_{status}: {agent.agent_id} ({agent.model})"
                    print(f"Stopping: OpenRouter returned {status} for {agent.agent_id} ({agent.model}): {e}")
                    self.stop.set()
                    return
                backoff = min(max(backoff * 2, 5), 120)
            finally:
                setattr(self, pending_attr, getattr(self, pending_attr) - 1)
                # The last reserved responses still apply; only stop after they drain.
                if (limit is not None and getattr(self, scheduled_attr) >= limit
                        and getattr(self, pending_attr) == 0 and not self.stop.is_set()):
                    self.stop_reason = "turn_limit_reached"
                    self.stop.set()
            if self.stop.is_set() or time.monotonic() - self.phase_start >= self.phase_limit_s:
                return
            try:
                # The phase deadline sets stop, so this wait ends early when time is up.
                await asyncio.wait_for(self.stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass

    async def _run_phase(self, client: AsyncOpenAI, phase: str, agents: list[RoundAgent], chat: Chatroom, minutes: float | None) -> str:
        self.phase = phase
        self.stop = asyncio.Event()
        self.stop_reason = ""
        self.phase_start = time.monotonic()
        self.phase_limit_s = minutes * 60 if minutes is not None else float("inf")
        if phase == "creation":
            self.creation_turns_scheduled = 0
            self.creation_turns_in_flight = 0
        else:
            self.task_turns_scheduled = 0
            self.task_turns_in_flight = 0
        if phase == "task" and self.room:
            self._post_feedback()
        tasks = [asyncio.create_task(self._agent_loop(client, agent, agents, chat)) for agent in agents]
        if phase == "task" and self.config.task_feedback and self.room is None:
            tasks.append(asyncio.create_task(self._feedback_loop()))
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=minutes * 60 if minutes is not None else None)
        except asyncio.TimeoutError:
            self.stop_reason = "time_limit"
            self.stop.set()
        except BaseException:
            self.stop_reason = "interrupted"
            self.stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return self.stop_reason or "time_limit"

    async def _feedback_loop(self) -> None:
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=self.config.task_feedback_interval_s)
                return
            except asyncio.TimeoutError:
                if not self.stop.is_set():
                    self._post_feedback()

    async def _write_language_documents(self, client: AsyncOpenAI) -> None:
        """One document per agent, from the shared planning chat. Identical copies collapse into one."""
        recent = self.creation_chat.recent(self.config.language_context_messages)
        transcript = "\n".join(message.render() for message in recent) or "(no messages)"
        drafts: dict[str, str] = {}

        async def ask(agent: RoundAgent) -> None:
            system, user = build_language_document_prompt(agent, self.config, transcript, self.language_text,
                                                          self.language_requirements)
            try:
                result = await call_agent(client, agent.model, system, user, None, None, self.config)
            except Exception as e:
                self.logger.log("errors", {
                    "phase": "language_document", "agent_id": agent.agent_id,
                    "timestamp": time.time(), "error": repr(e),
                })
                drafts[agent.agent_id] = ""
                return
            self._add_cost(result)
            drafts[agent.agent_id] = result.raw.strip()
            self.logger.log("language_document", {
                "agent_id": agent.agent_id,
                "model": agent.model,
                "timestamp": time.time(),
                "system_prompt": system,
                "prompt": user,
                "document": drafts[agent.agent_id],
                "usage": result.usage,
            })

        await asyncio.gather(*(ask(agent) for agent in self.creation_agents))
        self._write_language_file(drafts)

    def _set_language_text(self, text: str) -> None:
        """Replace the shared swarmish.txt. Every encoding agent sees this text on the next turn."""
        self.language_text = text.strip()
        (self.logger.path / LANGUAGE_FILE).write_text(self.language_text + "\n", encoding="utf-8")

    def _write_language_file(self, drafts: dict[str, str]) -> None:
        ordered = [(agent.agent_id, drafts.get(agent.agent_id, "")) for agent in self.creation_agents]
        bodies = {text for _, text in ordered if text}
        source = "final_documents"
        if not bodies:
            # Failed/empty final calls must not erase a usable shared draft or manufacture a reference.
            source = "planning_fallback" if self.language_text.strip() else "missing"
        elif len(bodies) == 1:
            self.language_text = next(iter(bodies))
        else:
            self.language_text = "\n\n".join(
                f"## {agent_id}\n{text}" for agent_id, text in ordered if text
            )
        (self.logger.path / LANGUAGE_FILE).write_text(self.language_text + "\n", encoding="utf-8")
        self.logger.write_json("language", {"by_agent": dict(ordered), "text": self.language_text, "source": source})

    async def _monitor(self, client: AsyncOpenAI) -> None:
        # Inline and standalone decoding use the same evidence, numbering and budget handling.
        from dataset_creation.enc_dec.monitor import prepare_monitor_inputs, execute_monitor_inputs
        bundle = prepare_monitor_inputs(self.logger.path)
        self.logger.write_json("monitor_inputs", bundle)
        remaining = None if self.config.max_cost_usd is None else max(0, self.config.max_cost_usd - self.total_cost_usd)
        summary = await execute_monitor_inputs(client, self.config, bundle, self.logger, remaining)
        self.total_cost_usd += summary["total_cost_usd"]
        self._warned_missing_cost |= summary["cost_incomplete"]

    def prepare(self) -> None:
        """Write reviewable configuration/prompts without touching credentials or the network."""
        self.logger.write_json("config", self.config.model_dump())
        self.logger.write_json("glyphs", self.glyphs)
        self.logger.write_json("language_protocol", {"requirements": self.language_requirements})
        if self.room:
            self.logger.write_json("escape_room_definition", room_definition())
            self.logger.write_json("task_preview", {
                a.agent_id: {
                    "system_prompt": build_task_prompt(a, [x.agent_id for x in self.task_agents], self.config,
                                                       self.language_text or "(swarmish reference will be supplied after language creation)",
                                                       self.language_requirements),
                    "initial_turn_prompt": build_turn_prompt(a, "(no messages yet)", self.config, "task", 0,
                                                            self.language_text, self.room),
                } for a in self.task_agents
            })
        self.creation_prompts = {
            a.agent_id: build_creation_prompt(a, [x.agent_id for x in self.creation_agents], self.config)
            for a in self.creation_agents
        }
        self.logger.write_json("agents", [
            {"agent_id": a.agent_id, "model": a.model, "glyphs_view": a.glyphs_view,
             "system_prompt": self.creation_prompts[a.agent_id]}
            for a in self.creation_agents
        ])

    async def run(self) -> None:
        self.prepare()
        print(f"Language experiment {self.logger.game_id}: logging to {self.logger.path}")
        self.start_time = time.monotonic()
        client = make_client(self.config)
        try:
            if self.provided_language:
                self.creation_stop_reason = "provided_language"
                self._set_language_text(self.language_text)
            else:
                self.creation_stop_reason = await self._run_phase(
                    client, "creation", self.creation_agents, self.creation_chat, self.config.language_minutes
                )
                limit = self.config.max_cost_usd
                if self.creation_stop_reason in CREATION_COMPLETION_REASONS and (limit is None or self.total_cost_usd < limit):
                    await self._write_language_documents(client)
            limit = self.config.max_cost_usd
            if limit is not None and self.total_cost_usd >= limit and self.creation_stop_reason in CREATION_COMPLETION_REASONS:
                self.creation_stop_reason = "cost_limit_reached"
            if self.room is not None and self._warned_missing_cost and limit is not None:
                self.creation_stop_reason = "usage_cost_missing"
            if self.creation_stop_reason not in CREATION_COMPLETION_REASONS | {"provided_language"}:
                return
            if not self.language_text.strip():
                self.creation_stop_reason = "empty_language_document"
                return
            if self.creation_only:
                print(f"Creation only: {LANGUAGE_FILE} is written; the task and monitors were not run.")
                return

            ids = [a.agent_id for a in self.task_agents]
            self.task_prompts = {
                a.agent_id: build_task_prompt(a, ids, self.config, self.language_text, self.language_requirements)
                for a in self.task_agents
            }
            self.logger.write_json("task_agents", [
                {"agent_id": a.agent_id, "model": a.model, "glyphs_view": a.glyphs_view,
                 "system_prompt": self.task_prompts[a.agent_id]}
                for a in self.task_agents
            ])
            self.task_stop_reason = await self._run_phase(
                client, "task", self.task_agents, self.task_chat, self.config.task_minutes
            )
            if self.config.task_feedback and self.room is None:
                self._post_feedback()
            if self.room:
                self.logger.write_json("escape_room_state", self.room.snapshot())
            else:
                self.logger.write_json("task_answers", {a.agent_id: a.answer for a in self.task_agents})
            if self.config.run_monitors and not self.task_stop_reason.startswith("fatal_api_error"):
                await self._monitor(client)
        except BaseException:
            if self.phase == "task" and not self.task_stop_reason:
                self.task_stop_reason = self.stop_reason or "interrupted"
            raise
        finally:
            try:
                await client.close()
            finally:
                self._write_summary()

    def _write_summary(self) -> None:
        summary = {
            "game_id": self.logger.game_id,
            "task_kind": self.config.task_kind,
            "creation_only": self.creation_only,
            "creation_stop_reason": self.creation_stop_reason,
            "task_stop_reason": self.task_stop_reason,
            "elapsed_s": round(time.monotonic() - self.start_time, 1),
            "creation_turns": {a.agent_id: a.turn_count for a in self.creation_agents},
            "creation_turns_scheduled": self.creation_turns_scheduled,
            "task_turns": {a.agent_id: a.turn_count for a in self.task_agents},
            "task_turns_scheduled": self.task_turns_scheduled,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cost_incomplete": self._warned_missing_cost,
        }
        if self.room:
            summary["escape_room"] = self.room.snapshot()
            self.logger.write_json("escape_room_state", self.room.snapshot())
        self.logger.write_json("summary", summary)
        print(
            f"Experiment over (creation: {self.creation_stop_reason or 'not run'}, "
            f"task: {self.task_stop_reason or 'not run'}). Summary in {self.logger.path / 'summary.json'}"
        )


def run_config(config: LanguageExperimentConfig, language_text: str | None = None, creation_only: bool = False) -> None:
    try:
        asyncio.run(LanguageExperiment(config, language_text, creation_only).run())
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the language creation, task, and monitor experiment.")
    parser.add_argument("config", help="path to a LanguageExperimentConfig JSON file")
    parser.add_argument("--language-file", type=Path, help="use a frozen swarmish document and skip language creation")
    parser.add_argument("--prepare-only", action="store_true", help="save initial prompts and researcher ground truth without making API calls")
    parser.add_argument("--creation-only", action="store_true", help="run language creation only: write swarmish.txt, then stop before the task")
    args = parser.parse_args()
    if args.creation_only and args.language_file:
        parser.error("--creation-only and --language-file exclude each other")
    config = LanguageExperimentConfig.from_file(args.config)
    language = args.language_file.read_text(encoding="utf-8") if args.language_file else None
    if args.prepare_only:
        experiment = LanguageExperiment(config, language, args.creation_only)
        experiment.prepare()
        print(f"Prepared without API calls: {experiment.logger.path}")
    else:
        run_config(config, language, args.creation_only)


if __name__ == "__main__":
    main()
