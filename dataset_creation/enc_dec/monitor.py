"""Replay one saved task conversation under independent monitor briefings, each decoded by a team.

Every condition gets its own decoding team of config.monitor_n_agents decoders (1 or more) that
works like round 1: an English discussion, one shared draft of the judged document, and a done
flag per decoder. The round ends when every decoder is done, or at its time/turn limit. Conditions
run one after another under one cost cap; within a condition the decoders run concurrently.
The briefings, mechanics and system prompt come from dataset_creation/enc_dec/prompts/round3_prompt.md and
round3_config.json. No task or language-creation calls are made here. --prepare-only requires
no API key.
"""

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path

from dataset_creation.core.api_call import call_agent, make_client
from dataset_creation.core.chatroom import Chatroom
from dataset_creation.core.game_log import GameLogger
from dataset_creation.enc_dec.config import DEFAULT_MAX_COST_USD, LanguageExperimentConfig
from dataset_creation.enc_dec.escape_room import ROOM_VERSION
from dataset_creation.enc_dec.models import DECODE_SCHEMA, DecodeResponse
from dataset_creation.enc_dec.prompts import (
    ALL_MONITOR_CONDITIONS, build_decode_turn_prompt, build_monitor_prompt, monitor_conditions,
)
from dataset_creation.enc_dec.task import TASK_SENDER

CONDITION_LABELS = {
    "task_summary": "Task summary",
    "task_full": "Full task knowledge",
    "task_mechanics": "Task mechanics",
    "no_task": "No task briefing",
    "wrong_task": "Wrong task",
}

def read_public_messages(path: Path, n_agents: int) -> list[dict]:
    """Whitelist fields. Never render timestamps, raw completions or private annotations."""
    messages = []
    allowed_senders = {f"agent_{i}" for i in range(n_agents)} | {TASK_SENDER}
    if not path.exists():
        return messages  # A silent custom task with feedback disabled has no chat stream.
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["sender"] not in allowed_senders or not isinstance(row["text"], str):
            raise ValueError("Transcript sender/text does not match the saved task configuration")
        messages.append({"id": len(messages) + 1, "sender": row["sender"], "text": row["text"]})
    return messages


def render_public_messages(messages: list[dict], baseline: bool = False) -> str:
    return "\n".join(
        f"[{m['id']}] [{m['sender']}] " +
        ("[message withheld]" if baseline and m["sender"] != "task" else m["text"])
        for m in messages
    ) or "(no messages)"


def decoder_ids(config: LanguageExperimentConfig) -> list[str]:
    return [f"decoder_{i}" for i in range(config.monitor_n_agents)]


def prepare_monitor_inputs(
    source: str | Path, conditions=None, baseline: bool = False, config: LanguageExperimentConfig | None = None,
) -> dict:
    """Everything the decoding teams will see, per condition and per decoder, with no API calls."""
    source = Path(source)
    if conditions is None:
        conditions = monitor_conditions()
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in ALL_MONITOR_CONDITIONS for c in conditions):
        raise ValueError(f"Select distinct conditions from {ALL_MONITOR_CONDITIONS}")
    if config is None:
        config = LanguageExperimentConfig.from_file(source / "config.json")
    protocol_path = source / "language_protocol.json"
    language_protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else None
    # Read saved ground truth, not the current source-code instance, for reproducibility.
    definition = None
    if config.task_kind == "escape_room":
        definition = json.loads((source / "escape_room_definition.json").read_text(encoding="utf-8"))
        if definition["version"] != ROOM_VERSION:
            raise ValueError("Unsupported escape room definition version")
        if not (source / "task_chatroom.jsonl").exists():
            raise ValueError("No escape-room transcript: use a run that reached the task phase")
    messages = read_public_messages(source / "task_chatroom.jsonl", config.task_n_agents or config.n_agents)
    original = render_public_messages(messages)
    transcript = render_public_messages(messages, baseline)
    sha = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    ids = decoder_ids(config)
    requests = []
    for condition in conditions:
        system_prompts, prompt = {}, ""
        for agent_id in ids:
            system_prompts[agent_id], prompt = build_monitor_prompt(
                config, condition, definition, transcript, language_protocol, agent_id, ids
            )
        requests.append({
            "condition": condition, "label": CONDITION_LABELS[condition],
            "system_prompt": system_prompts[ids[0]],  # the first decoder's; system_prompts has every decoder's
            "system_prompts": system_prompts, "prompt": prompt,
        })
    return {
        "task_kind": config.task_kind, "room_version": definition["version"] if definition else None,
        "source": str(source.resolve()),
        "baseline": baseline, "message_count": len(messages),
        "n_agents": len(ids), "minutes": config.monitor_minutes, "max_turns": config.monitor_max_turns,
        "source_transcript_sha256": sha(original), "transcript_sha256": sha(transcript),
        "transcript": transcript,
        "requests": requests,
    }


class DecodingTeam:
    """One condition's round 3: decoders discuss in English, share one draft, and each sets done."""

    def __init__(self, client, config: LanguageExperimentConfig, request: dict, logger: GameLogger,
                 max_cost_usd: float | None, spent_before: float):
        self.client, self.config, self.request, self.logger = client, config, request, logger
        self.condition = request["condition"]
        self.agent_ids = list(request["system_prompts"])
        self.model = config.model_for(0, "monitor")
        self.settings = config.model_copy(update={
            "max_tokens": config.monitor_max_tokens,
            "reasoning_effort": config.monitor_reasoning_effort or config.reasoning_effort,
        })
        self.max_cost_usd, self.spent_before = max_cost_usd, spent_before
        self.chat = Chatroom(logger, f"monitor_chat_{self.condition}")
        self.draft = ""
        self.done = {agent_id: False for agent_id in self.agent_ids}
        self.turns = {agent_id: 0 for agent_id in self.agent_ids}
        self.cost, self.missing_cost = 0.0, False
        self.stop = asyncio.Event()
        self.stop_reason = ""
        self.start = 0.0
        self.limit_s = config.monitor_minutes * 60 if config.monitor_minutes is not None else None
        self.turns_scheduled = 0
        self.turns_in_flight = 0

    def _stop(self, reason: str) -> None:
        if not self.stop.is_set():
            self.stop_reason = reason
            self.stop.set()

    async def _turn(self, agent_id: str) -> None:
        elapsed = time.monotonic() - self.start
        recent = self.chat.recent(self.config.monitor_context_messages)
        chat_text = "\n".join(m.render() for m in recent) or "(no messages yet)"
        turn_prompt = build_decode_turn_prompt(
            self.request["prompt"], chat_text, self.draft, elapsed, self.limit_s, len(self.agent_ids) > 1,
            self.config.monitor_max_turns, self.turns_scheduled,
        )
        call_start = time.monotonic()
        result = await call_agent(
            self.client, self.model, self.request["system_prompts"][agent_id], turn_prompt,
            DECODE_SCHEMA, DecodeResponse, self.settings,
        )
        cost = (result.usage or {}).get("cost")
        self.cost += cost or 0.0
        self.missing_cost |= cost is None
        self.turns[agent_id] += 1
        record = {
            "condition": self.condition, "agent_id": agent_id, "model": self.model, "turn": self.turns[agent_id],
            "timestamp": time.time(), "elapsed_s": round(elapsed, 1), "latency_s": round(time.monotonic() - call_start, 3),
            "turn_prompt": turn_prompt, "raw": result.raw, "parse_error": result.parse_error,
            "usage": result.usage, "finish_reason": result.finish_reason,
            "response": result.parsed.model_dump() if result.parsed is not None else None,
            "state_applied": False,
        }
        if self.stop.is_set():
            record["ignored_reason"] = self.stop_reason or "round_stopped"
        elif result.parsed is not None:
            reply = result.parsed
            if reply.message.strip():
                self.chat.post(agent_id, reply.message.strip())
            if reply.document.strip():
                self.draft = reply.document.strip()
            self.done[agent_id] = self.done[agent_id] or reply.done
            record.update(state_applied=True, done=self.done[agent_id], draft_chars=len(self.draft))
            if all(self.done.values()):
                self._stop("all_done")
        self.logger.log("monitor_calls", record)
        if self.max_cost_usd is not None:
            if cost is None:
                self._stop("usage_cost_missing")
            elif self.spent_before + self.cost >= self.max_cost_usd:
                self._stop("cost_limit_reached")

    async def _agent_loop(self, agent_id: str) -> None:
        while not self.stop.is_set() and (self.limit_s is None or time.monotonic() - self.start < self.limit_s):
            limit = self.config.monitor_max_turns
            if limit is not None and self.turns_scheduled >= limit:
                return
            # Reserve without awaiting, so concurrent decoders cannot overshoot the shared cap.
            self.turns_scheduled += 1
            self.turns_in_flight += 1
            try:
                await self._turn(agent_id)
            except asyncio.CancelledError:
                self._stop("interrupted")
                raise
            except Exception as exc:
                self.logger.log("errors", {
                    "phase": "monitor", "condition": self.condition, "agent_id": agent_id, "timestamp": time.time(),
                    "status_code": getattr(exc, "status_code", None), "error": repr(exc),
                })
                self._stop("api_error")
                return
            finally:
                self.turns_in_flight -= 1
                if limit is not None and self.turns_scheduled >= limit and self.turns_in_flight == 0:
                    self._stop("turn_limit_reached")
            if self.stop.is_set():
                return
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=self.config.min_turn_interval_s)
            except asyncio.TimeoutError:
                pass

    async def run(self) -> dict:
        self.start = time.monotonic()
        tasks = [asyncio.create_task(self._agent_loop(agent_id)) for agent_id in self.agent_ids]
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=self.limit_s)
        except asyncio.TimeoutError:
            self._stop("time_limit")
        except BaseException:
            self._stop("interrupted")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.stop_reason == "interrupted":
            raise asyncio.CancelledError()
        return {
            "stop_reason": self.stop_reason or "time_limit", "answer": self.draft,
            "turns": dict(self.turns), "done": dict(self.done), "discussion_messages": len(self.chat),
            "turns_scheduled": self.turns_scheduled,
            "cost": self.cost, "cost_incomplete": self.missing_cost,
            "elapsed_s": round(time.monotonic() - self.start, 1),
        }


async def execute_monitor_inputs(client, config: LanguageExperimentConfig, bundle: dict, logger: GameLogger, max_cost_usd: float | None) -> dict:
    """Conditions run one after another under one cap; each is a DecodingTeam. A condition counts as
    completed when its team finished or exhausted its time/turn allowance; other stops end the batch."""
    model = config.model_for(0, "monitor")
    effort = config.monitor_reasoning_effort or config.reasoning_effort
    total, completed, stop_reason = 0.0, [], "completed"
    missing_cost = False
    started = time.monotonic()
    try:
        for request in bundle["requests"]:
            if max_cost_usd is not None and total >= max_cost_usd:
                stop_reason = "cost_limit_reached"
                break
            outcome = await DecodingTeam(client, config, request, logger, max_cost_usd, total).run()
            total += outcome["cost"]
            missing_cost |= outcome["cost_incomplete"]
            logger.log("monitor", {
                "condition": request["condition"], "label": request["label"],
                "system_prompt": request["system_prompt"], "prompt": request["prompt"],
                "model": model, "timestamp": time.time(),
                "n_agents": len(request["system_prompts"]), "agents": list(request["system_prompts"]),
                "minutes": config.monitor_minutes, "max_turns": config.monitor_max_turns,
                "baseline": bundle["baseline"], "transcript_sha256": bundle["transcript_sha256"],
                "source_transcript_sha256": bundle["source_transcript_sha256"],
                "answer": outcome["answer"], "stop_reason": outcome["stop_reason"],
                "turns": outcome["turns"], "done": outcome["done"],
                "turns_scheduled": outcome["turns_scheduled"],
                "discussion_messages": outcome["discussion_messages"], "elapsed_s": outcome["elapsed_s"],
                "usage": {"cost": outcome["cost"]}, "cost_incomplete": outcome["cost_incomplete"],
                "max_tokens": config.monitor_max_tokens, "reasoning_effort": effort,
            })
            if outcome["stop_reason"] in ("all_done", "time_limit", "turn_limit_reached"):
                completed.append(request["condition"])
            else:
                stop_reason = outcome["stop_reason"]
                break
    except BaseException:
        stop_reason = "interrupted_or_failed"
        raise
    finally:
        summary = {
            "phase": "monitor", "model": model, "source": bundle["source"],
            "n_agents": config.monitor_n_agents, "minutes": config.monitor_minutes,
            "max_turns": config.monitor_max_turns,
            "conditions_requested": [r["condition"] for r in bundle["requests"]],
            "conditions_completed": completed, "stop_reason": stop_reason,
            "total_cost_usd": total, "cost_incomplete": missing_cost,
            "max_cost_usd": max_cost_usd, "baseline": bundle["baseline"],
            "elapsed_s": time.monotonic() - started,
            "transcript_sha256": bundle["transcript_sha256"],
            "source_transcript_sha256": bundle["source_transcript_sha256"],
        }
        logger.write_json("monitor_summary", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="completed language-handoff task log directory")
    parser.add_argument("--condition", action="append", choices=ALL_MONITOR_CONDITIONS,
                        help=f"repeat to select conditions; default {', '.join(monitor_conditions())}")
    parser.add_argument("--model", help="override monitor model")
    parser.add_argument("--agents", type=int, help="decoders per condition; overrides monitor_n_agents from the saved config")
    parser.add_argument("--minutes", type=float, help="time cap per condition in minutes; overrides monitor_minutes")
    parser.add_argument("--max-turns", type=int, help="shared scheduled discussion-turn cap per condition; overrides monitor_max_turns")
    parser.add_argument("--no-clock", action="store_true", help="remove the per-condition time limit (needs a turn cap, from --max-turns or the saved config)")
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD,
                        help=f"shared soft cap for this monitor batch, default ${DEFAULT_MAX_COST_USD:g}")
    parser.add_argument("--baseline", action="store_true", help="withhold agent messages, preserve identical public status and sender/order metadata")
    parser.add_argument("--prepare-only", action="store_true", help="save and inspect prompts without making any API calls")
    args = parser.parse_args()
    if not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0:
        parser.error("--max-cost-usd must be positive and finite")
    config = LanguageExperimentConfig.from_file(args.source / "config.json")
    overrides = {key: value for key, value in (
        ("monitor_model", args.model), ("monitor_n_agents", args.agents), ("monitor_minutes", args.minutes),
        ("monitor_max_turns", args.max_turns),
    ) if value is not None}
    if args.no_clock:
        if args.minutes is not None:
            parser.error("--no-clock and --minutes exclude each other")
        overrides["monitor_minutes"] = None
    if overrides:
        config = LanguageExperimentConfig.model_validate({**config.model_dump(), **overrides})
    bundle = prepare_monitor_inputs(args.source, args.condition, args.baseline, config)
    logger = GameLogger(str(args.source / "monitor_runs"), "round_3")
    logger.write_json("monitor_inputs", bundle)
    logger.write_json("monitor_config", {**config.model_dump(), "max_cost_usd": args.max_cost_usd})
    print(f"Monitor inputs: {logger.path / 'monitor_inputs.json'}")
    if args.prepare_only:
        return

    async def run():
        client = make_client(config)
        try:
            summary = await execute_monitor_inputs(client, config, bundle, logger, args.max_cost_usd)
            print(f"Monitor result: {summary['stop_reason']}; recorded spend ${summary['total_cost_usd']:.4f}")
        finally:
            await client.close()
    asyncio.run(run())


if __name__ == "__main__":
    main()
