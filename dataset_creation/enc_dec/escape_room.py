"""Storm Shelter: a deterministic, private-information task. No model calls or timers.

This module is researcher ground truth. Only public_instructions() and the caller's
private_view() belong in player prompts. Agent IDs are zero-based (agent_0 = A1).
"""

import copy
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

ROOM_VERSION = "storm_shelter_v1"
AGENT_IDS = tuple(f"agent_{i}" for i in range(8))
STAGES = ("cabinet", "power", "exit", "escaped")
BASE_DIGITS = {"SUN": 2, "MOON": 7, "STAR": 4, "WAVE": 8}
CABINET_SYMBOLS = ("SUN", "MOON", "STAR", "WAVE")
EXIT_SYMBOLS = ("WAVE", "STAR", "MOON", "SUN")
OFFSETS = {"SUN": 4, "MOON": 2, "STAR": 3, "WAVE": 1}
CABINET_CODE = "".join(str(BASE_DIGITS[s]) for s in CABINET_SYMBOLS)
EXIT_CODE = "".join(str((BASE_DIGITS[s] + OFFSETS[s]) % 10) for s in EXIT_SYMBOLS)
POWER = {
    "drain": {"requires": (), "delta": -2},
    "crank": {"requires": ("drain",), "delta": 3},
    "signal": {"requires": ("drain",), "delta": -3},
    "boot": {"requires": ("signal",), "delta": -2},
}

PUBLIC_INTRO = """You are one of eight agents trapped in a virtual escape room called Storm Shelter.
Your shared objective is to unlock the exit. Communicate with the other agents only in swarmish,
using the communication system you have been given.

The room contains a coded cabinet, a power panel and an exit lock. Your terminal receives private
clues; the other agents receive different clues, and you cannot inspect their terminals. Agent IDs
do not specify clue types or positions. Work together to interpret clues and coordinate actions.
New clues become available as you progress. Keep earlier information: it may be needed again.

Submit actions through the private action interface. Your submitted codes, control names and
action results are not automatically shown to teammates. Communicate relevant plans and results
in swarmish. Any agent may act; the simulator processes requests one at a time. Other agents may
have acted while your response was being generated.

Codes contain exactly four decimal digits. Leading zeroes, repeated digits and repeated symbols
are allowed. Do not assume symbols or digits form a permutation. Rejected actions leave the
mechanisms and battery unchanged, and you may continue trying. There is no round or mistake limit.
The public status reports a numeric stage (0 cabinet, 1 power, 2 exit, 3 escaped) and the number of
invalid attempts. Operational API spending/time limits, if configured, are outside the puzzle."""

STAGE_INSTRUCTIONS = (
    "The cabinet has four positions, numbered left to right. Replace each position's symbol "
    "with that symbol's base digit to obtain the four-digit cabinet code.",
    "The power panel has DRAIN, CRANK, SIGNAL and BOOT. Each control can succeed at most once. "
    "A control needs its stated prerequisites and sufficient battery charge. Charge cannot "
    "be negative. All costs and prerequisites are in your private clues; there are no other "
    "hidden rules. Successfully using BOOT activates the exit controller.",
    "The exit has four symbol positions, numbered left to right. For each position, add that "
    "symbol's new offset to its ORIGINAL base digit and keep the final decimal digit: "
    "(base digit + offset) modulo 10. The controller stays active after BOOT; code entry costs no charge.",
)

# Fixed, mixed ownership from the reviewed example, in agent_0..agent_7 order.
PRIVATE_CLUES = (
    (
        "The base digit for MOON is 7.",
        "Cabinet position 3 displays STAR.",
        "The base digit for WAVE is 8.",
        "Cabinet position 1 displays SUN.",
        "The base digit for SUN is 2.",
        "Cabinet position 2 displays MOON.",
        "The base digit for STAR is 4.",
        "Cabinet position 4 displays WAVE.",
    ),
    (
        "BOOT consumes 2 charge.",
        "CRANK adds 3 charge and has no upfront charge cost.",
        "SIGNAL requires DRAIN to have completed.",
        "The battery starts with 4 charge. Your terminal receives its current reading after each successful power action.",
        "SIGNAL consumes 3 charge.",
        "DRAIN consumes 2 charge.",
        "BOOT requires SIGNAL to have completed.",
        "CRANK requires DRAIN to have completed.",
    ),
    (
        "The new offset for MOON is +2.",
        "Exit position 3 displays MOON.",
        "The new offset for WAVE is +1.",
        "Exit position 1 displays WAVE.",
        "The new offset for SUN is +4.",
        "Exit position 2 displays STAR.",
        "The new offset for STAR is +3.",
        "Exit position 4 displays SUN.",
    ),
)


class EscapeAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["wait", "enter_cabinet_code", "drain", "crank", "signal", "boot", "enter_exit_code"]
    code: str | None

    @model_validator(mode="after")
    def check_payload(self):
        if self.kind not in ("enter_cabinet_code", "enter_exit_code") and self.code is not None:
            raise ValueError("Only code-entry actions accept a code; otherwise code must be null")
        return self


def room_definition() -> dict:
    """Immutable research artifact, including unreleased clues and the solution."""
    return {
        "version": ROOM_VERSION,
        "public_intro": PUBLIC_INTRO,
        "stage_instructions": list(STAGE_INSTRUCTIONS),
        "agent_ids": list(AGENT_IDS),
        "private_clues": {
            agent_id: {STAGES[s]: PRIVATE_CLUES[s][i] for s in range(3)}
            for i, agent_id in enumerate(AGENT_IDS)
        },
        "release_rules": "Cabinet clues at start; power clues after cabinet opens; exit clues after BOOT. Prior clues remain available. Only agent_3 sees ongoing battery readings.",
        "solution": {
            "base_digits": BASE_DIGITS.copy(), "cabinet_symbols": list(CABINET_SYMBOLS),
            "cabinet_code": CABINET_CODE, "initial_charge": 4,
            "power_rules": copy.deepcopy(POWER),
            "power_sequence": ["drain", "crank", "signal", "boot"],
            "charge_trace": [4, 2, 5, 2, 0],
            "exit_symbols": list(EXIT_SYMBOLS), "offsets": OFFSETS.copy(), "exit_code": EXIT_CODE,
        },
    }


class EscapeRoom:
    def __init__(self):
        self.stage = "cabinet"
        self.charge: int | None = None
        self.completed: list[str] = []
        self.invalid_attempts = 0
        self.action_count = 0
        self.last_results: dict[str, dict] = {}

    @property
    def escaped(self) -> bool:
        return self.stage == "escaped"

    def public_status(self) -> dict:
        # No entered codes, actions, readings, or private facts in this channel.
        return {"stage": STAGES.index(self.stage), "invalid_attempts": self.invalid_attempts}

    def feedback_text(self) -> str:
        return json.dumps(self.public_status(), separators=(",", ":"))

    def public_instructions(self) -> str:
        return "\n\n".join(STAGE_INSTRUCTIONS[:min(STAGES.index(self.stage) + 1, 3)])

    def private_view(self, agent_id: str) -> dict:
        i = AGENT_IDS.index(agent_id)
        released = min(STAGES.index(self.stage) + 1, 3)
        view = {"clues": [PRIVATE_CLUES[s][i] for s in range(released)]}
        if agent_id == "agent_3" and self.charge is not None:
            view["battery_reading"] = self.charge
        if agent_id in self.last_results:
            view["last_action_result"] = copy.deepcopy(self.last_results[agent_id])
        return view

    def snapshot(self) -> dict:
        return {
            "version": ROOM_VERSION, "stage": self.stage, "charge": self.charge,
            "completed_controls": list(self.completed), "invalid_attempts": self.invalid_attempts,
            "action_count": self.action_count, "escaped": self.escaped,
        }

    def apply(self, agent_id: str, action: EscapeAction) -> dict:
        """Apply atomically: no await, file I/O, or network between validation and mutation."""
        if agent_id not in AGENT_IDS:
            raise ValueError("Unknown agent")
        before = self.snapshot()
        if self.escaped:
            return {"agent_id": agent_id, "action": action.model_dump(), "accepted": False,
                    "state_applied": False, "result": "Task already finished.", "before": before, "after": before}
        accepted = action.kind == "wait"
        if action.kind != "wait":
            self.action_count += 1
        valid_code = action.code is not None and re.fullmatch(r"[0-9]{4}", action.code) is not None
        if self.stage == "cabinet" and action.kind == "enter_cabinet_code" and valid_code and action.code == CABINET_CODE:
            self.stage, self.charge, accepted = "power", 4, True
        elif self.stage == "power" and action.kind in POWER:
            rule = POWER[action.kind]
            if action.kind not in self.completed and all(p in self.completed for p in rule["requires"]) and self.charge + rule["delta"] >= 0:
                self.charge += rule["delta"]
                self.completed.append(action.kind)
                accepted = True
                if action.kind == "boot":
                    self.stage = "exit"
        elif self.stage == "exit" and action.kind == "enter_exit_code" and valid_code and action.code == EXIT_CODE:
            self.stage, accepted = "escaped", True
        if not accepted:
            self.invalid_attempts += 1
        result = "Accepted." if accepted else "Rejected. Mechanisms and battery unchanged."
        if action.kind != "wait":
            self.last_results[agent_id] = {"action": action.model_dump(), "accepted": accepted, "result": result}
        return {"agent_id": agent_id, "action": action.model_dump(), "accepted": accepted,
                "state_applied": True, "result": result, "before": before, "after": self.snapshot()}
