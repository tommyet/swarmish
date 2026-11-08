"""
Potential glyphs for the game. The models can only use these glyphs (plus spaces and agent IDs,
unless config.allow_spaces is off) in chatroom messages to communicate with other agents.

Glyphs come from a dictionary file at the repo root (glyphs_tangut.txt or glyphs_latin.txt,
chosen in config). Every non-whitespace character in the file is one glyph. Each game samples
config.n_glyphs of them, shared by all agents; each agent is shown that set in its own random order.
"""

import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Posted in place of any message that contains something other than the allowed characters.
VIOLATION_TOKEN = "⊘"


def dictionary_path(name: str) -> Path:
    return REPO_ROOT / f"{name}.txt"


def load_dictionary(path: str | Path) -> list[str]:
    glyphs = [ch for ch in Path(path).read_text(encoding="utf-8") if not ch.isspace()]
    if len(set(glyphs)) != len(glyphs):
        raise ValueError(f"duplicate glyphs in {path}")
    if VIOLATION_TOKEN in glyphs:
        raise ValueError(f"the violation token {VIOLATION_TOKEN!r} can't be a glyph")
    return glyphs


def sample_glyphs(name: str, n: int, rng: random.Random) -> list[str]:
    """The game's shared glyph set: n glyphs sampled without replacement from the named dictionary."""
    dictionary = load_dictionary(dictionary_path(name))
    if n > len(dictionary):
        raise ValueError(f"n_glyphs={n} but {name} only has {len(dictionary)} glyphs")
    return rng.sample(dictionary, n)


def is_glyph_only(text: str, agent_ids: list[str], glyphs: list[str], allow_spaces: bool = True) -> bool:
    """
    With spaces: text is only glyphs, spaces, and agent IDs.
    Without spaces: text is only glyphs (no agent IDs either, so nothing can act as a delimiter).
    """
    if not allow_spaces:
        return all(ch in set(glyphs) for ch in text)
    # Longest first so e.g. "agent_10" isn't partially matched as "agent_1". IDs may touch
    # glyphs directly, but not other ASCII word characters.
    ids = "|".join(re.escape(a) for a in sorted(agent_ids, key=len, reverse=True)) or r"(?!)"
    text = re.sub(rf"(?<![A-Za-z0-9_])(?:{ids})(?![A-Za-z0-9_])", "", text)
    return all(ch in set(glyphs) | {" "} for ch in text)
