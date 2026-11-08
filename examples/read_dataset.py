#!/usr/bin/env python3
"""Read public Swarmish conversations; English references require explicit opt-in.

This example uses only the Python standard library and works from any directory.
Import ``iter_messages`` or ``load_messages`` to use the same records in an experiment.
Neither function reads a language reference, private clues, or task answers.
"""

import argparse
import json
import sys
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def _root(root=None):
    return Path(root).resolve() if root is not None else DEFAULT_ROOT


def _json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError("{}:{}: blank JSONL record".format(path, line_number))
            yield json.loads(line)


def list_conversations(root=None):
    """Return the manifest's conversation inventory, without reading any content."""
    return _json(_root(root) / "manifest.json")["conversations"]


def _conversation_directory(conversation_id, root=None):
    dataset_root = _root(root)
    matches = [row for row in list_conversations(dataset_root)
               if row["id"] == conversation_id]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicated conversation ID: {}".format(conversation_id))
    directory = (dataset_root / matches[0]["path"]).resolve()
    try:
        directory.relative_to(dataset_root)
    except ValueError:
        raise ValueError("Conversation path is outside the dataset")
    return directory


def load_translations(conversation_id, root=None):
    """Explicitly load reference-assisted English annotations, indexed by message ID.

    An empty mapping means the conversation has no English reference. References
    preserve uncertainty and should not be treated as independently verified truth.
    """
    path = _conversation_directory(conversation_id, root) / "translations.jsonl"
    if not path.exists():
        return {}
    records = list(_jsonl(path))
    translations = {row["id"]: row for row in records}
    if len(translations) != len(records):
        raise ValueError("Duplicate translation message IDs")
    return translations


def iter_messages(conversation_id, root=None, english=False, agents_only=False):
    """Yield public message dictionaries in their original order.

    Original IDs remain unchanged when ``agents_only=True``. ``english=True``
    adds a ``translation`` dictionary (or None if no reference is available).
    The default reads only public ``messages.jsonl`` and the manifest inventory.
    """
    directory = _conversation_directory(conversation_id, root)
    translations = load_translations(conversation_id, root) if english else None
    for row in _jsonl(directory / "messages.jsonl"):
        if agents_only and row["sender_kind"] != "agent":
            continue
        if english:
            row["translation"] = translations.get(row["id"])
        yield row


def load_messages(conversation_id, root=None, english=False, agents_only=False):
    """Return ``iter_messages`` as a list, convenient for small experiments."""
    return list(iter_messages(conversation_id, root, english, agents_only))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conversation_id", nargs="?", default="R01_sol_seed1")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Dataset repository root")
    parser.add_argument("--english", action="store_true",
                        help="Also print reference-assisted English and uncertainty")
    parser.add_argument("--agents-only", action="store_true",
                        help="Omit public task/system messages without renumbering")
    parser.add_argument("--limit", type=int,
                        help="Print at most this many messages")
    parser.add_argument("--list", action="store_true", help="List conversation IDs")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        if args.list:
            for row in list_conversations(args.root):
                print(row["id"])
            return 0
        for index, row in enumerate(iter_messages(
                args.conversation_id, args.root, args.english, args.agents_only)):
            if args.limit is not None and index >= args.limit:
                break
            print("[{id:04d}] {sender}: {text}".format(**row))
            if args.english:
                reference = row["translation"]
                if reference is None:
                    print("  English: no reference available")
                else:
                    print("  English ({confidence}): {english}".format(**reference))
                    for ambiguity in reference.get("ambiguities", []):
                        print("  Uncertainty: {}".format(ambiguity))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Could not read dataset: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
