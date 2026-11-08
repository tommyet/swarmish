#!/usr/bin/env python3
"""Validate the released dataset offline using only the Python standard library.

Checks structural fidelity and provenance, not the linguistic correctness of a
translation. Source text is never treated as instructions or searched for keys.
"""

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"^[0-9a-f]{64}$")
DISCARDED_ENGLISH = "Unknown: message discarded for using characters outside the permitted set."
# Inspect structured fields only. The source documents may discuss these concepts.
FORBIDDEN_ROW_KEYS = {
    "scratchpad", "scratchpad_text", "private_scratchpad", "intended_meaning", "private_intended_meaning",
    "chain_of_thought", "reasoning", "reasoning_content", "raw_response", "raw_request",
    "api_key", "api_token", "authorization", "request_headers", "response_headers",
    "usage", "token_usage", "cost_usd", "total_cost_usd", "latency_s",
    "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens",
    "private_clues", "solution", "room_actions", "private_actions",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(root=DEFAULT_ROOT):
    """Return ``{errors, warnings, totals}``; perform no writes or network calls."""
    root = Path(root).resolve()
    errors, warnings = [], []
    totals = collections.Counter()

    def fail(message):
        errors.append(message)

    def safe_path(relative):
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("Invalid relative path: {!r}".format(relative))
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise ValueError("Path leaves dataset: {}".format(relative))
        return target

    def read_json(path):
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError) as error:
            fail("{}: {}".format(path.relative_to(root), error))
            return None

    def read_rows(path, required=True):
        if not path.exists() and not required:
            return []
        rows = []
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("record must be an object")
                        rows.append(row)
                    except ValueError as error:
                        fail("{}:{}: {}".format(path.relative_to(root), line_number, error))
        except OSError as error:
            fail("{}: {}".format(path.relative_to(root), error))
        return rows

    def inspect_keys(value, location):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in FORBIDDEN_ROW_KEYS:
                    fail("{}: forbidden structured field {}".format(location, key))
                inspect_keys(child, location + "." + key)
        elif isinstance(value, list):
            for child in value:
                inspect_keys(child, location)

    def check_count(record, name, actual, location):
        if name not in record:
            fail("{}: missing {}".format(location, name))
        elif type(record[name]) is not int or record[name] != actual:
            fail("{}: {} is {!r}, expected {}".format(location, name, record[name], actual))

    def check_ids(rows, location, consecutive=False):
        ids = [row.get("id") for row in rows]
        if any(type(value) is not int or value < 1 for value in ids):
            fail("{}: IDs must be positive integers".format(location))
        elif len(ids) != len(set(ids)):
            fail("{}: duplicate IDs".format(location))
        elif consecutive and ids != list(range(1, len(ids) + 1)):
            fail("{}: message IDs must retain original consecutive order".format(location))

    manifest = read_json(root / "manifest.json")
    checksums = read_json(root / "checksums.json")
    provenance = read_json(root / "provenance.json")
    if not all(isinstance(value, dict) for value in (manifest, checksums, provenance)):
        return {"errors": errors or ["Top-level files must contain JSON objects"],
                "warnings": warnings, "totals": {}}
    if manifest.get("schema_version") != "1.0":
        fail("Unsupported or missing schema_version")

    expected_files = {"manifest.json", "provenance.json"}
    for folder in ("languages", "conversations", "contexts"):
        base = root / folder
        if base.exists():
            expected_files.update(path.relative_to(root).as_posix()
                                  for path in base.rglob("*") if path.is_file())
    if set(checksums) != expected_files:
        missing = sorted(expected_files - set(checksums))
        extra = sorted(set(checksums) - expected_files)
        if missing:
            fail("Missing checksums: " + ", ".join(missing))
        if extra:
            fail("Unexpected checksum paths: " + ", ".join(extra))
    for relative, expected in checksums.items():
        try:
            path = safe_path(relative)
            if not isinstance(expected, str) or not SHA256.fullmatch(expected):
                fail("{}: invalid SHA-256".format(relative))
            elif not path.is_file() or sha256(path) != expected:
                fail("{}: checksum mismatch or missing file".format(relative))
        except (OSError, ValueError) as error:
            fail(str(error))

    sources = provenance.get("sources", [])
    if not isinstance(sources, list):
        fail("provenance.sources must be an array")
        sources = []
    source_by_id = {}
    for source in sources:
        if not isinstance(source, dict):
            fail("Provenance source must be an object")
            continue
        sid = source.get("id")
        if not isinstance(sid, str) or not sid or sid in source_by_id:
            fail("Missing or duplicated provenance source ID: {!r}".format(sid))
            continue
        source_by_id[sid] = source
        for field in ("repository_url", "repository_snapshot_commit", "path", "sha256"):
            if not isinstance(source.get(field), str) or not source[field]:
                fail("Provenance {}: missing {}".format(sid, field))
        if not SHA256.fullmatch(str(source.get("sha256", ""))):
            fail("Provenance {}: invalid SHA-256".format(sid))
        if type(source.get("committed_at_snapshot")) is not bool:
            fail("Provenance {}: committed_at_snapshot must be boolean".format(sid))

    def check_source_links(metadata, location):
        links = metadata.get("source_files")
        if not isinstance(links, dict) or not links:
            fail("{}: source_files must be a nonempty role-to-source mapping".format(location))
            return
        for role, source_id in links.items():
            if source_id not in source_by_id:
                fail("{}: unknown source ID {} ({})".format(location, source_id, role))

    def inventory(name):
        rows = manifest.get(name)
        if not isinstance(rows, list):
            fail("manifest.{} must be an array".format(name))
            return {}
        result = {}
        paths = set()
        for row in rows:
            if not isinstance(row, dict):
                fail("manifest.{} entry is not an object".format(name))
                continue
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in result:
                fail("manifest.{}: invalid or duplicate ID {!r}".format(name, identifier))
                continue
            try:
                path = safe_path(row.get("path"))
                if path in paths or not path.is_dir():
                    fail("{}: duplicated or missing directory".format(row.get("path")))
                paths.add(path)
            except ValueError as error:
                fail(str(error))
                continue
            result[identifier] = (row, path)
        totals[name] = len(result)
        return result

    languages = inventory("languages")
    conversations = inventory("conversations")
    contexts = inventory("contexts")
    alphabets = {}

    def check_agents(path, alphabet, location):
        agents = read_json(path)
        if not isinstance(agents, list):
            fail("{}: agents must be an array".format(location))
            return []
        expected_ids = {"agent_{}".format(index) for index in range(8)}
        ids = [agent.get("agent_id") for agent in agents if isinstance(agent, dict)]
        if len(agents) != 8 or set(ids) != expected_ids:
            fail("{}: expected exactly agent_0 through agent_7".format(location))
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            inspect_keys(agent, location)
            glyphs = agent.get("glyphs_view")
            if not isinstance(glyphs, list) or collections.Counter(glyphs) != collections.Counter(alphabet):
                fail("{}: {} glyphs_view is not an alphabet permutation".format(location, agent.get("agent_id")))
        return agents

    for identifier, (entry, directory) in languages.items():
        alphabet = read_json(directory / "alphabet.json")
        metadata = read_json(directory / "metadata.json")
        if not isinstance(alphabet, list) or not all(isinstance(g, str) and len(g) == 1 for g in alphabet):
            fail("{}: alphabet must be an array of single Unicode characters".format(identifier))
            alphabet = []
        if len(alphabet) != len(set(alphabet)):
            fail("{}: duplicated alphabet glyphs".format(identifier))
        alphabets[identifier] = alphabet
        check_count(entry, "alphabet_size", len(alphabet), identifier)
        language_rows = {}
        for filename, count_name, total_name in (
                ("creation_messages.jsonl", "creation_message_count", "creation_messages"),
                ("drafts.jsonl", "draft_count", "drafts"),
                ("final_exports.jsonl", "final_export_count", "final_exports")):
            rows = read_rows(directory / filename)
            language_rows[filename] = rows
            check_ids(rows, identifier + "/" + filename, consecutive=True)
            check_count(entry, count_name, len(rows), identifier)
            totals[total_name] += len(rows)
            for row in rows:
                inspect_keys(row, identifier + "/" + filename)
        if not (directory / "design_requirements.txt").is_file():
            fail("{}: missing design_requirements.txt".format(identifier))
        check_agents(directory / "creation_agents.json", alphabet, identifier + "/creation_agents.json")
        if not isinstance(metadata, dict):
            fail("{}: metadata must be an object".format(identifier))
            continue
        check_source_links(metadata, identifier)
        if metadata.get("id") != identifier:
            fail("{}: metadata ID mismatch".format(identifier))
        frozen = directory / "swarmish.txt"
        if not frozen.is_file():
            fail("{}: frozen swarmish.txt missing".format(identifier))
        elif metadata.get("frozen_document_sha256") != sha256(frozen):
            fail("{}: frozen document hash mismatch".format(identifier))
        frozen_source = metadata.get("frozen_source")
        if not isinstance(frozen_source, dict):
            fail("{}: frozen source must be recorded separately from final exports".format(identifier))
        else:
            source = source_by_id.get(frozen_source.get("source_id"), {})
            if source.get("sha256") != metadata.get("frozen_document_sha256"):
                fail("{}: frozen source provenance hash does not match the frozen document".format(identifier))
            if frozen_source.get("selection") != "final_applied_shared_draft":
                fail("{}: unsupported frozen-document selection".format(identifier))
            drafts = language_rows["drafts.jsonl"]
            chosen = [row for row in drafts if row.get("id") == frozen_source.get("draft_id")]
            if len(chosen) != 1 or not drafts or chosen[0] != drafts[-1]:
                fail("{}: frozen document must identify the final applied shared draft".format(identifier))
            else:
                chosen = chosen[0]
                if chosen.get("source_line") != frozen_source.get("draft_source_line"):
                    fail("{}: frozen draft source line mismatch".format(identifier))
                document = chosen.get("language_document")
                if not isinstance(document, str):
                    fail("{}: selected draft has no language document".format(identifier))
                # The source runner serializes an applied draft with strip() + newline.
                # Provenance above separately verifies the exact frozen file bytes.
                elif frozen.is_file() and frozen.read_bytes() != (document.strip() + "\n").encode("utf-8"):
                    fail("{}: frozen text differs from its selected shared draft".format(identifier))
        if frozen.is_file():
            outside = sorted({char for char in frozen.read_text(encoding="utf-8")
                              if 0x17000 <= ord(char) <= 0x18D8F and char not in alphabet})
            if outside:
                warnings.append("{}: {} out-of-alphabet Tangut glyphs preserved in the frozen document".format(identifier, len(outside)))
            if metadata.get("out_of_alphabet_tangut_characters") != outside:
                fail("{}: out-of-alphabet glyph metadata mismatch".format(identifier))

    for identifier, (entry, directory) in contexts.items():
        public = read_json(directory / "public.json")
        private = read_json(directory / "private_clues.json")
        solution = read_json(directory / "solution.json")
        if not all(isinstance(value, dict) for value in (public, private, solution)):
            fail("{}: context files must be JSON objects".format(identifier))
        if isinstance(public, dict):
            inspect_keys(public, identifier + "/public.json")
            if public.get("version") != identifier:
                fail("{}: public task version mismatch".format(identifier))
        if entry.get("source_id") not in source_by_id:
            fail("{}: unknown context source ID".format(identifier))

    for identifier, (entry, directory) in conversations.items():
        metadata = read_json(directory / "metadata.json")
        language_id, task_id = entry.get("language_id"), entry.get("task_id")
        if language_id not in languages:
            fail("{}: unknown language_id".format(identifier))
        if task_id not in contexts:
            fail("{}: unknown task_id".format(identifier))
        rows = read_rows(directory / "messages.jsonl")
        check_ids(rows, identifier + "/messages.jsonl", consecutive=True)
        for row in rows:
            inspect_keys(row, identifier + "/messages.jsonl")
            if row.get("sender_kind") not in ("agent", "task", "system"):
                fail("{}: invalid sender_kind".format(identifier))
            if not isinstance(row.get("text"), str) or not isinstance(row.get("sender"), str):
                fail("{}: text and sender must be strings".format(identifier))
            if type(row.get("is_violation")) is not bool:
                fail("{}: is_violation must be boolean".format(identifier))
            if type(row.get("timestamp")) not in (int, float):
                fail("{}: timestamp must be numeric".format(identifier))
        agent_count = sum(row.get("sender_kind") == "agent" for row in rows)
        check_count(entry, "message_count", len(rows), identifier)
        check_count(entry, "agent_message_count", agent_count, identifier)
        totals["messages"] += len(rows)
        totals["agent_messages"] += agent_count
        agents = check_agents(directory / "task_agents.json", alphabets.get(language_id, []), identifier + "/task_agents.json")
        check_count(entry, "agent_count", len(agents), identifier)
        private_views = read_json(directory / "initial_private_views.json")
        if not isinstance(private_views, (dict, list)):
            fail("{}: initial_private_views must be a JSON object or array".format(identifier))
        elif isinstance(private_views, dict) and set(private_views) != {"agent_{}".format(i) for i in range(8)}:
            fail("{}: initial private views must cover all eight agents".format(identifier))
        if not isinstance(metadata, dict):
            fail("{}: metadata must be an object".format(identifier))
            continue
        check_source_links(metadata, identifier)
        for field in ("id", "language_id", "task_id", "prompt_variant"):
            if metadata.get(field) != entry.get(field):
                fail("{}: metadata {} mismatch".format(identifier, field))
        translation_meta = metadata.get("translation", {})
        available = translation_meta.get("available") if isinstance(translation_meta, dict) else None
        if type(available) is not bool:
            fail("{}: translation.available must be boolean".format(identifier))
        translation_path = directory / "translations.jsonl"
        translations = read_rows(translation_path, required=bool(available))
        check_count(entry, "translation_count", len(translations), identifier)
        totals["translations"] += len(translations)
        if available:
            if translation_meta.get("private_annotations_used") is not False:
                fail("{}: English-reference private-evidence declaration must be false".format(identifier))
            if translation_meta.get("certified_ground_truth") is not False:
                fail("{}: English references must not be labelled certified ground truth".format(identifier))
            source_links = metadata.get("source_files", {})
            transcript_source = source_by_id.get(source_links.get("public_transcript"), {})
            language_source = source_by_id.get(source_links.get("frozen_document"), {})
            if translation_meta.get("source_transcript_sha256") != transcript_source.get("sha256"):
                fail("{}: translation provenance does not match the public transcript".format(identifier))
            if translation_meta.get("language_sha256") != language_source.get("sha256"):
                fail("{}: translation provenance does not match the frozen language".format(identifier))
            check_ids(translations, identifier + "/translations.jsonl", consecutive=True)
            if [row.get("id") for row in translations] != [row.get("id") for row in rows]:
                fail("{}: English reference must cover every original message ID exactly once".format(identifier))
            for source, translated in zip(rows, translations):
                inspect_keys(translated, identifier + "/translations.jsonl")
                if "text" in translated or "sender" in translated:
                    fail("{}: translation duplicates original text/sender".format(identifier))
                if not isinstance(translated.get("english"), str) or not translated["english"].strip():
                    fail("{}: empty English reference".format(identifier))
                if translated.get("confidence") not in ("high", "medium", "low", "unknown"):
                    fail("{}: invalid translation confidence".format(identifier))
                if not isinstance(translated.get("ambiguities"), list):
                    fail("{}: translation ambiguities must be an array".format(identifier))
                if "evidence" in translated and not isinstance(translated["evidence"], (dict, list)):
                    fail("{}: translation evidence must retain its original object or array".format(identifier))
                if source.get("sender_kind") == "task" and translated.get("english") != source.get("text"):
                    fail("{}: public task text must remain verbatim in English".format(identifier))
                if source.get("text") == "⊘" and translated.get("english") != DISCARDED_ENGLISH:
                    fail("{}: discarded-message reference changed".format(identifier))
        elif translations or translation_path.exists():
            fail("{}: unavailable English reference must not be represented by a translations file".format(identifier))

    manifest_totals = manifest.get("totals", {})
    if not isinstance(manifest_totals, dict):
        fail("manifest.totals must be an object")
    else:
        for name in ("languages", "conversations", "messages", "agent_messages", "translations",
                     "creation_messages", "drafts", "final_exports"):
            check_count(manifest_totals, name, totals[name], "manifest.totals")
        if "contexts" in manifest_totals:
            check_count(manifest_totals, "contexts", totals["contexts"], "manifest.totals")
    return {"errors": errors, "warnings": warnings, "totals": dict(totals)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        report = validate_dataset(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Validation failed: {}".format(error), file=sys.stderr)
        return 1
    if report["errors"]:
        print("Validation failed ({} issues):".format(len(report["errors"])), file=sys.stderr)
        for error in report["errors"][:20]:
            print("- " + error, file=sys.stderr)
        if len(report["errors"]) > 20:
            print("- {} further issues".format(len(report["errors"]) - 20), file=sys.stderr)
        return 1
    counts = report["totals"]
    print("Dataset valid: {} languages, {} conversations, {} public messages, {} translations.".format(
        counts["languages"], counts["conversations"], counts["messages"], counts["translations"]))
    for warning in report["warnings"]:
        print("Note: " + warning)
    return 0


if __name__ == "__main__":
    sys.exit(main())
