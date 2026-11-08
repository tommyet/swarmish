#!/usr/bin/env python3
"""Export the first Swarmish corpus from local experiment snapshots (stdlib only).

Sources are read-only. Only explicitly selected fields are published; raw call logs,
private notes, action records, evaluation outputs, costs, and summaries are excluded.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

SCHEMA_VERSION = "1.0"
SOURCE_URL = "https://github.com/erafkin/swarm-glyphs"
BATCH = Path("experiments/tangut_200g_250l_200t")
LANGUAGES = [
    "E01_sol_seed1", "E02_sol_seed2", "E03_sol_seed3", "E04_sol_seed4",
    "E05_sol_seed5", "E06_haiku_seed1", "E07_gemini_seed1", "E08_deepseek_seed1",
]
ORIGINAL = {"T01_sol": LANGUAGES[0], "T02_haiku": LANGUAGES[5],
            "T03_gemini": LANGUAGES[6], "T04_deepseek": LANGUAGES[7]}
CREATION_SETTINGS = (
    "model", "agent_models", "n_agents", "seed", "temperature", "reasoning_effort",
    "max_tokens", "allow_spaces", "glyph_dictionary", "n_glyphs", "language_minutes",
    "language_max_turns", "language_max_lines", "language_context_messages",
    "chat_context_messages", "min_turn_interval_s",
)
TASK_SETTINGS = (
    "model", "agent_models", "task_model", "n_agents", "task_n_agents", "seed",
    "temperature", "reasoning_effort", "max_tokens", "allow_spaces", "task_kind",
    "task_minutes", "task_max_turns", "task_feedback", "task_feedback_interval_s",
    "task_win_condition", "task_prompt", "chat_context_messages", "min_turn_interval_s",
)
TRANSLATION_FIELDS = {
    "id", "english", "confidence", "ambiguities", "evidence", "literal_gloss",
    "alternative_readings", "alternatives", "departures_from_reference",
    "handle_resolution", "public_evidence_message_ids", "deduplicated_from_public_message_id",
    "entry_type", "entry_kind",
}
TRANSLATION_METADATA = (
    "method", "private_annotations_used", "translator", "review_notes",
    "translation_notes", "translation_limitations", "translation_method",
    "alias_bindings", "reference_issues", "translation_scope",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def select(value, keys):
    return {key: value[key] for key in keys if key in value}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    # Do not silently renumber a transcript containing blank physical lines.
    return [(number, json.loads(line)) for number, line in
            enumerate(path.read_text(encoding="utf-8").splitlines(), 1)]


def one_raw(folder):
    candidates = sorted(p for p in (folder / "raw").iterdir() if p.is_dir())
    if len(candidates) != 1:
        raise ValueError(f"Expected one raw run in {folder.name}, found {len(candidates)}")
    return candidates[0]


class Exporter:
    def __init__(self, source, references, output):
        self.source = source.resolve()
        self.references = references.resolve()
        self.output = output.resolve()
        if any(self.output == p or p in self.output.parents or self.output in p.parents
               for p in (self.source, self.references)):
            raise ValueError("Output must be separate from both source checkouts")
        self.sources = {}
        self.files = set()
        self.commits = {}
        for root in (self.source, self.references):
            self.commits[root] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

    def source_id(self, path):
        path = path.resolve()
        root = next((r for r in (self.source, self.references) if r in path.parents), None)
        if root is None:
            raise ValueError("Source outside configured checkouts")
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        sha = digest(content)
        key = digest((self.commits[root] + ":" + relative + ":" + sha).encode())[:20]
        if key not in self.sources:
            stored = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=root,
                                    capture_output=True)
            self.sources[key] = {
                "id": key, "repository_url": SOURCE_URL,
                "repository_snapshot_commit": self.commits[root], "path": relative,
                "sha256": sha,
                "committed_at_snapshot": stored.returncode == 0 and stored.stdout == content,
            }
        return key

    def write(self, path, data):
        path = Path(path)
        dest = self.output / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        self.files.add(path.as_posix())

    def json(self, path, data):
        self.write(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())

    def jsonl(self, path, data):
        self.write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data).encode())

    def messages(self, path):
        output = []
        for number, row in rows(path):
            if set(row) != {"timestamp", "sender", "text", "is_violation"}:
                raise ValueError(f"Unexpected public message fields: {path.name}:{number}")
            kind = "agent" if row["sender"].startswith("agent_") else (
                "task" if row["sender"] == "task" else "system")
            output.append({"id": number, "sender_kind": kind, **row})
        return output

    def language(self, language_id, handoff, raw, frozen_name="swarmish.txt", pilot=False):
        folder = Path("languages") / language_id
        frozen_path = handoff / frozen_name
        frozen = frozen_path.read_bytes()
        alphabet = read_json(handoff / "glyphs.json")
        config = read_json(handoff / "config.json")
        requirements = read_json(handoff / "language_protocol.json")["requirements"]
        creation = self.messages(raw / "chatroom.jsonl")
        drafts = []
        for number, row in rows(raw / "agent_calls.jsonl"):
            if row.get("state_applied") is True and row.get("language_document"):
                drafts.append({"id": len(drafts) + 1, "source_line": number,
                               **select(row, ("timestamp", "agent_id", "model", "turn", "language_document"))})
        if not drafts or (drafts[-1]["language_document"].strip() + "\n").encode() != frozen:
            raise ValueError(f"Frozen language is not the final applied draft: {language_id}")
        exports = [{"id": number, "source_line": number,
                    **select(row, ("agent_id", "model", "timestamp", "system_prompt", "prompt", "document"))}
                   for number, row in rows(raw / "language_document.jsonl")]
        agents = read_json(raw / "agents.json")
        agents = [select(row, ("agent_id", "model", "glyphs_view", "system_prompt")) for row in agents]
        settings = select(config, CREATION_SETTINGS)
        limit = re.search(r"under (\d+) lines", requirements)
        settings["language_max_lines_from_protocol"] = int(limit[1]) if limit else None
        sources = {"frozen_document": self.source_id(frozen_path)}
        for role, path in {
            "alphabet": handoff / "glyphs.json", "configuration": handoff / "config.json",
            "requirements": handoff / "language_protocol.json", "creation_messages": raw / "chatroom.jsonl",
            "draft_history": raw / "agent_calls.jsonl", "final_exports": raw / "language_document.jsonl",
            "creation_agents": raw / "agents.json",
        }.items():
            sources[role] = self.source_id(path)
        outside = sorted({ch for ch in frozen.decode() if
                          (0x17000 <= ord(ch) <= 0x18AFF or 0x18D00 <= ord(ch) <= 0x18D8F)
                          and ch not in alphabet})
        notes = ["Frozen document is the final applied shared draft, not the concatenated independent final exports."]
        if outside:
            notes.append("The frozen document contains Tangut characters outside its declared alphabet; preserved unchanged.")
        if language_id == LANGUAGES[0] or pilot:
            notes.append("Actual design requirements specify a 500-line limit and a time-bounded creation session.")
        metadata = {
            "id": language_id, "collection": "latin_pilot" if pilot else "tangut",
            "model_family": "sol" if pilot else language_id.split("_")[1],
            "model": config["model"], "rng_seed": config["seed"],
            "seed_label": None if pilot else int(language_id.rsplit("seed", 1)[1]),
            "settings": settings, "alphabet_size": len(alphabet),
            "frozen_document_sha256": digest(frozen),
            "frozen_source": {"selection": "final_applied_shared_draft", "source_id": sources["frozen_document"],
                              "draft_id": drafts[-1]["id"], "draft_source_line": drafts[-1]["source_line"]},
            "line_count": len(frozen.decode().splitlines()),
            "out_of_alphabet_tangut_characters": outside,
            "notes": notes, "source_files": sources,
        }
        self.write(folder / "swarmish.txt", frozen)
        self.json(folder / "alphabet.json", alphabet)
        self.json(folder / "metadata.json", metadata)
        self.write(folder / "design_requirements.txt", (requirements + "\n").encode())
        self.jsonl(folder / "creation_messages.jsonl", creation)
        self.jsonl(folder / "drafts.jsonl", drafts)
        self.jsonl(folder / "final_exports.jsonl", exports)
        self.json(folder / "creation_agents.json", agents)
        return {"id": language_id, "path": folder.as_posix(), "collection": metadata["collection"],
                "model_family": metadata["model_family"], "alphabet_size": len(alphabet),
                "creation_message_count": len(creation), "draft_count": len(drafts),
                "final_export_count": len(exports)}

    def conversation(self, conversation_id, language_id, raw, variant, translation_path=None):
        folder = Path("conversations") / conversation_id
        public_path = raw / "task_chatroom.jsonl"
        public = self.messages(public_path)
        frozen = (self.output / "languages" / language_id / "swarmish.txt").read_bytes()
        if (raw / "swarmish.txt").read_bytes() != frozen:
            raise ValueError(f"Conversation language mismatch: {conversation_id}")
        definition = read_json(raw / "escape_room_definition.json")
        if definition != self.definition:
            raise ValueError(f"Task definition differs: {conversation_id}")
        sources = {"public_transcript": self.source_id(public_path),
                   "task_definition": self.source_id(raw / "escape_room_definition.json"),
                   "frozen_document": self.source_id(raw / "swarmish.txt")}
        references = []
        translation = {"available": False, "reason": "No public-reference English translation is available for this pilot."}
        if translation_path:
            source = read_json(translation_path)
            if source["source_transcript_sha256"] != digest(public_path.read_bytes()):
                raise ValueError(f"Reference transcript hash mismatch: {conversation_id}")
            if source["language_sha256"] != digest(frozen):
                raise ValueError(f"Reference language hash mismatch: {conversation_id}")
            if len(source["messages"]) != len(public):
                raise ValueError(f"Incomplete translation coverage: {conversation_id}")
            for message, reference in zip(public, source["messages"]):
                if any(message[key] != reference[key] for key in ("id", "sender", "text")):
                    raise ValueError(f"Reference alignment mismatch: {conversation_id}:{message['id']}")
                unknown = set(reference) - TRANSLATION_FIELDS - {"sender", "text"}
                if unknown:
                    raise ValueError(f"Unreviewed translation annotations: {unknown}")
                references.append({key: value for key, value in reference.items() if key in TRANSLATION_FIELDS})
            if source.get("private_annotations_used") is not False:
                raise ValueError(f"Reference does not declare public-only inputs: {conversation_id}")
            translation = {"available": True, **select(source, TRANSLATION_METADATA),
                           "certified_ground_truth": False,
                           "source_transcript_sha256": source["source_transcript_sha256"],
                           "language_sha256": source["language_sha256"]}
            sources["english_reference"] = self.source_id(translation_path)
            self.jsonl(folder / "translations.jsonl", references)
        config = read_json(raw / "config.json")
        sources["configuration"] = self.source_id(raw / "config.json")
        agents = [select(row, ("agent_id", "model", "glyphs_view", "system_prompt"))
                  for row in read_json(raw / "task_agents.json")]
        sources["task_agents"] = self.source_id(raw / "task_agents.json")
        # Only the initial prompt: no accumulated scratchpad or private actions.
        initial = {agent: {"initial_turn_prompt": row["initial_turn_prompt"]}
                   for agent, row in read_json(raw / "task_preview.json").items()}
        sources["initial_private_views"] = self.source_id(raw / "task_preview.json")
        self.jsonl(folder / "messages.jsonl", public)
        self.json(folder / "task_agents.json", agents)
        self.json(folder / "initial_private_views.json", initial)
        self.json(folder / "metadata.json", {
            "id": conversation_id, "language_id": language_id, "task_id": "storm_shelter_v1",
            "prompt_variant": variant, "settings": select(config, TASK_SETTINGS),
            "translation": translation, "source_files": sources,
            "context_note": "Initial private views contain agent-specific clues. Load them only for an explicitly chosen context condition.",
        })
        return {"id": conversation_id, "path": folder.as_posix(), "language_id": language_id,
                "task_id": "storm_shelter_v1", "prompt_variant": variant,
                "message_count": len(public), "agent_message_count": sum(r["sender_kind"] == "agent" for r in public),
                "translation_count": len(references), "agent_count": len(agents)}

    def build(self):
        # Refuse stale inventories instead of silently leaving orphaned corpus files.
        for name in ("languages", "conversations", "contexts"):
            if (self.output / name).exists():
                raise ValueError("Build into a fresh output directory; existing corpus directories are not overwritten")
        batch = self.references / BATCH
        language_entries = []
        for language_id in LANGUAGES:
            source = batch / "encoding" / language_id
            raw = self.source / "logs/20261107_181546_5949aefe" if language_id == LANGUAGES[0] else one_raw(source)
            language_entries.append(self.language(language_id, source / "handoff", raw))
        pilot_creation = self.source / "logs/20261107_151252_315264d9"
        language_entries.append(self.language("pilot_sol_latin", pilot_creation, pilot_creation,
                                              "swarmish_shared_final_114.txt", pilot=True))
        context_source = one_raw(batch / "gameplay/R01_sol_seed1") / "escape_room_definition.json"
        self.definition = read_json(context_source)
        context_folder = Path("contexts/storm_shelter_v1")
        self.json(context_folder / "public.json", select(self.definition,
                  ("version", "public_intro", "stage_instructions", "agent_ids")))
        self.json(context_folder / "private_clues.json", select(self.definition,
                  ("version", "private_clues", "release_rules")))
        self.json(context_folder / "solution.json", {"version": self.definition["version"],
                  "description": "Static task answer key, not an observed run outcome.", "solution": self.definition["solution"]})
        conversation_entries = []
        for language_id in LANGUAGES:
            conversation_id = "R" + language_id[1:]
            source = batch / "gameplay" / conversation_id
            conversation_entries.append(self.conversation(conversation_id, language_id, one_raw(source),
                                                          "revised", source / "english/translation.json"))
        for conversation_id, language_id in ORIGINAL.items():
            source = batch / "gameplay/archive/original_prompt_20261107" / conversation_id
            conversation_entries.append(self.conversation(conversation_id, language_id, one_raw(source),
                                                          "original", source / "english/translation.json"))
        conversation_entries.append(self.conversation("pilot_sol_latin_task", "pilot_sol_latin",
            self.source / "logs/20261107_153016_59aeb789", "pilot"))
        totals = {"languages": len(language_entries), "conversations": len(conversation_entries)}
        for total, field in (("messages", "message_count"), ("agent_messages", "agent_message_count"),
                             ("translations", "translation_count")):
            totals[total] = sum(row[field] for row in conversation_entries)
        for total, field in (("creation_messages", "creation_message_count"), ("drafts", "draft_count"),
                             ("final_exports", "final_export_count")):
            totals[total] = sum(row[field] for row in language_entries)
        self.json("manifest.json", {"schema_version": SCHEMA_VERSION, "name": "Swarmish",
            "languages": language_entries, "conversations": conversation_entries,
            "contexts": [{"id": "storm_shelter_v1", "path": context_folder.as_posix(),
                          "source_id": self.source_id(context_source)}], "totals": totals})
        self.json("provenance.json", {"schema_version": SCHEMA_VERSION,
            "note": "Snapshot commits identify source checkouts. A source file is available at that commit only when committed_at_snapshot is true; otherwise its content hash identifies the local artifact used.",
            "sources": sorted(self.sources.values(), key=lambda row: (row["path"], row["repository_snapshot_commit"]))})
        hashes = {name: digest((self.output / name).read_bytes()) for name in sorted(self.files)}
        self.json("checksums.json", hashes)
        print(json.dumps(totals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Local swarm-glyphs snapshot with creation/pilot logs")
    parser.add_argument("--references", required=True, type=Path, help="Local snapshot with complete R/T public transcripts and English references")
    parser.add_argument("--output", required=True, type=Path, help="Output directory without existing corpus folders")
    args = parser.parse_args()
    Exporter(args.source, args.references, args.output).build()


if __name__ == "__main__":
    main()
