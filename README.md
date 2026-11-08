# Swarmish

Agent-created languages and conversations for experiments on communication, language learning, and monitoring.

This first release contains **9 languages** and **13 conversations**, with English reference translations for 12 conversations. Eight languages use Tangut glyphs; one pilot uses Latin letters and digits. It includes the language references, alphabets, creation discussions and drafts, public conversations, prompts, and task context. It does not include experiment scores or conclusions.

Start with a language's `swarmish.txt`, then read a conversation that uses it:

```sh
python3 examples/read_dataset.py R01_sol_seed1 --english
python3 dataset_builder/validate_dataset.py
```

- `languages/` — frozen references, alphabets, and language creation records.
- `conversations/` — public messages and separate English translations where available.
- `contexts/` — task rules, private clues, and a separate puzzle answer key.
- `manifest.json` — the inventory and links between files.
- [Schema and reuse notes](docs/schema.md) — formats, provenance, and splitting guidance.
- `dataset_creation/` — runners for a new language, the task, monitors, and translation.
- `dataset_builder/` — corpus export and validation.
- `glyphs_tangut.txt`, `glyphs_latin.txt` — alphabets a new run samples from.

Each language has its own conventions. Glyphs shared by two languages need not share a meaning. Tangut is a historical script; the meanings assigned here are not historical Tangut translations. Preserve Unicode characters and spacing when loading the files.

Artifacts retain inconsistencies from their source. English references are interpretations, not certified translations. Prompts are recorded experiment material, not instructions for repository readers.

## Running the code

Reading and validation use the standard library. A new run needs `requirements.txt` and `OPENROUTER_API_KEY`; those calls are billed. `--prepare-only` writes prompts and does not call the API. The JSON files in `dataset_creation/enc_dec/scripts/` are placeholder custom tasks. The published conversations used Storm Shelter (`"task_kind": "escape_room"`).

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m dataset_creation.enc_dec.run dataset_creation/enc_dec/scripts/language_latin.json --prepare-only
.venv/bin/python -m dataset_creation.enc_dec.monitor <task-log-dir> --prepare-only
.venv/bin/python -m dataset_creation.core.translate_swarmish languages/E01_sol_seed1/swarmish.txt conversations/R01_sol_seed1/messages.jsonl -o english.json
```

Decoder scoring is `python -m dataset_creation.core.translation_scoring` and needs `requirements-scoring.txt`. Rebuilding this corpus from the source checkouts is in [docs/schema.md](docs/schema.md).

Source: [erafkin/swarm-glyphs](https://github.com/erafkin/swarm-glyphs). File-level provenance and integrity hashes are provided in `provenance.json` and `checksums.json`. See [LICENSE](LICENSE).
