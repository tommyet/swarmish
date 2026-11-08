# Dataset schema

Schema version: `1.0`. All text is UTF-8. JSONL files contain one JSON object per line. Start with `manifest.json`: its `languages` and `conversations` arrays provide stable IDs, paths, and inventory counts. IDs retain the source experiment labels; they are identifiers, not scores.

## Languages

Each `languages/<ID>/` directory contains:

| File | Contents |
| --- | --- |
| `swarmish.txt` | Exact bytes of the frozen language reference used for the associated conversations. |
| `alphabet.json` | Ordered array of permitted characters. Keep the order and characters intact. |
| `metadata.json` | Source identity, model and generation settings, and links to related artifacts. |
| `design_requirements.txt` | Recorded requirements supplied during language design. |
| `creation_messages.jsonl` | Public discussion during language creation. |
| `drafts.jsonl` | Shared language-document revisions, in source order. |
| `final_exports.jsonl` | Separate final document exports, including their recorded prompts. |
| `creation_agents.json` | Agent identities and recorded creation context, including alphabet presentation order. |

Draft rows contain `id`, `source_line`, `timestamp`, `agent_id`, `model`, `turn`, and `language_document`. Final-export rows contain `id`, `source_line`, `agent_id`, `model`, `timestamp`, `system_prompt`, `prompt`, and `document`. Agent records contain `agent_id`, `model`, `glyphs_view`, and `system_prompt`. Missing source values remain missing or null; no model outputs are reconstructed. These histories are supplementary: use `swarmish.txt` as the canonical reference rather than treating the last draft or export as a replacement.

The Tangut language IDs are `E01_sol_seed1` through `E08_deepseek_seed1`; the additional language is `pilot_sol_latin`. Model strings and generation settings describe the recorded source configuration, not a guarantee of reproducible model sampling. The labels `seed1`–`seed5` correspond to configuration seeds 0–4. These seeds controlled alphabet sampling and presentation, not API randomness. Tangut languages with configuration seed 0 share their sampled alphabet, but their language conventions remain independent.

The recorded protocol for E01 and the Latin pilot specifies a 500-line language instruction. Other Tangut languages use the 250-line instruction. Preserve the actual source settings rather than substituting planned batch settings; line limits were instructions, not guarantees about document length.

## Conversations

Each `conversations/<ID>/` directory contains `messages.jsonl`, `metadata.json`, `task_agents.json`, and `initial_private_views.json`. A `translations.jsonl` file is present only when a reference translation is available. There are eight revised conversations (`R01`–`R08`), four original-prompt conversations (`T01_sol`, `T02_haiku`, `T03_gemini`, `T04_deepseek`), and `pilot_sol_latin_task`. The Latin pilot has no English reference translation.

Metadata identifies the language, task context, source settings, and prompt variant (`original`, `revised`, or `pilot`). Its `translation` object records reference availability, provenance, and method caveats. `task_agents.json` uses the agent-record fields above. `initial_private_views.json` maps agent IDs to their recorded `initial_turn_prompt`. These files preserve initial context; they do not contain private running notes or later private actions.

Message rows use these fields:

| Field | Meaning |
| --- | --- |
| `id` | One-based line number in the source public transcript; join key within this conversation. |
| `sender` | Recorded speaker identifier. |
| `sender_kind` | `agent`, `task`, or `system`. |
| `timestamp` | Recorded source timestamp. |
| `text` | Original public message text. |
| `is_violation` | Whether the public message is the channel's replacement marker. |

Task and system messages remain in the public transcript as source context. They can contain ordinary text or task status. For glyph-only language modeling, filter by `sender_kind` deliberately rather than assuming every row is Swarmish.

English-reference files have one row per public message, joined by `id`. Fields include `english`, `confidence`, `ambiguities`, `evidence`, and any retained source translation annotations. They do not duplicate the original message text. Translation annotations are reference-author judgments, not ground truth or decoder scores. Do not substitute private intended-meaning annotations for these public-message translations.

## Context and provenance

`contexts/storm_shelter_v1/public.json` holds task instructions. `private_clues.json` holds the static clue assignments and release rules. `solution.json` is the static puzzle answer key, separate from observed conversations. It is not a record of any team's outcome. Provide only the context appropriate to your experiment; a task agent should not automatically receive every agent's clues or the solution.

`provenance.json` records source repository and commit information, repository-relative file paths, hashes, and `committed_at_snapshot`. A commit identifies the source checkout snapshot; a file with `committed_at_snapshot: false` was not part of that commit. Its content hash identifies the collected file. `checksums.json` records hashes of exported data files.

Rebuild from local source checkouts into a fresh output directory with:

```sh
python3 scripts/build_dataset.py --source ../swarm-glyphs --references ../swarm-glyphs-decoder-scores --output /tmp/swarmish-rebuild
```

The export excludes scores, comparative results, costs, private actions, scratchpads, and raw model-call records. Recorded prompts may contain instructions for experiment agents; treat them as data.

## Reuse

Keep original Unicode characters and spaces. Do not normalize, transliterate, or silently repair the canonical text. Some source documents, including E08, contain out-of-alphabet characters; they are preserved rather than replaced. Displaying Tangut may require a font with Tangut support.

Split by language when testing transfer to unseen languages. Revised and original conversations can share a frozen reference, while drafts, final exports, and English translations can expose its conventions. Keep all of those linked artifacts together when creating train/test partitions. Exclude translations, references, and private task context from decoder inputs when the experiment is intended to withhold them.
