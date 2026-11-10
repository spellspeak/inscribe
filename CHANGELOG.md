# Changelog

## Unreleased

- `inscribe init`, `inscribe check` and `inscribe plan`.
- Recipes for SpellSpeak Glyph 2B and Sigil 4B.
- Two example games.
- A backend interface, with Modal as the first backend: each step is a sandbox and the store is a volume.
- `inscribe mix`, `train`, `export`, `pull`, `status`, `logs` and `stop`.
- `inscribe data`: a teacher model writes each scene's lines on the backend, and code checks and a judge keep the best line of each scene.
- A step that is interrupted says so in its result.
- Time estimates for the teacher and the export come from measured runs.
- `inscribe extract`: a strong model reads the prose in `lore/` into a game sheet, each answer validated, with a report of what to check first and what the sheet could not hold.
- `inscribe eval`: scores the trained model and writes the report card to `out/report.md`. Lore, prices, run-time characters and retention by key words; later story told early and the character's voice by a judge model.
- `inscribe train` now also answers 78 general questions before and after training, which the report card's retention line reads.
- `inscribe chat`: talk to a character with the exported file on your own machine, through llama.cpp's `llama-server`.
- `lora.yaml` names the two outside models: `llm.strong_model` and `llm.fast_model`.
- A run-time character in `npcs.yaml` that does not fit the sheet is reported as such.
