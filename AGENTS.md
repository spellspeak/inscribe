# Working on Inscribe

Inscribe trains a game LoRA for SpellSpeak Glyph and Sigil. Read `README.md` first for what works today.

## Rules

- **The plan step is byte-stable.** The same sheet, run-time characters, seed and package version must give the same files. `tests/test_plan.py` checks it, in a fresh process too. A change that moves those bytes changes what every game trains on, so it is a deliberate change with its own changelog line.
- **`src/inscribe/runtime/` is the code that ships with the models.** It renders the exact prompt the models were trained on. Do not edit it here.
- **Recipes hold measured settings.** A number in `src/inscribe/recipes/` came from a run. Do not change one without a run to back it.
- **Say what is measured and what is an estimate.** The forecast is an estimate and says so. Never print an illustrative number as if it were a result.
- **Nothing reads outside the package.** No step may need a file that is not in this repo, the project folder or a published download.
- **A step that spends money asks first**, forecasts its cost, and honours `budget.max_usd`.

## Checks

```bash
uv run pytest
```
