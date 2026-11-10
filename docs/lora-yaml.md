# lora.yaml

`lora.yaml` says what to train and where. Anything it leaves out comes from the base model's recipe, the settings that model was measured with. A setting not listed here is an error.

Override a setting for one command with `--set`, for example `inscribe run --set train.epochs=2`.

```yaml
game: my_game
base: glyph-2b
replay:
  to_one: 5
budget:
  max_usd: 80
```

## Settings

| Setting | Default | What |
|---|---|---|
| `game` | required | The game's id. It must match `game_id` in the sheet |
| `base` | `glyph-2b` | `glyph-2b`, `sigil-4b`, or a list of both. Each one is trained, exported and scored |
| `seed` | `0` | The same seed, sheet and package version give the same scenes and the same replay rows |
| `backend.name` | `modal` | Where the GPU steps run. Modal is the only backend so far |
| `backend.environment` | your workspace's default | A Modal environment |
| `backend.app_prefix` | `inscribe` | The start of the app and volume name. The game's id follows it |
| `backend.usd_per_hour` | `{H100: 4.5, L4: 0.8}` | Prices for the forecast. Estimates, not quotes: set your own |
| `llm.key_env` | `OPENAI_API_KEY` | The environment variable that holds your key. The key is read on your machine and sent nowhere else |
| `llm.strong_model` | the one the recipes were measured with | Reads lore into a sheet, and judges spoilers and voice. The template's `lora.yaml` names it |
| `llm.fast_model` | the one the recipes were measured with | Checks the teacher's lines |
| `replay.source` | `hf://spellspeak/replay-pack@0.1` | The replay pack: the [published one](https://huggingface.co/datasets/spellspeak/replay-pack) at a fixed tag. Or another `hf://<dataset>[@revision]`, or `store:<folder>` for a pack already on the project's volume |
| `replay.to_one` | `4` | Replay rows per game row. Never below 3 |
| `teacher.candidates` | `4` | Lines the teacher writes per scene, 1 to 8. The checks keep the best one, or none |
| `train.epochs` | the recipe's (3) | Passes over the mix |
| `train.rank` | the recipe's (16) | The LoRA's rank. Alpha follows it at twice the rank |
| `train.learning_rate` | the recipe's (2e-4) | |
| `export.quant` | the recipe's | GGUF types to export. Glyph 2B ships Q5_K_M and Sigil 4B Q4_K_M |
| `budget.max_usd` | `80` | A command stops before GPU time whose forecast would pass this |

The measured settings are the recipe files in `src/inscribe/recipes/`. A number there came from a run. Changing `train.*` trains something nobody has measured: read the report card.
