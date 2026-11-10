# SpellSpeak Inscribe

Inscribe trains a game LoRA for [SpellSpeak Glyph 2B](https://huggingface.co/jptaylor/spellspeak-glyph-2b) and [SpellSpeak Sigil 4B](https://huggingface.co/jptaylor/spellspeak-sigil-4b) from your game's lore and character sheets.

Glyph and Sigil write one line of dialogue for a game character, from what that character knows. Out of the box they know nothing about your world: the game puts what matters in the prompt. A game LoRA puts the world itself in the weights, so every character knows the town's history, its prices and its sayings without being told each time, and speaks in your game's style.

You write the lore. Inscribe turns it into training scenes, has a teacher model answer them, checks the answers, trains the LoRA on your own GPU account, exports a GGUF file and scores it.

## Status

This is an early build. You can check a game sheet, plan the data, train on Modal and export a GGUF file. The step that writes the training lines for your own game does not exist yet, so training only works with rows you already have.

| Step | Command | What it does | Status |
|---|---|---|---|
| Start | `inscribe init` | Makes a game project from a template | works |
| Check | `inscribe check` | Validates the game sheet, shows what would be built, forecasts the run | works |
| Plan | `inscribe plan` | Builds the training scenes, the held-out questions and the world files | works |
| Extract | | Reads prose lore into a game sheet for you to check | not built |
| Data | | A teacher model answers the scenes. The answers are checked and mixed with general examples | not built |
| Train | `inscribe train` | Trains the LoRA on your Modal account and merges it | works, given training rows |
| Follow | `inscribe status`, `logs`, `stop`, `pull` | The steps a project has launched, and their files | works |
| Export | `inscribe export` | Makes the GGUF file and answers the held-out questions with it | works |
| Score | | Lore recalled, spoilers told early, skills kept | not built |

Until Extract exists you write the game sheet by hand. [docs/game-sheet.md](docs/game-sheet.md) describes every field.

## Try it

It needs Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd spellspeak-inscribe
uv sync
uv run inscribe init my-game
cd my-game
uv run inscribe check
uv run inscribe plan
```

Or look at a complete game first:

```bash
cd examples/brackenhythe
uv run inscribe check
uv run inscribe check --lore
```

`check` on that game prints:

```
brackenhythe (game-sheet-0.2)
  6 characters, 28 places and things, 158 lore items (48 marked high), 16 secrets, 19 priced offers, 13 speech examples, 5 story stages
  3 run-time characters in npcs.yaml

Scenes for the teacher to answer: 974
  lore 707, story 123, ordinary 57, secret 53, boundary 34 (219 of them misheard twins)
Training lines taken straight from the sheet's speech examples: 13
Held-out questions, never trained on: lore 384 on 98 items, spoilers 36, run-time characters 36, prices 38
```

## A game project

```
my-game/
  lora.yaml                 what to train and where
  lore/                     world.md, characters.md, story.md, style.md: prose, written by your team
  sheet/game-sheet.yaml     the game sheet: what every step reads. Checked by a person
  npcs.yaml                 optional: characters your game makes up at run time, for the checks
  build/                    generated, safe to delete
  out/                      the GGUF file, the adapter and the report
```

`lora.yaml` is short. Anything it leaves out comes from the base model's recipe, the settings that model was measured with. Override one setting for one run with `--set`, for example `inscribe check --set base=sigil-4b`.

```yaml
game: my_game
base: glyph-2b            # glyph-2b, sigil-4b, or a list of both
backend:
  name: modal
budget:
  max_usd: 80
```

## Where the GPU steps run

The steps that need a GPU run on your own Modal account. Each step is one container that starts, does its work and ends, so nothing stays running. Its files live on a Modal volume named after the game, which is also where the base model is cached. A step keeps running if your terminal closes: `inscribe status` and `inscribe logs` find it again.

Install the Modal extra and sign in once:

```bash
uv sync --extra modal
uv run modal token new
```

`inscribe train` says what it will cost and asks before it starts. `--max-steps 5` runs five steps and saves nothing: it shows that the base model loads, the data reads and the first losses look right, for well under a dollar.

The same weights give slightly different numbers on different machines. Two identical five-step checks gave first losses up to 0.003 apart, and worded 88 of 620 test replies differently. Compare runs with that in mind.

## What `plan` builds

Everything is built by code from the sheet. No model is called, and the same sheet always gives the same files.

- **Scenes for the teacher** (`scenarios.jsonl`). Players asking about the lore in many phrasings. Things from our world the characters should not know, and things they should. Each story stage, with the later stages forbidden. Secrets, asked by someone who may hear them and someone who may not. Ordinary talk. About a quarter have a misheard twin, as speech recognition would garble them.
- **Lines from the sheet** (`style-examples.jsonl`). Your speech examples, trained on as written.
- **Held-out questions** (`recall.json`). Different phrasings from the training scenes, with answer keys, used only to score the trained model.
- **World files** (`worlds/`). The game at each story stage, and a copy with each character's confidants in the room.
- **Quantization text** (`imatrix.txt`). The game's own prompts, so the small file keeps what matters to this game.

`inscribe check --lore` lists the lore lines that get no scene of their own, with the reason. A line such as "Her son runs it." gives the planner nothing to ask about. Writing it as a full fact fixes it.

## What a game LoRA does not do

- **Lore in the weights is a preference, not a guarantee.** Measure recall, and keep facts that must be right in the prompt.
- **A secret in the weights is not safe.** A player can talk it out. Keep real secrets in the game and hand them over when the story allows.
- **It does not track the game's state.** What has happened, who is in the room and what a thing costs today come from the game, each turn.

## Layout of this repo

| Folder | What |
|---|---|
| `src/inscribe/cli.py`, `config.py`, `project.py` | The command, `lora.yaml`, where a project's files are |
| `src/inscribe/plan/` | The plan step: scenes, world files, held-out questions and their scoring |
| `src/inscribe/scenarios/`, `context/`, `schemas/`, `sheets/` | Scenario generators, the view a character gets of the world, the file formats |
| `src/inscribe/runtime/` | The prompt format, the same code that ships with the models |
| `src/inscribe/train/`, `export/`, `stages/` | The training and export steps, and the small program that runs any step inside its container |
| `src/inscribe/backends/` | What a backend is, Modal as the first one, and a local one the tests use |
| `src/inscribe/recipes/`, `images.py` | One file per base model with its pinned revision and measured settings, and the pinned images the steps run in |
| `examples/` | Two complete games: lore, sheet and run-time characters |
| `tests/` | `uv run pytest` |

## Licence

Apache-2.0, as are Glyph, Sigil and their base model.
