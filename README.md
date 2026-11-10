# SpellSpeak Inscribe

Inscribe trains a game LoRA for [SpellSpeak Glyph 2B](https://huggingface.co/jptaylor/spellspeak-glyph-2b) and [SpellSpeak Sigil 4B](https://huggingface.co/jptaylor/spellspeak-sigil-4b) from your game's lore and character sheets.

Glyph and Sigil write one line of dialogue for a game character, from what that character knows. Out of the box they know nothing about your world: the game puts what matters in the prompt. A game LoRA puts the world itself in the weights, so every character knows the town's history, its prices and its sayings without being told each time, and speaks in your game's style.

You write the lore. Inscribe turns it into training scenes, has a teacher model answer them, checks the answers, trains the LoRA on your own GPU account, exports a GGUF file and scores it.

## Status

This is an early build. Every step from prose lore to a scored GGUF file works: it reads your lore into a game sheet for you to check, plans the data, has a teacher write the training lines and checks them, mixes them, trains on Modal, exports a GGUF file, scores it and writes a report card.

One thing is missing: the replay pack the mix draws on is not published yet.

| Step | Command | What it does | Status |
|---|---|---|---|
| Start | `inscribe init` | Makes a game project from a template | works |
| Extract | `inscribe extract` | Reads prose lore into a game sheet for you to check | works |
| Check | `inscribe check` | Validates the game sheet, shows what would be built, forecasts the run | works |
| Plan | `inscribe plan` | Builds the training scenes, the held-out questions and the world files | works |
| Data | `inscribe data` | A teacher model answers the scenes on your Modal account, and the answers are checked | works |
| Mix | `inscribe mix` | Mixes the game's rows with general rows, so the model keeps the skills it had | works; the replay pack it draws on is not published yet |
| Train | `inscribe train` | Trains the LoRA on your Modal account and merges it | works |
| Export | `inscribe export` | Makes the GGUF file and answers the held-out questions with it | works |
| Score | `inscribe eval` | Lore recalled, later story told early, the character's voice, skills kept. Writes the report card | works |
| Talk | `inscribe chat` | Talk to a character with the exported file, on your machine | works |
| All of it | `inscribe run` | Plan, data, mix, train, export and eval in order, skipping a step that is already done | works |
| Follow | `inscribe status`, `logs`, `stop`, `pull` | The steps a project has launched, and their files | works |

`inscribe extract` drafts the game sheet from your prose, or you write it by hand. Either way, read it before you train: every later step works from the sheet. [docs/game-sheet.md](docs/game-sheet.md) describes every field.

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

The whole path, once your prose is in `lore/`:

```bash
uv run inscribe extract     # prose to a game sheet. Read the sheet before going on
uv run inscribe check
uv run inscribe run         # plan, data, mix, train, export, eval, on your Modal account. Writes out/report.md
uv run inscribe pull
uv run inscribe chat        # lists the characters; then: inscribe chat <name>
```

`inscribe run` lists the steps with what each will cost, and asks once. A step that already finished with the same inputs is skipped, and a training or export step still running from an earlier launch is followed. So after a failure, a closed laptop or an edit to the sheet, you run it again. `--from export` starts at a step and `--to data` stops after one. Each step is also a command of its own, in the table above.

`extract`, `data` and `eval` call an outside model and need `OPENAI_API_KEY` in your shell. Every command that spends money says about how much and asks first.

`check` on the example game prints:

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

`lora.yaml` is short. Anything it leaves out comes from the base model's recipe, the settings that model was measured with. Override one setting for one run with `--set`, for example `inscribe check --set base=sigil-4b`. [docs/lora-yaml.md](docs/lora-yaml.md) lists every setting.

```yaml
game: my_game
base: glyph-2b            # glyph-2b, sigil-4b, or a list of both
backend:
  name: modal
budget:
  max_usd: 80
```

## On the example game

One run, on the 2B model, through `inscribe data`, `mix`, `train`, `export` and `eval`, from the game sheet in `examples/brackenhythe`.

| Step | What came out | Time |
|---|---|---|
| Data | The teacher wrote 3,896 lines for 974 scenes. The checks kept 727 | 10 minutes on one H100, then the judge, about $0.54 |
| Mix | 5,400 rows: 740 of the game's, 4,660 replayed | seconds |
| Train | The LoRA and the merged model | 80 minutes on one H100, 64 of them training |
| Export | A 2.91 GB file | 12 minutes on one L4 |
| Score | The report card | a minute on your machine, and the judge, about $0.38 |

| Held-out questions | Base model, before training | After, from the 2.91 GB file |
|---|---:|---:|
| Important lore items recalled, of 33 | 0 | 32 |
| All lore items recalled, of 98 | 1 | 76 |
| Sayings given word for word, of 20 questions | 0 | 16 |
| A later story stage given away, of 36 questions (by key words) | 0 | 0 |
| Run-time characters answering from the lore, of 36 questions | 0 | 26, and no secret told |
| Prices read from the character's own sheet, of 38 questions | 38 | 37 |

"Before" is the base model at full precision, in the training container.

`inscribe eval` wrote this card for the run:

| Check | Line | This run | |
|---|---|---|---|
| Important lore recalled | at least 80% | 32 of 33 | met |
| Later story told early | none | 0 of 36 | met |
| Run-time characters tell no secret | none told | 0 told in 36 replies | met |
| Sounds more like the character than the base model | preferred beyond chance | preferred in 54 of 80 pairs, the base in 20, 6 ties | met |
| Keeps what the base model could do | no gated set lost | know-how 25 to 25 of 29, following the player 11 to 10 of 12 | met |

The card cannot see a small loss of general skill. On a general test of ours that is not public yet, this LoRA passed 203 of 309 scenes, and the base model 205, a difference inside the noise of that test.

This is one run. A second LoRA, trained with the same settings on lines from the pipeline this tool is lifted from, recalled 31 of the 33 important items and passed 199 of the 309 scenes. Expect that much spread between two runs.

## Why most of the training mix is not your game

A LoRA trained on a game's rows alone forgets how to do what the base model could. So about six rows in seven of the mix are replay: general rows the base model was trained on. `inscribe mix` takes four of them for each of the game's rows, and a fixed number more from the families a game LoRA loses first: things from the real world, long conversations, and who speaks next in a group.

The general rows come from a replay pack, about 55,000 rows already rendered as text. The same seed always draws the same rows. The pack is not published yet, so today `inscribe mix` only works where the pack is.

## What `data` does

`inscribe data` has a teacher model write four lines for every scene, then keeps the best line of each scene, or none.

- **The teacher** is an open-weights model, Qwen3.8-27B at 8 bits, on one H100 of your Modal account. It sees the scene as the game model will see it, the lore that scene leaves out of the prompt on purpose (so the game model learns to carry it in its weights), and a note on style and on what a good reply does here. It never sees what the character does not know, or the answer key. The first run downloads about 30 GB to the project's volume.
- **Code checks** run on your machine. A line is dropped when it is empty or cut off, quotes the question back, has markup, is too long, misses a word the scene needs or says one it must not, or is a copy of a line seen twice already.
- **A judge model** then reads the lines that are not plain lore, from your machine. A line it faults is dropped. This needs `OPENAI_API_KEY` in your shell, and the key stays on your machine. On the example game the judge read 925 lines and dropped 396 of them.
- Of the lines left, a scene keeps the one the judge likes best. A line too close to one already kept is dropped.

`build/data/rejected.jsonl` holds every dropped line with its reason. Read some of the kept lines before you train: the model will talk like them. On the example game, 2 of the 727 kept lines open by saying the player's question back, a teacher habit the checks only catch when the question is in quotes.

`inscribe data --limit 20` writes lines for 20 scenes only, as a rehearsal. The teacher still takes about four minutes to load.

## What `extract` does

`inscribe extract` reads `lore/world.md`, `characters.md`, `story.md` and `style.md` and drafts `sheet/game-sheet.yaml`.

- **Three calls to a strong model**, named in `lora.yaml` as `llm.strong_model`. The first reads the world and the story. The second reads the characters, and never sees the story file. The third copies the speech examples. Each answer comes back in a fixed shape.
- **Every answer is checked** before the next call: the sheet must validate, a world must build from it, and none of the words your story says must not be said early may appear anywhere in it. A failed check goes back to the model once with the error. If it fails again, the command stops and writes nothing.
- **It reports what it left out.** `sheet/extract-report.md` lists what your documents say that the sheet cannot hold, such as writers' notes and anything that changes as the story moves. It also lists items to look at first.
- **It never replaces your work.** A sheet you have edited is left alone, and the new draft goes beside it as `game-sheet.extracted.yaml`.

On the example game's 56,000 characters of lore it took a little over three minutes and $0.26, with no repair needed. `inscribe extract --dry-run` shows the forecast and sends nothing.

The reader is told never to invent, and to copy your own words. It can still file a fact in the wrong place, and two runs do not give the same sheet. Two runs on the example game's lore agreed on the 6 characters, all 28 places and things, the 5 story stages, the 13 speech examples and the 5 wordings to be said exactly. They shared 128 lore items word for word, of 156 and 158. One found 20 secrets and the other 16. One returned 18 world rules and the other 46: it had cut a section of plain facts into single sentences and filed them as rules. The report flags a long rule list and a rule that leans on the line before it. Read the whole sheet all the same.

`npcs.yaml` is never read: characters your game makes up at run time must not enter the sheet.

## What `eval` checks

`inscribe eval` scores the answers the training and export steps left behind, and writes `out/report.md`. It needs no GPU.

| Line on the card | How it is measured |
|---|---|
| Important lore recalled, at least 80% | Held-out phrasings of each lore item marked `importance: high`, answered by the exported file, scored by key words |
| Later story told early, none | A judge model reads each reply to a what-comes-next question, with the story so far and the stages still to come |
| Run-time characters tell no secret | Characters given only in the prompt are asked about the world. No reply may use the key words of a core character's secret |
| Sounds more like the character than the base model | The judge sees the base model's reply and the LoRA's to the same line, unlabelled and in a random order, 80 pairs. Met when the LoRA wins beyond chance |
| Keeps what the base model could do | 78 general questions, answered before and after training. Not met when know-how loses more than 3 of 29, or following the player more than 2 of 12. For Sigil 4B, real-world facts count too |

The card also gives all lore items, the wordings to be said exactly, prices, and sample replies for every line: lore the model now says, lore it missed, each flagged spoiler with the judge's reason, and every general question it lost with both answers. When a line is not met it says what to try first.

The two judged lines cost about $0.40 a model on the example game. `inscribe eval --no-judge` skips them, costs nothing, and marks them not judged.

How far to trust it:

- **The key-word lines are exact.** We scored an earlier LoRA of the example game with this step and with the pipeline it is lifted from. Every key-word number and the retention verdict were the same.
- **The judge is a model, and moves a little.** On the same replies, two reads flagged the same single line of 36 on one set and none on the other. They preferred the LoRA's voice in 53 of 80 pairs and in 51.
- **Key words are blunt.** In the example run the model answered a question about storm clouds with "we're getting a thunderstorm" and lost the point, because the key is "storm" as a whole word. The card prints the lost questions so you can judge them yourself.
- **It does not check general skill or leaks into other games.** The card says so on every run. See [What a game LoRA does not do](#what-a-game-lora-does-not-do).

## Talk to a character

```bash
uv run inscribe pull
uv run inscribe chat maren
```

`chat` runs the exported file on your own machine. It needs `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp), build b11146 or later, on your PATH. The prompt is built as in training: what the character knows from the sheet, then the conversation so far. `--stage 3` puts the character at the third story stage. `--say "..."` says one line and prints the reply, for scripts.

From the example run's 2.91 GB file, on an Apple-silicon laptop:

```
Player: What's a pint cost?
Maren Tolliver: One pip, duck.
Player: And who runs the Tallyhouse?
Maren Tolliver: The Reckoners run it.
Player: Can I pay with my phone?
Maren Tolliver: What is a phone?
```

## Where the GPU steps run

The steps that need a GPU run on your own Modal account. Each step is one container that starts, does its work and ends, so nothing stays running. Its files live on a Modal volume named after the game, which is also where the base model is cached. A step keeps running if your terminal closes: `inscribe status` and `inscribe logs` find it again.

Install the Modal extra and sign in once:

```bash
uv sync --extra modal
uv run modal token new
```

`inscribe train` says what it will cost and asks before it starts. `--max-steps 5` runs five steps and saves nothing: it shows that the base model loads, the data reads and the first losses look right, for well under a dollar.

A step can be cut short from outside, for example when Modal takes its machine back. One of our exports ended that way after 15 minutes and ran cleanly the second time. `inscribe status` then shows the step as failed, with "interrupted" as the reason when the step had time to write one. Nothing of that step is kept: run the command again.

The same weights give slightly different numbers on different machines. Two identical five-step checks gave first losses up to 0.003 apart, and worded 88 of 620 test replies differently. Compare runs with that in mind.

## What `plan` builds

Everything is built by code from the sheet. No model is called, and the same sheet always gives the same files.

- **Scenes for the teacher** (`scenarios.jsonl`). Players asking about the lore in many phrasings. Things from our world the characters should not know, and things they should. Each story stage, with the later stages forbidden. Secrets, asked by someone who may hear them and someone who may not. Ordinary talk. About a quarter have a misheard twin, as speech recognition would garble them.
- **Lines from the sheet** (`style-examples.jsonl`). Your speech examples, trained on as written.
- **Held-out questions** (`recall.json`). Different phrasings from the training scenes, with answer keys, used only to score the trained model. `inscribe train` answers them before and after training, together with 78 general questions that show what the model kept.
- **World files** (`worlds/`). The game at each story stage, and a copy with each character's confidants in the room.
- **Quantization text** (`imatrix.txt`). The game's own prompts, so the small file keeps what matters to this game.

`inscribe check --lore` lists the lore lines that get no scene of their own, with the reason. A line such as "Her son runs it." gives the planner nothing to ask about. Writing it as a full fact fixes it.

## What a game LoRA does not do

- **Lore in the weights is a preference, not a guarantee.** Measure recall, and keep facts that must be right in the prompt.
- **A secret in the weights is not safe.** A player can talk it out. Keep real secrets in the game and hand them over when the story allows.
- **It does not track the game's state.** What has happened, who is in the room and what a thing costs today come from the game, each turn.

## With a coding agent

[skills/train-game-lora/SKILL.md](skills/train-game-lora/SKILL.md) is a skill a coding agent can load. It walks the agent through the path above, and says what stays with you: reading the game sheet, and agreeing to each spend.

## Layout of this repo

| Folder | What |
|---|---|
| `src/inscribe/cli.py`, `config.py`, `project.py` | The command, `lora.yaml`, where a project's files are |
| `src/inscribe/plan/` | The plan step: scenes, world files, held-out questions and their scoring |
| `src/inscribe/scenarios/`, `context/`, `schemas/`, `sheets/` | Scenario generators, the view a character gets of the world, the file formats, and the lore reader |
| `src/inscribe/runtime/` | The prompt format, the same code that ships with the models |
| `src/inscribe/data/`, `judges/` | The teacher, the checks on its lines, the two outside models' clients, and the mix |
| `src/inscribe/evaluate/`, `chat.py` | The scoring step and its report card, and talking to a character with the exported file |
| `src/inscribe/train/`, `export/`, `stages/` | The training and export steps, and the small program that runs any step inside its container |
| `src/inscribe/backends/` | What a backend is, Modal as the first one, and a local one the tests use |
| `src/inscribe/recipes/`, `images.py` | One file per base model with its pinned revision and measured settings, and the pinned images the steps run in |
| `examples/` | Two complete games: lore, sheet and run-time characters |
| `docs/`, `skills/` | Every field of the game sheet, every setting of `lora.yaml`, and the skill for coding agents |
| `tests/` | `uv run pytest` |

## Licence

Apache-2.0, as are Glyph, Sigil and their base model.
