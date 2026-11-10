---
name: train-game-lora
description: Train a game LoRA for SpellSpeak Glyph 2B or Sigil 4B with Inscribe, from a game's prose lore to a scored GGUF file. Use when someone wants their game's characters to know its world and speak in its style, asks to train, retrain or score a game LoRA, or asks why a line on the report card is not met.
---

# Train a game LoRA with Inscribe

Inscribe turns a game's lore into a LoRA for SpellSpeak Glyph 2B or Sigil 4B. You drive the `inscribe` command for the person. They own two things you must not take over: reading the game sheet, and agreeing to each spend.

Run every command from the game project folder. In a clone of the Inscribe repo, put `uv run` in front.

## Before you start

- `inscribe --version` works.
- For the GPU steps: the Modal extra is installed and `modal token new` has been run.
- For `extract`, `data` and `eval`: `OPENAI_API_KEY` is set in the shell. Never print it, and never write it to a file.
- Nothing to set for the replay pack: the mix step downloads the published one on the backend.

## The path

1. **Start.** `inscribe init <name>` makes the folder. The person writes `lore/world.md`, `characters.md`, `story.md` and `style.md`.
2. **Draft the sheet.** `inscribe extract --dry-run` shows the cost. `inscribe extract` writes `sheet/game-sheet.yaml` and `sheet/extract-report.md`. It costs well under a dollar for 50,000 characters of lore.
3. **Have the sheet read.** Show the person the report's "Items to check first" and "What the sheet could not hold". Ask them to read the sheet. Do not go on until they say it is right. Things the reader gets wrong:
   - a long list of world rules, some cut out of plain facts. Rules go into every prompt, so keep only what is and is not possible in the world;
   - a secret split over several knowledge items, one of them without its `secret` block;
   - a fact that only becomes true later in the story. That is live state and belongs to the game, not the sheet.
   The file `npcs.yaml` from the template describes the template's world. Replace it with the game's own run-time characters or delete it.
4. **Check.** `inscribe check` validates the sheet and forecasts the run. `inscribe check --lore` lists lore lines that get no training scene, usually because the line means nothing alone. Fix those lines in the sheet.
5. **Run.** `inscribe run` lists every step with its cost and asks once. Tell the person the total before you answer for them, and never pass `--yes` without their say. Training takes one to two hours. The GPU steps run on Modal and keep running if the terminal closes. Run `inscribe run` again to pick up where it was: finished steps are skipped, and a training or export step that is still running is followed. The teacher step is the exception: if the terminal closes while it writes, it is run again.
6. **Read the card.** `out/report.md`. Give the person each line as met or not met with its number, then the section "What this card does not tell you". Do not round a not-met line up.
7. **Try it.** `inscribe pull` brings the GGUF file to `out/`. `inscribe chat <name>` talks to a character with it. It needs `llama-server` from llama.cpp on the PATH.

## When a line on the card is not met

Read the sample replies under the line before changing anything. The judge that reads spoilers and voice is a model and is sometimes wrong. The key-word checks are blunt and can miss a right answer worded differently.

| Line | Try first |
|---|---|
| Important lore recalled | `inscribe check --lore`. Mark what players must hear `importance: high`. Rewrite lines that mean nothing alone |
| Later story told early | Look for a later stage's fact in `lore` or in an early stage's `summary`, and move it out |
| Run-time characters tell no secret | Every knowledge item that carries the secret needs its `secret` block. Nothing in `lore` may state it |
| Sounds more like the character | A fuller `voice` line and a few `speech_examples` for that character |
| Keeps what the base model could do | More replay (`replay.to_one`), then fewer epochs (`train.epochs`), then a lower rank (`train.rank`). One change at a time |

After a change to the sheet or to `lora.yaml`, run `inscribe run` again. It redoes only the steps whose inputs changed.

## Do not

- Do not train on a sheet nobody has read.
- Do not put the story's gated secret into the sheet in any field. The game hands it over at run time.
- Do not tell the person a LoRA is good because the card passed. The card does not check general ability, and says so.
- Do not treat lore in the weights as guaranteed, or a secret in the weights as safe.
- Do not stop a step you did not start. `inscribe status` shows what this project launched.

## Useful commands

```bash
inscribe status            # the steps this project has launched
inscribe logs train        # a step's output
inscribe stop              # stop this project's running steps
inscribe run --from export # start at a step, taking the earlier ones as done
inscribe eval --no-judge   # the key-word lines only, free
inscribe train --max-steps 5   # a cheap check that the base model loads and the data reads
```
