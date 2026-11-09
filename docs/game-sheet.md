# The game sheet

`sheet/game-sheet.yaml` holds your game in a form the pipeline reads. Every training scene and every check is built from it, so a person should read it through before anything is trained.

Rules that apply everywhere:

- A line of text is one sentence or phrase on one line, at most 160 characters.
- An id is lower case, starts with a letter, and uses only letters, digits and `_`.
- A level is one of `very_low`, `low`, `neutral`, `high`, `very_high`.
- A field not listed here is an error. `inscribe check` names the field and the problem.

Three kinds of content never reach a prompt. They exist only to be trained into the LoRA: `lore`, `story`, and each character's `backstory` and `speech_examples`.

## Top level

| Field | Required | What |
|---|---|---|
| `schema_version` | yes | `game-sheet-0.2` |
| `game_id` | yes | An id. It must match `game` in `lora.yaml` |
| `setting` | yes | See below |
| `rules` | | Things that are always true of the world. They are in every character's prompt |
| `things` | | Places, buildings, objects, factions and creatures |
| `characters` | yes | The core characters |
| `lore` | | The facts to put in the weights |
| `story` | | The stages of the story, in order |
| `extensions` | | Extra relationship kinds, feelings or verbs your game uses, at most 8 |

## Setting

| Field | Required | What |
|---|---|---|
| `kind` | yes | `closed_fictional`, `historical`, `contemporary` or `custom` |
| `technology` | yes | One line on the level of technology |
| `magic_policy` | yes | `absent`, `present` or `restricted` |
| `boundary` | | What the characters know of the real world |

The boundary has `real_world` (`none`: only the `allowed` list exists here; `listed`: the same, and the list is required; `all`: the real world is this world), an `allowed` list and a `denied` list. Characters meet a denied thing with in-character puzzlement and treat an allowed one as ordinary. Scenes are built for both.

## Things

Each has an `id`, a `kind` (`location`, `structure`, `object`, `faction` or `creature`), a `name`, and optionally `inside`, the id of the thing it is in.

## Characters

| Field | Required | What |
|---|---|---|
| `id`, `name` | yes | |
| `voice` | yes | How they talk: pace, favourite words, what they call the player |
| `role` | | What they do |
| `location` | | The id of the place they are usually found |
| `values`, `likes`, `dislikes` | | Lists of lines |
| `goal` | | An id naming what they want |
| `relationships` | | See below |
| `knows` | | What they always know, secrets included |
| `offers` | | What they sell or do for money |
| `never_discuss` | | Topics they will not talk about, whoever asks |
| `backstory` | | Lines about their past. LoRA only |
| `speech_examples` | | A `player` line and this character's `reply`. Trained on as written. LoRA only |

A relationship has `who` (a character id, a faction id or `player`), optionally a `kind` (`friend`, `rival`, `employer`, `merchant` and so on), levels for `trust`, `affection`, `fear`, `respect` and `familiarity`, and an `opinion` in the character's own words.

A thing they know has `about` (a thing or character id), `what` (an id naming the fact), a `value` (the fact, in one line) and `how` they know it (`self`, `told`, `witnessed` or `inferred`). To make it a secret, add `secret` with `may_tell` (the ids of who may hear it) and a `cover_story` (what they say to everyone else).

An offer has `what`, a `kind` (`item` or `service`), a `price` in the world's own money ("two coppers"), and `available`.

## Lore

Each item is a line, or a mapping with `text`, an `importance` (`high`, `normal` or `low`) and `exact`.

- Mark `importance: high` on what every player must hear. Those items get more scenes, and the scorecard's main line counts them.
- Set `exact: true` on a saying or motto that must come back word for word.
- Write each item as a full fact. "Her son runs it." cannot be asked about on its own.

## Story

Each stage has a `stage` (a short name) and a `summary`. The game tells the model which stage the player is at. The LoRA learns what each stage means, and the spoiler checks ask characters at an early stage what comes next.

## Run-time characters

Characters your game makes up at run time are not in the sheet. Put a few in `npcs.yaml` (the template has one). They are never trained on. The checks hand each one to the model in the prompt alone and ask it about your world.
