"""Game worlds: a checked game sheet becomes world-0.3 files.

    world = game_world(sheet)                      # the base world, no story position
    world = game_world(sheet, stage=1)             # at a story stage (index, name or StoryStage)
    world = game_world(sheet, group=True)          # every character with its confidants present (GS-02 shape)
    world = game_world(sheet, dynamic_npcs=npcs)   # plus prompt-only characters, for the checks

Each world is `inscribe.sheets.build.world_from_sheets(sheet, live)` with the live state the pipeline supplies:
the story position, default scenes (each core character at its usual place, a time of day, and in the group
copy the characters its secrets name as confidants), and two kinds of fact nobody observes:

- lore facts (`lore_NNN`): every lore item of the sheet, as a fact of the game's home place. No character observes
  them, so no prompt ever carries them (lore is LoRA-only, game-sheet 0.1); the scenarios hand them to the teacher
  with `teacher_only` and name them in `hidden_fact_ids`.
- stage facts (`stage_N`): one per story stage, so a case at an earlier stage can forbid the later ones
  (`forbidden_claim_ids`: the spoiler), and the judge sees them as hidden truth.

Run-time characters (dynamic NPCs) are not in the sheet and never in the training worlds. For the probe they are added
here as plain prompt-only characters: name, role, voice, a few facts they were given, relationships to the core
characters, and a scene. Their view then renders through the ordinary compiler, with nothing beyond those facts.

World ids: `<game>` (base), `<game>_group`, `<game>_sN` (stage N, 1-based). Cases name them in
`student_input_ref` (`worldgen:<world_id>#<actor>`), so `inscribe.scenarios.core.resolve_generated` and
`student_view` resolve a case against the world file of that id, unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from inscribe.schemas.game_sheet import CharacterSheet, GameSheet, StoryStage
from inscribe.runtime.vocab import Level
from inscribe.schemas.world import PLAYER, World
from inscribe.sheets.build import world_from_sheets

GENERATOR_VERSION = "game-lora-0.1"
LORE_SOURCE = "game_sheet_lore"
STORY_SOURCE = "game_sheet_story"
NPC_SOURCE = "dynamic_npc"
TIMES = ("morning", "midday", "afternoon", "evening")


@dataclass(frozen=True)
class NPCFact:
    """Something a dynamic NPC was given: `about` is a thing or character id of the sheet, or the NPC's own id."""

    about: str
    what: str
    value: str
    how: str = "self"  # self | told | witnessed | inferred


@dataclass(frozen=True)
class NPCRelationship:
    who: str  # a core character id, a faction id or "player"
    kind: str | None = None
    trust: Level = "neutral"
    affection: Level = "neutral"
    fear: Level = "neutral"
    respect: Level | None = None
    opinion: str | None = None


@dataclass(frozen=True)
class DynamicNPC:
    """A prompt-only character: what a scene designer hands the game with a paragraph (a project's `npcs.yaml`)."""

    id: str
    name: str
    role: str
    voice: str
    location: str | None = None
    values: tuple[str, ...] = ()
    likes: tuple[str, ...] = ()
    dislikes: tuple[str, ...] = ()
    facts: tuple[NPCFact, ...] = ()
    relationships: tuple[NPCRelationship, ...] = field(default_factory=tuple)
    time_of_day: str | None = None
    situation: str | None = None


def lore_fact_id(n: int) -> str:
    return f"lore_{n:03d}"


def stage_fact_id(n: int) -> str:
    """n is 1-based: the first stage is stage_1."""
    return f"stage_{n}"


def world_id_for(game_id: str, stage: int | None = None, group: bool = False) -> str:
    """stage is the 0-based index of the stage in `sheet.story`."""
    if stage is not None:
        return f"{game_id}_s{stage + 1}"
    return f"{game_id}_group" if group else game_id


def home_id(sheet: GameSheet) -> str:
    """The entity lore facts are about: the thing named after the game, else the first location, else the first
    thing, else the first character."""
    for t in sheet.things:
        if t.id == sheet.game_id:
            return t.id
    for t in sheet.things:
        if t.kind == "location":
            return t.id
    if sheet.things:
        return sheet.things[0].id
    return sheet.characters[0].id


def lore_facts(sheet: GameSheet) -> list[dict]:
    home = home_id(sheet)
    return [{"fact_id": lore_fact_id(n), "subject": home, "predicate": lore_fact_id(n), "value": item.text,
             "valid_from_revision": 0, "source": LORE_SOURCE} for n, item in enumerate(sheet.lore_items())]


def stage_facts(sheet: GameSheet) -> list[dict]:
    home = home_id(sheet)
    return [{"fact_id": stage_fact_id(n + 1), "subject": home, "predicate": f"story_stage_{n + 1}",
             "value": st.stage + (f". {st.summary}" if st.summary else ""), "valid_from_revision": 0, "source": STORY_SOURCE}
            for n, st in enumerate(sheet.story)]


def resolve_stage(sheet: GameSheet, stage: int | str | StoryStage | None) -> tuple[int, StoryStage] | None:
    if stage is None:
        return None
    if isinstance(stage, StoryStage):
        stage = stage.stage
    if isinstance(stage, int):
        if not 0 <= stage < len(sheet.story):
            raise ValueError(f"{sheet.game_id} has {len(sheet.story)} stages; no stage index {stage}")
        return stage, sheet.story[stage]
    for n, st in enumerate(sheet.story):
        if st.stage == stage:
            return n, st
    raise ValueError(f"{sheet.game_id} has no story stage named {stage!r}")


def confidants(c: CharacterSheet, sheet: GameSheet) -> list[str]:
    """Core characters this character's secrets may be told to, in sheet order, each once."""
    core = {x.id for x in sheet.characters}
    out: list[str] = []
    for k in c.knows:
        for who in (k.secret.may_tell if k.secret else []):
            if who in core and who != c.id and who not in out:
                out.append(who)
    return out


def default_scenes(sheet: GameSheet, stage_index: int | None = None, group: bool = False) -> dict[str, dict]:
    """Each core character at its usual place (the sheet's location) with a time of day that varies by character
    and stage; in the group copy, its confidants are present and can hear."""
    scenes = {}
    for i, c in enumerate(sheet.characters):
        present = [{"who": who, "can_hear": True} for who in confidants(c, sheet)] if group else []
        scenes[c.id] = {"time_of_day": TIMES[(i + (stage_index or 0)) % len(TIMES)], "present": present}
    return scenes


def _npc_entries(sheet: GameSheet, raw: dict, npcs: tuple[DynamicNPC, ...] | list[DynamicNPC]) -> None:
    ids = set(raw["entities"]) | {PLAYER}
    for npc in npcs:
        if npc.id in raw["entities"]:
            raise ValueError(f"dynamic NPC {npc.id} is already in the world; dynamic NPCs must never be in the sheet")
        if npc.location is not None and npc.location not in raw["entities"]:
            raise ValueError(f"dynamic NPC {npc.id}: unknown location {npc.location}")
        raw["entities"][npc.id] = {"kind": "character", "name": npc.name, "role": npc.role,
                                   **({"location": npc.location} if npc.location else {}),
                                   "profile": {"voice": npc.voice, "values": list(npc.values), "likes": list(npc.likes),
                                               "dislikes": list(npc.dislikes)}}
        observations = []
        for i, f in enumerate(npc.facts):
            about = f.about if f.about in ids else npc.id
            fid = f"{npc.id}_{i}"
            raw["facts"].append({"fact_id": fid, "subject": about, "predicate": f.what, "value": f.value,
                                 "valid_from_revision": 0, "source": NPC_SOURCE})
            observations.append({"fact_id": fid, "learned_at_revision": 0, "acquisition": f.how})
        raw["actor_knowledge"][npc.id] = {"observations": observations}
        for r in npc.relationships:
            raw["relationships"].append({"source": npc.id, "target": r.who, "trust": r.trust, "affection": r.affection,
                                         "fear": r.fear, **({"kind": r.kind} if r.kind else {}),
                                         **({"respect": r.respect} if r.respect else {}),
                                         **({"opinion": r.opinion} if r.opinion else {})})
        raw["scenes"][npc.id] = {"time_of_day": npc.time_of_day, "situation": npc.situation, "present": []}


def game_world(sheet: GameSheet, stage: int | str | StoryStage | None = None, dynamic_npcs: tuple[DynamicNPC, ...] | list[DynamicNPC] = (),
               *, group: bool = False) -> World:
    """The world-0.3 for a sheet, at a story stage, with the pipeline's live state (module note)."""
    if stage is not None and group:
        raise ValueError("a world is either a stage world or the group copy, not both")
    resolved = resolve_stage(sheet, stage)
    live: dict = {"facts": lore_facts(sheet) + stage_facts(sheet),
                  "scenes": default_scenes(sheet, resolved[0] if resolved else None, group)}
    if resolved is not None:
        n, st = resolved
        live["story"] = {"stage": st.stage, **({"summary": st.summary} if st.summary else {})}
    raw = world_from_sheets(sheet, live).model_dump()
    raw["schema_version"] = "world-0.3"  # the stage facts and the boundary are 0.3 state even when no stage is set
    raw["world_id"] = world_id_for(sheet.game_id, resolved[0] if resolved else None, group)
    if dynamic_npcs:
        _npc_entries(sheet, raw, dynamic_npcs)
    return World.model_validate(raw)


def game_worlds(sheet: GameSheet) -> dict[str, World]:
    """Every world the training cases name: the base world, the group copy and one world per story stage."""
    worlds = {sheet.game_id: game_world(sheet), world_id_for(sheet.game_id, group=True): game_world(sheet, group=True)}
    for n in range(len(sheet.story)):
        worlds[world_id_for(sheet.game_id, n)] = game_world(sheet, n)
    return worlds
