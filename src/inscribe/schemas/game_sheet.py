"""The game sheet: what a game creator writes once per game, before training the game LoRA.

Two layers:
- Stable, in this sheet: the setting, places and groups, each character's identity, voice, likes, long-standing
  relationships, what it always knows (secrets with who may hear them and a cover story), what it offers, and
  topics it never discusses. Lore, backstory and speech examples feed the LoRA only and never reach a prompt.
- Live, from the game at runtime: events, history with the player, scene, the player as perceived, current
  relationships and state. Live state is merged on top of the sheet (`inscribe.sheets.build`), and
  live wins where they disagree.

game-sheet-0.2: lore items may carry an importance and whether they must be recalled
word for word (what the creator marks important gets more data); the sheet may hold the shape of the story
arc (stages, LoRA only; the live position comes from the game); the setting may carry the boundary policy for
real-world knowledge. A 0.1 sheet still validates and may not use these.
"""
from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from inscribe.runtime.output import speakability_issues
from inscribe.runtime.vocab import Extension, Id, Level, Obligation, Phrase, ShortPhrase, Strict
from inscribe.schemas.world import Offer, Setting

SHEET_VERSION = "game-sheet-0.2"
Importance = Literal["low", "normal", "high"]


class LoreItem(Strict):
    """One piece of stable lore for the LoRA (0.2). `importance` scales how much data it gets; `exact` means
    the wording itself matters (a name, a date, a line of scripture) and recall is scored word for word."""

    text: Phrase
    importance: Importance = "normal"
    exact: bool = False


class StoryStage(Strict):
    """One stage of the story arc, for the LoRA (0.2). The live position names one of these stages."""

    stage: ShortPhrase
    summary: Phrase | None = None


class SpeechExample(Strict):
    """How the character talks, for the LoRA's style data. The reply must pass the output rule."""

    player: Phrase
    reply: Phrase

    @model_validator(mode="after")
    def _speakable(self) -> "SpeechExample":
        issues = speakability_issues(self.reply)
        if issues:
            raise ValueError(f"speech example is not speakable: {self.reply!r}: " + "; ".join(issues))
        return self


class SheetRelationship(Strict):
    """A long-standing relationship. `who` is a character, a group, or "player"; `kind` says what they are to
    this character; free text is written from this character's side ("you" is this character)."""

    who: str
    kind: Id | None = None
    trust: Level = "neutral"
    affection: Level = "neutral"
    fear: Level = "neutral"
    respect: Level | None = None
    familiarity: Level | None = None
    extra: dict[Id, Level] = {}
    obligations: list[Obligation] = []
    opinion: Phrase | None = None


class Secret(Strict):
    """Who this character may tell (character ids or "player"; empty means nobody), and what it says instead."""

    may_tell: list[str] = []
    cover_story: Phrase | None = None


class Knowledge(Strict):
    """Something this character always knows: `about` (a thing, a character or "player") `what`: `value`."""

    about: str
    what: Id
    value: Phrase
    how: Literal["self", "told", "witnessed", "inferred"] = "self"
    secret: Secret | None = None


class Thing(Strict):
    id: Id
    kind: Literal["location", "structure", "object", "faction", "creature"]
    name: Phrase
    inside: Id | None = None


class CharacterSheet(Strict):
    id: Id
    name: Phrase
    role: Phrase | None = None
    location: Id | None = None
    voice: Phrase
    values: list[Phrase] = []
    goal: Id | None = None
    likes: list[Phrase] = []
    dislikes: list[Phrase] = []
    relationships: list[SheetRelationship] = []
    knows: list[Knowledge] = []
    offers: list[Offer] = []
    never_discuss: list[Phrase] = []
    backstory: list[Phrase] = []  # LoRA only
    speech_examples: list[SpeechExample] = []  # LoRA only


class GameSheet(Strict):
    schema_version: Literal["game-sheet-0.1", "game-sheet-0.2"]
    game_id: Id
    setting: Setting
    rules: list[Phrase] = []
    things: list[Thing] = []
    extensions: list[Extension] = []
    characters: list[CharacterSheet]
    lore: list[Phrase | LoreItem] = []  # LoRA only; 0.2: items may carry importance and exact
    story: list[StoryStage] = []  # LoRA only (0.2): the shape of the arc

    @model_validator(mode="after")
    def _unique_ids(self) -> "GameSheet":
        ids = [t.id for t in self.things] + [c.id for c in self.characters]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError("duplicate ids in game sheet: " + ", ".join(dupes))
        if self.schema_version == "game-sheet-0.1":
            used = [name for name, hit in (("lore items with importance", any(isinstance(x, LoreItem) for x in self.lore)),
                                           ("story", bool(self.story)), ("setting.boundary", self.setting.boundary is not None)) if hit]
            if used:
                raise ValueError("game-sheet-0.1 cannot use " + ", ".join(used) + "; set schema_version: game-sheet-0.2")
        return self

    def lore_items(self) -> list[LoreItem]:
        """Every lore entry as a LoreItem (plain strings are normal importance, not exact)."""
        return [x if isinstance(x, LoreItem) else LoreItem(text=x) for x in self.lore]
