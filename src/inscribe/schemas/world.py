"""The world schema.

The authoritative world: what is actually true, plus what each character knows,
believes, and may disclose. This is a generator/simulator representation. It is
never shown to the model directly; the context compiler projects one actor's view
from it (`inscribe.context.compiler`).

It holds character profiles, world rules
and relationships so a single file can drive the no-leak test.

world-0.2 (NPC snapshot 0.1) adds, all optional:
- relationships with a kind, respect, familiarity, obligations, an opinion, and groups as targets;
- events (who did what to whom), and per character which events it knows about and how;
- history with the player, per character (owned by the game's save system; the harness only reads it);
- per-character state (activity, mood) and the game's extension values.
The developer sends this truth plus who knows what; the compiler projects each character's view.
A world-0.1 file still loads, and may not use the 0.2 fields.

world-0.3 adds, both optional and gated to 0.3 files:
- setting.boundary: the policy for real-world knowledge;
- story: where the story is right now; the shape of the arc lives in the game LoRA.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inscribe.runtime.vocab import (
    CORE_BY_KIND,
    EXTENSION_LIMIT,
    Confidence,
    Extension,
    Feeling,
    HistoryEvent,
    HistoryItem,
    Id,
    Level,
    Mood,
    Obligation,
    Phrase,
    Reputation,
    Salience,
    ShortPhrase,
    Source,
    TaskStatus,
    When,
)

SCHEMA_VERSION = "world-0.3"
SCHEMA_VERSIONS = ("world-0.1", "world-0.2", "world-0.3")
PLAYER = "player"

Revision = Annotated[int, Field(ge=0)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Boundary(Strict):
    """The world's policy for real-world knowledge (0.3). `none`: nothing from the real world exists
    here except `allowed`. `listed`: only `allowed` does. `all`: the real world is this world, except
    `denied`. `denied` can name specific traps under any blanket. The creator sets it in the game sheet."""

    real_world: Literal["all", "none", "listed"]
    allowed: list[Phrase] = []
    denied: list[Phrase] = []

    @model_validator(mode="after")
    def _listed_needs_a_list(self) -> "Boundary":
        if self.real_world == "listed" and not self.allowed:
            raise ValueError("a listed boundary needs at least one allowed item")
        return self


class StoryPosition(Strict):
    """Where the story is right now (0.3): a short stage name the game LoRA's data keys on, and an
    optional one-line summary. Live state; fixed when a conversation starts."""

    stage: ShortPhrase
    summary: Phrase | None = None


class Setting(Strict):
    kind: Literal["closed_fictional", "historical", "contemporary", "custom"]
    technology: str
    magic_policy: Literal["absent", "present", "restricted"]
    language: Literal["en"] = "en"
    boundary: Boundary | None = None  # world-0.3


class CharacterProfile(Strict):
    voice: str
    current_goal: str | None = None
    values: list[str] = []
    likes: list[Phrase] = []  # 0.2: stable preferences, such as gift tastes in a life sim
    dislikes: list[Phrase] = []


class Entity(Strict):
    kind: Literal["character", "location", "structure", "object", "faction", "creature"]
    name: str
    role: str | None = None
    location: Id | None = None
    profile: CharacterProfile | None = None


class Fact(Strict):
    """Something actually true in the simulation from a given revision onward.
    In world-0.2 the subject may also be the player ("the player is a police inspector")."""

    fact_id: Id
    subject: Id
    predicate: Id
    value: str
    valid_from_revision: Revision
    source: str
    information_class: Id | None = None


class Observation(Strict):
    """A true fact this actor has access to, and how they got it."""

    fact_id: Id
    learned_at_revision: Revision
    acquisition: Literal["witnessed", "told", "self", "inferred"]


class Belief(Strict):
    """Something the actor thinks is true. It may be stale or simply wrong."""

    subject: Id
    predicate: Id
    value: str
    confidence: Literal["certain", "likely", "tentative", "doubtful"]
    acquisition: Literal["witnessed", "told", "hearsay", "assumed", "inferred"]
    learned_at_revision: Revision | None = None


class DisclosureRule(Strict):
    """Information this actor holds but may only share with the listed audience.

    `cover_story` (0.2): what the actor says instead when asked. An NPC knowingly lies only
    when its snapshot gives it a cover story; otherwise it deflects.
    """

    rule_id: Id
    fact_ids: list[Id] = []
    event_ids: list[Id] = []
    information_class: Id | None = None
    permitted_audience: list[str] = []
    cover_story: Phrase | None = None

    @model_validator(mode="after")
    def _targets_something(self) -> "DisclosureRule":
        if not self.fact_ids and not self.event_ids and not self.information_class:
            raise ValueError(f"disclosure rule {self.rule_id} must name fact_ids, event_ids or an information_class")
        return self


class EventKnowledge(Strict):
    """An event this actor knows about (or has heard of), and how (0.2)."""

    event_id: Id
    source: Source
    told_by: str | None = None  # who told them: a character id or "player"; required when source is told_by
    confidence: Confidence = "certain"
    learned_at_revision: Revision
    salience: Salience | None = None  # this actor's own importance; default: the event's
    feeling: Feeling = "neutral"

    @model_validator(mode="after")
    def _told_by(self) -> "EventKnowledge":
        if (self.source == "told_by") != (self.told_by is not None):
            raise ValueError(f"event knowledge {self.event_id}: told_by is needed exactly when source is told_by")
        return self


class ActorKnowledge(Strict):
    observations: list[Observation] = []
    beliefs: list[Belief] = []
    disclosure_rules: list[DisclosureRule] = []
    events: list[EventKnowledge] = []


class Relationship(Strict):
    """How `source` feels about `target` (a character, the player, or a group). Only the source sees it.

    0.1 fields: trust, affection, fear. 0.2 adds kind, respect, familiarity, extension feelings,
    obligations, a one-line opinion, salience and when they last interacted.
    `kind` says what the target is to the source: a shop owner's relationship to a customer has
    kind customer. Free text (opinion, obligation details) is written from the source's side:
    "you" is the source.
    """

    source: Id
    target: str
    kind: Id | None = None
    trust: Level = "neutral"
    affection: Level = "neutral"
    fear: Level = "neutral"
    respect: Level | None = None
    familiarity: Level | None = None
    extra: dict[Id, Level] = {}
    obligations: list[Obligation] = []
    opinion: Phrase | None = None
    salience: Salience = "medium"
    last_interaction: When | None = None

    def uses_02(self) -> bool:
        return bool(self.kind or self.respect or self.familiarity or self.extra or self.obligations or self.opinion
                    or self.salience != "medium" or self.last_interaction)


class Event(Strict):
    """Something that happened, or that is rumoured to have happened (0.2).

    Read as: `by` `verb` `whom` `what` at `where` `detail`. `when` is relative to now and supplied by
    the game. `happened: false` marks a rumour or lie about something that never happened; characters
    can still believe it, and the flag itself never reaches a view.
    """

    event_id: Id
    verb: Id
    by: str
    whom: str | None = None
    what: Phrase | None = None
    where: Id | None = None
    detail: Phrase | None = None
    at_revision: Revision
    when: When
    at_time: ShortPhrase | None = None  # optional precise time from the game, e.g. "around nine last night"
    happened: bool = True
    information_class: Id | None = None
    salience: Salience = "medium"


class ActorState(Strict):
    """Section A, what the character is doing and feeling now (0.2). Mood is any short word or phrase
    the game's engine tracks (open-ended). If it is left out the harness derives one."""

    activity: Phrase | None = None
    mood: Mood | None = None


class Presence(Strict):
    """Someone else in the scene (section E): a character in this world, or a description of people the
    game does not model ("two guards"). Who can hear decides what may be said (harness-managed)."""

    who: Id | None = None
    description: Phrase | None = None
    can_hear: bool = True
    doing: Phrase | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "Presence":
        if (self.who is None) == (self.description is None):
            raise ValueError("a presence needs exactly one of who or description")
        return self


class Scene(Strict):
    """Section E for one character: when and where the interaction happens, and who else is there."""

    time_of_day: ShortPhrase | None = None
    situation: Phrase | None = None
    present: list[Presence] = []


class PlayerTask(Strict):
    task: Phrase
    status: TaskStatus


class PlayerSeen(Strict):
    """Section F for one character: the player as this character perceives them, not as they truly are."""

    known_as: Phrase | None = None
    appearance: Phrase | None = None
    condition: Phrase | None = None
    carrying: list[Phrase] = []
    reputation: Reputation | None = None
    tasks: list[PlayerTask] = []


class Offer(Strict):
    what: Phrase
    kind: Literal["item", "service"] = "item"
    price: Phrase | None = None
    available: bool = True
    note: Phrase | None = None


class Offers(Strict):
    """Section H for one character: what it can offer, and topics it never discusses whatever is asked."""

    offers: list[Offer] = []
    never_discuss: list[Phrase] = []


class PlayerHistory(Strict):
    """Section D for one character. Owned by the game's save system; the harness only reads it."""

    items: list[HistoryItem] = []
    visits: Annotated[int, Field(ge=0)] | None = None
    last_seen: When | None = None


class WorldRule(Strict):
    rule_id: Id
    statement: str


class World(Strict):
    schema_version: Literal["world-0.1", "world-0.2", "world-0.3"]
    world_id: Id
    world_seed: int
    setting: Setting
    state_revision: Revision
    entities: dict[Id, Entity]
    facts: list[Fact]
    rules: list[WorldRule] = []
    relationships: list[Relationship] = []
    actor_knowledge: dict[Id, ActorKnowledge] = {}
    # world-0.2
    events: list[Event] = []
    actor_state: dict[Id, ActorState] = {}
    player_history: dict[Id, PlayerHistory] = {}
    extensions: list[Extension] = []
    scenes: dict[Id, Scene] = {}
    player_seen: dict[Id, PlayerSeen] = {}
    offers: dict[Id, Offers] = {}
    # world-0.3
    story: StoryPosition | None = None

    def fact(self, fact_id: str) -> Fact:
        return next(f for f in self.facts if f.fact_id == fact_id)

    def event(self, event_id: str) -> Event:
        return next(e for e in self.events if e.event_id == event_id)

    def characters(self) -> list[str]:
        return [eid for eid, e in self.entities.items() if e.kind == "character"]

    def allowed(self, kind: str) -> set[str]:
        """Core vocabulary of one kind plus the values this game declared."""
        return set(CORE_BY_KIND[kind]) | {x.name for x in self.extensions if x.kind == kind}

    def _uses_02(self) -> list[str]:
        used = [name for name in ("events", "actor_state", "player_history", "extensions", "scenes", "player_seen", "offers")
                if getattr(self, name)]
        if any(r.uses_02() for r in self.relationships):
            used.append("relationship fields")
        for k in self.actor_knowledge.values():
            if k.events:
                used.append("actor_knowledge.events")
            if any(r.event_ids or r.cover_story for r in k.disclosure_rules):
                used.append("disclosure event_ids or cover_story")
        return used

    def _uses_03(self) -> list[str]:
        return [name for name, used in (("setting.boundary", self.setting.boundary), ("story", self.story)) if used]

    @model_validator(mode="after")
    def _integrity(self) -> "World":
        errors: list[str] = []
        ents, chars = self.entities, set(self.characters())
        if self.schema_version != "world-0.3" and self._uses_03():
            errors.append(f"{self.schema_version} cannot use " + ", ".join(self._uses_03()) + "; set schema_version: world-0.3")

        def dupes(ids: list[str]) -> set[str]:
            return {i for i in ids if ids.count(i) > 1}

        for d in dupes([f.fact_id for f in self.facts]):
            errors.append(f"duplicate fact_id {d}")
        for d in dupes([r.rule_id for r in self.rules]):
            errors.append(f"duplicate world rule_id {d}")

        for eid, e in ents.items():
            if e.location and e.location not in ents:
                errors.append(f"entity {eid}: unknown location {e.location}")
            if e.profile and e.kind != "character":
                errors.append(f"entity {eid}: only characters have a profile")
            if e.kind == "character" and not e.profile:
                errors.append(f"entity {eid}: characters need a profile")

        facts = {f.fact_id: f for f in self.facts}
        for f in self.facts:
            if f.subject not in ents and not (f.subject == PLAYER and self.schema_version != "world-0.1"):
                errors.append(f"fact {f.fact_id}: unknown subject {f.subject}")
            if f.valid_from_revision > self.state_revision:
                errors.append(f"fact {f.fact_id}: valid_from_revision is after state_revision")

        audience_ok = chars | {PLAYER}
        groups = {eid for eid, e in ents.items() if e.kind == "faction"}
        events = {e.event_id: e for e in self.events}

        if self.schema_version == "world-0.1" and self._uses_02():
            errors.append("world-0.1 cannot use " + ", ".join(sorted(set(self._uses_02()))) + "; set schema_version: world-0.2")

        # Extensions: a capped list, each value new to its kind (snapshot 0.1 bounds).
        if len(self.extensions) > EXTENSION_LIMIT:
            errors.append(f"{len(self.extensions)} extension values; the limit is {EXTENSION_LIMIT} per game")
        for d in dupes([x.name for x in self.extensions]):
            errors.append(f"duplicate extension {d}")
        for x in self.extensions:
            if x.name in CORE_BY_KIND[x.kind]:
                errors.append(f"extension {x.name} is already a core {x.kind}")
        verbs, kinds, feelings = (self.allowed(k) for k in ("verb", "relationship_kind", "feeling"))

        for d in dupes([e.event_id for e in self.events]):
            errors.append(f"duplicate event_id {d}")
        for e in self.events:
            if e.verb not in verbs:
                errors.append(f"event {e.event_id}: verb {e.verb} is not core and not a declared extension")
            for role, who in (("by", e.by), ("whom", e.whom)):
                if who is not None and who not in ents and who != PLAYER:
                    errors.append(f"event {e.event_id}: unknown {role} {who}")
            if e.where is not None and e.where not in ents:
                errors.append(f"event {e.event_id}: unknown where {e.where}")
            if e.at_revision > self.state_revision:
                errors.append(f"event {e.event_id}: at_revision is after state_revision")

        for actor, st in self.actor_state.items():
            if actor not in chars:
                errors.append(f"actor_state: {actor} is not a character")
        for section in ("scenes", "player_seen", "offers"):
            for actor in getattr(self, section):
                if actor not in chars:
                    errors.append(f"{section}: {actor} is not a character")
        for actor, scene in self.scenes.items():
            for p in scene.present:
                if p.who is not None and (p.who not in chars or p.who == actor):
                    errors.append(f"scenes {actor}: present {p.who} is not another character")
        for actor, hist in self.player_history.items():
            if actor not in chars:
                errors.append(f"player_history: {actor} is not a character")
            for item in hist.items:
                if isinstance(item, HistoryEvent) and item.verb not in verbs:
                    errors.append(f"player_history {actor}: verb {item.verb} is not core and not a declared extension")
        for actor, k in self.actor_knowledge.items():
            if actor not in chars:
                errors.append(f"actor_knowledge: {actor} is not a character")
            for o in k.observations:
                f = facts.get(o.fact_id)
                if f is None:
                    errors.append(f"{actor}: observation of unknown fact {o.fact_id}")
                    continue
                if not f.valid_from_revision <= o.learned_at_revision <= self.state_revision:
                    errors.append(f"{actor}: observation of {o.fact_id} learned outside its valid revisions")
            for b in k.beliefs:
                if b.subject not in ents and not (b.subject == PLAYER and self.schema_version != "world-0.1"):
                    errors.append(f"{actor}: belief about unknown subject {b.subject}")
                if b.learned_at_revision is not None and b.learned_at_revision > self.state_revision:
                    errors.append(f"{actor}: belief learned after state_revision")
            for ek in k.events:
                ev = events.get(ek.event_id)
                if ev is None:
                    errors.append(f"{actor}: knows unknown event {ek.event_id}")
                    continue
                if not ev.at_revision <= ek.learned_at_revision <= self.state_revision:
                    errors.append(f"{actor}: event {ek.event_id} learned outside its valid revisions")
                if ek.told_by is not None and ek.told_by not in audience_ok:
                    errors.append(f"{actor}: event {ek.event_id} told_by unknown {ek.told_by}")
            for d in dupes([ek.event_id for ek in k.events]):
                errors.append(f"{actor}: event {d} known twice")
            for r in k.disclosure_rules:
                for fid in r.fact_ids:
                    if fid not in facts:
                        errors.append(f"{actor}: disclosure rule {r.rule_id} names unknown fact {fid}")
                for eid in r.event_ids:
                    if eid not in events:
                        errors.append(f"{actor}: disclosure rule {r.rule_id} names unknown event {eid}")
                for a in r.permitted_audience:
                    if a not in audience_ok:
                        errors.append(f"{actor}: disclosure rule {r.rule_id} has unknown audience {a}")

        for rel in self.relationships:
            if rel.source not in chars:
                errors.append(f"relationship source {rel.source} is not a character")
            if rel.target not in audience_ok | groups:
                errors.append(f"relationship target {rel.target} is not a character, the player or a group")
            if rel.target == rel.source:
                errors.append(f"relationship {rel.source} -> itself")
            if rel.kind is not None and rel.kind not in kinds:
                errors.append(f"relationship {rel.source} -> {rel.target}: kind {rel.kind} is not core and not a declared extension")
            for dim in rel.extra:
                if dim not in feelings:
                    errors.append(f"relationship {rel.source} -> {rel.target}: feeling {dim} is not a declared extension")
                elif dim in CORE_BY_KIND["feeling"]:
                    errors.append(f"relationship {rel.source} -> {rel.target}: {dim} is a core feeling, not an extra")
        for d in dupes([f"{r.source}>{r.target}" for r in self.relationships]):
            errors.append(f"duplicate relationship {d}")

        if errors:
            raise ValueError("world integrity failed: " + "; ".join(errors))
        return self
