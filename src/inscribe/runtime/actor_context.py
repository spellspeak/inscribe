from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from inscribe.runtime.vocab import Confidence, Feeling, HistoryItem, Level, Obligation, Reputation, Salience, Source, TaskStatus, When
PROTOCOL_VERSION = 'actor-context-0.3'

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

class ActorIdentity(Strict):
    id: str
    name: str
    role: str | None = None
    voice: str
    current_goal: str | None = None
    values: list[str] = []
    likes: list[str] = []
    dislikes: list[str] = []

class BoundaryView(Strict):
    real_world: Literal['all', 'none', 'listed']
    allowed: list[str] = []
    denied: list[str] = []

    @model_validator(mode='after')
    def _listed_needs_a_list(self) -> 'BoundaryView':
        if self.real_world == 'listed' and (not self.allowed):
            raise ValueError('a listed boundary needs at least one allowed item')
        return self

class StoryView(Strict):
    stage: str
    summary: str | None = None

class SettingView(Strict):
    knowledge_policy: Literal['closed_fictional', 'historical', 'contemporary', 'custom']
    technology: str
    magic: Literal['absent', 'present', 'restricted']
    boundary: BoundaryView | None = None

class KnownItem(Strict):
    subject: str
    subject_name: str
    predicate: str
    value: str
    source_kind: Literal['witnessed', 'told', 'self', 'inferred']
    learned_at_revision: int
    may_disclose: bool = True
    cover_story: str | None = None
    private_only: bool = False
    tell_only: list[str] = []

class BeliefItem(Strict):
    subject: str
    subject_name: str
    predicate: str
    value: str
    confidence: Literal['certain', 'likely', 'tentative', 'doubtful']
    source_kind: Literal['witnessed', 'told', 'hearsay', 'assumed', 'inferred']

class RelationshipView(Strict):
    trust: Level = 'neutral'
    affection: Level = 'neutral'
    fear: Level = 'neutral'
    kind: str | None = None
    respect: Level | None = None
    familiarity: Level | None = None
    extra: dict[str, Level] = {}
    obligations: list[Obligation] = []
    opinion: str | None = None

class PersonView(RelationshipView):
    who: str
    is_group: bool = False

class EventView(Strict):
    by: str
    verb: str
    whom: str | None = None
    what: str | None = None
    where: str | None = None
    detail: str | None = None
    when: When
    at_time: str | None = None
    source: Source
    told_by: str | None = None
    confidence: Confidence
    feeling: Feeling = 'neutral'
    salience: Salience = 'medium'
    may_disclose: bool = True
    cover_story: str | None = None
    private_only: bool = False
    tell_only: list[str] = []

class PresenceView(Strict):
    who: str
    can_hear: bool = True
    doing: str | None = None

class PlayerTaskView(Strict):
    task: str
    status: TaskStatus

class PlayerSeenView(Strict):
    known_as: str | None = None
    appearance: str | None = None
    condition: str | None = None
    carrying: list[str] = []
    reputation: Reputation | None = None
    tasks: list[PlayerTaskView] = []

class OfferView(Strict):
    what: str
    kind: Literal['item', 'service'] = 'item'
    price: str | None = None
    available: bool = True
    note: str | None = None

class ActionResult(Strict):
    action: str
    status: Literal['committed', 'rejected']
    reason: str | None = None
    details: dict[str, str] = {}

class OutputPolicy(Strict):
    language: Literal['en'] = 'en'
    response_style: Literal['brief_conversation', 'explanatory'] = 'brief_conversation'
    max_output_tokens: int = Field(default=80, ge=8, le=400)

class Turn(Strict):
    speaker: Literal['player', 'actor', 'game', 'other']
    text: str
    mode: Literal['voice', 'typed'] | None = None
    interrupts_previous: bool = False
    name: str | None = None
    to: str | None = None

    @model_validator(mode='after')
    def _player_fields(self) -> 'Turn':
        if self.speaker != 'player' and (self.mode is not None or self.interrupts_previous):
            raise ValueError('actor, game and other turns carry no mode or interruption flag')
        if (self.speaker == 'other') != (self.name is not None):
            raise ValueError('an other turn needs a name; player, actor and game turns must not have one')
        if self.speaker == 'game' and self.to is not None:
            raise ValueError('a game update is not said to anyone')
        if self.speaker == 'other' and self.to == self.name:
            raise ValueError(f'{self.name} cannot speak to themselves')
        return self

    def addressed_elsewhere(self) -> bool:
        return self.speaker == 'player' and self.to not in (None, 'you')

class ActorContext(Strict):
    protocol_version: Literal['actor-context-0.1', 'actor-context-0.2', 'actor-context-0.3'] = PROTOCOL_VERSION
    world_id: str
    session_id: str
    state_revision: int
    actor: ActorIdentity
    location_name: str | None = None
    setting: SettingView
    world_rules: list[str] = []
    known: list[KnownItem] = []
    beliefs: list[BeliefItem] = []
    relationship_to_player: RelationshipView = RelationshipView()
    action_results: list[ActionResult] = []
    output_policy: OutputPolicy = OutputPolicy()
    people: list[PersonView] = []
    events: list[EventView] = []
    player_history: list[HistoryItem] = []
    history_visits: int | None = None
    history_last_seen: When | None = None
    activity: str | None = None
    mood: str | None = None
    glossary: dict[str, str] = {}
    time_of_day: str | None = None
    situation: str | None = None
    present: list[PresenceView] = []
    player_seen: PlayerSeenView | None = None
    offers: list[OfferView] = []
    never_discuss: list[str] = []
    omitted: dict[str, int] = {}
    story: StoryView | None = None
