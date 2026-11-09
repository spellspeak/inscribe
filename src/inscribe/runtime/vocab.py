from __future__ import annotations
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
SNAPSHOT_VERSION = 'npc-snapshot-0.1'
Level = Literal['very_low', 'low', 'neutral', 'high', 'very_high']
LEVELS: tuple[str, ...] = Level.__args__
FEELING_DIMENSIONS = ('trust', 'affection', 'respect', 'fear', 'familiarity')
RELATIONSHIP_KINDS = ('parent', 'child', 'sibling', 'spouse', 'relative', 'partner', 'friend', 'acquaintance', 'stranger', 'rival', 'enemy', 'employer', 'employee', 'mentor', 'apprentice', 'superior', 'subordinate', 'ally', 'customer', 'merchant', 'neighbour')
ObligationType = Literal['owes_favour', 'is_owed_favour', 'owes_money', 'is_owed_money', 'promised', 'was_promised']
Source = Literal['witnessed', 'overheard', 'told_by', 'rumour', 'read', 'inferred', 'common_knowledge', 'own_action', 'assumed']
Confidence = Literal['certain', 'likely', 'tentative', 'doubtful']
When = Literal['just_now', 'earlier_today', 'yesterday', 'this_week', 'long_ago']
WHEN: tuple[str, ...] = When.__args__
Salience = Literal['low', 'medium', 'high', 'defining']
SALIENCE: tuple[str, ...] = Salience.__args__
Feeling = Literal['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
FEELING_SCORE = {'very_negative': -2, 'negative': -1, 'neutral': 0, 'positive': 1, 'very_positive': 2}
TRAINING_MOODS = ('calm', 'content', 'cheerful', 'excited', 'proud', 'amused', 'relieved', 'hopeful', 'grateful', 'curious', 'tired', 'bored', 'distracted', 'impatient', 'nervous', 'worried', 'afraid', 'sad', 'grieving', 'lonely', 'irritated', 'angry', 'bitter', 'suspicious', 'embarrassed', 'guilty', 'confused')
Mood = Annotated[str, StringConstraints(min_length=1, max_length=40, pattern='^[^\\n]*$')]
ShortPhrase = Mood
TRAINING_TIMES_OF_DAY = ('dawn', 'morning', 'midday', 'afternoon', 'evening', 'night', 'late at night')
VERB_GROUPS: dict[str, tuple[str, ...]] = {'social': ('greeted', 'talked_to', 'helped', 'insulted', 'threatened', 'lied_to', 'promised', 'broke_promise', 'betrayed', 'apologised', 'gave_gift'), 'trade': ('bought', 'sold', 'paid', 'refused_to_pay', 'borrowed', 'lent', 'stole', 'bribed'), 'conflict': ('attacked', 'injured', 'killed', 'defended', 'fled', 'arrested', 'escaped', 'captured'), 'movement': ('arrived', 'left', 'entered', 'trespassed'), 'objects': ('found', 'lost', 'broke', 'repaired', 'made', 'used', 'hid'), 'information': ('revealed', 'discovered', 'accused', 'confessed', 'reported'), 'tasks': ('asked_for_help', 'agreed_to_help', 'completed_task', 'failed_task', 'abandoned_task'), 'world': ('changed_state', 'announced', 'died', 'disaster')}
VERBS: tuple[str, ...] = tuple((v for group in VERB_GROUPS.values() for v in group))
Reputation = Literal['unknown', 'disliked', 'neutral', 'liked', 'renowned']
REPUTATION: tuple[str, ...] = Reputation.__args__
TaskStatus = Literal['not_offered', 'offered', 'active', 'done', 'failed', 'reward_given']
TASK_STATUS: tuple[str, ...] = TaskStatus.__args__
CAPS: dict[str, int] = {'people': 12, 'events': 20, 'player_history': 8, 'facts': 20, 'present': 8, 'offers': 12, 'never_discuss': 8, 'player_tasks': 8}
EXTENSION_LIMIT = 8
ExtensionKind = Literal['verb', 'relationship_kind', 'feeling']
CORE_BY_KIND: dict[str, tuple[str, ...]] = {'verb': VERBS, 'relationship_kind': RELATIONSHIP_KINDS, 'feeling': FEELING_DIMENSIONS}
Id = Annotated[str, StringConstraints(pattern='^[a-z][a-z0-9_]*$', max_length=64)]
Phrase = Annotated[str, StringConstraints(min_length=1, max_length=160, pattern='^[^\\n]*$')]

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Extension(Strict):
    name: Id
    kind: ExtensionKind
    meaning: Phrase

class Obligation(Strict):
    type: ObligationType
    detail: Phrase | None = None

class HistoryEvent(Strict):
    type: Literal['event'] = 'event'
    by: Literal['player', 'self']
    verb: Id
    what: Phrase | None = None
    detail: Phrase | None = None
    when: When
    feeling: Feeling = 'neutral'
    salience: Salience = 'medium'

class HistoryTold(Strict):
    type: Literal['told'] = 'told'
    said: Phrase
    when: When
    salience: Salience = 'medium'

class HistoryPromise(Strict):
    type: Literal['promise'] = 'promise'
    by: Literal['player', 'self']
    what: Phrase
    status: Literal['open', 'kept', 'broken']
    when: When
    salience: Salience = 'medium'
HistoryItem = Annotated[Union[HistoryEvent, HistoryTold, HistoryPromise], Field(discriminator='type')]

def rank_key(salience: str, when: str | None, tiebreak: str) -> tuple:
    recency = WHEN.index(when) if when in WHEN else len(WHEN)
    return (-SALIENCE.index(salience), recency, tiebreak)
