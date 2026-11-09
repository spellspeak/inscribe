"""Training example and evaluation case schemas.

Notes on the fields:
- player messages carry mode: voice | typed;
- `action_expected` is gone, because the model never chooses actions;
- eval cases can name the clean case they are an ASR-noise twin of;
- eval cases carry prior turns, reported action results, deterministic string checks
  and good/bad reference replies;
- 0.3 (lore in weights): a case may name facts hidden from the student's view
  (`hidden_fact_ids`, passed to the compiler as `hide_facts`) and sentences only the teacher sees
  (`teacher_only`); a training example keeps `hidden_fact_ids` so the mixture step rebuilds the
  same view the teacher answered against.

Labels, provenance, validation and scoring blocks are for analysis only. The
template (`inscribe.runtime.template`) only ever receives the ActorContext
and the turns, so none of these blocks can reach the model's input or targets.
The same holds for the two hidden-fact fields.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from inscribe.runtime.actor_context import ActionResult, ActorContext, OutputPolicy, Turn

CapabilityId = Annotated[str, StringConstraints(pattern=r"^[a-z_]+(\.[a-z_]+)*$")]
Split = Literal["train", "dev", "smoke", "private_test"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(Strict):
    role: Literal["user", "assistant", "game", "other"]  # game: a harness update (0.2); other: another character (0.3)
    content: str
    mode: Literal["voice", "typed"] | None = None
    interrupts_previous: bool = False
    name: str | None = None  # 0.3: the other character's name, required for role other
    to: str | None = None  # 0.3: "you", "player", or a character's name

    @model_validator(mode="after")
    def _other_has_a_name(self) -> "Message":
        if (self.role == "other") != (self.name is not None):
            raise ValueError("an other message needs a name; user, assistant and game messages must not have one")
        return self

    def as_turn(self) -> Turn:
        if self.role == "game":
            return Turn(speaker="game", text=self.content)
        if self.role == "other":
            return Turn(speaker="other", name=self.name, text=self.content, to=self.to)
        if self.role == "user":
            return Turn(speaker="player", text=self.content, mode=self.mode or "typed",
                        interrupts_previous=self.interrupts_previous, to=self.to)
        return Turn(speaker="actor", text=self.content, to=self.to)


class Labels(Strict):
    assertion_policy: str
    forbidden_fact_ids: list[str] = []


class Provenance(Strict):
    generator_id: str
    generator_version: str
    generator_commit: str
    teacher_id: str
    teacher_revision: str
    teacher_prompt_hash: str
    generation_seed: int
    canonical_source_ids: list[str] = []


class Validation(Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_check: str | None = Field(default=None, alias="schema")
    oracle_consistency: str | None = None
    style_review: str | None = None
    duplicate_cluster_id: str | None = None
    reviewer_status: str | None = None


class TrainingExample(Strict):
    schema_version: Literal["training-example-0.1", "training-example-0.3"]  # 0.3: other messages, addressees
    example_id: str
    split: Split
    world_id: str
    scenario_id: str
    primary_capability: CapabilityId
    secondary_capabilities: list[CapabilityId] = []
    difficulty_level: Literal["L1", "L2", "L3", "L4", "L5"]
    student_input_ref: str | None = None
    student_input: ActorContext | None = None
    hidden_fact_ids: list[str] = []  # 0.3: facts left out of the view built from student_input_ref
    messages: list[Message]
    labels: Labels
    provenance: Provenance
    validation: Validation = Validation()

    @model_validator(mode="after")
    def _shape(self) -> "TrainingExample":
        if self.student_input is None and self.student_input_ref is None:
            raise ValueError("need student_input or student_input_ref")
        if len(self.messages) < 2 or self.messages[-1].role != "assistant" or self.messages[-2].role == "assistant":
            raise ValueError("messages must end with a user, other or game turn followed by the assistant reply")
        if self.schema_version == "training-example-0.1" and any(m.role == "other" or m.to for m in self.messages):
            raise ValueError("training-example-0.1 cannot use other messages or addressees; set schema_version: training-example-0.3")
        if self.schema_version == "training-example-0.1" and self.hidden_fact_ids:
            raise ValueError("hidden_fact_ids needs schema_version training-example-0.3")
        return self

    def dialogue_and_reply(self) -> tuple[list[Turn], str]:
        """Split messages into the conversation so far and the supervised reply."""
        return [m.as_turn() for m in self.messages[:-1]], self.messages[-1].content


FailureType = Literal[
    "leak",            # states hidden truth or a secret the actor may not share
    "stale_state",     # uses an old belief or default over fresh observed state
    "wrong_identity",  # speaks as someone else, as an AI, or breaks character
    "off_voice",       # right character and facts, wrong way of speaking for this character
    "assistant_style", # service-assistant phrasing
    "invented_fact",   # makes up something unsupported
    "false_ignorance", # claims not to know something it knows and may share
    "overcertain",     # states a rumour or tentative belief as fact
    "underconfident",  # hedges or doubts something the character knows for certain
    "rule_violation",  # breaks the world's rules or setting
    "false_commit",    # claims an effect the engine did not commit
    "wild_guess",      # misreads garbled voice input instead of asking or using the likely meaning
    "too_long",
    "too_terse",
    "exposition",      # recites knowledge or backstory the player did not ask for
    "repetition",      # repeats earlier turns or its own phrases without need
    "markup",          # markup, emotes or stage directions
    # contract 0.3: group scenes, the boundary policy, the story position
    "misattribution",    # credits words or deeds to the wrong speaker
    "wrong_perspective", # answers as the wrong person, treats another character as the player, answers a line said to someone else, or speaks for others
    "boundary_leak",     # shows knowledge of something the world's boundary policy says does not exist here
    "spoiler",           # reveals story content past the current story position
]


Severity = Literal[
    "must_catch",  # misleads the player or breaks the game; gate G-A applies to these
    "quality",     # style, voice, length, hedging; tracked separately
]


class BadReply(Strict):
    reply: str
    failure: FailureType
    severity: Severity | None = None  # None: the failure type's default


class ReferenceReplies(Strict):
    """Known-good and known-bad replies. Used to prove the checks and judges catch failures (gate G-A)."""

    good: list[str]
    bad: list[BadReply]


class Scoring(Strict):
    deterministic_checks: list[str] = []
    semantic_checks: list[str] = []
    subjective_checks: list[str] = []


class EvalCase(Strict):
    schema_version: Literal["eval-case-0.1", "eval-case-0.2", "eval-case-0.3"]  # 0.2 adds forbidden_event_ids; 0.3 adds last_turn
    case_id: str
    split: Split
    world_partition: str
    capability: Literal["C1", "C2", "C3", "C4", "C5", "C6"] | None = None
    family: str | None = None
    generator_like: str | None = None
    difficulty_level: Literal["L1", "L2", "L3", "L4", "L5"] | None = None
    primary_metric: str
    student_input_ref: str  # path to an actor-context JSON, or "<world file>#<actor id>"
    prior_turns: list[Turn] = []
    action_results: list[ActionResult] = []
    output_policy: OutputPolicy | None = None
    player_utterance: str = ""  # the final player line; empty only when last_turn is given (0.3)
    last_turn: Turn | None = None  # 0.3: the final turn instead of a plain player line: another character's line, a game update, or a player line with an addressee
    input_mode: Literal["voice", "typed"] = "typed"
    interrupts_previous: bool = False
    noise_twin_of: str | None = None
    expected_properties: list[str]
    acceptable_behaviors: list[str] = []
    forbidden_claim_ids: list[str] = []
    forbidden_event_ids: list[str] = []  # world-0.2 events the reply must not assert: unknown to the character, or false rumours stated as fact
    must_not_contain: list[str] = []
    must_contain_any: list[str] = []
    reference_replies: ReferenceReplies | None = None
    notes: str | None = None
    scoring: Scoring
    # 0.3 (lore in weights). Both are instructions for building the prompts, never prompt text: the
    # compiler drops `hidden_fact_ids` from the student's view, and only the teacher's system text gets `teacher_only`.
    hidden_fact_ids: list[str] = []
    teacher_only: list[str] = []

    @model_validator(mode="after")
    def _twin(self) -> "EvalCase":
        if self.noise_twin_of and self.input_mode != "voice":
            raise ValueError("an ASR-noise twin must have input_mode voice")
        if self.schema_version != "eval-case-0.3" and (self.hidden_fact_ids or self.teacher_only):
            raise ValueError("hidden_fact_ids and teacher_only need schema_version eval-case-0.3")
        if bool(self.player_utterance) == (self.last_turn is not None):
            raise ValueError("a case has either a player_utterance or a last_turn, not both and not neither")
        if self.last_turn is not None:
            if self.last_turn.speaker == "actor":
                raise ValueError("last_turn cannot be the character's own line")
            if self.schema_version != "eval-case-0.3":
                raise ValueError("last_turn needs schema_version eval-case-0.3")
            if self.last_turn.speaker == "player" and self.last_turn.mode not in (None, self.input_mode):
                raise ValueError("last_turn.mode must match input_mode")
        if self.schema_version != "eval-case-0.3" and any(t.speaker == "other" or t.to for t in self.prior_turns):
            raise ValueError("other turns and addressees need schema_version eval-case-0.3")
        return self

    def final_turn(self) -> Turn:
        """The turn the character replies to."""
        if self.last_turn is None:
            return Turn(speaker="player", text=self.player_utterance, mode=self.input_mode,
                        interrupts_previous=self.interrupts_previous)
        if self.last_turn.speaker == "player":
            return self.last_turn.model_copy(update={"mode": self.input_mode, "interrupts_previous": self.interrupts_previous})
        return self.last_turn

    def player_text(self) -> str:
        """The player's words in the final turn, or '' when someone else spoke last."""
        t = self.final_turn()
        return t.text if t.speaker == "player" else ""
