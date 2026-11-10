"""The questions a judge is asked about one reply, and how its answers are read.

Each expected property of a scene becomes one yes-or-no question about the reply, and so does each kind of failure
that can apply. A judge gives every question a probability that its answer is yes. A reply's fail probability is the
worst of them: the least likely property, or the most likely failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from inscribe.cases import ResolvedCase
from inscribe.judges.rubric import FAILURE_DEFINITIONS, conversation_lines, hidden_truth
from inscribe.runtime.template import render_system

Polarity = Literal["holds", "violated", "permission", "present"]
# Each failure as a question (yes means the failure is there) and what a "no" means. What a "yes" means is the
# rubric's own definition.
FAILURE_QUESTIONS: dict[str, tuple[str, str]] = {
    "leak": (
        "Does `reply` reveal a fact listed in `hidden_truth` or a secret that `character_sheet` says the character must not tell?",
        "The reply keeps every hidden fact and secret back.",
    ),
    "stale_state": (
        "Does `reply` rely on information that newer information in `character_sheet` or `conversation` has replaced?",
        "The reply matches the freshest information the character has, including anything under \"What just happened\".",
    ),
    "wrong_identity": (
        "Does `reply` break character?",
        "The reply is spoken by the character described in `character_sheet`, inside the scene.",
    ),
    "off_voice": (
        "Does `reply` sound unlike the voice described in `character_sheet`?",
        "The wording, tone and register fit the described voice.",
    ),
    "assistant_style": (
        "Does `reply` sound like a service assistant?",
        "The reply sounds like a person in the scene talking to the player.",
    ),
    "invented_fact": (
        "Does `reply` assert a fact about the world that is unsupported by `character_sheet` and `conversation`?",
        "Every claim in the reply is supported by the character sheet or the conversation, or is framed as the character's own guess.",
    ),
    "false_ignorance": (
        "Does `reply` claim ignorance of something that `character_sheet` lists as known and shareable?",
        "The reply shares what the character knows, or the character really lacks that knowledge, or it is a secret the character must keep.",
    ),
    "overcertain": (
        "Does `reply` state a rumour, guess or tentative belief as settled fact?",
        "Beliefs that `character_sheet` marks as tentative, likely, hearsay or assumed are hedged, or the reply states only what the character knows for certain.",
    ),
    "underconfident": (
        "Does `reply` hedge or doubt something the character knows for certain?",
        "The reply states what the character knows plainly.",
    ),
    "rule_violation": (
        "Does `reply` break a rule of this world, or bring in knowledge or technology the setting excludes?",
        "The reply fits the world rules and setting in `character_sheet`.",
    ),
    "false_commit": (
        "Does `reply` claim an action or effect happened that is absent from the results the game reported in `character_sheet`?",
        "Every completed action the reply mentions is reported under \"What just happened\", or the reply only offers or proposes.",
    ),
    "wild_guess": (
        "Does `reply` confidently answer a misheard version of the player's last line?",
        "The player's last line is clear, or the reply answers its likely meaning, or the reply asks what the player meant.",
    ),
    "too_long": (
        "Is `reply` much longer than the situation needs?",
        "The length suits the moment.",
    ),
    "too_terse": (
        "Is `reply` too short to be useful?",
        "The reply gives the player something usable, even if it is brief or an in-character refusal.",
    ),
    "exposition": (
        "Does `reply` recite unrequested knowledge or backstory?",
        "Everything in the reply answers or follows from what the player said.",
    ),
    "repetition": (
        "Does `reply` needlessly repeat earlier lines from `conversation` or its own phrases?",
        "The reply says something new, or repeats only what the moment needs.",
    ),
    "markup": (
        "Does `reply` contain markup, emotes or stage directions?",
        "The reply is plain spoken words. Normal punctuation and quotes are fine.",
    ),
    # Asked only when the scene has the feature: other speakers, a boundary policy, a story position.
    "misattribution": (
        "Check every 'you said', 'he said', 'she told you', 'as you asked' and the like in `reply` against `conversation`. "
        "Does `reply` credit words or deeds to the wrong person, or treat something one person said as if another had said it?",
        "Everything the reply credits to someone was said or done by that person in the conversation, or the reply credits nothing.",
    ),
    "wrong_perspective": (
        "`last_line` says who spoke last and to whom, and `who_is_speaking` says who the character is. Does `reply` answer a line "
        "said to someone else as if it had been said to the character, reply as if the character were the player, treat another "
        "character as if they were the player, or put words in the mouth of the player or of another character?",
        "The reply is the character's own words, aimed at whoever spoke to it; a line said to someone else gets at most a bystander's reaction.",
    ),
    "boundary_leak": (
        "Does `reply` show knowledge of something that `character_sheet` says does not exist in this world or that the character has never heard of?",
        "The reply treats such things as unknown, in character, and uses only what the world allows.",
    ),
    "spoiler": (
        "Does `reply` reveal something about the story that lies past the story position in `character_sheet`, or that the character could not know yet?",
        "The reply stays within what the character knows at this point in the story.",
    ),
}
# With a boundary policy, false ignorance also covers things the policy says exist here: a fantasy innkeeper who
# has never heard of a horse when horses are allowed, a shop owner who has never heard of wifi when the real world is this world.
BOUNDARY_FALSE_IGNORANCE = (
    "Does `reply` claim ignorance of something that `character_sheet` lists as known and shareable, or of something that "
    "`world_boundary` says exists in this world (an allowed thing, or any ordinary real-world thing when the real world is this world)?",
    "The reply shares what the character knows, treats things that exist in this world as familiar, or the character really lacks that "
    "knowledge, or it is a secret the character must keep.",
)


def applicable_failures(rc) -> list[str]:
    """Every failure question, minus the ones a scene has no feature for: other speakers and addressees, a boundary
    policy, a story position."""
    group = any(t.speaker == "other" or t.to for t in rc.dialogue)
    boundary = rc.context.setting.boundary is not None
    story = rc.context.story is not None
    wanted = {"misattribution": group, "wrong_perspective": group, "boundary_leak": boundary, "spoiler": story,
              # a question the player asks again may get the same answer; repeating it is the point there
              "repetition": "gives_the_same_answer_as_before" not in rc.case.expected_properties}
    return [f for f in FAILURE_QUESTIONS if wanted.get(f, True)]


def failure_yes_criterion(failure: str) -> str:
    """The rubric definition as a sentence about the reply."""
    if failure == "off_voice":
        return "The reply has the right character and facts, but is not the way this character speaks."
    definition = FAILURE_DEFINITIONS[failure]
    verb = definition.split()[0]
    return f"The reply {definition}." if verb.endswith("s") else f"The reply is {definition}."


def failure_question(failure: str, boundary: bool = False) -> dict:
    instruction, _ = BOUNDARY_FALSE_IGNORANCE if (failure == "false_ignorance" and boundary) else FAILURE_QUESTIONS[failure]
    return {"instructions": instruction}
_STOPWORDS = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for"}
_FIXED_CASE = {"ai": "AI", "id": "ID", **{d.lower(): d for d in (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")}}


def _is_third_person_verb(word: str) -> bool:
    return len(word) > 2 and word.endswith("s") and not word.endswith(("ous", "ss", "us", "is"))


def _base_verb(word: str) -> str:
    if word == "does":
        return "do"
    if word.endswith("ies"):
        return word[:-1] if len(word) <= 4 else word[:-3] + "y"   # ties -> tie, applies -> apply
    if word.endswith(("sses", "shes", "ches", "xes", "zes", "oes")):
        return word[:-2]                                           # dismisses -> dismiss
    return word[:-1]


def name_map(rc: ResolvedCase) -> dict[str, str]:
    """Lowercase word -> its capitalised form, from the world's entity names (property ids are lowercase)."""
    out: dict[str, str] = {}
    for ent in rc.world.entities.values():
        for tok in str(getattr(ent, "name", "")).split():
            tok = tok.strip(".,'\"")
            if len(tok) > 1 and tok[0].isupper() and tok.lower() not in _STOPWORDS:
                out[tok.lower()] = tok
    return out


def property_question(prop: str, names: dict[str, str] | None = None) -> tuple[str, Polarity]:
    """Turn a property id into one yes/no question about `reply`, and say how its answer reads.

    holds: yes = property holds. violated: the property is a prohibition, so the question asks
    for the forbidden thing and yes = property broken. permission: the property only allows
    something, so no answer can fail the reply.
    """
    names = {**_FIXED_CASE, **(names or {})}
    w = prop.split("_")

    def text(words: list[str]) -> str:
        return " ".join(names.get(x, x) for x in words)

    if w[:3] == ["does", "not", "know"] and len(w) > 3:
        return f"Does `reply` show that the character knows {text(w[3:])}?", "violated"
    if w[:2] == ["does", "not"] and len(w) > 2:
        return f"Does `reply` {text(w[2:])}?", "violated"
    if w[0] == "stays" and len(w) == 2 and w[1] in names and names[w[1]][0].isupper():
        return f"Does `reply` stay in character as {names[w[1]]}?", "holds"
    if w[0] == "never" and len(w) > 1 and _is_third_person_verb(w[1]):
        return f"Does `reply` {_base_verb(w[1])} {text(w[2:])}".rstrip() + "?", "violated"
    if w[0] == "no" and len(w) > 1:
        return f"Does `reply` contain {text(w[1:])}?", "violated"
    if w[0] == "not" and len(w) > 1:
        return f"Is `reply` {text(w[1:])}?", "violated"
    if w[0] == "neither" and "nor" in w[2:]:
        i = w.index("nor")
        verbs = f"{_base_verb(w[1])} or {_base_verb(w[i + 1])}"
        return f"Does `reply` {verbs} {text(w[i + 2:])}".rstrip() + "?", "violated"
    if w[0] == "may" and len(w) > 1:
        return f"Does `reply` {text(w[1:])}?", "permission"
    if _is_third_person_verb(w[0]):
        verb = _base_verb(w[0])
        if len(w) > 2 and w[1] in ("no", "nothing"):
            obj = "any" if w[1] == "no" else "anything"
            return f"Does `reply` {verb} {obj} {text(w[2:])}?", "violated"
        return f"Does `reply` {verb} {text(w[1:])}".rstrip() + "?", "holds"
    # An adjective phrase ("shrewd_and_clipped", "dry_not_preachy").
    return f"Is `reply` {text(w)}?".replace(" not ", " rather than "), "holds"


@dataclass(frozen=True)
class QuestionKey:
    kind: Literal["property", "failure"]
    name: str           # property id or failure type
    polarity: Polarity

    def as_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "polarity": self.polarity}


@dataclass(frozen=True)
class JudgeRequest:
    item_id: str
    body: dict                       # the state, and the questions by id
    key: dict[str, QuestionKey] = field(default_factory=dict)


def contract_03_state(rc: ResolvedCase) -> dict:
    """Extra state for scenes with other speakers, a boundary policy or a story position: who the last line was said
    to, the world's boundary, the story position. Stated as facts the judge would otherwise have to infer from the
    rendered text. Scenes without these features get no extra keys."""
    me = rc.context.actor.name
    extra: dict = {}
    if any(t.speaker == "other" or t.to for t in rc.dialogue):
        last = rc.dialogue[-1]
        by = "the player" if last.speaker == "player" else last.name if last.speaker == "other" else "the game"
        if last.speaker == "game":
            said_to = "nobody; it is a game update"
        elif last.to in (None, ""):
            said_to = f"the room, not to {me} in particular"
        elif last.to == "you":
            said_to = f"{me} (the character)"
        elif last.to == "player":
            said_to = f"the player, not to {me}"
        else:
            said_to = f"{last.to}, not to {me}"
        others = sorted({t.name for t in rc.dialogue if t.speaker == "other"})
        roles = {e.name: e.role for e in rc.world.entities.values() if getattr(e, "role", None)}
        described = [f"{n} ({roles[n]})" if n in roles else n for n in others]
        extra["last_line"] = f"spoken by {by} to {said_to}."
        extra["who_is_speaking"] = (f"{me}" + (f" ({roles[me]})" if me in roles else "") + " is the character replying. The player is a separate person. "
                                    + (f"Also in the conversation: {', '.join(described)}." if described else ""))
    b = rc.context.setting.boundary
    if b is not None:
        parts = []
        if b.real_world == "none":
            parts.append("nothing from the real world exists in this world" + (f" except: {', '.join(b.allowed)}" if b.allowed else ""))
        elif b.real_world == "listed":
            parts.append(f"of the real world only these exist here: {', '.join(b.allowed)}")
        else:
            parts.append("the real world is this world, so ordinary real-world things exist and are known")
        if b.denied:
            parts.append(f"these do not exist here and the character has never heard of them: {', '.join(b.denied)}")
        extra["world_boundary"] = "; ".join(parts) + "."
    if rc.context.story is not None:
        extra["story_position"] = rc.context.story.stage + (f": {rc.context.story.summary}" if rc.context.story.summary else "")
    return extra


def build_state(rc: ResolvedCase, reply: str, extra_sheet: str = "") -> dict:
    # acceptable_alternatives: answers the scene allows, so a judge does not fail a reply for taking one of them.
    # extra_sheet: sentences the character knows that no view renders (lore the model carries in its weights), so the
    # judge does not call a correct lore answer invented.
    return {
        "acceptable_alternatives": [b.replace("_", " ") for b in rc.case.acceptable_behaviors] or ["none listed"],
        "character_sheet": render_system(rc.context) + (f"\nAlso known to you:\n{extra_sheet}" if extra_sheet else ""),
        "conversation": conversation_lines(rc),
        "hidden_truth": hidden_truth(rc) or ["none for this case"],
        "reply": reply,
        **contract_03_state(rc),
    }


def build_request(rc: ResolvedCase, reply: str, item_id: str, failures: list[str] | None = None, extra_sheet: str = "") -> JudgeRequest:
    """One reply to judge: the scene's state, one question per expected property and one per failure that applies.
    `failures`: the failure questions to ask; default every applicable one."""
    names = name_map(rc)
    questions: dict[str, dict] = {}
    key: dict[str, QuestionKey] = {}
    entries = [("property", p) for p in rc.case.expected_properties] + [("failure", f) for f in (applicable_failures(rc) if failures is None else failures)]
    for n, (kind, name) in enumerate(entries, start=1):
        qid = f"q{n:02d}"
        if kind == "property":
            instruction, polarity = property_question(name, names)
            questions[qid] = {"instructions": instruction}
        else:
            questions[qid], polarity = failure_question(name, boundary=rc.context.setting.boundary is not None), "present"
        key[qid] = QuestionKey(kind, name, polarity)
    return JudgeRequest(item_id, {"state": build_state(rc, reply, extra_sheet), "questions": questions}, key)


def holds_probability(p: float, polarity: Polarity) -> float:
    """The probability that a property holds, from the probability that its question's answer is yes."""
    return 1.0 - p if polarity == "violated" else p


def fail_probability(answers: dict, key: dict[str, QuestionKey]) -> float:
    """max(1 - min property-holds, max failure-present). Permissions are ignored."""
    holds = [holds_probability(answers[q], k.polarity)
             for q, k in key.items() if k.kind == "property" and k.polarity != "permission"]
    present = [answers[q] for q, k in key.items() if k.kind == "failure"]
    return max(1.0 - min(holds) if holds else 0.0, max(present) if present else 0.0)


def flagged_failures(answers: dict, key: dict[str, QuestionKey], threshold: float) -> list[str]:
    return sorted(k.name for q, k in key.items() if k.kind == "failure" and answers[q] >= threshold)


def top_question(answers: dict, key: dict[str, QuestionKey]) -> tuple[str, float]:
    """The question that drives the fail probability, as (description, its fail contribution)."""
    best = ("", -1.0)
    for q, k in key.items():
        if k.kind == "property" and k.polarity == "permission":
            continue
        p = answers[q] if k.kind == "failure" else 1.0 - holds_probability(answers[q], k.polarity)
        if p > best[1]:
            best = (f"{k.kind}:{k.name}", p)
    return best


assert set(FAILURE_QUESTIONS) == set(FAILURE_DEFINITIONS), "one question per failure type"
