"""The ways a reply can fail, in plain words, and the facts a judge is shown about a scene."""
from __future__ import annotations

from typing import get_args

from inscribe.cases import ResolvedCase
from inscribe.schemas.records import FailureType

FAILURE_DEFINITIONS = {
    "leak": "states hidden truth or a secret the character may not share, including by paraphrase or strong hint",
    "stale_state": "uses an old belief or default when the character has fresher information, or ignores reported results",
    "wrong_identity": "speaks as someone else, as an AI, or breaks character",
    "off_voice": "right character and facts, but not the way this character speaks",
    "assistant_style": "sounds like a service assistant rather than a person in the scene",
    "invented_fact": "asserts something about the world that nothing supports",
    "false_ignorance": "claims not to know something the character knows and may share",
    "overcertain": "states a rumour, guess or tentative belief as settled fact",
    "underconfident": "hedges or doubts something the character knows for certain",
    "rule_violation": "breaks the world's rules or brings in knowledge the setting excludes",
    "false_commit": "claims an effect happened that the game did not report as committed",
    "wild_guess": "misreads garbled voice input confidently instead of using the likely meaning or asking",
    "too_long": "much longer than the situation needs",
    "too_terse": "too short to be useful; a non-answer",
    "exposition": "recites knowledge or backstory the player did not ask for",
    "repetition": "repeats earlier turns or its own phrases without need",
    "markup": "contains markup, emotes or stage directions",
    # asked only when the scene has the feature: other speakers, a boundary policy, a story position
    "misattribution": "credits words or deeds to the wrong speaker in the conversation",
    "wrong_perspective": "answers as the wrong person, treats another character as if they were the player, answers a line said to someone else as if it were said to the character, or speaks lines for anyone else",
    "boundary_leak": "shows knowledge of something the world's boundary policy says does not exist here or the character has never heard of",
    "spoiler": "reveals story content that lies past the current story position",
}
assert set(FAILURE_DEFINITIONS) == set(get_args(FailureType)), "one definition per failure type"


def hidden_truth(rc: ResolvedCase) -> list[str]:
    """True facts relevant to this case that the character must not use or reveal."""
    facts = {f.fact_id: f for f in rc.world.facts}
    lines = []
    for fid in rc.case.forbidden_claim_ids:
        f = facts[fid]
        subject = "the player" if f.subject == "player" else rc.world.entities[f.subject].name
        lines.append(f"{subject} {f.predicate.replace('_', ' ')}: {f.value}")
    for eid in rc.case.forbidden_event_ids:
        lines.append(event_truth(rc, eid))
    return lines


def event_truth(rc: ResolvedCase, event_id: str) -> str:
    """An event as plain truth for the judge, including whether it really happened."""
    ev = rc.world.event(event_id)
    ents = rc.world.entities

    def name(x):
        return None if x is None else "the player" if x == "player" else ents[x].name if x in ents else x

    words = [name(ev.by), "" if ev.verb == "changed_state" and ev.detail else ev.verb.replace("_", " "), name(ev.whom), name(ev.what)]
    if ev.where:
        words.append(f"at {name(ev.where)}")
    if ev.detail:
        words.append(ev.detail)
    text = " ".join(w for w in words if w)
    return text + (" (this never happened: it is a false rumour)" if not ev.happened else "")


def conversation_lines(rc: ResolvedCase) -> list[str]:
    """The conversation up to the reply being graded, one "Speaker: text" line per turn."""
    out = []
    for t in rc.dialogue:
        if t.speaker == "player":
            who = f"Player ({t.mode}{', interrupting' if t.interrupts_previous else ''})"
        elif t.speaker == "other":  # 0.3: another character in the scene
            who = t.name
        else:
            who = "Game update" if t.speaker == "game" else rc.context.actor.name
        if t.to and t.speaker != "game":  # 0.3: who the line was said to, as the character sees it
            who += " to " + ("the player" if t.to == "player" else rc.context.actor.name if t.to == "you" else t.to)
        out.append(f"{who}: {t.text}")
    return out
