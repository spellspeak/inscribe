"""The spoiler check by the strong model. For every reply to a what-comes-next question, the judge reads what the
character knows from the start, the story so far and the stages still to come, and says whether the reply reveals,
hints at or presupposes anything that belongs only to the later stages.

The key-word check in `plan/recall.py` bans a handful of rare words per stage and cannot carry a "never". This can,
for a few cents a run.

    lines = spoiler_requests(bundle, sheet, replies)       # replies: {id: the spoken line}
    report = judge(bundle, sheet, replies, api_key)        # told early, with the judge's evidence and a 95% upper bound

The judge is shown the game's lore and the character's own facts as well as the stages. Without them a premise looks
like a later stage: lore that says the player was hired "to write the bid" and a late stage "with the bid they wrote"
made an early mention of the bid look like a spoiler.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from inscribe.judges import strong
from inscribe.plan.recall import clean_reply, wilson_upper
from inscribe.schemas.game_sheet import GameSheet

OUTPUT_FORECAST_TOKENS = 300
VERSION = "spoiler-judge-0.2"

SYSTEM = """You are checking a game character's line for spoilers. You will see what the character knows from the start (the game's lore and the character's own facts), the story stages that have already happened and the stages still to come, then a question the player asked the character and the character's one-line reply. The character stands at the end of the stages that have happened and knows nothing of the later ones.

Answer whether the reply reveals, hints at or presupposes any event, decision, outcome or fact that belongs only to the later stages. Restating what is known from the start or what has already happened, refusing, deflecting, guessing in a general way ("things will get worse") or stating the character's own hopes is not a spoiler. A later stage mentioning something known from the start does not make it a later fact. Naming a later event, a later decision, who will do what, how things turn out, or a fact the later stages introduce is a spoiler, even if the words differ."""


class SpoilerVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reveals_later_stage: bool
    evidence: str


SCHEMA = strong.strict_schema(SpoilerVerdict)


def _stage_lines(sheet: GameSheet, upto: int) -> tuple[str, str]:
    done = "\n".join(f"Stage {i + 1}, {st.stage}: {st.summary or ''}".rstrip(": ") for i, st in enumerate(sheet.story[: upto + 1]))
    later = "\n".join(f"Stage {i + 1}, {st.stage}: {st.summary or ''}".rstrip(": ") for i, st in enumerate(sheet.story) if i > upto)
    return done, later


def known_from_start(sheet: GameSheet, actor: str) -> str:
    """The game's lore and the asked character's own facts: role, what they know, what they think of others."""
    lore = [x if isinstance(x, str) else x.text for x in sheet.lore]
    c = next((c for c in sheet.characters if c.id == actor), None)
    own = []
    if c is not None:
        own += [f"{c.name}: {c.role}"] if c.role else []
        own += [f"{c.name} knows: {k.about} {k.what.replace('_', ' ')}: {k.value}" for k in c.knows]
        own += [f"{c.name} on {r.who}: {r.opinion}" for r in c.relationships if r.opinion]
    return "\n".join(f"- {x}" for x in lore + own)


def spoiler_requests(bundle: dict, sheet: GameSheet, replies: dict[str, str], model: str = strong.MODEL) -> list[dict]:
    """One request per answered what-comes-next question."""
    names = {c.id: c.name for c in sheet.characters}
    lines = []
    for r in bundle["spoiler"]:
        if r["id"] not in replies:
            continue
        done, later = _stage_lines(sheet, r["stage_index"])
        user = (f"Known from the start (the character knows these):\n{known_from_start(sheet, r['actor'])}\n\n"
                f"Story so far (the character knows these):\n{done}\n\nStill to come (the character knows none of this):\n{later}\n\n"
                f"The player asked {names.get(r['actor'], r['actor'])}: \"{r['question']}\"\nThe character replied: \"{replies[r['id']]}\"\n\n"
                "Does the reply reveal, hint at or presuppose anything from the stages still to come?")
        lines.append(strong.request(r["id"], [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], SCHEMA, "spoiler_verdict", model))
    return lines


def spoken(rows: list[dict]) -> dict[str, str]:
    """Reply rows ([{id, text}]) as {id: the spoken line}."""
    return {r["id"]: clean_reply(r.get("text") or "") for r in rows}


def judge(bundle: dict, sheet: GameSheet, replies: dict[str, str], api_key: str = "", max_usd: float = 1.0, model: str = strong.MODEL,
          concurrency: int = 8, answer=None) -> dict:
    """Ask the judge about every answered what-comes-next question. `answer` replaces the paid call (tests)."""
    lines = spoiler_requests(bundle, sheet, replies, model)
    forecast = strong.forecast_usd(lines, OUTPUT_FORECAST_TOKENS)
    if forecast > max_usd:
        raise RuntimeError(f"the spoiler check would cost about ${forecast:.2f}, over its cap of ${max_usd:.2f}")
    answer = answer or (lambda ls: strong.answer(ls, api_key, concurrency=concurrency))
    questions = {r["id"]: r["question"] for r in bundle["spoiler"]}
    told, errors, usage = [], [], []
    for line in answer(lines):
        p = strong.parse(line)
        usage.append(p.usage)
        if p.error or not p.text:
            errors.append({"id": p.custom_id, "error": p.error})
            continue
        v = SpoilerVerdict.model_validate_json(p.text)
        if v.reveals_later_stage:
            told.append({"id": p.custom_id, "asked": questions.get(p.custom_id, ""), "reply": replies.get(p.custom_id, "")[:200], "evidence": v.evidence})
    n = len(lines) - len(errors)
    return {"judge": model, "version": VERSION, "prompts": n, "told_early": len(told), "upper_bound_95": wilson_upper(len(told), n),
            "told": told, "errors": errors, "forecast_usd": round(forecast, 3), "cost_usd": round(strong.cost_usd(usage), 3)}
