"""Read a game's prose lore into a game sheet, for a person to check.

    sheet, info = extract(lore_dir, "my_game", api_key)        # three paid calls to the strong model
    f = forecast(read_documents(lore_dir), "my_game")          # the cost from the document sizes; nothing is sent

Reads world.md, characters.md, story.md and style.md. It never reads npcs.yaml: characters the game makes up at run
time exist only as prompt text and must not enter the sheet.

Three calls, each answered in a fixed JSON shape whose fields are plain strings and choices. The sheet's own length
and id rules are checked afterwards.

1. world: world.md and story.md give the setting with its boundary, the rules, things, vocabulary extensions, lore
   items (importance and exactness from the designer's own marks), story stages, and the words the story says must not
   be said early.
2. characters: characters.md, plus the setting, things and extensions from call 1, gives the characters with their
   knowledge, secrets (who may be told, and a cover story), relationships, offers, never-discuss topics and backstory.
3. style: style.md, plus the character ids, gives speech examples: explicit player line and reply pairs only.

After each call the sheet so far is validated: the schema, then a world is built from it, then a scan for the
story's gated words. On an error the error text goes back to the model once for a repair pass. If the result is still
invalid the run fails with the error. What the documents state that the sheet cannot hold comes back as
`could_not_place` and goes into the report.

Money: every call is paid. The cost is forecast from the document sizes before anything is sent, and the run stops if
the forecast, with one repair pass per call, would pass the cap.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import re
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from inscribe.judges import strong
from inscribe.project import LORE_FILES
from inscribe.runtime.vocab import EXTENSION_LIMIT, FEELING_DIMENSIONS, LEVELS, RELATIONSHIP_KINDS, Level, ObligationType
from inscribe.schemas.game_sheet import SHEET_VERSION, GameSheet
from inscribe.sheets.build import world_from_sheets

DOCS = LORE_FILES
STAGES = ("world", "characters", "style")
DOCS_BY_STAGE = {"world": ("world.md", "story.md"), "characters": ("characters.md",), "style": ("style.md",)}

MAX_OUTPUT_TOKENS = 32_000  # ceiling per call: a full character list is several thousand visible tokens plus reasoning
# Output tokens assumed per call for the forecast (visible JSON plus reasoning). An assumption, not a measurement;
# the report shows the measured figure next to it.
ASSUMED_OUTPUT_TOKENS = {"world": 12_000, "characters": 16_000, "style": 5_000}
DEFAULT_MAX_USD = 3.0
DEFAULT_EFFORT = "medium"
TEXT_LIMIT = 150  # what the prompt asks for; the sheet allows 160
HOW = ("self", "told", "witnessed", "inferred")
OBLIGATIONS: tuple[str, ...] = ObligationType.__args__


class ExtractionError(RuntimeError):
    """The model's output did not become a valid sheet, even after the repair pass."""


class BudgetError(RuntimeError):
    """The forecast or the running total would pass the cap. Raised before the call is made."""


# --- what the model returns: plain strings and choices, so the strict schema can express every field ----------
class X(BaseModel):
    model_config = ConfigDict(extra="forbid")


class XBoundary(X):
    real_world: Literal["all", "none", "listed"]
    allowed: list[str]
    denied: list[str]


class XSetting(X):
    kind: Literal["closed_fictional", "historical", "contemporary", "custom"]
    technology: str
    magic_policy: Literal["absent", "present", "restricted"]
    boundary: XBoundary | None


class XThing(X):
    id: str
    kind: Literal["location", "structure", "object", "faction", "creature"]
    name: str
    inside: str | None


class XExtension(X):
    name: str
    kind: Literal["verb", "relationship_kind", "feeling"]
    meaning: str


class XLore(X):
    text: str
    importance: Literal["low", "normal", "high"]
    exact: bool


class XStage(X):
    stage: str
    summary: str | None


class XWorld(X):
    """Call 1."""

    setting: XSetting
    rules: list[str]
    things: list[XThing]
    extensions: list[XExtension]
    lore: list[XLore]
    story: list[XStage]
    gated_terms: list[str]
    could_not_place: list[str]


class XObligation(X):
    type: ObligationType
    detail: str | None


class XExtraFeeling(X):
    feeling: str
    level: Level


class XRelationship(X):
    who: str
    kind: str | None
    trust: Level
    affection: Level
    fear: Level
    respect: Level | None
    familiarity: Level | None
    extra: list[XExtraFeeling]
    obligations: list[XObligation]
    opinion: str | None


class XSecret(X):
    may_tell: list[str]
    cover_story: str | None


class XKnowledge(X):
    about: str
    what: str
    value: str
    how: Literal["self", "told", "witnessed", "inferred"]
    secret: XSecret | None


class XOffer(X):
    what: str
    kind: Literal["item", "service"]
    price: str | None
    available: bool
    note: str | None


class XCharacter(X):
    id: str
    name: str
    role: str | None
    location: str | None
    voice: str
    values: list[str]
    goal: str | None
    likes: list[str]
    dislikes: list[str]
    relationships: list[XRelationship]
    knows: list[XKnowledge]
    offers: list[XOffer]
    never_discuss: list[str]
    backstory: list[str]


class XCharacters(X):
    """Call 2."""

    characters: list[XCharacter]
    extensions: list[XExtension]
    could_not_place: list[str]


class XSpeech(X):
    character_id: str
    player: str
    reply: str


class XStyle(X):
    """Call 3."""

    speech_examples: list[XSpeech]
    could_not_place: list[str]


MODELS = {"world": XWorld, "characters": XCharacters, "style": XStyle}
SCHEMAS = {stage: strong.strict_schema(model) for stage, model in MODELS.items()}


# --- prompts --------------------------------------------------------------------------------------------------
SYSTEM = f"""You extract a game sheet from a game studio's prose design documents. The sheet is a fixed form the studio will confirm by hand, so it must hold only what the documents state. You are an extractor, not a writer.

Rules:
1. Never invent. Every value must be stated in the documents or follow directly from them. When a field has no support in the text, leave it empty (null or an empty list) rather than guess. Put anything the documents state that the form cannot hold into could_not_place, one short line each, so the studio can see what was left out.
2. Ids are lowercase snake_case ASCII: letters, digits and underscores, starting with a letter, at most 64 characters (for example hythe, heron_and_hook, npc_maren_tolliver). Character ids start with npc_. Every reference to an id (inside, location, who, about, may_tell, character_id) must be an id defined in this sheet, or the word player.
3. Use only the vocabulary given for levels, relationship kinds, obligation types, knowledge sources, thing kinds and importance. If the documents need a relationship kind or feeling the core vocabulary lacks, declare it as an extension (at most {EXTENSION_LIMIT} per game) and then use it; otherwise leave the kind null.
4. Every text value is a single line of at most {TEXT_LIMIT} characters, in the documents' own words. Each item is one self-contained fact or statement that means something on its own: a date stays with its event, a number with what it counts, a condition with its consequence. Prefer the designer's own sentence as the item; do not split a sentence into clauses. When a passage holds several facts, make one item per fact, and never a fragment that means nothing alone. No markdown, no bullet characters, no line breaks.
5. Importance and exactness come only from the designer's own marks in the prose, never from your judgement. "Players must hear this" (or should hear, or the like) means importance high. "Say it exactly" (or that a name, number or wording must be exact) means importance high and exact true. "Background colour" means importance low. No mark means normal. exact applies to the wording the mark points at (a quoted line, a name, a number): put that wording in its own item and mark it exact; the sentence around it is not exact.
6. A character's secret is a knowledge item with a secret block. may_tell lists the ids of the characters the text says already know it or may be told it (or player); "nobody may be told" with no one named is an empty list. cover_story is the cover story the text gives, as one complete sentence, word for word where it is quoted. When a secret needs more than one item, every one of its items carries the same secret block. When the text says a character knows another character's secret, that character gets it as a secret too: may_tell lists the secret's owner and anyone else the text names as knowing, never the character itself, and cover_story only when the text gives this character one.
7. The story's gated secret (the spoiler the documents say must not be revealed before a given stage) never enters the sheet in any field: not lore, knowledge, backstory, relationships, story summaries, rules, offers or could_not_place. Describe story stages by what the player is doing and what has changed in the town, without the revelation. The game supplies gated facts at runtime.
8. Whatever changes with the story (what a character knows or will say from a later stage onward, what the player has done, prices that move) is live state, not the sheet. Leave it out of the sheet's fields and note it in could_not_place as live state, with no spoiler in it.
9. Speech examples come only from explicit player-line and reply pairs in the documents, copied word for word. Never write a player line for a sample reply that has none.
10. Extract each fact once. The documents often restate the body in a numbered list of lore facts; when a fact appears in both, keep one item in the numbered list's wording with the numbered list's mark, and drop the body's version even when its words differ. The exception is the documents' own rules section: every line of it is a rule, and a rule that also carries an importance mark is a lore item as well (rules reach the runtime prompt, lore reaches the weights). Writers' notes, tone notes and document housekeeping (draft numbers, what the designers want writers to feel, instructions to writers) are neither lore nor rules; mention them in could_not_place in one line.
11. could_not_place is a short summary of what was left out, grouped: one line per section, or per character and topic (for example "story.md, stage 2: what each character knows and says", or "Maren: three sample lines with no player line: ..."). Never copy the documents line by line."""

VOCAB = {
    "levels": list(LEVELS),
    "relationship_kinds": list(RELATIONSHIP_KINDS),
    "obligation_types": list(OBLIGATIONS),
    "knowledge_how": list(HOW),
    "core_feelings": list(FEELING_DIMENSIONS),
    "thing_kinds": ["location", "structure", "object", "faction", "creature"],
    "importance": ["low", "normal", "high"],
}

GUIDE = {
    "world": {
        "setting.kind": "closed_fictional (its own world), historical, contemporary (the real world now, possibly with one twist) or custom",
        "setting.technology": "one line on the technology level in the documents' words",
        "setting.magic_policy": "absent, present or restricted",
        "setting.boundary": "the documents' policy for real-world knowledge. real_world: none (nothing from the real world exists here except the allowed list), listed (only the allowed list exists), all (this is the real world, except the denied list). allowed and denied: short items copied from the documents' own lists. null if the documents give no policy",
        "rules": "what is and is not possible in this world: every line of the documents' own rules section, one rule per item, whether or not it also carries a mark (rule 10). Nothing from outside that section",
        "things": "the named places, structures, factions, notable objects and creatures that play refers to by name: somewhere a character can be, a group a character belongs to, an object or animal the story turns on. Not the goods in a price list and not the ordinary furniture of the setting (those are lore or offers). inside: the id of the thing it is in, when the documents say so, else null",
        "extensions": "relationship kinds, feelings or verbs the game needs beyond the core vocabulary, each with its one-line meaning",
        "lore": "the stable facts of the world for the game's lore: one self-contained fact per item (rule 4), each fact once (rule 10), importance and exact from the marks (rule 5). History, calendar, geography, factions, money and prices belong here when the documents state them",
        "story": "the stages of the story arc in order: a short stage name and a one-line summary of what the player is doing and what has changed, with no gated secret (rule 7)",
        "gated_terms": "the words, phrases and whole sentences the story document says nobody says before the gate (for example 'nobody says the words X, Y' or 'nobody says that Z'), copied from it, so the pipeline can check that nothing leaked. Empty if the documents name none",
        "could_not_place": "everything the documents state that none of the fields above can hold (rule 1), except the gated secret",
    },
    "characters": {
        "characters": "the core characters only, in document order. id: npc_ plus the name. role: one line. location: the id of the thing where the character is usually found, from the things given, else null. voice: how they talk, in two to four sentences, keeping every tone word, habit, tic and quoted phrase the documents give (short sentences, counts out loud, soft and level), and adding the turns of phrase the documents put in their mouth elsewhere; drop nothing the documents say about their speech, invent nothing values, likes, dislikes: from the sections of those names, one item each. goal: a snake_case id for a standing goal the documents state, else null",
        "characters[].relationships": "from the relationship table: who is a character id, a faction id or player; trust, affection, fear and respect as written (a level with a qualifier like 'high, falling' is the level); familiarity null unless given; opinion is the 'in their words' line; obligations only where the text states a debt, favour or promise",
        "characters[].knows": "what the character knows from the start of the game, including what they know that others do not and their secret. about: a thing id, a character id or player. what: a snake_case predicate. value: one line. how: self, told, witnessed or inferred. secret: rule 6. Knowledge that begins at a later stage is live state (rule 8)",
        "characters[].offers": "from the Offers paragraph: what, item or service, price as written (null when free or not for sale), available true unless the text says otherwise, note only for a condition that belongs to that one offer. A condition that applies to all of a character's offers (coin only, no credit, regulars first) is stated once, in values or backstory, never repeated on every offer",
        "characters[].never_discuss": "topics the text says the character never discusses, whatever is asked, each as a short noun phrase naming the topic (for example: who he treats), not a sentence",
        "characters[].backstory": "stable facts about the character's past and habits that no other field holds, one line each",
        "extensions": "any relationship kind or feeling you needed beyond the core vocabulary and the extensions already declared",
        "could_not_place": "what the documents state about these characters that the fields cannot hold, grouped per character (rule 11): sample lines without a player line (quoted), live-state knowledge (rule 8), anything else; never the gated secret",
    },
    "style": {
        "speech_examples": "every explicit player-line and reply pair in the document, copied word for word, with the id of the character who replies from the list given. Skip a pair whose speaker is not in the list",
        "could_not_place": "the style rules and anything else in the document the sheet cannot hold",
    },
}


def read_documents(lore_dir: pathlib.Path) -> dict[str, str]:
    """The four documents, by name."""
    docs = {}
    for name in DOCS:
        path = pathlib.Path(lore_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"{lore_dir} has no {name}")
        docs[name] = path.read_text()
    return docs


def build_messages(stage: str, docs: dict[str, str], game_id: str, context: dict | None = None) -> list[dict]:
    """The system and user messages for one call. Only that stage's documents are included."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage}")
    body = {
        "task": f"Extract the {stage} part of the game sheet for the game '{game_id}' from the documents below.",
        "documents": {name: docs[name] for name in DOCS_BY_STAGE[stage]},
        "already_extracted": context or {},
        "vocabulary": VOCAB,
        "fields": GUIDE[stage],
    }
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(body, indent=1, ensure_ascii=False)}]


def repair_messages(stage: str, docs: dict[str, str], game_id: str, context: dict | None, previous: dict, errors: str) -> list[dict]:
    messages = build_messages(stage, docs, game_id, context)
    body = json.loads(messages[1]["content"])
    body["task"] = (f"Your previous extraction of the {stage} part for '{game_id}' failed validation. Return the whole "
                    f"corrected object. Change only what the errors need; keep everything else as it was. Never resolve an "
                    f"error by inventing: drop or shorten an item the documents do not support.")
    body["validation_errors"] = errors
    body["your_previous_output"] = previous
    messages[1]["content"] = json.dumps(body, indent=1, ensure_ascii=False)
    return messages


def request_line(custom_id: str, messages: list[dict], stage: str, effort: str, model: str = strong.MODEL) -> dict:
    line = strong.request(custom_id, messages, SCHEMAS[stage], f"game_sheet_{stage}", model)
    line["body"]["reasoning"] = {"effort": effort}
    line["body"]["max_output_tokens"] = MAX_OUTPUT_TOKENS
    return line


# --- mapping the model's answer onto the sheet ---------------------------------------------------------------
_WS = re.compile(r"\s+")


def _text(s: str | None) -> str | None:
    if s is None:
        return None
    s = _WS.sub(" ", s).strip()
    return s or None


def to_id(s: str | None) -> str | None:
    """Normalise an id the model wrote (case, spaces, hyphens). Applied to every id field and reference alike, so
    references stay aligned; anything still wrong is caught by validation."""
    if s is None:
        return None
    s = s.strip()
    if s == "player":
        return s
    s = re.sub(r"[\s\-]+", "_", s.lower())
    return s or None


_id = to_id


def _texts(items: list[str]) -> list[str]:
    return [t for t in (_text(x) for x in items) if t]


def _opt(d: dict, **fields) -> dict:
    d.update({k: v for k, v in fields.items() if v is not None and v != [] and v != {}})
    return d


def _ext(x: XExtension) -> dict:
    return {"name": _id(x.name), "kind": x.kind, "meaning": _text(x.meaning)}


def _character(c: XCharacter) -> dict:
    out = _opt({"id": _id(c.id), "name": _text(c.name)}, role=_text(c.role), location=_id(c.location))
    out["voice"] = _text(c.voice) or ""
    _opt(out, values=_texts(c.values), goal=_id(c.goal), likes=_texts(c.likes), dislikes=_texts(c.dislikes))
    rels = []
    for r in c.relationships:
        rel = {"who": _id(r.who), "trust": r.trust, "affection": r.affection, "fear": r.fear}
        _opt(rel, kind=_id(r.kind), respect=r.respect, familiarity=r.familiarity,
             extra={_id(e.feeling): e.level for e in r.extra} or None,
             obligations=[_opt({"type": o.type}, detail=_text(o.detail)) for o in r.obligations], opinion=_text(r.opinion))
        rels.append(rel)
    knows = []
    for k in c.knows:
        item = {"about": _id(k.about), "what": _id(k.what), "value": _text(k.value) or "", "how": k.how}
        if k.secret is not None:
            cover = _text(k.secret.cover_story)
            if cover and cover[-1] in ",;:":  # a quote cut mid-sentence by the document; the line is said as a sentence
                cover = cover[:-1] + "."
            item["secret"] = _opt({"may_tell": [_id(w) for w in k.secret.may_tell if _id(w)]}, cover_story=cover)
        knows.append(item)
    offers = [_opt({"what": _text(o.what) or "", "kind": o.kind, "available": o.available}, price=_text(o.price), note=_text(o.note))
              for o in c.offers]
    return _opt(out, relationships=rels, knows=knows, offers=offers, never_discuss=_texts(c.never_discuss), backstory=_texts(c.backstory))


def to_sheet(game_id: str, world: XWorld, characters: XCharacters | None = None, style: XStyle | None = None) -> dict:
    """The game sheet as a plain mapping, ready for GameSheet.model_validate."""
    setting = {"kind": world.setting.kind, "technology": _text(world.setting.technology) or "", "magic_policy": world.setting.magic_policy}
    if world.setting.boundary is not None:
        b = world.setting.boundary
        setting["boundary"] = _opt({"real_world": b.real_world}, allowed=_texts(b.allowed), denied=_texts(b.denied))
    extensions: dict[str, dict] = {}
    for x in list(world.extensions) + (list(characters.extensions) if characters else []):
        e = _ext(x)
        extensions.setdefault(e["name"], e)
    sheet = {"schema_version": SHEET_VERSION, "game_id": game_id, "setting": setting}
    _opt(sheet, rules=_texts(world.rules),
         things=[_opt({"id": _id(t.id), "kind": t.kind, "name": _text(t.name) or ""}, inside=_id(t.inside)) for t in world.things],
         extensions=list(extensions.values()))
    sheet["characters"] = [_character(c) for c in characters.characters] if characters else []
    if style:
        by_id = {c["id"]: c for c in sheet["characters"]}
        for s in style.speech_examples:
            cid = _id(s.character_id)
            if cid not in by_id:
                raise ValueError(f"speech example for unknown character {s.character_id!r}: {s.reply[:60]!r}")
            by_id[cid].setdefault("speech_examples", []).append({"player": _text(s.player) or "", "reply": _text(s.reply) or ""})
    _opt(sheet, lore=[_opt({"text": _text(item.text) or "", "importance": item.importance, "exact": item.exact}) for item in world.lore],
         story=[_opt({"stage": _text(st.stage) or ""}, summary=_text(st.summary)) for st in world.story])
    return sheet


def _strings(node: Any):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def leaks(sheet: dict, gated_terms: list[str]) -> list[str]:
    """Gated terms (from the story document) that appear anywhere in the sheet, whatever the case."""
    found = []
    for term in gated_terms:
        t = term.strip().lower()
        if len(t) < 3:
            continue
        hits = [s for s in _strings(sheet) if t in s.lower()]
        if hits:
            found.append(f"gated term {term!r} appears in the sheet: {hits[0][:100]!r}")
    return found


def validate(sheet: dict, gated_terms: list[str]) -> GameSheet:
    """The schema, then a world from it, then the gated-term scan. Raises ValueError with every problem found."""
    try:
        game = GameSheet.model_validate(sheet)
        world_from_sheets(game)
    except (ValidationError, ValueError) as exc:
        raise ValueError(str(exc)) from None
    found = leaks(sheet, gated_terms)
    if found:
        raise ValueError("; ".join(found))
    return game


# --- cost ----------------------------------------------------------------------------------------------------
def _tokens(messages: list[dict]) -> int:
    return math.ceil(sum(len(m["content"]) for m in messages) / strong.CHARS_PER_TOKEN) + strong.MESSAGE_OVERHEAD_TOKENS * len(messages)


def usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 1e6 * strong.USD_PER_M_INPUT + output_tokens / 1e6 * strong.USD_PER_M_OUTPUT


def forecast(docs: dict[str, str], game_id: str) -> dict:
    """Cost from the document sizes, before anything is sent. A repair pass resends the documents plus the previous
    output, so the ceiling counts every call twice. The context passed between calls is small and ignored here."""
    calls = {}
    for stage in STAGES:
        tin = _tokens(build_messages(stage, docs, game_id)) + math.ceil(len(json.dumps(SCHEMAS[stage])) / strong.CHARS_PER_TOKEN)
        tout = ASSUMED_OUTPUT_TOKENS[stage]
        calls[stage] = {"input_tokens": tin, "output_tokens_assumed": tout, "usd": round(usd(tin, tout), 3),
                        "usd_with_repair": round(usd(2 * tin + tout // 2, 2 * tout), 3)}
    first = sum(c["usd"] for c in calls.values())
    return {"calls": calls, "usd_first_pass": round(first, 3), "usd_with_repairs": round(sum(c["usd_with_repair"] for c in calls.values()), 3),
            "usd_ceiling": round(sum(usd(2 * c["input_tokens"] + c["output_tokens_assumed"] // 2, 2 * MAX_OUTPUT_TOKENS) for c in calls.values()), 2),
            "document_chars": {name: len(text) for name, text in docs.items()}}


# --- the run -------------------------------------------------------------------------------------------------
Answer = Callable[[list[dict]], list[dict]]


def _usage_row(stage: str, repair: bool, usage: dict, model: str | None) -> dict:
    tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return {"stage": stage, "repair": repair, "input_tokens": tin, "output_tokens": tout,
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
            "usd": round(usd(tin, tout), 4), "model": model}


class Extractor:
    """One game's extraction. `answer` takes the requests and returns their answers: the paid call, or a stand-in."""

    def __init__(self, lore_dir: pathlib.Path, game_id: str, answer: Answer, max_usd: float = DEFAULT_MAX_USD, effort: str = DEFAULT_EFFORT,
                 model: str = strong.MODEL):
        self.lore_dir = pathlib.Path(lore_dir)
        self.game_id = to_id(game_id)
        self.docs = read_documents(self.lore_dir)
        self.max_usd, self.effort, self.answer, self.model = max_usd, effort, answer, model
        self.calls: list[dict] = []
        self.raw: dict[str, Any] = {}
        self.forecast = forecast(self.docs, self.game_id)

    @property
    def spent(self) -> float:
        return sum(c["usd"] for c in self.calls)

    def _call(self, stage: str, messages: list[dict], repair: bool) -> X:
        cid = f"{self.game_id}.{stage}" + (".repair" if repair else "")
        line = request_line(cid, messages, stage, self.effort, self.model)
        this = usd(_tokens(messages), ASSUMED_OUTPUT_TOKENS[stage] * (2 if repair else 1))
        if self.spent + this > self.max_usd:
            raise BudgetError(f"stopping before {cid}: spent ${self.spent:.2f}, this call about ${this:.2f}, cap ${self.max_usd:.2f}")
        out = self.answer([line])
        if len(out) != 1 or out[0].get("custom_id") != cid:
            raise ExtractionError(f"{cid}: the answer function returned {len(out)} lines")
        p = strong.parse(out[0])
        self.calls.append(_usage_row(stage, repair, p.usage, p.model))
        if p.error:
            raise ExtractionError(f"{cid}: {p.error}")
        try:
            parsed = MODELS[stage].model_validate_json(p.text)
        except ValidationError as exc:
            raise ExtractionError(f"{cid}: the answer does not match the output schema: {str(exc)[:500]}") from None
        self.raw[cid] = parsed.model_dump()
        return parsed

    def _stage(self, stage: str, context: dict | None, check: Callable[[X], GameSheet]) -> X:
        """One call, validated; on an error one repair call with the error text; then fail."""
        parsed = self._call(stage, build_messages(stage, self.docs, self.game_id, context), repair=False)
        try:
            check(parsed)
            return parsed
        except ValueError as exc:
            errors = str(exc)[:4000]
        repaired = self._call(stage, repair_messages(stage, self.docs, self.game_id, context, parsed.model_dump(), errors), repair=True)
        try:
            check(repaired)
        except ValueError as exc:
            raise ExtractionError(f"{self.game_id}.{stage}: still invalid after the repair pass: {str(exc)[:2000]}") from None
        return repaired

    def run(self) -> tuple[GameSheet, dict]:
        if self.forecast["usd_with_repairs"] > self.max_usd:
            raise BudgetError(f"forecast ${self.forecast['usd_with_repairs']:.2f} with repairs (first pass ${self.forecast['usd_first_pass']:.2f}) "
                              f"is over the cap of ${self.max_usd:.2f}; nothing was sent")
        gated: list[str] = []

        def check_world(w: XWorld) -> GameSheet:
            gated[:] = [t for t in w.gated_terms if t.strip()]
            return validate(to_sheet(self.game_id, w), gated)

        world = self._stage("world", None, check_world)
        context = {"setting": world.setting.model_dump(), "things": [t.model_dump() for t in world.things],
                   "extensions": [x.model_dump() for x in world.extensions], "story_stages": [s.stage for s in world.story]}
        characters = self._stage("characters", context, lambda c: validate(to_sheet(self.game_id, world, c), gated))
        context = {"characters": [{"id": _id(c.id), "name": c.name} for c in characters.characters]}
        style = self._stage("style", context, lambda s: validate(to_sheet(self.game_id, world, characters, s), gated))
        sheet_dict = to_sheet(self.game_id, world, characters, style)
        sheet = validate(sheet_dict, gated)
        info = {
            "game_id": self.game_id, "model": next((c["model"] for c in self.calls if c["model"]), self.model), "effort": self.effort,
            "calls": self.calls, "cost_usd": round(self.spent, 4), "forecast": self.forecast, "max_usd": self.max_usd,
            "repairs": sum(1 for c in self.calls if c["repair"]), "counts": counts(sheet),
            "could_not_place": {"world": world.could_not_place, "characters": characters.could_not_place, "style": style.could_not_place},
            "gated_terms_held": len(gated), "sheet": sheet_dict, "to_check": items_to_check(sheet) + rules_to_check(sheet),
        }
        return sheet, info


def items_to_check(sheet: GameSheet) -> list[str]:
    """Cheap flags for the person checking the sheet, computed from the sheet alone: a knowledge item with no secret
    block beside a secret of the same character about the same subject (a secret split across items that lost its
    block on one of them), and a character with no speech example."""
    flags = []
    for c in sheet.characters:
        secret_subjects = {k.about for k in c.knows if k.secret is not None}
        for k in c.knows:
            if k.secret is None and k.about in secret_subjects:
                flags.append(f"{c.id}: `{k.what}` has no secret block but sits beside a secret about `{k.about}`; is it part of that secret?")
        if not c.speech_examples:
            flags.append(f"{c.id}: no speech example in style.md")
    return flags


RULES_MANY = 25  # the example games have 8 and 18
LEANS_ON_THE_LINE_BEFORE = ("It", "They", "He", "She")


def rules_to_check(sheet: GameSheet) -> list[str]:
    """Flags on the world rules. Every rule goes into every prompt, and a lore item that is also a rule gets no training
    scene of its own, so the list should be short and each line should stand alone. One run of the reader on the
    example game returned 18 rules and another 46, the second with a whole section's facts cut into sentences."""
    flags = []
    if len(sheet.rules) > RULES_MANY:
        flags.append(f"rules: {len(sheet.rules)} world rules. Every rule goes into every prompt, and a lore item that is also a rule is not taught as "
                     "lore. Keep what is and is not possible in this world, and leave plain facts to `lore`.")
    for r in sheet.rules:
        if r.split()[0] in LEANS_ON_THE_LINE_BEFORE:
            flags.append(f"rules: `{r}` leans on the line before it. A rule is read on its own: join the two, or drop it.")
    return flags


def counts(sheet: GameSheet) -> dict:
    lore = sheet.lore_items()
    return {
        "things": len(sheet.things), "characters": len(sheet.characters), "rules": len(sheet.rules), "extensions": len(sheet.extensions),
        "facts": sum(len(c.knows) for c in sheet.characters),
        "secrets": sum(1 for c in sheet.characters for k in c.knows if k.secret is not None),
        "relationships": sum(len(c.relationships) for c in sheet.characters), "offers": sum(len(c.offers) for c in sheet.characters),
        "never_discuss": sum(len(c.never_discuss) for c in sheet.characters), "backstory": sum(len(c.backstory) for c in sheet.characters),
        "lore": len(lore), "lore_by_importance": {level: sum(1 for x in lore if x.importance == level) for level in ("high", "normal", "low")},
        "exact": sum(1 for x in lore if x.exact), "story_stages": len(sheet.story),
        "speech_examples": sum(len(c.speech_examples) for c in sheet.characters),
    }


def paid_answer(api_key: str) -> Answer:
    """The paid call. One request at a time, a long timeout and a few retries: a timed-out call is billed again, so
    the wait is long rather than the retries many."""
    return lambda lines: strong.answer(lines, api_key, concurrency=1, timeout=900.0, retries=5)


def extract(lore_dir: pathlib.Path | str, game_id: str, api_key: str, max_usd: float = DEFAULT_MAX_USD, effort: str = DEFAULT_EFFORT,
            model: str = strong.MODEL) -> tuple[GameSheet, dict]:
    """Read one game's lore into a sheet. Paid: three calls, or up to six with repairs."""
    return Extractor(pathlib.Path(lore_dir), game_id, paid_answer(api_key), max_usd=max_usd, effort=effort, model=model).run()


# --- files ---------------------------------------------------------------------------------------------------
def sheet_yaml(sheet: GameSheet, info: dict) -> str:
    head = (f"# {info['game_id']}: game sheet read from the prose in lore/ by {info['model']}, {dt.date.today()}.\n"
            f"# Check it by hand before you train. The reader never invents, but it can misplace a fact. The report beside it lists what to look at first.\n"
            f"# Defaults are left out (neutral levels, normal importance, exact false, available true).\n")
    return head + yaml.safe_dump(sheet.model_dump(exclude_none=True, exclude_defaults=True), sort_keys=False, allow_unicode=True, width=120)


def report_md(info: dict) -> str:
    c = info["counts"]
    f = info["forecast"]
    lines = [f"# {info['game_id']}: what the lore reader found", "",
             f"Read {dt.date.today()} by {info['model']} (reasoning effort {info['effort']}) from {', '.join(DOCS)}. "
             f"The sheet validates and builds a world. It is for you to check before training turns it into examples.", "",
             "## Counts", "", "| What | Count |", "|---|---|",
             f"| Things (places, structures, objects, factions, creatures) | {c['things']} |",
             f"| Characters | {c['characters']} |",
             f"| Facts a character knows from the start | {c['facts']} |",
             f"| Of which secrets (with who may hear them and a cover story) | {c['secrets']} |",
             f"| Relationships | {c['relationships']} |",
             f"| Offers | {c['offers']} |",
             f"| Never-discuss topics | {c['never_discuss']} |",
             f"| Backstory lines | {c['backstory']} |",
             f"| World rules | {c['rules']} |",
             f"| Vocabulary extensions | {c['extensions']} |",
             f"| Lore items | {c['lore']} (high {c['lore_by_importance']['high']}, normal {c['lore_by_importance']['normal']}, low {c['lore_by_importance']['low']}) |",
             f"| Lore items to recall word for word (exact) | {c['exact']} |",
             f"| Story stages | {c['story_stages']} |",
             f"| Speech examples | {c['speech_examples']} |", "",
             "## Items to check first", "",
             "Flags computed from the sheet alone. The whole sheet still needs reading: the reader never invents, but it can "
             "place a fact on the wrong side of a line the documents draw (a secret that begins later in the story, a clue filed as "
             "plain knowledge, two wordings of one fact)."]
    lines += [f"- {flag}" for flag in info.get("to_check", [])] or ["- nothing flagged"]
    lines += ["", "## Held back", "",
              "The story's gated secret is kept out of every field: the characters call never sees story.md, and the world call is told to "
              "leave the secret out. " + (f"The {info['gated_terms_held']} terms the story marks as unsayable before the gate were scanned for "
                                          "across the whole sheet: none found." if info["gated_terms_held"] else
                                          "The story names no single terms to scan for, so the scan had nothing to check. Read the sheet for the secret yourself.")
              + " Knowledge that begins at a later stage is live state and is listed below, not in the sheet. Your game supplies gated facts at run time.", "",
              "## What the documents state that the sheet could not hold", ""]
    for stage, items in info["could_not_place"].items():
        lines.append(f"From {' and '.join(DOCS_BY_STAGE[stage])}:")
        lines += [f"- {item}" for item in items] or ["- nothing"]
        lines.append("")
    lines += ["## Cost", "", "| Call | Input tokens | Output tokens | Of which reasoning | USD |", "|---|---|---|---|---|"]
    for call in info["calls"]:
        name = call["stage"] + (" (repair)" if call["repair"] else "")
        lines.append(f"| {name} | {call['input_tokens']:,} | {call['output_tokens']:,} | {call['reasoning_tokens']:,} | ${call['usd']:.3f} |")
    lines += ["", f"Actual: ${info['cost_usd']:.3f} (${strong.USD_PER_M_INPUT:.0f} per million input tokens, ${strong.USD_PER_M_OUTPUT:.0f} per million output). "
              f"Forecast from the document sizes: ${f['usd_first_pass']:.2f} for one pass, ${f['usd_with_repairs']:.2f} with a repair pass per call "
              f"(output tokens assumed, not measured). Cap: ${info['max_usd']:.2f}. Repair passes used: {info['repairs']}. One run.", ""]
    return "\n".join(lines)


def raw_json(info: dict, answers: dict) -> str:
    """What the run cost and what the model answered, for the record."""
    keep = ("model", "effort", "calls", "cost_usd", "max_usd", "repairs", "gated_terms_held", "could_not_place", "forecast")
    return json.dumps({**{k: info[k] for k in keep}, "answers": answers}, indent=1, ensure_ascii=False) + "\n"
