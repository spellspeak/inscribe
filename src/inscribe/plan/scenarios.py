"""Scenes for a game LoRA: a checked game sheet becomes eval-case-0.3 records in the generators' style.

    cases = game_scenarios(sheet)        # list of EvalCase dicts on the worlds of `inscribe.plan.world`

Six kinds, each marked by its generator id (`kind_of(case)` names them):

| Kind | Generator | What it asks | Answer key |
|---|---|---|---|
| lore | LW-01 | a lore item of the sheet, hidden from the student's view (`hidden_fact_ids`) and given to the teacher (`teacher_only`), in several phrasings to the characters who would know it; importance high: 3 knowers and 3 phrasings, normal 2 and 2, low 1 and 1 | key words of the item (`must_contain_any`); exact items the exact wording |
| style | LORE | the player line of a speech example; the good reply IS the example, so these never go to the teacher (`is_style`) | the reply itself |
| boundary | RW-01 | a denied real-world thing (in-character ignorance) and an allowed one (used naturally), on three characters | no explanation of the thing, no lecture about the world (`must_not_contain`) |
| story | SP-01 | at every stage, what is happening now and what comes next, with the later stages as `forbidden_claim_ids` and their words as `must_not_contain` (the spoiler) | stays at the current stage |
| secret | EP-07, GS-02 | the player asks about a secret (deflect or cover story); a confidant asks while the player listens (GS-02's shape, on the group world) | the secret's key words never said (`must_not_contain`), `forbidden_claim_ids` |
| ordinary | ID-01, RB-01, BR-01, ST-01, ST-07, SO-01, SO-02, SO-03, SO-07, RB-05 | plain character behaviour on this world, reusing the thin-slice and catalogue generators | theirs |

Every case has expected properties in the existing style, `must_contain_any` or `must_not_contain` where code can
check, and reference replies where cheap. Noisy twins (RB-02) on about a third of the cases where the player speaks
last. Everything is deterministic in (sheet, seed). Lore questions never contain their answer: a key word that would
sit in the question is dropped, and a character whose own prompt shows a key is never asked it (as LW-01's
`gives_away`), so the only way to answer is from the weights.

A lore question is shaped from the item's own sentence (`lore_asks`): a price line asks the price, "X is Y" asks what X
is, "There is no X" asks whether there is one, "X: Y" asks for the rest, an exact item is asked through the sentence
that quotes it, and anything else asks about its subject (a proper name or the leading noun phrase). An item whose
subject is a pronoun or a fragment of the item before it gets no question and is counted as skipped.

Why not `scenarios_for_world` whole: its BR-01 parses prices as "3 coins" and raises on "one pip"; WR-04 takes a
genre's rules as the only rules there are, which is false on a world with its own (Brackenhythe has no mind
reading; a negative control would call saying so an invented rule). The generators that hold are called one by one.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, replace

from inscribe.runtime.output import speakability_issues
from inscribe.runtime.template import render_system
from inscribe.schemas.game_sheet import CharacterSheet, GameSheet, LoreItem
from inscribe.schemas.records import EvalCase
from inscribe.schemas.world import World
from inscribe.checks import _contains as says
from inscribe.context.compiler import project_actor_view
from inscribe.scenarios.catalogue import GREETINGS, catalogue_scenarios
from inscribe.scenarios.group import ALLOWED, DENIED, IGNORANCE, LECTURE, ORDINARY, _bad, _cap, add_twins
from inscribe.scenarios.knowledge import crosses_firewall, forms
from inscribe.scenarios.core import _case, _id, ep06_hearsay, id01_identity, number_forms, rb01_twins, so05_correction, st04_change
from inscribe.plan.world import GENERATOR_VERSION, game_worlds, lore_fact_id, stage_fact_id, world_id_for

SCHEMA = "eval-case-0.3"
STYLE_GENERATOR = "LORE"  # `inscribe.data.examples.CAPABILITY_IDS`: identity.embedded_game_lore, the id for LoRA-only game data
KIND_OF_GENERATOR = {"LW-01": "lore", STYLE_GENERATOR: "style", "RW-01": "boundary", "SP-01": "story", "EP-07": "secret", "GS-02": "secret"}
PER_IMPORTANCE = {"high": (3, 3), "normal": (2, 2), "low": (1, 1)}  # (knowers, phrasings)
GENRE_FOR_KIND = {"closed_fictional": "fantasy_town", "historical": "frontier_town", "contemporary": "modern_town", "custom": "modern_town"}
DUNNO = "Couldn't tell you. No idea, sorry."
BOUNDARY_CHARACTERS = 3
DENIED_PER_CHARACTER = 8
ALLOWED_PER_CHARACTER = 6
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")

# Words that never carry a lore answer on their own: function words, the vocabulary of a design document, and words
# so common in these sheets that a reply would use them for any reason. Never a key.
STOP = frozenset("""
a an the and or but nor so yet for of to in on at by with from into onto over under above below between among through
during before after since until till about against along around across behind beyond beside near off out up down past
than as if then else when where while whether because although though unless once also just only even still again
i me my mine we us our ours you your yours he him his she her hers it its they them their theirs who whom whose which
what that this these those there here why how all any both each few more most other some such no not own same too very
am is are was were be been being have has had having do does did doing done will would shall should can could may might
must ought need dare used
say says said saying tell tells told telling ask asks asked know knows knew known think thinks thought see sees saw seen
look looks go goes went gone come comes came get gets got make makes made take takes took give gives gave keep keeps kept
let put run runs ran want wants mean means meant find found call calls called read write hear show turn leave left stay
work play live died pay pays paid buy buys sell sells cut cuts open opens close closed stop start hold holds held carry
bring walk talk speak sit stand wait feel try answer refuse mind add admit allow arrive avoid feed learn lose loses move
need offer remember send set sleep use watch win hide break clean crack heal counts count counted quote quotes
nothing nobody none nowhere anyone anybody anything anywhere everyone everybody everything everywhere someone somebody
something somewhere people person man men woman women boy girl child children mother father son daughter brother
sister wife husband friend friends family
one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen
nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million first second third fourth fifth
sixth seventh eighth ninth tenth last next half quarter
good bad big small large little old new young long short high low early late hot cold warm cool dry wet great real
true false right wrong full empty whole different every much many less least best worse worst better well far near fast
slow hard soft quick plain simple free dead alive safe sure certain ready clear quiet loud dark light bright heavy
fresh honest happy sad angry tired strong weak rich poor cheap dear deep own other another following
day days night nights morning evening afternoon noon midnight dawn dusk week weeks month months year years hour hours
minute minutes time times today tomorrow yesterday tonight season seasons
way ways thing things kind kinds sort sorts part parts place places lot lots bit bits point points end ends side sides
house houses home room rooms door doors window windows wall walls floor roof step steps road street town city village
world story game player players character characters lore rule rules setting rather almost always never often
sometimes usually mostly ever fine now ok okay perhaps maybe
""".split())
# Function words only: a noun phrase needs one word that is not one of these to be a topic.
FUNCTION = frozenset("""
a an the and or but nor so yet for of to in on at by with from into onto over under above below between among through
during before after since until till about against along around across behind beyond beside near off out up down past
than as if then else when where while whether because although though unless once also just only even still again
i me my mine we us our ours you your yours he him his she her hers it its they them their theirs who whom whose which
what that this these those there here why how all any both each few more most other some such no not own same too very
nothing nobody none nowhere anyone anybody anything anywhere everyone everybody everything everywhere someone somebody
something somewhere people person children following
""".split())
# Heads a question cannot name: pronouns and fragments that point at the item before.
PRONOUNS = frozenset("it he she they we i you his her their its our my your him them us me".split())
PRONOUN_HEADS = PRONOUNS | frozenset("then people children everyone everybody somebody someone nobody characters player we".split())
NO_HEAD = PRONOUN_HEADS | frozenset("there this that these those what who which nothing everything no none one some upstairs downstairs".split())
ARTICLES = frozenset("the a an this that these those".split())
TITLES = frozenset("dr mr mrs ms miss sir sergeant captain doctor lord lady".split())
QUANTIFIERS = frozenset("every each all most some many any".split())
TAIL_WORDS = frozenset("now then still always often already again that which who".split())
# Generic nouns that cannot be a topic on their own ("the thing", "the town") even though they are not function words.
HEAD_STOP = FUNCTION | frozenset("""
thing things way ways kind kinds sort sorts part parts bit bits lot lots end ends side sides point points place places
town city village world story game player players character characters lore rule rules setting number
""".split())
# Verb-like words that end a leading noun phrase, and never a key.
VERBS = frozenset("""
is are was were be been has have had do does did can could will would may might must shall should cannot
glows glow warms warm greys grey fades fade rings ring thuds thud comes come goes go keeps keep holds hold meets meet
runs run means mean opens open closes close sits sit stands stand works work makes make takes take gives
give gets get needs need costs cost spreads spread teaches teach counts count calls call says say tells tell uses use
pays pay earns earn hangs hang flashes flash moved moves move went began begins begin sells sell buys buy wants want
knows know thinks think puts put lights light starts start started applies apply dates date fixes fix feels feel
describes describe treats treat loses lose fetches fetch demands demand becomes become owns own charges charge drowned
drown floods flood kills kill killed signed sign folded fold flooded hired hire lent lend owes owe founded found
wrecked wreck stopped stop opened closed covered cover wears wear carry carries sees see sat sleeps sleep happened
happens happen smashed broke breaks walked walks walk ran turned turns turn passed passes pass kept took gave made
told asked said came found left stood held paid bought sold let put sent hid hidden won lost built met became
grew grows grow fell falls fall rose rises rise read reads believes believe claims claim picks pick
describe describes applies apply learned learns learn reaches reach wets wet eats eat swims swim
""".split())
NUMBER_WORDS = {w: n for n in range(100) for w in number_forms(n)[1:]}
NUMBER_WORDS.update({"hundred": 100, "thousand": 1000})
_WORD = re.compile(r"[A-Z]\.\d+|[A-Za-z][A-Za-z']*(?:-[A-Za-z]+)*|\d+(?:[.,]\d+)*")
_WORD_OR_PUNCT = re.compile(r"[A-Z]\.\d+|[A-Za-z][A-Za-z']*(?:-[A-Za-z]+)*|\d+(?:[.,]\d+)*|[.,;:!?()\"“”]")
_CAP = re.compile(r"^[A-Z]")
_YEAR_TOKEN = re.compile(r"^[A-Z]\.\d+$")
PUNCT = frozenset(".,;:!?()\"“”")

# --- phrasings. Training pools and held-out pools never share a template, so recall prompts never reuse a question.
PRICE_ASKS = ("What does {x} cost here?", "How much is {x}?", "What would {x} set me back?", "What's the going price for {x}?")
PRICE_HELD = ("If I wanted {x}, what would I pay?", "Any idea what {x} goes for?")
NEG_ASKS = ("Is there any {x} here?", "Where would I find {x} round here?", "Who do I see about {x}?")
NEG_HELD = ("Does this place have any {x}?", "Any {x} to be had here?")
CLAUSE_ASKS = ("{X}, I'm told. Tell me more.", "{X}, they say. What exactly?", "I heard {x}. Go on?")
CLAUSE_HELD = ("{X}, someone said. Can you finish that for me?", "Is it right that {x}? What's the rest of it?")
COPULA_ASKS = ("What {v} {x}?", "What's {x}, exactly?", "Tell me about {x}.", "What do you know about {x}?")
COPULA_HELD = ("What can you tell me about {x}?", "{X}. What {v} that?")
TOPIC_ASKS = ("What do you know about {x}?", "Tell me about {x}.", "What's the story with {x}?", "Anything I should know about {x}?")
TOPIC_HELD = ("What can you tell me about {x}?", "Remind me about {x}.")
EXACT_ASKS = {"number": ("{C}. What's the exact figure?", "{C}. Give me the number exactly as it's said.", "How many, exactly? {C}."),
              "when": ("{C}. When was that, exactly?", "{C}. The exact date, as everyone says it.", "When exactly? {C}."),
              "words": ("{C}. What are the exact words?", "{C}. How does it go, word for word?", "Word for word, how does it go? {C}."),
              "name": ("{C}. What's it called, exactly?", "{C}. What's the exact name?", "The exact name, please. {C}.")}
EXACT_HELD = {"number": ("{C}. Say the number the way everyone here says it.", "{C}. The figure, exactly, if you'd be so kind."),
              "when": ("{C}. Say the date the way everyone here says it.", "{C}. The day, exactly, if you'd be so kind."),
              "words": ("{C}. Say it for me exactly as it's said.", "{C}. The exact line, please."),
              "name": ("{C}. The exact name, please.", "{C}. Remind me what it's called, exactly.")}
EXACT_TOPIC_ASKS = ("Give me the exact wording about {x}.", "What's the exact line about {x}?", "How is it said exactly, about {x}?")
EXACT_TOPIC_HELD = ("Word for word, what do people say about {x}?", "The exact words about {x}, please.")
EXACT_GENERIC_ASKS = {"words": ("There's a line people here say exactly. How does it go?", "Say the exact words everyone here knows.",
                                "What's the one line this place is known for, word for word?"),
                      "name": ("There's a name everyone here says exactly. What is it?", "The name everyone here knows. Say it exactly.",
                               "What's the name this place is known for, exactly?"),
                      "number": ("What's the number everyone here quotes exactly?", "There's a figure everyone here knows. Say it exactly.",
                                 "The number this place is known for. What is it, exactly?"),
                      "when": ("What's the date everyone here knows exactly?", "There's a day everyone here can name. Say it exactly.",
                               "The date this place is known for. When, exactly?")}
EXACT_GENERIC_HELD = {"words": ("Word for word, what's the line everyone here can recite?", "The exact words, the ones everyone knows. Go on."),
                      "name": ("Word for word, what's the name everyone here knows?", "The exact name, the one everyone knows. Go on."),
                      "number": ("Word for word, what's the figure everyone here knows?", "The exact number, the one everyone quotes. Go on."),
                      "when": ("Word for word, what's the date everyone here knows?", "The exact day, the one everyone names. Go on.")}
WHO_ASKS = ("Who is {x}?", "Who's {x}, exactly?", "Tell me about {x}.", "What do you know about {x}?")
WHO_HELD = ("What can you tell me about {x}?", "Remind me who {x} is.")
FINE_ASKS = ("Are {x} all right here?", "Can I get away with {x} here?", "Is there anything wrong with {x} here?")
FINE_HELD = ("Do {x} work here, or do they get stuck?", "What about {x}? Allowed?")
CAN_ASKS = ("Can anyone {who} {what}?", "Is there anyone {who} who can {what}?", "Could I {what}, {who}?")
CAN_HELD = ("Has anyone {who} ever managed to {what}?", "Suppose I tried to {what} {who}. What happens?")
DENIED_ASKS = ("Is there any {x} round here?", "What do you know about {x}?")
ALLOWED_ASKS = ("Do you know where I could get some {x}?", "Is there {x} to be had here?")
STATE_ASKS = ("How are things in town just now?", "What's the mood here these days?", "What's happening around here at the moment?", "What's the latest?")
NEXT_ASKS = ("What do you reckon happens next?", "Where's all this heading, do you think?", "How does this end, in your view?")
NEXT_HELD = ("What comes after this, then?", "Tell me how it all turns out.", "You must have an idea where this is going. Go on.")
SECRET_ASKS = ("I've heard talk about {s} {t}. What's the truth of it?", "Straight answer: what do you know about {s} {t}?")
CONFIDANT_ASK = "Any change about the {t}, or is it as it was? You can tell me."
DEFLECT = "That's not a thing I talk about. Ask me something else."


def kind_of(case: dict) -> str:
    return KIND_OF_GENERATOR.get(case["generator_like"], "ordinary")


def is_style(case: dict) -> bool:
    """A style case carries its own good reply and never goes to the teacher."""
    return case["generator_like"] == STYLE_GENERATOR and case.get("family") == "style"


def style_reply(case: dict) -> str:
    return case["reference_replies"]["good"][0]


# --- words ------------------------------------------------------------------------------------------------------
def tokens(text: str) -> list[str]:
    return _WORD.findall(text.replace("’", "'"))


def words(text: str) -> set[str]:
    """Lowercase words of a text; a hyphenated word counts whole and in its parts (as the whole-word check matches)."""
    out = set()
    for t in tokens(text):
        low = t.lower().strip("'")
        out.add(low)
        if "-" in low:
            out.update(p for p in low.split("-") if p)
    return out


def _number_keys(tok: str) -> list[str]:
    """Ways a reply may say a number token: "40" -> 40, forty; "6.50" -> 6.50, six fifty; "1,200" -> 1,200, 1200, twelve hundred."""
    low = tok.lower()
    if low in NUMBER_WORDS:
        n = NUMBER_WORDS[low]
        return number_forms(n) if n < 100 else [low]
    if "-" in low and all(p in NUMBER_WORDS for p in low.split("-")):  # forty-one
        tens, ones = (NUMBER_WORDS[p] for p in low.split("-"))
        return number_forms(tens + ones) if tens + ones < 100 else [low]
    if re.fullmatch(r"\d+", tok):
        n = int(tok)
        return number_forms(n) if n < 100 else [tok]
    if re.fullmatch(r"\d+\.\d+", tok):
        whole, frac = tok.split(".")
        out = [tok]
        if int(whole) < 100 and int(frac) < 100:
            out.append(f"{number_forms(int(whole))[-1]} {number_forms(int(frac))[-1]}")
        return out
    if re.fullmatch(r"\d{1,3}(,\d{3})+", tok):
        plain = tok.replace(",", "")
        out = [tok, plain]
        if plain.endswith("00") and int(plain) < 10000:
            out.append(f"{number_forms(int(plain) // 100)[-1]} hundred")
        return out
    return []


def _is_number(tok: str) -> bool:
    return bool(_number_keys(tok)) or tok.lower() == "point"


def _is_verb(low: str) -> bool:
    if low in VERBS:
        return True
    for end in ("s", "es", "ed", "d", "ing"):
        if low.endswith(end) and (low[: -len(end)] in VERBS or low[: -len(end)] + "e" in VERBS):
            return True
    return False


def _plural(noun: str) -> str:
    return noun if noun.endswith("s") else noun + "s"


# --- lore -------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LoreAsk:
    """One lore item with everything its cases and its recall prompts need."""

    index: int
    item: LoreItem
    pattern: str  # price | exact | negation | nobody_can | fine | colon | copula | who | topic | none
    topic: str  # what the questions name (never the answer)
    keys: tuple[str, ...]  # base key words; `must_contain_any` is these with their endings
    asks: tuple[str, ...]  # training phrasings
    held_out: tuple[str, ...]  # recall phrasings, never used in training
    stated: str  # expected property
    good: str | None  # a speakable reply that states the item, when the item itself is one
    knowers: tuple[str, ...]  # characters who may be asked it, best first

    @property
    def fact_id(self) -> str:
        return lore_fact_id(self.index)

    @property
    def checks(self) -> list[str]:
        """`must_contain_any`: the exact wording for exact items; otherwise the keys with their common endings."""
        if self.item.exact:
            return list(self.keys)
        return forms(list(self.keys))


class _Names:
    """What the sheet treats as proper nouns: thing and character names, months, and words that are capitalised
    mid-sentence somewhere in the sheet and never written in lowercase (Idonea, Hythe; not spar, which is both)."""

    def __init__(self, sheet: GameSheet):
        self.names = ({t.lower() for x in sheet.things for t in tokens(x.name) if t[:1].isupper()}
                      | {t.lower() for c in sheet.characters for t in tokens(c.name) if t[:1].isupper()}) - STOP
        self.character_words = [words(c.name) - STOP - TITLES for c in sheet.characters]
        mid, lower = set(), set()
        for text in [i.text for i in sheet.lore_items()] + list(sheet.rules) + [character_text(c) for c in sheet.characters]:
            toks = _WORD_OR_PUNCT.findall(text.replace("’", "'"))
            for k, t in enumerate(toks):
                if t in PUNCT:
                    continue
                if _CAP.match(t) and k and toks[k - 1] not in PUNCT and t.lower() not in STOP:
                    mid.add(t.lower().removesuffix("'s"))
                elif t[:1].islower():
                    lower.add(t.lower().removesuffix("'s"))
        self.proper = self.names | set(MONTHS) | (mid - lower)

    def is_proper(self, tok: str) -> bool:
        return tok.lower().strip("'").removesuffix("'s") in self.proper or bool(_YEAR_TOKEN.match(tok))

    def lower_first(self, text: str) -> str:
        """The first letter lowered unless the text starts with a name, or with a capitalised run (Dim Day)."""
        toks = tokens(text)[:2]
        if toks and toks[0].lower() not in ARTICLES and (self.is_proper(toks[0]) or (len(toks) == 2 and _CAP.match(toks[0]) and _CAP.match(toks[1]))):
            return text
        return text[:1].lower() + text[1:]

    def fill(self, templates: tuple[str, ...], x: str, verb: str = "is") -> list[str]:
        low = self.lower_first(x)
        return [re.sub(r"\s+", " ", t.format(x=low, X=_cap(low), C=_cap(low), v=verb)).replace(" ,", ",").replace(" ?", "?") for t in templates]


def _price_match(text: str) -> tuple[str, str] | None:
    m = re.match(r"^(?P<item>[^:]{3,70}):\s*(?P<price>(?:[^,]|,(?=\d{3}))+?)(?:,\s*(?P<rest>.*))?\.?$", text)
    if not m or len(tokens(m["item"])) > 8:
        return None
    price = m["price"].strip().rstrip(".")
    if not any(_number_keys(t) for t in tokens(price)) and not re.search(r"[£$€]", price):
        return None
    return m["item"].strip(), price


def _price_keys(price: str) -> list[str]:
    keys: list[str] = []
    toks = tokens(price.replace("£", " ").replace("$", " ").replace("€", " "))
    for t in toks:
        for k in _number_keys(t):
            if k not in keys:
                keys.append(k)
    if toks and toks[0].lower() in ("a", "an") and len(toks) > 1 and toks[1].isalpha():
        keys = [f"a {toks[1]}", f"one {toks[1]}", f"1 {toks[1]}"] + keys
    return keys


def _strip_phrase(phrase: str) -> str:
    """An exact item as a key: quotes and the final full stop off, inner punctuation kept."""
    return phrase.strip().strip("\"“”'").rstrip(".").strip()


def _exact_type(phrase: str) -> str:
    toks = tokens(phrase)
    if any(t.lower() in MONTHS for t in toks) or any(t.lower() in ("night", "morning", "twelfth", "day") for t in toks):
        return "when"
    if any(_is_number(t) for t in toks):
        return "number"
    inner = re.sub(r"[A-Z]\.\d+", "", phrase).rstrip(".!?")
    return "words" if re.search(r"[.,!?]", inner) or len(toks) > 6 else "name"


def _punct_tokens(text: str) -> list[str]:
    return _WORD_OR_PUNCT.findall(text.replace("’", "'"))


def _proper_phrase(text: str, names: _Names) -> str | None:
    """The longest run of capitalised words (joined by of/the/and), with a leading "the" when the text has one and up
    to two plain nouns that follow it (the Shalecombe parish line). A sentence-initial word starts a run only when it
    is a known name or the next word is capitalised; a mid-sentence capitalised stop word only when the next is too."""
    toks = _punct_tokens(text)
    best: tuple[int, int] | None = None
    i = 0
    while i < len(toks):
        t = toks[i]
        cap = bool(_CAP.match(t)) and t not in PUNCT and not _is_number(t) and not _YEAR_TOKEN.match(t)
        next_cap = i + 1 < len(toks) and bool(_CAP.match(toks[i + 1])) and toks[i + 1] not in PUNCT and not _is_number(toks[i + 1])
        after_punct = i == 0 or toks[i - 1] in PUNCT
        starts = cap and (names.is_proper(t) or (not after_punct and t.lower() not in STOP) or (not after_punct and next_cap) or (after_punct and next_cap and t.lower() not in STOP))
        if starts:
            j = i + 1
            while j < len(toks) and ((_CAP.match(toks[j]) and toks[j] not in PUNCT and not _is_number(toks[j]))
                                     or toks[j] in ("of", "the", "and") or _YEAR_TOKEN.match(toks[j])):
                j += 1
            while j - 1 > i and toks[j - 1] in ("of", "the", "and"):
                j -= 1
            if best is None or j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    if best is None:
        return None
    i, j = best
    run = toks[i:j]
    if i > 0 and toks[i - 1].lower() == "the" and run[0].lower() != "the":
        extra: list[str] = []
        for u in toks[j: j + 2]:
            if u in PUNCT or not u.isalpha() or not u[:1].islower() or u.lower() in STOP or _is_verb(u.lower()):
                break
            extra.append(u)
        after = toks[j + len(extra)] if j + len(extra) < len(toks) else None  # the plain nouns belong to the name only before a verb
        if extra and not (after is not None and after not in PUNCT and _is_verb(after.lower())):
            extra = []
        return " ".join(["the", *run, *extra])
    return " ".join(run)


def _head(text: str) -> tuple[list[str], str | None, list[str]]:
    """(leading noun phrase tokens, the verb that ended it or None, the tokens after the verb up to punctuation)."""
    toks = _punct_tokens(text)
    for k, t in enumerate(toks):
        if k and (t in PUNCT or _is_verb(t.lower())):
            head = toks[:k]
            verb = None if t in PUNCT else t
            tail: list[str] = []
            for u in toks[k + 1:] if verb else []:
                if u in PUNCT:
                    break
                tail.append(u)
            return head, verb, tail
    return toks, None, []


def _content(toks: list[str]) -> bool:
    return any(t.lower() not in HEAD_STOP and t.isalpha() and len(t) >= 3 for t in toks)


def _noun_phrase(toks: list[str]) -> str | None:
    """A usable noun phrase from head tokens: at most six words with a content word; no pronoun head or pronoun inside;
    a trailing adverb or relative pronoun dropped; a quantified head ("every window") made plural ("the windows")."""
    toks = list(toks)
    while toks and toks[-1].lower() in TAIL_WORDS:
        toks.pop()
    if not toks or len(toks) > 6 or toks[0].lower() in NO_HEAD or any(t.lower() in PRONOUNS for t in toks[1:]) or not _content(toks):
        return None
    if toks[0].lower() in QUANTIFIERS and len(toks) >= 2:
        return " ".join(["the", _plural(toks[1]), *toks[2:]])
    if toks[0].lower() == "half" and len(toks) >= 2 and toks[1].lower() == "the":
        return " ".join(toks[1:])
    return " ".join(toks)


def _tail_phrase(tail: list[str]) -> str | None:
    """The noun phrase after the verb, cut at the first preposition or conjunction after a content word."""
    out: list[str] = []
    for t in tail:
        if out and _content(out) and t.lower() in ("in", "on", "at", "for", "of", "to", "from", "with", "and", "by", "than", "as", "that", "which", "who"):
            break
        out.append(t)
    return _noun_phrase(out)


def _topic(text: str, names: _Names) -> tuple[str, str]:
    """(what a question can name without giving the answer, the question mode). Modes: copula ("X is Y" with an
    article or a name as X), who (X is a character), topic (anything else), none (nothing usable: a pronoun or a
    fragment of the item before it, or a topic that would say the whole item)."""
    head, verb, tail = _head(text)
    if head and head[0].lower() in PRONOUN_HEADS:
        return "", "none"
    phrase = _noun_phrase(head)
    mode = "topic"
    if phrase and _is_number(head[0]):
        proper = _proper_phrase(" ".join(tail), names) if tail else None
        phrase = proper or _tail_phrase(tail)
    elif phrase:
        copula = verb is not None and verb.lower() in ("is", "are")
        if tail and tail[0].lower() in ("there", "here", "it"):
            return "", "none"
        if copula and all(_CAP.match(t) for t in head) and any(wd & words(phrase) for wd in names.character_words):
            mode = "who"
        elif copula and (head[0].lower() in ARTICLES or names.is_proper(head[0])) and head[0].lower() not in QUANTIFIERS:
            mode = "copula"
    else:
        phrase = _proper_phrase(text, names)
    if not phrase:
        return "", "none"
    content = {w for w in words(text) if w not in HEAD_STOP and not _is_verb(w) and w.isalpha()}
    if content and content <= words(phrase):  # the topic would say the whole item
        return "", "none"
    return phrase, mode


def _rank_keys(text: str, exclude: set[str], df: dict[str, int], names: _Names, limit: int = 4) -> list[str]:
    """Key words of a lore sentence a reply that states it would use: numbers first, then proper nouns, then the rarest
    content words; never a word of the question (`exclude`), a verb, or a stop word."""
    toks = tokens(text)
    numbers: list[str] = []
    proper: list[tuple[int, str]] = []
    plain: list[tuple[int, str]] = []
    seen = set()
    for i, t in enumerate(toks):
        low = t.lower().strip("'")
        if low in exclude or low in seen:
            continue
        if _YEAR_TOKEN.match(t):
            numbers.append(t)
            seen.add(low)
        elif _number_keys(t):
            numbers += [k for k in _number_keys(t) if k.lower() not in exclude]
            seen.add(low)
        elif low in STOP or _is_verb(low):
            continue
        elif _CAP.match(t) and (i > 0 or names.is_proper(t)) and len(low) > 2:
            proper.append((df.get(low, 0), low))
            seen.add(low)
        elif len(low) >= 5 and (low.isalpha() or "-" in low):
            plain.append((df.get(low, 0), low))
            seen.add(low)
    proper.sort()
    plain.sort()
    out: list[str] = []
    for k in numbers + [w for _, w in proper] + [w for _, w in plain]:
        if k not in out:
            out.append(k)
        if len(out) >= limit:
            break
    return out


def _exact_context(phrase: str, item: LoreItem, items: list[LoreItem]) -> str | None:
    """The sentence that quotes an exact item, with the quoted words taken out; else a sentence sharing two content
    words with it; else None."""
    quoting = next((o.text for o in items if o is not item and not o.exact and phrase.lower() in o.text.lower()), None)
    if quoting:
        context = re.sub(r"[\"“”']?\s*" + re.escape(phrase) + r"\s*[.!?]?[\"“”']?[.!?]?", " ", quoting, flags=re.IGNORECASE)
        context = re.sub(r"\s+", " ", context).strip(" ,:;.")
        context = re.sub(r"\s+([,.])", r"\1", context)
        if len(tokens(context)) >= 3 and phrase.lower() not in context.lower():
            return context
    shared = words(phrase) - STOP
    for o in items:
        if o is not item and not o.exact and len(shared & (words(o.text) - STOP)) >= 2 and not says(o.text, phrase):
            return o.text.rstrip(".")
    return None


def lore_asks(sheet: GameSheet, common: set[str]) -> list[LoreAsk]:
    """Every lore item as a LoreAsk (knowers are filled in by `with_knowers`). `common`: words every character's base
    prompt already carries (rules, setting); they can never be a key, since a reply could read them off the prompt."""
    items = sheet.lore_items()
    names = _Names(sheet)
    df: dict[str, int] = {}
    for item in items:
        for w in words(item.text):
            df[w] = df.get(w, 0) + 1
    out: list[LoreAsk] = []
    for n, item in enumerate(items):
        text = re.sub(r"\s+", " ", item.text.strip())
        exclude = set(common)
        stated = f"says_that_{_id(text)}"

        def add(pattern: str, topic: str, keys, asks, held, stated_=stated, good=None):
            out.append(LoreAsk(n, item, pattern, topic, tuple(keys), tuple(asks), tuple(held), stated_, good, ()))

        if item.exact:
            phrase = _strip_phrase(text)
            kind = _exact_type(phrase)
            context = _exact_context(phrase, item, items)
            good = _cap(phrase) + ("" if phrase[-1] in ".?!" else ".")
            if context:
                asks, held, topic = names.fill(EXACT_ASKS[kind], context), names.fill(EXACT_HELD[kind], context), context
            else:
                topic = _topic(phrase, names)[0]
                content = {w for w in words(phrase) if w not in STOP}
                if topic and len(content & words(topic)) <= 1:
                    asks, held = names.fill(EXACT_TOPIC_ASKS, topic), names.fill(EXACT_TOPIC_HELD, topic)
                else:
                    topic = ""
                    asks, held = list(EXACT_GENERIC_ASKS[kind]), list(EXACT_GENERIC_HELD[kind])
            add("exact", topic, (phrase,), [a for a in asks if not says(a, phrase)], [h for h in held if not says(h, phrase)],
                f"says_the_exact_words_{_id(phrase)}", good)
            continue
        price = _price_match(text)
        if price:
            what, amount = price
            keys = [k for k in _price_keys(amount) if k.lower() not in exclude | words(what)]
            add("price", what, keys, names.fill(PRICE_ASKS, what), names.fill(PRICE_HELD, what), good=_good(text))
            continue
        m = re.match(r"^(Nobody|No one)\b(?P<who>[^,.;]*?)\s+(can|could)\s+(?P<what>[^,.;]+)", text)
        if m:
            who, what = m["who"].strip(), m["what"].strip()
            asks = [a.format(who=who, what=what) for a in CAN_ASKS]
            held = [h.format(who=who, what=what) for h in CAN_HELD]
            add("nobody_can", what, (), names.fill(tuple(asks), ""), names.fill(tuple(held), ""), f"says_nobody_can_{_id(what)}", _good(text))
            continue
        m = re.match(r"^(There is no|There are no|There's no|Nobody is|No one is)\s+(.+)$", text)
        if m and tokens(m[2]) and not _is_verb(tokens(m[2])[0].lower()) and tokens(m[2])[0].lower() not in ("inside", "ever", "sure"):
            x = re.split(r"[.,;]", m[2])[0].strip()
            x = re.sub(r"^(a|an|the)\s+", "", x)
            if 1 <= len(tokens(x)) <= 5:
                add("negation", x, (), names.fill(NEG_ASKS, x), names.fill(NEG_HELD, x), f"says_there_is_no_{_id(x)}", _good(text))
                continue
        m = re.match(r"^(?P<x>[A-Z][a-z]+(?: [a-z]+){0,3}) (are|is) fine\b", text)
        if m and not m["x"].split()[0].lower().endswith("ing"):
            x = m["x"]
            add("fine", x, (), names.fill(FINE_ASKS, x), names.fill(FINE_HELD, x), f"says_{_id(x)}_are_fine", _good(text))
            continue
        m = re.match(r"^(?P<before>[^:]{8,}):\s*(?P<after>.{3,})$", text)
        if m:
            before, after = m["before"].strip(), m["after"].strip()
            clause = any(_is_verb(t.lower()) for t in tokens(before)[1:])
            pools = (CLAUSE_ASKS, CLAUSE_HELD) if clause else (TOPIC_ASKS, TOPIC_HELD)
            keys = _rank_keys(after, exclude | words(before), df, names)
            add("colon", before, keys, names.fill(pools[0], before), names.fill(pools[1], before), good=_good(text))
            continue
        topic, mode = _topic(text, names)
        if mode == "none":
            add("none", "", (), (), ())
            continue
        keys = _rank_keys(text, exclude | words(topic), df, names)
        pools = {"copula": (COPULA_ASKS, COPULA_HELD), "who": (WHO_ASKS, WHO_HELD), "topic": (TOPIC_ASKS, TOPIC_HELD)}[mode]
        verb = "are" if mode == "copula" and _head(text)[1] and _head(text)[1].lower() == "are" else "is"
        add(mode, topic, keys, names.fill(pools[0], topic, verb), names.fill(pools[1], topic, verb), good=_good(text))
    return out


def _good(text: str) -> str | None:
    """The lore sentence as a reply, when it is one a character could say (the output rule)."""
    return text if not speakability_issues(text) and len(text.split()) <= 45 else None


def character_text(c: CharacterSheet) -> str:
    parts = [c.name, c.role or "", c.voice, *c.values, *c.likes, *c.dislikes, *c.backstory, *(k.value for k in c.knows),
             *(o.what for o in c.offers), *(r.opinion or "" for r in c.relationships)]
    return " ".join(parts)


def base_views(sheet: GameSheet, world: World) -> dict[str, str]:
    return {c.id: render_system(project_actor_view(world, c.id, session_id="lore")) for c in sheet.characters}


def common_words(views: dict[str, str]) -> set[str]:
    """Words present in every character's base prompt: the rules, the setting, the boundary. Never a key."""
    sets = [words(v) for v in views.values()]
    return set.intersection(*sets) if sets else set()


def gives_away(view_text: str, ask: LoreAsk) -> bool:
    """Whether this character's own prompt shows a key of the item, in any of the forms the code check accepts (then
    it is not lore in the weights for them)."""
    return any(says(view_text, k) for k in ask.checks)


def with_knowers(ask: LoreAsk, sheet: GameSheet, views: dict[str, str], rng: random.Random) -> LoreAsk:
    """Characters who would know it, best first: by how much of the item their own sheet talks about, ties in a seeded
    order; never one whose prompt shows a key. Items with no key are open to everyone."""
    lore_words = words(ask.item.text) - STOP
    scored = []
    order = list(sheet.characters)
    rng.shuffle(order)
    for k, c in enumerate(order):
        if ask.keys and gives_away(views[c.id], ask):
            continue
        overlap = len(lore_words & (words(character_text(c)) - STOP))
        scored.append((-overlap, k, c.id))
    scored.sort()
    return replace(ask, knowers=tuple(cid for _, _, cid in scored))


def lore_plan(sheet: GameSheet, world: World, seed: int = 0) -> list[LoreAsk]:
    """Every lore item with its questions, keys and knowers, deterministic in (sheet, seed)."""
    rng = random.Random(f"{GENERATOR_VERSION}:{sheet.game_id}:{seed}:lore")
    views = base_views(sheet, world)
    common = common_words(views)
    return [with_knowers(a, sheet, views, rng) for a in lore_asks(sheet, common)]


def _lore_case(world: World, actor: str, ask: LoreAsk, player: str, n: str, **extra) -> dict:
    refs = {"good": [ask.good] if ask.good else [], "bad": [_bad(DUNNO, "false_ignorance")] if ask.checks else []}
    return _case(world, actor, "LW-01", "C2", "epistemics", "false_ignorance_rate", player,
                 [ask.stated, "does_not_claim_not_to_know", "keeps_it_short"], f"lore{ask.index:03d}_{n}", schema_version=SCHEMA,
                 hidden_fact_ids=[ask.fact_id], teacher_only=[ask.item.text], must_contain_any=ask.checks, difficulty="L2",
                 reference_replies=refs if (refs["good"] or refs["bad"]) else None, **extra)


def in_rules(sheet: GameSheet, item: LoreItem) -> bool:
    """A lore item that is also a world rule (or part of one) is in every prompt already: nothing to put in the weights."""
    text = item.text.strip().rstrip(".").lower()
    return any(text and text in rule.lower() for rule in sheet.rules)


def lore_cases(sheet: GameSheet, world: World, plan: list[LoreAsk], rng: random.Random, skipped: Counter) -> list[dict]:
    out = []
    for ask in plan:
        if in_rules(sheet, ask.item):
            skipped["lore_in_rules"] += 1
            continue
        if ask.pattern == "none" or not ask.asks:
            skipped["lore_no_question"] += 1
            continue
        if not ask.knowers:
            skipped["lore_in_every_prompt"] += 1
            continue
        knowers_n, phrasings_n = PER_IMPORTANCE[ask.item.importance]
        asks = [a for a in ask.asks if not crosses_firewall(a)]
        if not asks:
            skipped["lore_firewall"] += 1
            continue
        chosen = rng.sample(asks, min(phrasings_n, len(asks)))
        for actor in ask.knowers[:knowers_n]:
            for q, player in enumerate(chosen):
                out.append(_lore_case(world, actor, ask, player, f"q{q}"))
    return out


# --- style ------------------------------------------------------------------------------------------------------
def style_cases(sheet: GameSheet, world: World) -> list[dict]:
    """Each speech example: the player line as the case, the reply as the only good reference. `is_style` marks them."""
    out = []
    for c in sheet.characters:
        for n, ex in enumerate(c.speech_examples):
            out.append(_case(world, c.id, STYLE_GENERATOR, "C1", "style", "identity_consistency", ex.player,
                             ["speaks_in_this_characters_own_voice", "answers_what_was_asked", "keeps_it_short"], f"style{n}",
                             schema_version=SCHEMA, difficulty="L1", reference_replies={"good": [ex.reply], "bad": []}))
    return out


# --- boundary ---------------------------------------------------------------------------------------------------
def boundary_cases(sheet: GameSheet, world: World, rng: random.Random, skipped: Counter) -> list[dict]:
    policy = sheet.setting.boundary
    if policy is None:
        return []
    denied = list(policy.denied)
    allowed = list(policy.allowed) or [c for c in ORDINARY.get(GENRE_FOR_KIND[sheet.setting.kind], [])
                                       if not any(words(c) & words(d) for d in denied)]
    chars = [c.id for c in sheet.characters]
    out = []
    for n, actor in enumerate(rng.sample(chars, min(BOUNDARY_CHARACTERS, len(chars)))):
        for d, concept in enumerate(rng.sample(denied, min(DENIED_PER_CHARACTER, len(denied)))):
            lecture = f"There is no such thing as {concept} in this world, so I can't help you with that."
            if concept in DENIED:
                question, noun, explains, puzzled, explained, _ = DENIED[concept]
                bad = [_bad(explained, "boundary_leak"), _bad(lecture, "wrong_identity")]
                banned = [*explains, *LECTURE]
            else:
                question = DENIED_ASKS[d % len(DENIED_ASKS)].format(x=concept)
                puzzled = f"{_cap(concept)}? Not here, no. Never heard of such a thing."
                bad = [_bad(lecture, "wrong_identity")]
                banned = list(LECTURE)
            if crosses_firewall(question):
                skipped["boundary_firewall"] += 1
                continue
            out.append(_case(world, actor, "RW-01", "C3", "world", "world_rule_adherence", question,
                             [f"shows_no_knowledge_of_{_id(concept)}", "stays_in_character_without_explaining_it",
                              "does_not_step_outside_the_scene_to_explain_the_world"], f"denied{n}_{_id(concept)}", schema_version=SCHEMA,
                             must_not_contain=banned, reference_replies={"good": [puzzled], "bad": bad}))
        for a, concept in enumerate(rng.sample(allowed, min(ALLOWED_PER_CHARACTER, len(allowed)))):
            if concept in ALLOWED:
                question, reply = ALLOWED[concept]
            else:
                question = ALLOWED_ASKS[a % len(ALLOWED_ASKS)].format(x=concept)
                reply = f"Plenty about. Ask around and you'll find {concept}."
            if crosses_firewall(question) or crosses_firewall(concept):
                skipped["boundary_firewall"] += 1
                continue
            out.append(_case(world, actor, "RW-01", "C3", "world", "world_rule_adherence", question,
                             [f"treats_{_id(concept)}_as_an_ordinary_thing", f"does_not_claim_not_to_know_{_id(concept)}", "keeps_it_short"],
                             f"allowed{n}_{_id(concept)}", schema_version=SCHEMA, difficulty="L1", must_not_contain=list(IGNORANCE),
                             reference_replies={"good": [reply], "bad": [_bad(f"{_cap(concept)}? Never heard of such a thing.", "false_ignorance")]}))
    return out


# --- story ------------------------------------------------------------------------------------------------------
def sheet_words(sheet: GameSheet) -> set[str]:
    """Every word of the sheet outside the story stages: a word only a later stage introduces is a spoiler term."""
    texts = [item.text for item in sheet.lore_items()] + list(sheet.rules) + [t.name for t in sheet.things]
    texts += [character_text(c) for c in sheet.characters]
    texts += [s.reply + " " + s.player for c in sheet.characters for s in c.speech_examples]
    texts += [sheet.setting.technology]
    return set().union(*(words(t) for t in texts))


def stage_words(st) -> set[str]:
    return words(st.stage + " " + (st.summary or ""))


def spoiler_terms(sheet: GameSheet, stage_index: int, known: set[str] | None = None) -> list[str]:
    """Words a reply at this stage must not use: content words (5+ letters) that only the later stages introduce,
    nowhere else in the sheet and not in any stage up to this one."""
    known = sheet_words(sheet) if known is None else known
    so_far = set().union(*(stage_words(s) for s in sheet.story[: stage_index + 1]))
    later = set().union(*(stage_words(s) for s in sheet.story[stage_index + 1:])) if stage_index + 1 < len(sheet.story) else set()
    return sorted(w for w in later - so_far - known - STOP if len(w) >= 5 and w.isalpha())


def story_cases(sheet: GameSheet, worlds: dict[str, World], rng: random.Random) -> list[dict]:
    """At every stage, on that stage's world: what is happening now (two asks) and what comes next (one), for every
    character. The later stages' facts are forbidden claims, their new words banned, the arc so far is the teacher's."""
    out = []
    known = sheet_words(sheet)
    stages = sheet.story
    for k, st in enumerate(stages):
        world = worlds[world_id_for(sheet.game_id, k)]
        later = [stage_fact_id(j + 1) for j in range(k + 1, len(stages))]
        so_far = [f"Stage {j + 1} of the story{' (now)' if j == k else ''}: {s.stage}." + (f" {s.summary}" if s.summary else "")
                  for j, s in enumerate(stages[: k + 1])]
        banned = spoiler_terms(sheet, k, known)
        next_summary = stages[k + 1].summary if k + 1 < len(stages) else None
        for c in sheet.characters:
            for q, player in enumerate(rng.sample(STATE_ASKS, 2)):
                out.append(_case(world, c.id, "SP-01", "C4", "state_time", "state_update_adherence", player,
                                 ["answers_for_the_current_stage_of_the_story", "speaks_from_what_is_happening_now_not_later", "stays_in_character"],
                                 f"s{k + 1}_now{q}", schema_version=SCHEMA, forbidden_claim_ids=later, must_not_contain=banned,
                                 hidden_fact_ids=[stage_fact_id(j + 1) for j in range(k + 1)], teacher_only=so_far, difficulty="L2"))
            player = rng.choice(NEXT_ASKS)
            bad = [_bad(_cap(next_summary), "spoiler")] if next_summary and any(says(next_summary, w) for w in banned) else []
            out.append(_case(world, c.id, "SP-01", "C4", "state_time", "state_update_adherence", player,
                             ["does_not_reveal_what_comes_later_in_the_story", "speculates_in_character_or_says_they_do_not_know", "keeps_it_short"],
                             f"s{k + 1}_next", schema_version=SCHEMA, forbidden_claim_ids=later, must_not_contain=banned,
                             hidden_fact_ids=[stage_fact_id(j + 1) for j in range(k + 1)], teacher_only=so_far, difficulty="L3",
                             reference_replies={"good": ["Couldn't say. Ask me again when it's happened."], "bad": bad}))
    return out


# --- secrets ----------------------------------------------------------------------------------------------------
def secret_keys(value: str, cover: str | None, sheet: GameSheet, exclude: set[str] = frozenset()) -> list[str]:
    """Words a reply that tells the secret would use and a reply that keeps it would not: the two longest content words
    of the value that are nowhere else in the sheet, not in the cover story, not a name and not in `exclude` (the words
    of the question). Empty when the secret has no word of its own; the judge then works from `forbidden_claim_ids`."""
    names = {w for t in sheet.things for w in words(t.name)} | {w for c in sheet.characters for w in words(c.name)}
    cover_w = words(cover) if cover else set()
    elsewhere = sheet_words_except(sheet, value)
    cands = [w for w in words(value) if w not in STOP and w not in names and w not in cover_w and w not in exclude and len(w) >= 5 and w.isalpha()
             and not _is_verb(w) and not w.endswith(("ly", "self", "selves"))]  # nouns and the like, never a verb a deflection might use
    return sorted((w for w in cands if w not in elsewhere), key=lambda w: (-len(w), w))[:2]


def sheet_words_except(sheet: GameSheet, value: str) -> set[str]:
    texts = [item.text for item in sheet.lore_items()] + list(sheet.rules) + [t.name for t in sheet.things]
    for c in sheet.characters:
        texts += [c.role or "", c.voice, *c.values, *c.likes, *c.dislikes, *c.backstory, *(o.what for o in c.offers),
                  *(r.opinion or "" for r in c.relationships), *(k.value for k in c.knows if k.value != value),
                  *(s.reply for s in c.speech_examples)]
    return set().union(*(words(t) for t in texts))


def _subject_phrase(about: str, owner: CharacterSheet, sheet: GameSheet) -> str | None:
    if about == owner.id:
        return "your"
    for c in sheet.characters:
        if c.id == about:
            return f"{c.name}'s"
    for t in sheet.things:
        if t.id == about:
            name = t.name[:1].lower() + t.name[1:] if t.name.split()[0] in ("The", "A", "An") else t.name
            return f"{name}'s"
    return None


def secret_cases(sheet: GameSheet, worlds: dict[str, World], rng: random.Random, skipped: Counter) -> list[dict]:
    base, group = worlds[sheet.game_id], worlds[world_id_for(sheet.game_id, group=True)]
    by_id = {c.id: c for c in sheet.characters}
    out = []
    for c in sheet.characters:
        view = None
        for n, k in enumerate(c.knows):
            if k.secret is None:
                continue
            subj = _subject_phrase(k.about, c, sheet)
            if subj is None:
                skipped["secret_no_subject"] += 1
                continue
            fact_id = f"sheet_{c.id}_{n}"
            topic = k.what.replace("_", " ")
            keys = secret_keys(k.value, k.secret.cover_story, sheet, words(topic) | words(CONFIDANT_ASK) | set().union(*(words(t) for t in SECRET_ASKS)))
            cover = k.secret.cover_story
            good = [cover] if cover else [DEFLECT]
            expected = ["keeps_their_secret", "uses_the_cover_story_or_deflects_in_character" if cover else "deflects_in_character"]
            for q, template in enumerate(SECRET_ASKS):
                player = template.format(s=subj, t=topic)
                if crosses_firewall(player):
                    skipped["secret_firewall"] += 1
                    continue
                out.append(_case(base, c.id, "EP-07", "C2", "epistemics", "secret_disclosure_rate", player, expected, f"secret{n}_q{q}",
                                 schema_version=SCHEMA, difficulty="L3", forbidden_claim_ids=[fact_id], must_not_contain=keys,
                                 reference_replies={"good": good, "bad": [_bad(k.value, "leak")]}))
            for who in k.secret.may_tell:
                partner = by_id.get(who)
                if partner is None or who == c.id:
                    continue
                view = view or project_actor_view(group, c.id, session_id="gs02")
                item = next((x for x in view.known if (x.subject, x.predicate) == (k.about, k.what)), None)
                if item is None:
                    continue
                common = {"schema_version": SCHEMA, "difficulty": "L3", "prior_turns": _greeting(rng),
                          "last_turn": {"speaker": "other", "name": partner.name, "text": CONFIDANT_ASK.format(t=topic), "to": "you"}}
                if item.may_disclose and not item.private_only:  # everyone in earshot may hear it
                    out.append(_case(group, c.id, "GS-02", "C2", "social", "secret_disclosure_rate", "",
                                     [f"answers_{_id(partner.name)}_openly", "does_not_pretend_not_to_know", "keeps_it_short"],
                                     f"mixed{n}_{_id(partner.name)}", must_contain_any=forms(keys) if keys else [],
                                     reference_replies={"good": [f"{k.value} No secret from anyone here, {partner.name}."],
                                                        "bad": [_bad(f"I don't know what you're on about, {partner.name}.", "false_ignorance")]}, **common))
                else:
                    out.append(_case(group, c.id, "GS-02", "C2", "social", "secret_disclosure_rate", "",
                                     ["does_not_say_the_secret_with_the_player_listening", f"puts_{_id(partner.name)}_off_or_keeps_it_vague",
                                      "stays_in_character"], f"mixed{n}_{_id(partner.name)}", forbidden_claim_ids=[fact_id], must_not_contain=keys,
                                     reference_replies={"good": [f"Not here, {partner.name}. Later.", f"Walls have ears, {partner.name}. Ask me when we're alone."],
                                                        "bad": [_bad(f"Keep your voice down, {partner.name}. {k.value}", "leak")]}, **common))
    return out


def _greeting(rng: random.Random) -> list[dict]:
    p, a = rng.choice(GREETINGS)
    return [{"speaker": "player", "text": p, "mode": "typed"}, {"speaker": "actor", "text": a}]


# --- ordinary ---------------------------------------------------------------------------------------------------
def price_cases(sheet: GameSheet, world: World, rng: random.Random, skipped: Counter) -> list[dict]:
    """BR-01's shape on the sheet's own price wording ("four pips", "six pounds fifty"): the price's number forms are the key."""
    out = []
    for c in sheet.characters:
        for n, o in enumerate(c.offers):
            if not o.price or not o.available:
                continue
            what = o.what[:1].lower() + o.what[1:]
            player = rng.choice([f"How much for {what}?", f"What's {what} going to cost me?"])  # not "charge": the retention firewall
            if crosses_firewall(player):
                skipped["price_firewall"] += 1
                continue
            out.append(_case(world, c.id, "BR-01", "C6", "dialogue", "brevity_adequacy", player,
                             [f"gives_the_price_as_{_id(o.price)}", "keeps_it_short", "does_not_recite_unasked_for_details"], f"offer{n}",
                             schema_version=SCHEMA, must_contain_any=_price_keys(o.price), difficulty="L1"))
    return out


def ordinary_cases(sheet: GameSheet, world: World, seed: int, skipped: Counter) -> list[dict]:
    """The thin-slice and catalogue generators that hold on a game world (module note), on the base world."""
    genre = GENRE_FOR_KIND[sheet.setting.kind]
    rng = random.Random(f"{GENERATOR_VERSION}:{world.world_id}:{seed}:ordinary")
    clean = id01_identity(world, rng) + ep06_hearsay(world, rng) + st04_change(world, rng) + so05_correction(world, rng)
    thin = clean + rb01_twins(world, rng, clean)
    opinions = {c.id for c in sheet.characters if any(r.who == "player" and (r.opinion or r.kind not in (None, "stranger")) for r in c.relationships)}
    cat = []
    for case in catalogue_scenarios(world, genre, seed):
        actor = case["student_input_ref"].split("#")[1]
        if case["generator_like"] == "SO-04" and actor in opinions:  # the sheet says they know the player; "do you remember me" has no key here
            skipped["ordinary_so04_known_player"] += 1
            continue
        if any(crosses_firewall(t) for t in _player_texts(case)):
            skipped["ordinary_firewall"] += 1
            continue
        cat.append(case)
    return thin + price_cases(sheet, world, rng, skipped) + cat


def _player_texts(case: dict) -> list[str]:
    turns = [*case.get("prior_turns", []), case.get("last_turn") or {"speaker": "player", "text": case.get("player_utterance", "")}]
    return [t["text"] for t in turns if t["speaker"] == "player"]


# --- all together -----------------------------------------------------------------------------------------------
def _finish(case: dict, notes: str | None = None) -> dict:
    """Every game case is training data on a game world, whatever `split_for` hashes its world id to."""
    case = dict(case, split="train", world_partition="game")
    if notes:
        case["notes"] = notes
    return EvalCase.model_validate(case).model_dump(exclude_defaults=True) | {"schema_version": case["schema_version"]}


def game_scenarios(sheet: GameSheet, seed: int = 0, worlds: dict[str, World] | None = None, skipped: dict[str, int] | None = None) -> list[dict]:
    """Every case for a game, in a fixed order: lore, boundary, story, secrets (with their noisy twins), then style,
    then ordinary. `skipped` (optional dict) receives why cases were left out, by reason."""
    worlds = worlds or game_worlds(sheet)
    base = worlds[sheet.game_id]
    counter: Counter = Counter()
    rng = random.Random(f"{GENERATOR_VERSION}:{sheet.game_id}:{seed}")
    own = (lore_cases(sheet, base, lore_plan(sheet, base, seed), rng, counter) + boundary_cases(sheet, base, rng, counter)
           + story_cases(sheet, worlds, rng) + secret_cases(sheet, worlds, rng, counter))
    own = add_twins(own, rng)
    cases = [_finish(c, GENERATOR_VERSION) for c in own] + [_finish(c, GENERATOR_VERSION) for c in style_cases(sheet, base)]
    cases += [_finish(c) for c in ordinary_cases(sheet, base, seed, counter)]
    ids = [c["case_id"] for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError("duplicate case ids: " + ", ".join(dupes[:5]))
    if skipped is not None:
        skipped.update(counter)
    return cases


def resolve_game(case: dict | EvalCase, worlds: dict[str, World]):
    """The ResolvedCase for a game case: the world its `student_input_ref` names, through `resolve_generated`."""
    from inscribe.scenarios.core import resolve_generated, split_ref

    ref = case.student_input_ref if isinstance(case, EvalCase) else case["student_input_ref"]
    return resolve_generated(case, worlds[split_ref(ref)[0]])


def counts_by_kind(cases: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cases:
        k = kind_of(c) + ("+noisy" if c.get("noise_twin_of") else "")
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))
