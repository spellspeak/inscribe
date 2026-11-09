"""Scenario generators for using what a character knows.

An early trained model lost two of eight general-knowledge answers: it said four loaves at three coins cost four coins,
and waved off storm clouds. The thin slice rewarded doubt and short answers, never confident use of knowledge.
These three generators add that, on the same generated worlds, with answer keys written here, not by the teacher.

| Generator | Capability | What it asks | Causal variable (the answer flips with it) |
|---|---|---|---|
| ID-02 | C1 | a practical question about the character's own trade | the character's role |
| EP-11 | C2 | everyday know-how any adult in the setting has; modern know-how | the setting's technology |
| BR-01 | C6 | the total for several of something it sells | the quantity and the price |

Retention firewall: no question here touches the 8 retention knowledge prompts (blades, phone charging, RAM, the moon,
the war, storm clouds, bread, floating ice). `FIREWALL` lists the words; a test checks every generated question.
Multi-item totals do exercise the same skill as prompt k7 (a sum at a counter); that overlap is known and declared.
"""
from __future__ import annotations

import random
import re

from inscribe.schemas.world import World
from inscribe.scenarios.core import _case, _id, number_forms

GENERATOR_VERSION = "scenariogen-0.2-knowledge"

FIREWALL = ("storm", "thunder", "lightning", "rain", "cloud", "bread", "loaf", "loaves", "ice", "float", "moon", "armstrong",
            "war", "1945", "charge", "charging", "ram", "storage", "quench", "temper", "harden", "blade")

# role -> [(question, words a correct answer uses: any one counts)]. Roles absent here get no ID-02 case.
TRADE: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "innkeeper": [("How do you keep the ale from going sour?", ("cool", "cold", "cellar", "clean", "seal", "cask", "fresh")),
                  ("What's the best way to get wine out of a tablecloth?", ("salt", "soak", "vinegar", "cold water", "lemon", "scrub"))],
    "blacksmith": [("How do you keep tools from rusting?", ("oil", "grease", "dry", "wax", "lard")),
                   ("What's the horn of the anvil for?", ("bend", "curve", "round", "shape", "hook", "ring"))],
    "healer": [("What's good for a bee sting?", ("scrape", "pull", "remove", "cold", "honey", "baking", "vinegar", "poultice", "onion")),
               ("How should I clean a cut?", ("wash", "clean", "boiled", "water", "spirit", "rinse"))],
    "miller": [("Why do millstones need dressing?", ("sharp", "groove", "furrow", "dull", "worn", "smooth", "grind")),
               ("Why do you keep the grain dry?", ("mould", "mold", "rot", "spoil", "damp", "sprout"))],
    "stablehand": [("How often does a horse need new shoes?", ("week", "month", "six", "eight", "4", "6", "8")),
                   ("How can you tell if a horse is lame?", ("limp", "nod", "bob", "uneven", "favour", "favor", "trot", "short"))],
    "baker": [("How do you know a pie is done?", ("golden", "brown", "bubble", "crust", "crisp")),
              ("What does yeast actually do?", ("rise", "gas", "bubble", "ferment", "puff", "air"))],
    "ferryman": [("How do you get across when the current's strong?", ("angle", "upstream", "against", "point", "slant", "rope", "line"))],
    "merchant": [("How can you tell good rope from bad?", ("fray", "twist", "tight", "rot", "smell", "strong", "even", "fibre", "fiber"))],
    "café owner": [("Why does espresso need the beans ground so fine?", ("pressure", "extract", "flavour", "flavor", "water", "slow", "through")),
                   ("How do you steam milk properly?", ("air", "froth", "foam", "wand", "swirl", "jug", "hot", "temperature"))],
    "phone repair tech": [("What should I do if I drop my phone in water?", ("off", "dry", "rice", "power", "don't")),
                          ("How do I stop my screen cracking?", ("case", "protector", "glass", "cover"))],
    "librarian": [("How do I find books on a subject here?", ("catalogue", "catalog", "shelf", "number", "dewey", "search", "computer", "section"))],
    "mechanic": [("How often should I change my oil?", ("mile", "month", "year", "thousand", "manual", "km", "kilometre", "kilometer")),
                 ("What does it mean if my brakes squeal?", ("pad", "worn", "replace", "wear"))],
    "nurse": [("How long should I wash my hands for?", ("twenty", "20", "seconds")),
              ("What do I do for a nosebleed?", ("forward", "pinch", "soft part", "ten minutes", "10 minutes"))],
    "shopkeeper": [("How do I keep milk fresh longer?", ("fridge", "cold", "cool", "chill"))],
    "personal trainer": [("How do I stop being so sore after a workout?", ("stretch", "warm", "water", "rest", "slow", "cool down", "build up", "gradual")),
                         ("How many days a week should I train?", ("three", "four", "two", "five", "3", "4", "rest"))],
    "teacher": [("How do I help my kid get better at reading?", ("read", "together", "every day", "aloud", "sound", "books"))],
    "delivery driver": [("What's the quickest way to plan a lot of stops?", ("route", "order", "map", "group", "loop", "area", "close"))],
    "quartermaster": [("How do you stop supplies going missing?", ("count", "log", "inventory", "record", "sign", "check", "lock", "audit"))],
    "botanist": [("Why are the grow lights that pinkish colour?", ("red", "blue", "light", "photosynthesis", "grow", "wavelength", "colours", "colors")),
                 ("How often do you water seedlings?", ("moist", "damp", "daily", "every day", "soil", "little", "often", "dry"))],
    "medic": [("What's the first thing to do if someone isn't breathing?", ("airway", "call", "chest", "compression", "cpr", "breath", "help", "check")),
              ("How do I treat a sprained wrist?", ("rest", "cold", "raise", "wrap", "elevate", "brace", "still"))],
    "engineer": [("What's the first thing you check when a pump stops?", ("power", "seal", "block", "filter", "pressure", "valve", "fuse", "clog")),
                 ("Why does that valve hiss?", ("pressure", "leak", "seal", "release", "gas", "air", "relief"))],
    "cargo handler": [("How do you stop crates shifting in transit?", ("strap", "tie", "lash", "secure", "net", "brace", "lock", "clamp"))],
    "comms officer": [("Why does the signal drop when we pass behind the planet?", ("line of sight", "block", "shadow", "in the way", "between", "relay"))],
    "cook": [("How do you stop rice sticking?", ("rinse", "wash", "starch", "stir", "oil", "water", "lid")),
             ("How do you know oil is hot enough to fry?", ("sizzle", "bubble", "shimmer", "drop", "test", "smoke", "crumb"))],
    "maintenance robot": [("What does a filter in the air system do?", ("dust", "particle", "clean", "trap", "catch", "filter"))],
    "bartender": [("What's the difference between a lager and an ale?", ("yeast", "top", "bottom", "cold", "warm", "ferment")),
                  ("Why do you shake some drinks and stir others?", ("cloud", "air", "froth", "chill", "dilute", "clear", "texture", "mix"))],
    "storekeeper": [("How do you keep flour safe from weevils?", ("dry", "seal", "tin", "bay", "airtight", "cool", "lid", "barrel"))],
    "prospector": [("How do you pan for gold?", ("swirl", "water", "heavy", "settle", "bottom", "gravel", "wash", "shake"))],
    "assayer": [("How can you tell real gold from fool's gold?", ("streak", "soft", "scratch", "brittle", "heavy", "weight", "cube", "crumble", "pyrite", "bend"))],
    "banker": [("What's interest on a loan, plainly?", ("extra", "borrow", "pay back", "percent", "on top", "cost", "more than"))],
    "nightclub singer": [("How do you look after your voice?", ("water", "warm", "honey", "rest", "tea", "steam", "scarf", "sleep"))],
    "pawnbroker": [("How can you tell if a watch is real gold?", ("stamp", "hallmark", "mark", "karat", "carat", "test", "acid", "magnet", "weight"))],
    "dockworker": [("How do you lift a heavy crate without hurting your back?", ("leg", "knee", "bend", "straight", "close", "squat"))],
    "boxer": [("How should I make a fist so I don't break my hand?", ("thumb", "outside", "knuckle", "tight", "wrist"))],
    "waitress": [("How do you carry so many plates at once?", ("arm", "forearm", "balance", "stack", "wrist", "practice", "palm"))],
    "harbour master": [("How do I know when the tide's high?", ("table", "chart", "times", "almanac", "moon", "twice", "twelve"))],
    "lighthouse keeper": [("Why does every lighthouse flash a different pattern?", ("pattern", "tell", "which", "identify", "know where", "recognise", "recognize", "own"))],
    "fishmonger": [("How can you tell if a fish is fresh?", ("eye", "clear", "gill", "red", "smell", "firm", "shiny", "bright"))],
    "boatwright": [("Why do you caulk a hull?", ("leak", "seam", "gap", "watertight", "seal", "water"))],
    "herbalist": [("What's chamomile good for?", ("sleep", "calm", "stomach", "relax", "nerve", "rest")),
                  ("What's mint good for?", ("stomach", "digest", "belly", "breath", "head", "cool", "sick"))],
    "net mender": [("How do you mend a hole in a net?", ("knot", "needle", "twine", "mesh", "tie", "bend", "cord"))],
    # action genres
    "wingman": [("How do you shake a missile on your tail?", ("chaff", "flare", "break", "turn", "notch", "dive"))],
    "flight lead": [("How do you keep the formation together in a fight?", ("call", "radio", "spacing", "support", "close", "check"))],
    "AWACS controller": [("How do you call out where a bandit is?", ("bearing", "range", "altitude", "angels", "bullseye", "clock"))],
    "deck chief": [("How do you launch a jet off the catapult?", ("catapult", "shuttle", "tension", "salute", "hook", "power"))],
    "rescue pilot": [("How do you pick up a downed pilot under fire?", ("smoke", "hover", "hoist", "cover", "fast", "low"))],
    "tanker pilot": [("How does refuelling in the air work?", ("boom", "basket", "probe", "contact", "drogue", "steady"))],
    "sniper": [("How do you allow for the wind on a long shot?", ("wind", "hold", "drift", "adjust", "flag", "dope"))],
    "radio operator": [("What do you do when the radio cuts out?", ("antenna", "frequency", "relay", "battery", "higher", "channel"))],
    "demolitions specialist": [("How do you breach a locked door safely?", ("charge", "cover", "clear", "back", "stack"))],
    "scout": [("How do you move without being seen?", ("cover", "slow", "low", "shadow", "quiet", "ground"))],
    "heavy gunner": [("How do you keep the gun from overheating?", ("burst", "barrel", "short", "change", "cool"))],
    "helicopter pilot": [("How do you land in a hot landing zone?", ("fast", "smoke", "low", "cover", "flare", "quick"))],
    "interpreter": [("How do you calm a frightened villager down?", ("slow", "calm", "quiet", "explain", "respect", "name"))],
    "driver": [("How do you drive through an ambush?", ("speed", "keep", "moving", "stop", "push", "through"))],
}

# Everyday know-how any adult in any setting here would have: (question, words a correct answer uses).
EVERYDAY: list[tuple[str, tuple[str, ...]]] = [
    ("How do I stop a cut from bleeding?", ("press", "pressure", "cloth", "bandage", "hold", "wrap", "tight")),
    ("What should I do for a burn on my hand?", ("cool", "cold", "water")),
    ("How can I find north without a compass?", ("sun", "star", "shadow", "moss")),
    ("How long do I boil an egg to get it hard?", ("minute", "ten", "eight", "nine", "twelve", "seven", "10", "8", "9", "12", "7")),
    ("How do I get a fire going with damp wood?", ("kindling", "dry", "tinder", "shaving", "small", "bark", "split")),
    ("How do I keep meat from spoiling?", ("salt", "smoke", "dry", "cool", "cold", "cure")),
    ("How do I get a stuck ring off my finger?", ("soap", "oil", "butter", "cold", "raise", "lift", "grease", "twist")),
    ("What do I do if a dog growls at me?", ("still", "slow", "back away", "don't run", "calm", "eye", "stand")),
    ("How do I keep moths out of my wool?", ("cedar", "lavender", "seal", "clean", "chest", "air")),
    ("How should I treat a twisted ankle?", ("rest", "raise", "cold", "wrap", "elevate", "bind", "off it")),
    ("How do I sharpen a knife?", ("stone", "whetstone", "angle", "strop", "steel", "file")),
    ("How do I get a splinter out?", ("needle", "tweezer", "pin", "squeeze", "soak", "pull")),
]

# Modern know-how: practical in present-day and near-future settings; unknown everywhere else.
MODERN = {"modern_town", "space_station"}
MODERN_KNOWHOW: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = [
    # (question, the thing, words a practical answer uses, words only a practical answer would use)
    ("How do I reset a wifi router?", "a wifi router", ("unplug", "button", "restart", "power", "switch", "plug", "reset", "pin"),
     ("unplug", "restart", "reboot", "paperclip")),
    ("How do I attach a photo to an email?", "email attachments", ("attach", "paperclip", "clip", "button", "icon", "drag", "select", "plus"),
     ("paperclip", "icon", "drag", "click")),
    ("How do I turn off notifications on my phone?", "phone notifications", ("settings", "toggle", "swipe", "app", "switch"),
     ("settings", "toggle", "swipe")),
    ("How do I back up the photos on my phone?", "phone backups", ("cloud", "drive", "computer", "laptop", "copy", "account", "sync"),
     ("cloud", "sync", "laptop")),
]

TOTAL_EXCLUDE = ("bread", "loaf", "pie")  # k7 is bread at a counter; keep totals off bread and baked goods


def forms(words) -> list[str]:
    """Each key word with its common endings, since the code check matches whole words ("minute" must match "minutes")."""
    out = []
    for w in words:
        head, _, last = w.rpartition(" ")
        if head or not last.isalpha():
            ends = [last]
        elif last.endswith("e"):
            ends = [last, last + "s", last + "d", last[:-1] + "ing"]
        elif last.endswith("y") and last[-2:-1] not in "aeiou":
            ends = [last, last[:-1] + "ies", last[:-1] + "ied", last + "ing"]
        elif last.endswith(("s", "x", "ch", "sh")):
            ends = [last, last + "es", last + "ed", last + "ing"]
        else:
            ends = [last, last + "s", last + "ed", last + "ing"]
        out += [f"{head} {e}".strip() for e in ends]
    return list(dict.fromkeys(out))


def plural(what: str) -> str | None:
    """'a slice of cake' -> 'slices of cake'; None for things that are not counted one by one."""
    m = re.match(r"^(a|an) (.+)$", what)
    if not m or " and " in what:
        return None
    head, sep, tail = re.match(r"^(.*?)((?: of | for ).*)?$", m.group(2)).group(1, 2, 2)
    words = head.split()
    noun = words[-1]
    irregular = {"knife": "knives", "battery": "batteries", "box": "boxes", "watch": "watches", "glass": "glasses"}
    words[-1] = irregular.get(noun, noun + ("es" if noun.endswith(("s", "x", "ch", "sh")) else "s"))
    return " ".join(words) + (sep or "")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def crosses_firewall(text: str) -> bool:
    words = _words(text)
    return bool((words | {w[:-1] for w in words if w.endswith("s")}) & set(FIREWALL))


def id02_trade(world: World, rng: random.Random) -> list[dict]:
    out = []
    for n, actor in enumerate(world.characters()):
        e = world.entities[actor]
        pool = TRADE.get(e.role or "")
        if not pool:
            continue
        question, words = rng.choice(pool)
        out.append(_case(world, actor, "ID-02", "C1", "identity", "identity_consistency", question,
                         [f"answers_from_their_work_as_{_id(e.role)}", "states_it_plainly_without_claiming_not_to_know", "keeps_it_short"],
                         f"trade{n}", difficulty="L1", must_contain_any=forms(words),
                         must_not_contain=["I don't know", "no idea", "not sure"]))
    return out


def ep11_knowhow(world: World, rng: random.Random, genre: str) -> list[dict]:
    chars = list(world.characters())
    out = []
    for n, actor in enumerate(rng.sample(chars, min(3, len(chars)))):
        question, words = rng.choice(EVERYDAY)
        out.append(_case(world, actor, "EP-11", "C2", "epistemics", "false_ignorance_rate", question,
                         ["gives_sensible_everyday_advice", "does_not_claim_not_to_know", "keeps_it_short"], f"everyday{n}",
                         difficulty="L1", must_contain_any=forms(words)))
    actor = rng.choice(chars)
    question, thing, words, only_practical = rng.choice(MODERN_KNOWHOW)
    if genre in MODERN:
        out.append(_case(world, actor, "EP-11", "C2", "epistemics", "false_ignorance_rate", question,
                         ["gives_practical_modern_advice", "does_not_feign_unfamiliarity_with_modern_technology", "keeps_it_short"],
                         "modern", difficulty="L2", must_contain_any=forms(words)))
    else:
        out.append(_case(world, actor, "EP-11", "C2", "epistemics", "appropriate_ignorance", question,
                         [f"does_not_know_what_{_id(thing)}_are" if thing.endswith("s") else f"does_not_know_what_{_id(thing)}_is",
                          "does_not_explain_how_it_works", "stays_in_character"],
                         "modern", difficulty="L2", must_not_contain=list(only_practical)))
    return out


def br01_totals(world: World, rng: random.Random) -> list[dict]:
    out = []
    for actor, offer_set in world.offers.items():
        for n, o in enumerate(offer_set.offers):
            if not o.price or not o.available or any(w in o.what.lower() for w in TOTAL_EXCLUDE):
                continue
            many = plural(o.what)
            if many is None:
                continue
            amount = int(o.price.split()[0])
            qty = rng.choice([2, 3, 4, 5])
            word = number_forms(qty)[1]
            player = rng.choice([f"I'll take {word} {many}. What do I owe you?", f"How much for {word} {many}?",
                                 f"What would {word} {many} come to?"])
            currency = o.price.split(maxsplit=1)[1]
            total = amount * qty
            out.append(_case(world, actor, "BR-01", "C6", "dialogue", "brevity_adequacy", player,
                             [f"gives_the_total_as_{total}_{_id(currency)}", "keeps_it_short"], f"total{n}", difficulty="L2",
                             must_contain_any=number_forms(total)))
    return out


def knowledge_scenarios(world: World, genre: str, seed: int = 0) -> list[dict]:
    rng = random.Random(f"{GENERATOR_VERSION}:{world.world_id}:{seed}")
    cases = id02_trade(world, rng) + ep11_knowhow(world, rng, genre) + br01_totals(world, rng)
    for c in cases:
        c["notes"] = GENERATOR_VERSION
    return cases
