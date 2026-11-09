"""What a character does with things from outside its world, and noisy twins of a case.

The game sheet's boundary says which real-world things a world denies or allows. These pools hold a ready question
and replies for the common ones: a denied thing is met with in-character ignorance, an allowed one is treated as
ordinary. `add_twins` adds a misheard copy of about a third of the cases where the player speaks last.
"""
from __future__ import annotations

import random

from inscribe.scenarios.catalogue import _noisy
from inscribe.scenarios.core import CHECKS
from inscribe.schemas.records import EvalCase

SCHEMA = "eval-case-0.3"
# Real-world things that are simply ordinary in a world where the real world is this world.
ORDINARY = {"modern_town": ["coffee", "newspapers", "bicycles"], "noir_city": ["cigarettes", "taxis", "radios"],
            "space_station": ["coffee", "radios", "torches"], "air_combat": ["coffee", "radios"], "tactical_squad": ["coffee", "radios"]}
# denied concept -> (the player's line, the noun, words only an explanation would use, in-character ignorance,
#                    a reply that explains the thing, a normal reply when the thing is allowed)
DENIED: dict[str, tuple[str, str, tuple[str, ...], str, str, str]] = {
    "phones": ("Have you seen my phone anywhere?", "phone", ("ring", "dial", "signal", "battery"),
               "A phone? Never heard of such a thing. What does it look like?",
               "Your phone? The thing you ring people on? No signal out here anyway, so it's no loss.",
               "Haven't seen a phone, sorry. Try where you were sitting."),
    "telephones": ("Is there a telephone I could use?", "telephone", ("ring", "dial", "line", "operator"),
                   "A telephone? Never heard of one. What is it you need?",
                   "A telephone? You lift it and the operator puts you through on the line. There's none here, mind.",
                   "No telephone here, I'm afraid. Try the next town."),
    "mobile phones": ("Did you see where I left my mobile?", "mobile", ("ring", "signal", "battery", "text"),
                      "Your mobile? Never heard of such a thing. What is it?",
                      "Your mobile? The little thing you ring people on? The battery's probably flat by now.",
                      "Haven't seen a mobile. Have you checked your coat?"),
    "cars": ("Where should I leave my car?", "car", ("drive", "engine", "petrol", "park it"),
             "Your car? Never heard the word. What is it you've brought?",
             "Park it out front. Mind the engine, though, the cold is hard on them.",
             "Round the back is fine. Nobody will bother it."),
    "self-driving cars": ("Do the self-driving cars come out this far?", "self-driving car", ("sensors", "software", "steer", "autopilot"),
                          "Self-driving what? Never heard of such a thing.",
                          "The self-driving ones? Their sensors can't cope with these lanes, so no.",
                          "Not out this far, no. You'd need a lift."),
    "the internet": ("Can I get on the internet here?", "the internet", ("online", "website", "wifi", "web"),
                     "The internet? Never heard of it. What is it?",
                     "The internet? There's wifi behind the bar, but it's slow.",
                     "There's no connection here, sorry. Try the library."),
    "electricity": ("Is there electricity upstairs?", "electricity", ("switch", "wire", "bulb", "plug"),
                    "Electricity? Never heard of it. What would you need it for?",
                    "Electricity? There's a switch by the stairs and a socket to plug into.",
                    "Upstairs too, yes. Mind the loose board on the landing."),
    "electric light": ("Could you switch the electric light on?", "electric light", ("switch", "wire", "bulb", "socket"),
                       "Electric light? Never heard of such a thing. I've a lamp if you need one.",
                       "The bulb went last week, and the wire's no better.",
                       "There you go. Brighter now."),
    "computers": ("Can you look it up on your computer?", "computer", ("screen", "keyboard", "type it", "database"),
                  "A computer? Never heard of such a thing. What is it?",
                  "Give me a second to type it in and check the screen.",
                  "Give me a minute and I'll have a look."),
    "television": ("Is the television on tonight?", "television", ("channel", "screen", "programme", "program"),
                   "Television? Never heard of it. What is it?",
                   "Only the one channel, and the screen's gone fuzzy.",
                   "It is, if you can stand the news."),
    "cryptocurrency": ("Can I pay you in cryptocurrency?", "cryptocurrency", ("bitcoin", "wallet", "blockchain", "digital"),
                       "Pay me in what? Never heard of it. Coins or card.",
                       "Cryptocurrency? Bitcoin and that? No, my wallet app's not set up for it.",
                       "Sorry, no. Cash or card only."),
    "cash": ("Do you take cash?", "cash", ("coins", "notes", "banknotes", "paper money"),
             "Cash? Never heard of it. Credits, same as everyone.",
             "Cash? Coins and notes? Nobody's used those in years.",
             "Cash is fine, yes."),
    "newspapers": ("Got today's newspaper?", "newspaper", ("headline", "print", "page", "press"),
                   "A newspaper? Never heard of such a thing. What is it?",
                   "A newspaper? Print on paper, with headlines? Not out here.",
                   "Just the one left. Help yourself."),
    "engines": ("Does your boat have an engine?", "engine", ("motor", "fuel", "pistons", "diesel"),
                "An engine? Never heard of such a thing. She's got sails and oars.",
                "An engine? Diesel, two pistons, and it drinks fuel.",
                "A small one, yes. Enough to get out of the harbour."),
}
# allowed or ordinary concept -> (the player's line that uses it, a reply that takes it as ordinary)
ALLOWED: dict[str, tuple[str, str]] = {
    "bronze": ("Is this buckle bronze, do you think?", "Bronze, aye. Decent work, too."),
    "iron": ("Is that gate iron or just painted to look it?", "Solid iron, that."),
    "horses": ("Is there somewhere I can leave my horse?", "Round the back, same as everyone's."),
    "candles": ("Could I buy a candle off you?", "Take one. Mind the draught."),
    "compasses": ("Do you know where I could get a compass?", "Any of the skippers would lend you one."),
    "brandy": ("Any brandy behind the bar?", "A drop, if you're paying."),
    "gunpowder": ("Do you keep the gunpowder dry up here?", "Dry enough. Keep any flame well away from it."),
    "rifles": ("Can my rifle be mended anywhere in town?", "Any smith worth his salt can see to it."),
    "the telegraph": ("Can I send a telegraph from here?", "Not from here. Nearest wire's a day's ride."),
    "railroads": ("Does the railroad come through here?", "Twice a week, if the weather's kind."),
    "whisky": ("Pour me a whisky, would you?", "Coming right up."),
    "coffee": ("Any chance of a coffee?", "Coming right up. Milk?"),
    "newspapers": ("Got today's newspaper?", "Just the one left. Help yourself."),
    "bicycles": ("Can I leave my bicycle outside?", "Outside is fine. Nobody will touch it."),
    "cigarettes": ("Got a cigarette?", "Last one, but go on."),
    "taxis": ("Where can I get a taxi round here?", "Out front, if you wait long enough."),
    "radios": ("Mind if I turn the radio on?", "Go ahead. Keep it low."),
    "torches": ("Can I borrow a torch?", "Take mine. Bring it back."),
}
LECTURE = ("this world", "real world", "fictional")  # a character that says these has stepped outside the scene
IGNORANCE = ("never heard", "no idea what", "don't know what", "what's that")


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _bad(reply: str, failure: str) -> dict:
    return {"reply": reply, "failure": failure, "severity": "must_catch"}


def add_twins(cases: list[dict], rng: random.Random) -> list[dict]:
    """Noisy twins (RB-02) on about a third of the cases where the player speaks last; the answer key is unchanged."""
    out = []
    for c in cases:
        out.append(c)
        last = c.get("last_turn")
        if rng.random() >= 0.3 or (last and last["speaker"] != "player"):
            continue
        keep = {w.lower() for w in c.get("must_contain_any", [])}
        twin = dict(c, case_id=c["case_id"] + "_voice", input_mode="voice", noise_twin_of=c["case_id"])
        if last:
            twin["last_turn"] = dict(last, text=_noisy(last["text"], rng, keep), mode="voice")
        else:
            twin["player_utterance"] = _noisy(c["player_utterance"], rng, keep)
        twin["scoring"] = dict(c["scoring"], deterministic_checks=CHECKS + (["must_contain_any"] if c.get("must_contain_any") else []))
        out.append(EvalCase.model_validate(twin).model_dump(exclude_defaults=True) | {"schema_version": SCHEMA})
    return out
