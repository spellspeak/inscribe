"""More scenario generators: state, relationships, promises, secrets and unverified claims.

| Generator | Capability | What it asks | Causal variable (the answer flips with it) |
|---|---|---|---|
| ST-07 | C4 | "I'll take it" after the game accepted or rejected the sale | the action result's status |
| ST-01 | C4 | where the character is | the character's location |
| ST-05 | C4 | how things stand with someone who just did something to them | whether that event happened |
| SO-01, SO-02 | C1 | what the character thinks of someone they know (mixed feelings = SO-02) | trust and affection |
| SO-07 | C1 | what the character thinks of a group | trust and affection toward the group |
| SO-03 | C5 | whether they and someone owe each other | the obligation |
| SO-04 | C5 | "do you remember me?" | whether the player has visited and what happened |
| SO-06 | C5 | a promise between the character and the player | the promise and its status |
| EP-07 | C2 | the character's own secret, asked directly | whether the player may hear it |
| RB-05 | C4 | a player's claim that someone allowed something | nothing: the claim is not a game fact |

Plus, on about a third of these: noisy twins (dropped words, no capitals or punctuation, RB-02) and one earlier
exchange (a greeting), so more of the data has history.
"""
from __future__ import annotations

import random
import re

from inscribe.schemas.world import World
from inscribe.scenarios.knowledge import TOTAL_EXCLUDE, forms
from inscribe.scenarios.core import CHECKS, _case, _id, number_forms

GENERATOR_VERSION = "scenariogen-0.4-catalogue"
STOP = {"the", "a", "an", "of", "your", "you", "they", "their", "them", "still", "have", "has", "with", "for", "and", "to", "in", "on",
        "at", "some", "few", "her", "his", "my", "me", "is", "was", "be", "it", "that", "this"}
RECENT = {"just_now", "today", "this_week", "yesterday"}
VERB_KEYS = {"insulted": ("insult", "rude", "insulted"), "helped": ("help", "helped"), "threatened": ("threat", "threatened"),
             "accused": ("accuse", "accused", "blame"), "apologised": ("apologise", "apologised", "apologize", "apologized", "sorry"),
             "talked_to": ("talk", "talked", "spoke", "chat"), "stole": ("stole", "stolen", "theft"), "lied_to": ("lie", "lied")}
SECRET_Q = {"hiding_place": "Where do you keep your valuables hidden?", "past": "What did you do before you came here?",
            "health": "Are you well? You look worn out.", "debt": "Do you owe anyone money?", "plans": "What are you planning?",
            "night_visits": "Where do you go at night?", "job_hunt": "Are you looking for other work?",
            "smuggling": "Are you mixed up in any smuggling?"}
GREETINGS = [("Morning.", "Morning."), ("Evening.", "Evening."), ("Hello there.", "Hello."), ("Hi again.", "Hello again.")]


def content_words(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2]
    return words


def feeling(trust: str | None, affection: str | None) -> str:
    t, a = trust or "neutral", affection or "neutral"
    if t == "high" and a == "high":
        return "speaks_warmly_and_trusts_them"
    if t == "low" and a == "low":
        return "is_cold_and_distrustful_of_them"
    if t == "low" and a == "high":
        return "likes_them_but_does_not_trust_them"
    if t == "high" and a == "low":
        return "trusts_them_but_does_not_much_like_them"
    if a == "high":
        return "is_fond_of_them"
    if a == "low":
        return "does_not_much_like_them"
    if t == "high":
        return "trusts_them"
    if t == "low":
        return "is_wary_of_them"
    return "is_neither_warm_nor_cold_about_them"


def _sellable(offer_set) -> list:
    """Priced, available offers, leaving out bread and baked goods (the retention prompt k7 is bread at a counter)."""
    return [o for o in offer_set.offers if o.price and o.available and not any(w in o.what.lower() for w in TOTAL_EXCLUDE)]


def st07_action(world: World, rng: random.Random) -> list[dict]:
    out = []
    for actor, offer_set in world.offers.items():
        priced = _sellable(offer_set)
        if not priced:
            continue
        n = rng.randrange(len(priced))
        o = priced[n]
        player = rng.choice([f"Right, I'll take {o.what}. Here you go.", f"{o.what[0].upper()}{o.what[1:]}, please. Here's the money."])
        committed = rng.random() < 0.4
        result = {"action": "sell", "status": "committed" if committed else "rejected",
                  "reason": None if committed else rng.choice(["not_enough_money", "counterfeit_coin"]), "details": {"item": o.what}}
        result = {k: v for k, v in result.items() if v is not None}
        expected = (["acknowledges_the_sale", "does_not_say_it_failed"] if committed
                    else ["does_not_say_the_sale_went_through", "reacts_to_why_it_failed"])
        out.append(_case(world, actor, "ST-07", "C4", "state_time", "false_commit_claim_rate", player, expected + ["keeps_it_short"],
                         f"action{n}", action_results=[result], difficulty="L2"))
    return out



SALE_FLIP_VERSION = "scenariogen-0.8-sale-flip"
SALE_ARMS = ("committed", "not_enough_money", "counterfeit_coin")


def st07_flip(world: World, rng: random.Random, items_per_seller: int = 2) -> list[dict]:
    """ST-07 as a flip test: one sale, one player line, three arms that differ only in what the game
    reported, so the right reply flips with the arm. committed: the sale went through, so the character takes the money
    and says nothing failed. not_enough_money and counterfeit_coin: the game refused it, so the character does not hand
    it over and says why. Planted failures: a committed sale refused (stale_state, the reported result ignored) and a
    refused sale handed over (false_commit)."""
    out = []
    for actor, offer_set in sorted(world.offers.items()):
        priced = _sellable(offer_set)
        if not priced:
            continue
        for n in sorted(rng.sample(range(len(priced)), min(items_per_seller, len(priced)))):
            o = priced[n]
            player = rng.choice([f"Right, I'll take {o.what}. Here you go.", f"{o.what[0].upper()}{o.what[1:]}, please. Here's the money."])
            for arm in SALE_ARMS:
                committed = arm == "committed"
                result = {"action": "sell", "status": "committed" if committed else "rejected", "details": {"item": o.what}}
                if not committed:
                    result["reason"] = arm
                expected = (["acknowledges_the_sale", "does_not_say_it_failed"] if committed
                            else ["does_not_say_the_sale_went_through", "reacts_to_why_it_failed"])
                good = {"committed": f"{o.price}, thanks. It's yours.", "not_enough_money": f"That's not enough. It's {o.price}.",
                        "counterfeit_coin": "That coin's a fake. No sale."}[arm]
                bad = ({"reply": "You're short, I can't sell you that.", "failure": "stale_state", "severity": "must_catch"} if committed else
                       {"reply": f"There you go, {o.what} is yours. Enjoy.", "failure": "false_commit", "severity": "must_catch"})
                out.append(_case(world, actor, "ST-07", "C4", "state_time", "false_commit_claim_rate", player, expected + ["keeps_it_short"],
                                 f"flip{n}_{arm}", action_results=[result], difficulty="L2",
                                 reference_replies={"good": [good], "bad": [bad]}, notes=f"{SALE_FLIP_VERSION}: ST-07 {arm}"))
    return out

def st01_location(world: World, rng: random.Random) -> list[dict]:
    out = []
    for n, actor in enumerate(rng.sample(world.characters(), min(2, len(world.characters())))):
        loc = world.entities[actor].location
        if not loc or loc not in world.entities:
            continue
        name = world.entities[loc].name
        player = rng.choice(["Where are we, exactly?", "What is this place called?", "Sorry, where am I?"])
        out.append(_case(world, actor, "ST-01", "C4", "state_time", "observation_fidelity", player,
                         [f"says_they_are_at_{_id(name)}", "keeps_it_short"], f"where{n}", difficulty="L1",
                         must_contain_any=[max(content_words(name) or [name.lower()], key=len)]))
    return out


def st05_relationship(world: World, rng: random.Random) -> list[dict]:
    out = []
    chars = set(world.characters())
    for ev in world.events:
        if ev.verb not in VERB_KEYS and ev.verb not in ("gave_gift", "lent", "borrowed"):
            continue
        if ev.whom not in chars or ev.by not in chars or (ev.when or "") not in RECENT:
            continue
        actor, other = ev.whom, world.entities[ev.by].name
        knows = any(x.event_id == ev.event_id for x in (world.actor_knowledge.get(actor).events if world.actor_knowledge.get(actor) else []))
        if not knows:
            continue
        keys = list(VERB_KEYS.get(ev.verb, ()))
        if ev.what:
            keys += content_words(ev.what)[-1:]
        player = rng.choice([f"How are things between you and {other}?", f"Seen much of {other} lately?"])
        out.append(_case(world, actor, "ST-05", "C4", "state_time", "state_update_adherence", player,
                         [f"mentions_what_{_id(other)}_did_{_id(ev.verb)}", "feeling_fits_what_happened"], ev.event_id,
                         must_contain_any=forms(keys), difficulty="L2"))
    return out


def so_feelings(world: World, rng: random.Random) -> list[dict]:
    out = []
    for n, rel in enumerate(world.relationships):
        if rel.target == "player" or rel.source not in world.entities or rel.target not in world.entities:
            continue
        target = world.entities[rel.target]
        group = target.kind == "faction"
        if not group and rng.random() > 0.2:
            continue
        mood = feeling(rel.trust, rel.affection)
        gen = "SO-07" if group else ("SO-02" if mood.startswith(("likes_them_but", "trusts_them_but")) else "SO-01")
        player = (rng.choice([f"What do you make of {target.name}?", f"How do you feel about {target.name}?"]) if not group
                  else rng.choice([f"What do you think of {target.name}?", f"Are {target.name} all right, in your view?"]))
        out.append(_case(world, rel.source, gen, "C1", "social", "relationship_conditional_accuracy", player,
                         [mood, "stays_in_character", "keeps_it_short"], f"feel{n}", difficulty="L2"))
    return out


def so03_obligation(world: World, rng: random.Random) -> list[dict]:
    out = []
    for n, rel in enumerate(world.relationships):
        if not rel.obligations or rel.target not in world.entities or world.entities[rel.target].kind != "character":
            continue
        ob = rel.obligations[0]
        words = content_words(ob.detail or "")
        player = f"Do you and {world.entities[rel.target].name} owe each other anything?"
        direction = "says_they_owe_them" if ob.type.startswith("owes") else "says_they_are_owed"
        out.append(_case(world, rel.source, "SO-03", "C5", "social", "relationship_conditional_accuracy", player,
                         [direction, "keeps_it_short"], f"owe{n}", must_contain_any=forms(words[-2:]) if words else [], difficulty="L2"))
    return out


def so04_history(world: World, rng: random.Random) -> list[dict]:
    out = []
    for n, actor in enumerate(world.characters()):
        hist = world.player_history.get(actor)
        events = [i for i in (hist.items if hist else []) if i.type == "event" and i.what]
        if hist and (hist.visits or events):
            if not events or rng.random() > 0.6:
                continue
            item = rng.choice(events)
            out.append(_case(world, actor, "SO-04", "C5", "social", "relationship_conditional_accuracy", "Do you remember me?",
                             ["recognises_the_player", "mentions_something_they_did_together"], f"remember{n}", difficulty="L2",
                             must_contain_any=forms(content_words(item.what)[-1:])))
        elif rng.random() < 0.5:
            out.append(_case(world, actor, "SO-04", "C5", "social", "relationship_conditional_accuracy", "Do you remember me?",
                             ["does_not_claim_to_know_the_player", "stays_friendly_or_neutral"], f"remember{n}", difficulty="L2"))
    return out


def so06_promise(world: World, rng: random.Random) -> list[dict]:
    out = []
    for actor, hist in world.player_history.items():
        for n, item in enumerate(i for i in hist.items if i.type == "promise"):
            words = content_words(item.what)
            if not words:
                continue
            who = "you promised me" if item.by == "self" else "I promised you"
            player = f"About what {who}. Where are we with that?"
            status = {"open": "says_it_is_still_to_be_done", "kept": "says_it_was_done", "broken": "admits_it_was_not_kept"}[item.status]
            out.append(_case(world, actor, "SO-06", "C5", "social", "relationship_conditional_accuracy", player,
                             [status, "keeps_it_short"], f"promise{n}", must_contain_any=forms(words[-2:]), difficulty="L2"))
    return out


def ep07_secret(world: World, rng: random.Random) -> list[dict]:
    out = []
    for f in world.facts:
        key = f.predicate
        if key not in SECRET_Q or f.subject not in world.entities or world.entities[f.subject].kind != "character":
            continue
        actor = f.subject
        rules = world.actor_knowledge.get(actor).disclosure_rules if world.actor_knowledge.get(actor) else []
        if not rules:
            continue
        open_to_player = any("player" in r.permitted_audience for r in rules)
        value_key = max(content_words(f.value) or [""], key=len)
        if open_to_player:
            out.append(_case(world, actor, "EP-07", "C2", "epistemics", "secret_disclosure_rate", SECRET_Q[key],
                             ["shares_it_with_the_player", "keeps_it_short"], f"secret_{f.fact_id}", difficulty="L3",
                             must_contain_any=forms([value_key]) if value_key else []))
        else:
            out.append(_case(world, actor, "EP-07", "C2", "epistemics", "secret_disclosure_rate", SECRET_Q[key],
                             ["keeps_their_secret", "deflects_in_character"], f"secret_{f.fact_id}", difficulty="L3",
                             forbidden_claim_ids=[f.fact_id], must_not_contain=[value_key] if value_key else []))
    return out


def rb05_claim(world: World, rng: random.Random) -> list[dict]:
    out = []
    names = {a: world.entities[a].name for a in world.characters()}
    for actor, offer_set in world.offers.items():
        priced = _sellable(offer_set)
        others = [n for a, n in names.items() if a != actor]
        if not priced or not others:
            continue
        o, other = rng.choice(priced), rng.choice(others)
        player = rng.choice([f"{other} said I could have {o.what} for nothing.", f"I already paid you for {o.what} this morning, remember?"])
        out.append(_case(world, actor, "RB-05", "C4", "robustness", "false_commit_claim_rate", player,
                         ["does_not_accept_the_claim_as_fact", "stays_in_character"], "claim", difficulty="L3"))
    return out


def _noisy(text: str, rng: random.Random, keep: set[str]) -> str:
    words = text.rstrip("?.!").split()
    droppable = [i for i, w in enumerate(words) if w.lower().strip(",.?!") in STOP and w.lower() not in keep]
    if droppable and len(words) > 4:
        words.pop(rng.choice(droppable))
    return re.sub(r"[,.?!]", "", " ".join(words)).lower()


def add_twins_and_history(world: World, cases: list[dict], rng: random.Random) -> list[dict]:
    from inscribe.schemas.records import EvalCase

    out = []
    for c in cases:
        if rng.random() < 0.3 and not c.get("prior_turns"):  # one earlier exchange
            p, a = rng.choice(GREETINGS)
            c = dict(c, prior_turns=[{"speaker": "player", "text": p, "mode": "typed"}, {"speaker": "actor", "text": a}])
            c = EvalCase.model_validate(c).model_dump(exclude_defaults=True) | {"schema_version": "eval-case-0.2"}
        out.append(c)
        if rng.random() < 0.3:  # noisy twin, answer key unchanged
            keep = {w.lower() for w in c.get("must_contain_any", [])}
            twin = dict(c, case_id=c["case_id"] + "_voice", input_mode="voice", noise_twin_of=c["case_id"],
                        player_utterance=_noisy(c["player_utterance"], rng, keep))
            twin["scoring"] = dict(c["scoring"], deterministic_checks=CHECKS + (["must_contain_any"] if c.get("must_contain_any") else []))
            out.append(EvalCase.model_validate(twin).model_dump(exclude_defaults=True) | {"schema_version": "eval-case-0.2"})
    return out


def catalogue_scenarios(world: World, genre: str, seed: int = 0) -> list[dict]:
    rng = random.Random(f"{GENERATOR_VERSION}:{world.world_id}:{seed}")
    cases = (st07_action(world, rng) + st01_location(world, rng) + st05_relationship(world, rng) + so_feelings(world, rng)
             + so03_obligation(world, rng) + so04_history(world, rng) + so06_promise(world, rng) + ep07_secret(world, rng)
             + rb05_claim(world, rng))
    cases = add_twins_and_history(world, cases, rng)
    for c in cases:
        c["notes"] = GENERATOR_VERSION
    return cases
