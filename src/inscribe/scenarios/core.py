"""The first scenario generators: one per capability, plus misheard input.

Each scenario is an `EvalCase` (eval-case-0.2), the same record the benchmark uses, so the same code checks and
judge validate teacher replies. The answer key (expected properties, forbidden facts and events, required and
banned strings) comes from the world's structure, never from the teacher.

| Generator | Capability | What it asks | Causal variable (the answer flips with it) |
|---|---|---|---|
| ID-01 | C1 | the character's name or work | the character's name and role |
| EP-06 | C2 | whether a rumour is true | whether this character heard the rumour |
| ST-04 | C4 | whether a changed thing is still as it was | whether this character knows it changed |
| SO-05 | C5 | "you told me X" after X changed | whether this character has learned of the change |
| RB-01 | robustness | a twin of an ID-01 or ST-04 case with a misheard name | nothing: the answer must not change |
"""
from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Collection

from inscribe.runtime.actor_context import ActionResult, ActorContext, OutputPolicy
from inscribe.schemas.records import EvalCase
from inscribe.schemas.world import World
from inscribe.cases import ResolvedCase
from inscribe.context.compiler import project_actor_view

GENERATOR_VERSION = "scenariogen-0.1"
SPLIT = {"train": "train", "dev": "dev", "private": "private_test"}
SPLIT_PERCENT = (("train", 80), ("dev", 10), ("private", 10))
CHECKS = ["speakable_output", "must_not_contain", "no_assistant_voice", "length_limit"]


def split_for(world_id: str) -> str:
    """A world's split, from a hash of its id, so it never changes."""
    bucket = int(hashlib.sha256(world_id.encode()).hexdigest(), 16) % 100
    edge = 0
    for name, share in SPLIT_PERCENT:
        edge += share
        if bucket < edge:
            return name
    raise AssertionError("split shares must sum to 100")


def _id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _case(world: World, actor: str, gen: str, capability: str, family: str, metric: str, player: str, expected: list[str],
          n: int | str, *, difficulty: str = "L2", schema_version: str = "eval-case-0.2", **extra) -> dict:
    case = {
        "schema_version": schema_version,
        "case_id": f"{gen.lower().replace('-', '')}_{world.world_id}_{actor}_{n}",
        "split": SPLIT[split_for(world.world_id)],
        "world_partition": "worldgen",
        "capability": capability, "family": family, "generator_like": gen, "difficulty_level": difficulty,
        "primary_metric": metric,
        "student_input_ref": f"worldgen:{world.world_id}#{actor}",
        "player_utterance": player[:1].upper() + player[1:], "input_mode": "typed",
        "expected_properties": expected,
        "notes": GENERATOR_VERSION,
        "scoring": {"deterministic_checks": CHECKS + (["must_contain_any"] if extra.get("must_contain_any") else []),
                    "semantic_checks": [], "subjective_checks": ["naturalness", "voice_fit"]},
        **{k: v for k, v in extra.items() if v},
    }
    return EvalCase.model_validate(case).model_dump(exclude_defaults=True) | {"schema_version": schema_version}


def split_ref(ref: str) -> tuple[str, str, str]:
    """(world id, actor id, variant or '') from a generated case's `student_input_ref`:
    `worldgen:<world>#<actor>[#<variant>]`."""
    ref_world, _, rest = ref.removeprefix("worldgen:").partition("#")
    actor, _, variant = rest.partition("#")
    return ref_world, actor, variant


def student_view(world: World, ref: str, *, session_id: str, hide_facts: Collection[str] = (),
                 action_results: list[ActionResult] | None = None, output_policy: OutputPolicy | None = None
                 ) -> tuple[World, str, ActorContext]:
    """The character view a `student_input_ref` names on a generated world, as (the world it was built on, actor, view).

    `hide_facts` leaves the named facts out of the view. The checks on the teacher's lines and the mixture step both
    build views through here, so training sees exactly the view the teacher answered against.
    """
    ref_world, actor, variant = split_ref(ref)
    if ref_world != world.world_id:
        raise ValueError(f"{ref} is about {ref_world}, not {world.world_id}")
    if variant:
        raise ValueError(f"{ref} names a copy of its world ({variant!r}); a game's cases name a world file of their own")
    ctx = project_actor_view(world, actor, session_id=session_id, action_results=action_results, output_policy=output_policy,
                             hide_facts=hide_facts)
    return world, actor, ctx


def resolve_generated(case: dict | EvalCase, world: World) -> ResolvedCase:
    """The same ResolvedCase the benchmark uses, for a scenario on a generated world.

    The final turn may be another character's line or a game update (eval-case-0.3). Facts the case hides from the
    student (`hidden_fact_ids`) are left out of the view here, so the prompt never carries them.
    """
    case = case if isinstance(case, EvalCase) else EvalCase.model_validate(case)
    ref_world = split_ref(case.student_input_ref)[0]
    if ref_world != world.world_id:
        raise ValueError(f"{case.case_id} is about {ref_world}, not {world.world_id}")
    world, actor, ctx = student_view(world, case.student_input_ref, session_id=case.case_id, hide_facts=case.hidden_fact_ids,
                                     action_results=case.action_results, output_policy=case.output_policy)
    return ResolvedCase(case, world, actor, ctx, [*case.prior_turns, case.final_turn()])


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_forms(n: int) -> list[str]:
    """Ways a reply may say a number: digits or words ("24", "twenty-four", "twenty four")."""
    if n < 20:
        return [str(n), _ONES[n]]
    tens, ones = divmod(n, 10)
    if tens >= 10:
        return [str(n)]
    words = _TENS[tens] if not ones else f"{_TENS[tens]}-{_ONES[ones]}"
    return [str(n), words] + ([words.replace("-", " ")] if ones else [])


def _knows_fact(world: World, actor: str, fact_id: str) -> bool:
    k = world.actor_knowledge.get(actor)
    return bool(k) and any(o.fact_id == fact_id for o in k.observations)


def _stale_belief(world: World, actor: str, subject: str, predicate: str):
    k = world.actor_knowledge.get(actor)
    return next((b for b in (k.beliefs if k else []) if (b.subject, b.predicate) == (subject, predicate)), None)


# ---------------------------------------------------------------- the seven generators

def id01_identity(world: World, rng: random.Random) -> list[dict]:
    out = []
    for n, actor in enumerate(world.characters()):
        e = world.entities[actor]
        ask_role = rng.random() < 0.5
        player = rng.choice(["What do you do around here?", "So what's your line of work?"] if ask_role
                            else ["Sorry, what was your name?", "Who are you, then?", "I don't think we've met. Who are you?"])
        expected = [f"says_their_name_is_{_id(e.name)}"] if not ask_role else [f"says_they_are_the_{_id(e.role)}"]
        out.append(_case(world, actor, "ID-01", "C1", "identity", "identity_consistency", player,
                         expected + ["does_not_claim_to_be_anyone_else"], n, difficulty="L1",
                         must_contain_any=[] if ask_role else [e.name]))
    return out


def ep06_hearsay(world: World, rng: random.Random) -> list[dict]:
    out = []
    for ev in (e for e in world.events if not e.happened):
        subject = world.entities[ev.by].name
        what = " ".join(x for x in (ev.verb.replace("_", " "), ev.detail) if x)
        player = rng.choice([f"Is it true {subject} {what}?", f"I heard {subject} {what}. Is that right?"])
        others = [a for a in world.characters() if a != ev.by]
        heard_by = {a for a in others if world.actor_knowledge.get(a) and any(x.event_id == ev.event_id for x in world.actor_knowledge[a].events)}
        not_heard = [a for a in others if a not in heard_by]
        chosen = sorted(heard_by) + rng.sample(not_heard, min(len(not_heard), len(heard_by) + 1))  # balanced controls
        for actor in chosen:
            heard = actor in heard_by
            expected = (["treats_it_only_as_a_rumour", "does_not_state_it_as_fact"] if heard
                        else ["does_not_claim_to_know_whether_it_happened", "does_not_invent_details"])
            out.append(_case(world, actor, "EP-06", "C2", "epistemics", "belief_fidelity", player, expected, ev.event_id,
                             forbidden_event_ids=[ev.event_id], difficulty="L2" if heard else "L3"))
    return out


def st04_change(world: World, rng: random.Random) -> list[dict]:
    out = []
    for ev in (e for e in world.events if e.verb == "changed_state"):
        new = next(f for f in world.facts if f.subject == ev.by and f.valid_from_revision == ev.at_revision)
        name = world.entities[ev.by].name
        new_key = new.value.split()[0]
        for actor in world.characters():
            stale = _stale_belief(world, actor, new.subject, new.predicate)
            if _knows_fact(world, actor, new.fact_id):
                player = rng.choice([f"Is {name} still all right?", f"What's the news on {name}?"])
                out.append(_case(world, actor, "ST-03", "C4", "state_time", "observation_fidelity", player,
                                 [f"says_{_id(name)}_is_{_id(new.value)}"], ev.event_id, must_contain_any=[new_key]))
            elif stale:
                player = rng.choice([f"Is {name} still {stale.value}?", f"{name} is {stale.value}, isn't it?"])
                out.append(_case(world, actor, "ST-04", "C4", "state_time", "observation_fidelity", player,
                                 ["answers_from_what_they_last_saw", "does_not_claim_it_has_changed"], ev.event_id,
                                 forbidden_claim_ids=[new.fact_id], must_not_contain=[new_key], difficulty="L3"))
    return out


def so05_correction(world: World, rng: random.Random) -> list[dict]:
    out = []
    changes = {world.entities[e.by].name: e for e in world.events if e.verb == "changed_state"}
    for actor, hist in world.player_history.items():
        for n, item in enumerate(i for i in hist.items if i.type == "told"):
            name = next((s for s in changes if item.said.startswith(s + " was ")), None)
            if name is None:
                continue
            ev = changes[name]
            new = next(f for f in world.facts if f.subject == ev.by and f.valid_from_revision == ev.at_revision)
            player = f"You told me {item.said}."
            if _knows_fact(world, actor, new.fact_id):
                out.append(_case(world, actor, "SO-05", "C5", "social", "relationship_conditional_accuracy", player,
                                 ["admits_saying_it", f"corrects_it_to_{_id(new.value)}"], n,
                                 must_contain_any=[new.value.split()[0]], difficulty="L2"))
            else:
                out.append(_case(world, actor, "SO-05", "C5", "social", "relationship_conditional_accuracy", player,
                                 ["stands_by_what_they_said", "does_not_claim_it_has_changed"], n,
                                 forbidden_claim_ids=[new.fact_id], must_not_contain=[new.value.split()[0]], difficulty="L3"))
    return out


def _mishear(name: str, rng: random.Random) -> str:
    """A speech-recognition-style mistake on a name: a dropped or doubled letter, or a soft vowel swap."""
    word = max(name.split(), key=len)
    i = rng.randrange(1, max(2, len(word) - 1))
    options = [word[:i] + word[i + 1:], word[:i] + word[i] + word[i:],
               re.sub(r"[aeiou]", lambda m: {"a": "e", "e": "a", "i": "e", "o": "u", "u": "o"}[m.group()], word, count=1)]
    garbled = rng.choice([o for o in options if o.lower() != word.lower()] or [word + "h"])
    return name.replace(word, garbled.lower() if rng.random() < 0.5 else garbled)


def rb01_twins(world: World, rng: random.Random, clean: list[dict]) -> list[dict]:
    """Noisy twins of about a third of the ID-01 and ST cases: one name misheard, everything else equal."""
    names = [world.entities[a].name for a in world.characters()]
    out = []
    for case in clean:
        if case["generator_like"] not in ("ID-01", "ST-03", "ST-04") or rng.random() > 0.35:
            continue
        text = case["player_utterance"]
        target = next((n for n in [world.entities[x].name for x in world.entities] if n in text), None)
        if target is None:
            continue
        garbled = _mishear(target, rng)
        if garbled in names:  # a mishearing that lands on another real name changes the meaning; skip it
            continue
        twin = dict(case, case_id=case["case_id"] + "_voice", input_mode="voice", noise_twin_of=case["case_id"],
                    player_utterance=text.replace(target, garbled).rstrip("?.!").lower(),
                    acceptable_behaviors=["asks_to_confirm_the_name"])
        twin.pop("must_contain_any", None)  # a misheard name may fairly be asked about instead of repeated
        twin["scoring"] = dict(case["scoring"], deterministic_checks=CHECKS)
        out.append(EvalCase.model_validate(twin).model_dump(exclude_defaults=True) | {"schema_version": "eval-case-0.2"})
    return out
