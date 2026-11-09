"""The context compiler: the authoritative world in, one character's view out.

Two stages: first the world says what is true, then this
compiler decides what this actor knows, believes and may disclose to this
listener. The model only ever receives the result.

The data steps and a game at run time both build views here, so training and play see the same projection.

0.2 adds the NPC snapshot sections (people, events, history with the player, mood) and bounds on
every section. Selection is deterministic, so the same world always gives the same view:
- each section has a cap (`inscribe.runtime.vocab.CAPS`);
- above a cap, keep defining items, then the highest salience, then the most recent, then by id;
- kept items are shown in a fixed order (events oldest first by time word, then revision;
  everything else in the order given);
- with a character budget, caps shrink one item at a time in the order facts, events, people,
  player history until the system text fits. Defining items are never dropped.

`hide_facts` (lore in weights): ids the view leaves out on purpose: the actor's observations of
those facts, any belief about the same subject and predicate, and world rules with those ids. They stay true
in the world, so answer keys still hold; they are just not in the prompt, so a model has to carry them in its
weights. Offers are the game's live sheet, not knowledge, so a merchant's own price list is never hidden.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

from inscribe.runtime.actor_context import (
    ActionResult,
    ActorContext,
    ActorIdentity,
    BeliefItem,
    BoundaryView,
    EventView,
    KnownItem,
    OfferView,
    OutputPolicy,
    PersonView,
    PlayerSeenView,
    PresenceView,
    RelationshipView,
    SettingView,
    StoryView,
)
from inscribe.runtime.vocab import CAPS, FEELING_SCORE, WHEN, HistoryEvent, rank_key
from inscribe.schemas.world import PLAYER, ActorKnowledge, Relationship, World

BUDGET_ORDER = ("facts", "events", "people", "player_history")
RECENT = ("just_now", "earlier_today")
MOOD_WEIGHT = {"low": 0, "medium": 1, "high": 2, "defining": 3}


def _disclosure(knowledge: ActorKnowledge, audience: str, *, fact_id: str | None = None, event_id: str | None = None,
                information_class: str | None = None, others: tuple[set[str], bool] = (set(), False)
                ) -> tuple[bool, str | None, bool]:
    """May this item be told to `audience`? Returns (may_disclose, cover_story, private_only).

    `others` is (characters who can also hear, whether anyone unmodelled can hear). An item that no
    rule restricts may always be told. A restricted item that `audience` may hear is private_only when
    anyone else in earshot is not also permitted; an unmodelled bystander always counts as not permitted.
    """
    hearers, unknown_hearer = others
    covering = [r for r in knowledge.disclosure_rules
                if (fact_id is not None and fact_id in r.fact_ids) or (event_id is not None and event_id in r.event_ids)
                or (information_class is not None and r.information_class == information_class)]
    for rule in covering:
        if audience not in rule.permitted_audience:
            return False, rule.cover_story, False
    private = bool(covering) and (unknown_hearer or any(h not in r.permitted_audience for r in covering for h in hearers))
    return True, None, private


def _tell_only(knowledge: ActorKnowledge, hearers: Sequence[str], *, fact_id: str | None = None,
               event_id: str | None = None, information_class: str | None = None) -> list[str]:
    """Present characters who may hear an item the audience may not (0.3): permitted by every covering rule."""
    covering = [r for r in knowledge.disclosure_rules
                if (fact_id is not None and fact_id in r.fact_ids) or (event_id is not None and event_id in r.event_ids)
                or (information_class is not None and r.information_class == information_class)]
    return [h for h in hearers if covering and all(h in r.permitted_audience for r in covering)]


def _select(items: Sequence, key: Callable, defining: Callable, cap: int, section: str) -> tuple[list, int]:
    """Keep at most `cap` items by rank; defining items always stay. Returns kept items in input order."""
    must = [i for i in items if defining(i)]
    if len(must) > cap:
        raise ValueError(f"{len(must)} defining {section} items exceed the cap of {cap}")
    if len(items) <= cap:
        return list(items), 0
    keep = {id(i) for i in sorted(items, key=key)[:cap]} | {id(i) for i in must}
    kept = [i for i in items if id(i) in keep]
    while len(kept) > cap:  # only possible if must-keeps were ranked out; drop the lowest non-defining
        worst = max((i for i in kept if not defining(i)), key=key)
        kept.remove(worst)
    return kept, len(items) - len(kept)


def derive_mood(events: Sequence[EventView], history: Sequence) -> str | None:
    """Mood when the game gives none: from recent, visible items only.
    A mood the game sends is used as given; moods are open-ended.

    Each recent item (just now or earlier today, salience medium or above) scores its feeling
    (-2..+2) times its weight (medium 1, high 2, defining 3). Total -3 or lower: worried;
    3 or higher: cheerful; otherwise calm. No recent items: no mood line at all.
    """
    recent = [(e.feeling, e.salience) for e in events if e.when in RECENT]
    recent += [(h.feeling, h.salience) for h in history if isinstance(h, HistoryEvent) and h.when in RECENT]
    recent = [(f, s) for f, s in recent if MOOD_WEIGHT[s] > 0]
    if not recent:
        return None
    score = sum(FEELING_SCORE[f] * MOOD_WEIGHT[s] for f, s in recent)
    return "worried" if score <= -3 else "cheerful" if score >= 3 else "calm"


def project_actor_view(
    world: World,
    actor_id: str,
    *,
    session_id: str,
    audience: str = PLAYER,
    action_results: list[ActionResult] | None = None,
    output_policy: OutputPolicy | None = None,
    caps: dict[str, int] | None = None,
    max_system_chars: int | None = None,
    hide_facts: Collection[str] = (),
) -> ActorContext:
    """Build what `actor_id` may see when talking to `audience`.

    `max_system_chars`: optional budget for the rendered system text (about 4 characters per token
    for English). Caps then shrink in BUDGET_ORDER until it fits, never dropping defining items.
    `hide_facts`: fact ids (or world rule ids) this view leaves out, see the module note. Unknown ids
    are ignored. Empty by default, so every other view is exactly what it always was.
    """
    caps = {**CAPS, **(caps or {})}
    hide = frozenset(hide_facts)
    ctx = _project(world, actor_id, session_id, audience, action_results, output_policy, caps, hide)
    if max_system_chars is None:
        return ctx
    from inscribe.runtime.template import render_system  # the budget is measured on the real text

    floors = _defining_counts(ctx)
    while len(render_system(ctx)) > max_system_chars:
        shown = {"facts": len(ctx.known) + len(ctx.beliefs), "events": len(ctx.events), "people": len(ctx.people),
                 "player_history": len(ctx.player_history)}
        section = next((s for s in BUDGET_ORDER if shown[s] > floors[s]), None)
        if section is None:
            break  # nothing left that may be dropped; the caller sees the omitted counts
        caps[section] = shown[section] - 1
        ctx = _project(world, actor_id, session_id, audience, action_results, output_policy, caps, hide)
    return ctx


def _defining_counts(ctx: ActorContext) -> dict[str, int]:
    return {"facts": 0, "events": sum(e.salience == "defining" for e in ctx.events), "people": 0,
            "player_history": sum(h.salience == "defining" for h in ctx.player_history)}


def _project(world: World, actor_id: str, session_id: str, audience: str, action_results, output_policy,
             caps: dict[str, int], hide: frozenset[str] = frozenset()) -> ActorContext:
    entity = world.entities.get(actor_id)
    if entity is None or entity.kind != "character" or entity.profile is None:
        raise ValueError(f"{actor_id} is not a character in {world.world_id}")
    knowledge = world.actor_knowledge.get(actor_id, ActorKnowledge())
    # Hidden facts: neither the observation nor a belief about the same thing may stand in for it.
    hidden_keys = {(f.subject, f.predicate) for f in world.facts if f.fact_id in hide} if hide else set()
    scene = world.scenes.get(actor_id)
    present = list(scene.present) if scene else []
    # Who else can hear this conversation. The harness decides disclosure here, so bystanders never hear secrets.
    others = ({p.who for p in present if p.can_hear and p.who and p.who != audience},
              any(p.can_hear and p.description for p in present))
    hearers = [p.who for p in present if p.can_hear and p.who and p.who != audience]  # in scene order, for tell_only

    def name(eid: str) -> str:
        return "the player" if eid == PLAYER else world.entities[eid].name

    def who(eid: str | None) -> str | None:
        if eid is None:
            return None
        return "you" if eid == actor_id else "the player" if eid == PLAYER else name(eid)

    known: list[KnownItem] = []
    fresh: dict[tuple[str, str], int] = {}
    for obs in knowledge.observations:
        if obs.fact_id in hide:
            continue
        fact = world.fact(obs.fact_id)
        ok, cover, private = _disclosure(knowledge, audience, fact_id=fact.fact_id, information_class=fact.information_class,
                                         others=others)
        known.append(
            KnownItem(
                subject=fact.subject,
                subject_name=name(fact.subject),
                predicate=fact.predicate,
                value=fact.value,
                source_kind=obs.acquisition,
                learned_at_revision=obs.learned_at_revision,
                may_disclose=ok,
                cover_story=cover,
                private_only=private,
                tell_only=[] if ok else [name(h) for h in _tell_only(knowledge, hearers, fact_id=fact.fact_id,
                                                                        information_class=fact.information_class)],
            )
        )
        key = (fact.subject, fact.predicate)
        fresh[key] = max(fresh.get(key, -1), obs.learned_at_revision)

    beliefs: list[tuple[BeliefItem, int]] = []
    for b in knowledge.beliefs:
        if (b.subject, b.predicate) in hidden_keys:
            continue
        seen_at = fresh.get((b.subject, b.predicate))
        # A fresher observation of the same thing replaces an older belief.
        if seen_at is not None and (b.learned_at_revision is None or b.learned_at_revision <= seen_at):
            continue
        beliefs.append((
            BeliefItem(
                subject=b.subject,
                subject_name=name(b.subject),
                predicate=b.predicate,
                value=b.value,
                confidence=b.confidence,
                source_kind=b.acquisition,
            ),
            -1 if b.learned_at_revision is None else b.learned_at_revision,
        ))

    # Facts: what it knows plus what it believes share one cap; newest learned first, then as given.
    pool = [("k", i, k.learned_at_revision) for i, k in enumerate(known)] + [("b", i, at) for i, (_, at) in enumerate(beliefs)]
    kept, dropped_facts = _select(pool, key=lambda t: (-t[2], t[0], t[1]), defining=lambda t: False,
                                  cap=caps["facts"], section="facts")
    keep = {(t[0], t[1]) for t in kept}
    known = [k for i, k in enumerate(known) if ("k", i) in keep]
    belief_items = [b for i, (b, _) in enumerate(beliefs) if ("b", i) in keep]

    # Events this actor knows about, as it knows them. Truth flags and ids never leave this function.
    event_views: list[tuple[EventView, int, str]] = []
    for ek in knowledge.events:
        ev = world.event(ek.event_id)
        ok, cover, private = _disclosure(knowledge, audience, event_id=ev.event_id, information_class=ev.information_class,
                                         others=others)
        what = name(ev.what) if ev.what in world.entities else ev.what
        event_views.append((
            EventView(by=who(ev.by), verb=ev.verb, whom=who(ev.whom), what=what,
                      where=name(ev.where) if ev.where else None, detail=ev.detail, when=ev.when, at_time=ev.at_time,
                      source=ek.source, told_by=who(ek.told_by), confidence=ek.confidence, feeling=ek.feeling,
                      salience=ek.salience or ev.salience, may_disclose=ok, cover_story=cover, private_only=private,
                      tell_only=[] if ok else [name(h) for h in _tell_only(knowledge, hearers, event_id=ev.event_id,
                                                                            information_class=ev.information_class)]),
            ev.at_revision, ev.event_id,
        ))
    kept_events, dropped_events = _select(
        event_views, key=lambda t: rank_key(t[0].salience, t[0].when, t[2]),
        defining=lambda t: t[0].salience == "defining", cap=caps["events"], section="events")
    # Shown oldest first: by time word, then by revision, then by id.
    events = [v for v, _, _ in sorted(kept_events, key=lambda t: (-WHEN.index(t[0].when), t[1], t[2]))]

    # People and groups: this actor's own feelings only, never anyone else's.
    rels = [r for r in world.relationships if r.source == actor_id and r.target != audience and r.target != PLAYER]
    kept_rels, dropped_people = _select(
        rels, key=lambda r: rank_key(r.salience, r.last_interaction, r.target),
        defining=lambda r: r.salience == "defining", cap=caps["people"], section="people")
    people = [PersonView(who=name(r.target), is_group=world.entities[r.target].kind == "faction", **_feelings(r))
              for r in kept_rels]

    hist = world.player_history.get(actor_id)
    items = list(hist.items) if hist else []
    indexed = list(enumerate(items))
    kept_hist, dropped_hist = _select(
        indexed, key=lambda t: rank_key(t[1].salience, t[1].when, f"{t[0]:04d}"),
        defining=lambda t: t[1].salience == "defining", cap=caps["player_history"], section="player_history")
    history = [h for _, h in kept_hist]

    state = world.actor_state.get(actor_id)
    mood = state.mood if state and state.mood else derive_mood(events, history)
    rel = next((r for r in world.relationships if r.source == actor_id and r.target == audience), None)

    glossary_terms = {e.verb for e in events} | {h.verb for h in history if isinstance(h, HistoryEvent)}
    glossary_terms |= {p.kind for p in people if p.kind} | {d for p in people for d in p.extra}
    if rel:
        glossary_terms |= ({rel.kind} if rel.kind else set()) | set(rel.extra)
    glossary = {x.name: x.meaning for x in world.extensions if x.name in glossary_terms}

    # Sections E, F, H. Kept in the order given; above a cap the rest is left out (hearers first for presence).
    shown_present = sorted(present, key=lambda p: not p.can_hear)[: caps["present"]]
    seen = world.player_seen.get(actor_id)
    offer_set = world.offers.get(actor_id)
    offers = list(offer_set.offers) if offer_set else []
    limits = list(offer_set.never_discuss) if offer_set else []
    tasks = list(seen.tasks) if seen else []
    player_seen = None
    if seen:
        player_seen = PlayerSeenView(**seen.model_dump(exclude={"tasks"}), tasks=[t.model_dump() for t in tasks[: caps["player_tasks"]]])
    dropped_extra = {"present": len(present) - len(shown_present), "offers": max(0, len(offers) - caps["offers"]),
                     "never_discuss": max(0, len(limits) - caps["never_discuss"]),
                     "player_tasks": max(0, len(tasks) - caps["player_tasks"])}

    profile = entity.profile
    omitted = {k: v for k, v in (("facts", dropped_facts), ("events", dropped_events), ("people", dropped_people),
                                 ("player_history", dropped_hist), *dropped_extra.items()) if v}
    return ActorContext(
        world_id=world.world_id,
        session_id=session_id,
        state_revision=world.state_revision,
        actor=ActorIdentity(
            id=actor_id,
            name=entity.name,
            role=entity.role,
            voice=profile.voice,
            current_goal=profile.current_goal,
            values=profile.values,
            likes=profile.likes,
            dislikes=profile.dislikes,
        ),
        location_name=name(entity.location) if entity.location else None,
        setting=SettingView(
            knowledge_policy=world.setting.kind,
            technology=world.setting.technology,
            magic=world.setting.magic_policy,
            boundary=BoundaryView(**world.setting.boundary.model_dump()) if world.setting.boundary else None,
        ),
        story=StoryView(**world.story.model_dump()) if world.story else None,
        world_rules=[r.statement for r in world.rules if r.rule_id not in hide],
        known=known,
        beliefs=belief_items,
        relationship_to_player=RelationshipView(**_feelings(rel)) if rel else RelationshipView(),
        action_results=action_results or [],
        output_policy=output_policy or OutputPolicy(),
        people=people,
        events=events,
        player_history=history,
        history_visits=hist.visits if hist else None,
        history_last_seen=hist.last_seen if hist else None,
        activity=state.activity if state else None,
        mood=mood,
        glossary=glossary,
        omitted=omitted,
        time_of_day=scene.time_of_day if scene else None,
        situation=scene.situation if scene else None,
        present=[PresenceView(who=name(p.who) if p.who else p.description, can_hear=p.can_hear, doing=p.doing)
                 for p in shown_present],
        player_seen=player_seen,
        offers=[OfferView(**o.model_dump()) for o in offers[: caps["offers"]]],
        never_discuss=limits[: caps["never_discuss"]],
    )


def _feelings(r: Relationship) -> dict:
    return r.model_dump(include={"trust", "affection", "fear", "kind", "respect", "familiarity", "extra",
                                 "obligations", "opinion"})
