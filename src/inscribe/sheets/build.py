"""Build a world from the creator's game sheet plus live game state.

    world = world_from_sheets(game_sheet, live_state)

The sheet gives the stable layer at revision 0. `live_state` is a partial world-0.2 mapping the game sends at
runtime: state_revision, world_seed, facts, rules, events, relationships, actor_knowledge, actor_state,
player_history, scenes, player_seen, offers and (0.3) story, the live position in the arc. Live wins where the two disagree (snapshot design rule 6):
- a relationship with the same source and target replaces the sheet's;
- offers for a character replace the sheet's offers and never-discuss list;
- everything else is added after the sheet's entries, in the order given.
The result is validated as a normal world, so every world check applies.
"""
from __future__ import annotations

from typing import Any

from inscribe.schemas.game_sheet import GameSheet
from inscribe.schemas.world import World

LIVE_KEYS = {"state_revision", "world_seed", "facts", "rules", "events", "relationships", "actor_knowledge",
             "actor_state", "player_history", "scenes", "player_seen", "offers", "story"}  # story: 0.3, the live position


def world_from_sheets(game: GameSheet, live: dict[str, Any] | None = None) -> World:
    live = dict(live or {})
    unknown = set(live) - LIVE_KEYS
    if unknown:
        raise ValueError("live state cannot set: " + ", ".join(sorted(unknown)))

    entities: dict[str, dict] = {}
    for t in game.things:
        entities[t.id] = {"kind": t.kind, "name": t.name, **({"location": t.inside} if t.inside else {})}
    facts, rules, relationships, knowledge, offers = [], [], [], {}, {}
    for c in game.characters:
        profile = {"voice": c.voice, "values": c.values, "likes": c.likes, "dislikes": c.dislikes,
                   **({"current_goal": c.goal} if c.goal else {})}
        entities[c.id] = {"kind": "character", "name": c.name, "profile": profile,
                          **({"role": c.role} if c.role else {}), **({"location": c.location} if c.location else {})}
        k = knowledge.setdefault(c.id, {"observations": [], "beliefs": [], "disclosure_rules": [], "events": []})
        for n, item in enumerate(c.knows):
            fid = f"sheet_{c.id}_{n}"
            facts.append({"fact_id": fid, "subject": item.about, "predicate": item.what, "value": item.value,
                          "valid_from_revision": 0, "source": "game_sheet"})
            k["observations"].append({"fact_id": fid, "learned_at_revision": 0, "acquisition": item.how})
            if item.secret:
                k["disclosure_rules"].append({"rule_id": f"sheet_secret_{n}", "fact_ids": [fid],
                                              "permitted_audience": item.secret.may_tell,
                                              **({"cover_story": item.secret.cover_story} if item.secret.cover_story else {})})
        for r in c.relationships:
            relationships.append({"source": c.id, "target": r.who, **r.model_dump(exclude={"who"}, exclude_none=True)})
        if c.offers or c.never_discuss:
            offers[c.id] = {"offers": [o.model_dump() for o in c.offers], "never_discuss": c.never_discuss}
    for i, statement in enumerate(game.rules):
        rules.append({"rule_id": f"sheet_rule_{i}", "statement": statement})

    # Live state on top.
    live_rels = {(r["source"], r["target"]): r for r in live.get("relationships", [])}
    relationships = [live_rels.pop((r["source"], r["target"]), r) for r in relationships] + list(live_rels.values())
    for actor, extra in live.get("actor_knowledge", {}).items():
        k = knowledge.setdefault(actor, {"observations": [], "beliefs": [], "disclosure_rules": [], "events": []})
        for key, items in extra.items():
            k[key] = k.get(key, []) + list(items)
    offers.update(live.get("offers", {}))

    story = live.get("story")
    version = "world-0.3" if (story or game.setting.boundary) else "world-0.2"
    return World.model_validate({
        "schema_version": version, "world_id": game.game_id, **({"story": story} if story else {}), "world_seed": live.get("world_seed", 0),
        "setting": game.setting.model_dump(), "state_revision": live.get("state_revision", 0),
        "entities": entities, "facts": facts + list(live.get("facts", [])),
        "rules": rules + list(live.get("rules", [])), "relationships": relationships,
        "actor_knowledge": {a: {key: v for key, v in k.items() if v} for a, k in knowledge.items()},
        "events": live.get("events", []), "actor_state": live.get("actor_state", {}),
        "player_history": live.get("player_history", {}), "scenes": live.get("scenes", {}),
        "player_seen": live.get("player_seen", {}), "offers": offers,
        "extensions": [x.model_dump() for x in game.extensions],
    })
