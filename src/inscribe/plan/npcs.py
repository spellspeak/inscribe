"""Characters a game makes up at run time, read from a project's `npcs.yaml`.

These are not in the game sheet and are never trained on. The game hands them to the model in the prompt alone: a
name, a role, a voice, a few facts and relationships. The checks ask them about the world, to see that a character
the model has never met still answers from the game's lore, and never says a core character's secret.

    npcs:
      - id: npc_ludo_brasque
        name: Ludo Brasque
        role: Factor for a downriver house.
        voice: Cheerful and slightly deaf.
        location: heron_and_hook            # a thing id from the sheet (optional)
        values: [...]                       # optional, as are likes and dislikes
        facts:
          - {about: ormsgate, what: spar_prices, value: A hearthblock fetches forty-five tallies.}
        relationships:
          - {who: npc_osmund_stannock, kind: merchant, trust: high, opinion: Courtesy itself.}
        time_of_day: evening
        situation: the inn on a market night
"""
from __future__ import annotations

import pathlib

import yaml

from inscribe.plan.world import DynamicNPC, NPCFact, NPCRelationship

_NPC_KEYS = {"id", "name", "role", "voice", "location", "values", "likes", "dislikes", "facts", "relationships", "time_of_day", "situation"}
_FACT_KEYS = {"about", "what", "value", "how"}
_REL_KEYS = {"who", "kind", "trust", "affection", "fear", "respect", "opinion"}


def _only(raw: dict, allowed: set[str], where: str) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}")
    return raw


def npc_from_dict(raw: dict) -> DynamicNPC:
    where = f"npc {raw.get('id', '?')!r}" if isinstance(raw, dict) else "npc"
    raw = dict(_only(raw, _NPC_KEYS, where))
    for key in ("id", "name", "role", "voice"):
        if not raw.get(key):
            raise ValueError(f"{where}: `{key}` is required")
    facts = tuple(NPCFact(**_only(f, _FACT_KEYS, f"{where}, fact {i + 1}")) for i, f in enumerate(raw.pop("facts", None) or []))
    rels = tuple(NPCRelationship(**_only(r, _REL_KEYS, f"{where}, relationship {i + 1}")) for i, r in enumerate(raw.pop("relationships", None) or []))
    lists = {k: tuple(raw.pop(k, None) or ()) for k in ("values", "likes", "dislikes")}
    return DynamicNPC(**raw, **lists, facts=facts, relationships=rels)


def load_npcs(path: str | pathlib.Path) -> tuple[DynamicNPC, ...]:
    """The NPCs of an `npcs.yaml`, or none when the file is not there."""
    path = pathlib.Path(path)
    if not path.exists():
        return ()
    raw = yaml.safe_load(path.read_text()) or {}
    rows = _only(raw, {"npcs"}, str(path)).get("npcs") or []
    npcs = tuple(npc_from_dict(r) for r in rows)
    ids = [n.id for n in npcs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: an NPC id is used twice")
    return npcs


def npcs_to_yaml(npcs: tuple[DynamicNPC, ...] | list[DynamicNPC]) -> str:
    """The same NPCs as `npcs.yaml` text, defaults left out."""
    rows = []
    for n in npcs:
        row: dict = {"id": n.id, "name": n.name, "role": n.role, "voice": n.voice}
        if n.location:
            row["location"] = n.location
        for key in ("values", "likes", "dislikes"):
            if getattr(n, key):
                row[key] = list(getattr(n, key))
        if n.facts:
            row["facts"] = [{"about": f.about, "what": f.what, "value": f.value, **({"how": f.how} if f.how != "self" else {})} for f in n.facts]
        if n.relationships:
            row["relationships"] = [{"who": r.who, **({"kind": r.kind} if r.kind else {}),
                                     **{k: getattr(r, k) for k in ("trust", "affection", "fear") if getattr(r, k) != "neutral"},
                                     **({"respect": r.respect} if r.respect else {}), **({"opinion": r.opinion} if r.opinion else {})}
                                    for r in n.relationships]
        if n.time_of_day:
            row["time_of_day"] = n.time_of_day
        if n.situation:
            row["situation"] = n.situation
        rows.append(row)
    return yaml.safe_dump({"npcs": rows}, sort_keys=False, allow_unicode=True, width=120)
