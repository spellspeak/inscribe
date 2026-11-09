"""Held-out questions for a game LoRA, and their scoring: lore recall, spoilers, run-time characters, prices.

    rows = lore_recall_prompts(sheet, fmt)                 # held-out phrasings per lore item, never used in training
    rows = spoiler_prompts(sheet, fmt)                     # per stage, asks that would reveal the later stages
    rows = dynamic_npc_prompts(sheet, npcs, fmt)           # lore questions put to a prompt-only run-time character
    rows = offer_prompts(sheet, fmt)                       # each character's priced offers, the other prices as must_not
    terms = leak_terms(sheet)                              # what a reply from another world must never mention
    bundle = recall_bundle(sheet, npcs, fmt)               # all of the above, what the plan step writes as recall.json
    rows = load_prompt_rows("recall.json", fmt.name)       # the prompts alone ({id, prompt, max_new}), for a container
    report = score(bundle, replies)                        # replies: [{"id", "text"}] from a model run

Every row is a prompt the training and export steps answer (`id`, `prompt`, `max_new`), rendered through the same compiler and template as play, plus its answer key: `expect_any` (key words a reply
that states the item uses; an exact item's wording, matched whole) and `must_not` (words a reply must not use: a later
stage's new words, a core character's secret). The prompt never carries the item: lore is in
no character's view, and the recall phrasings are a separate pool from training.

Scoring: lore recall per item and per importance, exact items word for word; spoilers told early
with a 95% upper bound (Wilson) so zero events still carry a number; dynamic NPCs scored on the same lore keys and on
never saying a secret. `score_file` also takes a training run's result (`retention_before`
and `retention_after` rows) or a replies JSONL, so the base model and the game LoRA are scored side by side.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import sys
from importlib import resources

from inscribe.runtime.actor_context import Turn
from inscribe.runtime.template import DEFAULT_FORMAT, ChatFormat, render_prompt, render_system
from inscribe.schemas.game_sheet import GameSheet
from inscribe.checks import _contains as says
from inscribe.context.compiler import project_actor_view
from inscribe.plan import scenarios as S
from inscribe.plan.world import DynamicNPC, GENERATOR_VERSION, game_world, world_id_for

MAX_NEW = 60
STOP_MARKERS = ("<turn|>", "<|turn>", "<|im_end|>", "<|im_start|>", "\n")


def _rng(sheet: GameSheet, seed: int, stream: str) -> random.Random:
    return random.Random(f"{GENERATOR_VERSION}:{sheet.game_id}:{seed}:{stream}")


def _row(world, actor: str, question: str, fmt: ChatFormat, session: str) -> dict:
    view = project_actor_view(world, actor, session_id=session)
    return {"world_id": world.world_id, "actor": actor, "question": question,
            "prompt": render_prompt(view, [Turn(speaker="player", text=question, mode="typed")], fmt), "max_new": MAX_NEW}


def lore_recall_prompts(sheet: GameSheet, fmt: ChatFormat = DEFAULT_FORMAT, seed: int = 0, phrasings: int = 2, knowers: int = 2) -> list[dict]:
    """Held-out phrasings per lore item (never used in training), each asked to up to `knowers` of the characters the
    training asked (best first), rendered on the base world. Items the pipeline could not ask, and items whose answer is
    yes or no (no key words to score), are left out."""
    world = game_world(sheet)
    rows = []
    for ask in S.lore_plan(sheet, world, seed):
        if ask.pattern == "none" or not ask.held_out or not ask.knowers or not ask.checks:
            continue  # an item without a code key (a yes-or-no item) is trained but cannot be scored by key words
        held = [h for h in ask.held_out if not S.crosses_firewall(h)][:phrasings]
        for actor in ask.knowers[:knowers]:
            for q, question in enumerate(held):
                rows.append({"id": f"lore.{sheet.game_id}.{ask.index:03d}.{q}.{actor}", "set": "lore", "lore_id": ask.fact_id,
                             "lore_index": ask.index, "importance": ask.item.importance, "exact": ask.item.exact, "pattern": ask.pattern,
                             **_row(world, actor, question, fmt, f"recall-{ask.index}-{q}-{actor}"), "expect_any": ask.checks, "must_not": []})
    return rows


def spoiler_prompts(sheet: GameSheet, fmt: ChatFormat = DEFAULT_FORMAT, seed: int = 0, per_stage: int = 3) -> list[dict]:
    """At every stage but the last, on that stage's world, `per_stage` characters are asked what comes next in held-out
    phrasings; `must_not` is the words only the later stages introduce (scenarios.spoiler_terms)."""
    rng = _rng(sheet, seed, "spoiler")
    known = S.sheet_words(sheet)
    common = foundation_vocabulary()
    rows = []
    chars = [c.id for c in sheet.characters]
    for k, st in enumerate(sheet.story[:-1]):
        world = game_world(sheet, k)
        # a later stage's words, minus plain vocabulary: "standing" in "the wall's still standing" is no spoiler.
        # Key words are a weak spoiler check either way (one to five words a stage); a judge question is the debt.
        banned = [w for w in S.spoiler_terms(sheet, k, known) if w.lower() not in common]
        for actor in rng.sample(chars, min(per_stage, len(chars))):
            for q, question in enumerate(S.NEXT_HELD):
                rows.append({"id": f"spoiler.{sheet.game_id}.s{k + 1}.{q}.{actor}", "set": "spoiler", "stage": st.stage, "stage_index": k,
                             **_row(world, actor, question, fmt, f"spoiler-{k}-{q}-{actor}"), "expect_any": [], "must_not": banned,
                             "later_stage_ids": [S.stage_fact_id(j + 1) for j in range(k + 1, len(sheet.story))]})
    return rows


def secret_terms(sheet: GameSheet) -> list[str]:
    """Key words of every core character's secrets (scenarios.secret_keys): what a dynamic NPC must never say. A key
    the foundation already says plainly ("country", from a headline) is dropped: it cannot mark a told secret."""
    common = foundation_vocabulary()
    out: list[str] = []
    for c in sheet.characters:
        for k in c.knows:
            if k.secret is not None:
                out += [w for w in S.secret_keys(k.value, k.secret.cover_story, sheet) if w not in out and not (w.islower() and w in common)]
    return out


def dynamic_npc_prompts(sheet: GameSheet, dynamic_npcs: tuple[DynamicNPC, ...] | list[DynamicNPC], fmt: ChatFormat = DEFAULT_FORMAT,
                        seed: int = 0, per_npc: int = 12) -> list[dict]:
    """Lore questions put to a prompt-only dynamic NPC: for each NPC, up to `per_npc` items of high importance (then
    normal) in a seeded order, one held-out phrasing each, on the base world with the NPCs added. `must_not` holds the
    core characters' secret terms: the NPC was never given them."""
    world = game_world(sheet, dynamic_npcs=dynamic_npcs)
    rng = _rng(sheet, seed, "dynamic")
    plan = [a for a in S.lore_plan(sheet, game_world(sheet), seed) if a.pattern != "none" and a.held_out and a.checks and not S.in_rules(sheet, a.item)]
    secrets = secret_terms(sheet)
    rows = []
    for npc in dynamic_npcs:
        view_text = render_system(project_actor_view(world, npc.id, session_id="npc"))
        order = sorted((a for a in plan if not S.gives_away(view_text, a)),  # never an item the NPC's own facts answer
                       key=lambda a: ({"high": 0, "normal": 1, "low": 2}[a.item.importance], rng.random()))
        for ask in order[:per_npc]:
            question = next((h for h in ask.held_out if not S.crosses_firewall(h)), None)
            if question is None:
                continue
            rows.append({"id": f"dynamic.{sheet.game_id}.{npc.id}.{ask.index:03d}", "set": "dynamic", "npc": npc.id, "lore_id": ask.fact_id,
                         "lore_index": ask.index, "importance": ask.item.importance, "exact": ask.item.exact,
                         **_row(world, npc.id, question, fmt, f"dynamic-{npc.id}-{ask.index}"), "expect_any": ask.checks, "must_not": secrets})
    return rows


OFFER_HELD = ("What do you ask for {x}?", "Remind me what {x} comes to.")


def offer_prompts(sheet: GameSheet, fmt: ChatFormat = DEFAULT_FORMAT, seed: int = 0) -> list[dict]:
    """Each character's priced offers, asked in two phrasings the training never uses (the game scenarios ask offers
    through BR-01 and the lore price lines through PRICE_ASKS). The key is the price's number words; the same
    character's other prices are what the reply must not give (Maren priced a pint at three pips, the stabling
    rate). Both prices and the offers sit in the prompt, so this measures reading the sheet, not recall."""
    world = game_world(sheet)
    names = S._Names(sheet)
    rows = []
    for c in sheet.characters:
        priced = [o for o in c.offers if o.price]
        keys = {o.what: S._price_keys(o.price) for o in priced}
        for i, o in enumerate(priced):
            own = keys[o.what]
            if not own:
                continue
            others = sorted({k for w, ks in keys.items() if w != o.what for k in ks if k not in own})
            for q, question in enumerate(names.fill(OFFER_HELD, o.what)):
                rows.append({"id": f"offer.{sheet.game_id}.{c.id}.{i:02d}.{q}", "set": "offer", "offer": o.what, "price": o.price,
                             **_row(world, c.id, question, fmt, f"offer-{c.id}-{i}-{q}"), "expect_any": own, "must_not": others})
    return rows


def leak_terms(sheet: GameSheet) -> list[str]:
    """Names and values of this game that a reply from another world must not mention (the lore_leak_terms shape):
    thing and character names, exact wordings, and the proper-noun keys of the lore items. A term that also belongs to
    the world being checked is no leak there, so a check drops those first. A bare word that is a common noun outside the
    game is not a signal ("master" as in harbour master, "throat", "the shelf"), so a single word is dropped when
    it is a role or title, or when the game only uses it as a named place ("the Throat") and the article form is a term."""
    names = S._Names(sheet)
    terms = {t.name for t in sheet.things} | {c.name for c in sheet.characters}
    terms |= {w for c in sheet.characters for w in S.tokens(c.name) if len(w) >= 5 and w.lower() not in S.TITLES and w.lower() not in ROLE_WORDS}
    for ask in S.lore_asks(sheet, set()):
        if ask.item.exact:
            terms.update(ask.keys)
        else:
            terms.update(k for k in ask.keys if S._YEAR_TOKEN.match(k) or (k.isalpha() and len(k) >= 6 and k.lower() in names.proper
                                                                           and k.lower() not in S.MONTHS and k.lower() not in WEEKDAYS
                                                                           and k.lower() not in ROLE_WORDS))
    named_places = {t.lower()[4:] for t in terms if t.lower().startswith("the ") and " " not in t[4:]}  # "the Throat" -> "throat"
    common = foundation_vocabulary()
    keep = []
    for t in terms:
        low = t.lower()
        if " " not in t and (low in named_places or (t.islower() and t in common)):
            continue  # a bare word the untouched model already says on other worlds is plain vocabulary
        if low.startswith("the ") and " " not in low[4:] and low[4:] in common:
            continue  # "the Shelf", "the Throat": other worlds say "the shelf is bare" and "clear the throat"
        keep.append(t)
    return sorted(keep, key=lambda t: (t.lower(), t))


_VOCAB: set[str] | None = None


def foundation_vocabulary() -> set[str]:
    """Plain words the base models already use on other worlds. A lore key that is in this list ("compact", "chance",
    "the shelf") cannot mark a leak or a spoiler. The list is frozen and ships with the package, so a game's answer
    keys do not change from one build to the next."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = set(resources.files("inscribe.plan").joinpath("common_words.txt").read_text().split())
    return _VOCAB


WEEKDAYS = frozenset("monday tuesday wednesday thursday friday saturday sunday mondays tuesdays wednesdays thursdays fridays saturdays sundays".split())
# Roles and titles a game may capitalise as names but other worlds use as plain words (harbour master, the warden).
ROLE_WORDS = frozenset("master mistress warden reeve speaker keeper steward captain sergeant doctor mayor sheriff constable bailiff "
                       "elder chief guard porter clerk marshal matron abbot prior brother sister father mother".split())


def recall_bundle(sheet: GameSheet, dynamic_npcs: tuple[DynamicNPC, ...] | list[DynamicNPC], fmt: ChatFormat = DEFAULT_FORMAT, seed: int = 0) -> dict:
    lore, spoiler, dynamic = lore_recall_prompts(sheet, fmt, seed), spoiler_prompts(sheet, fmt, seed), dynamic_npc_prompts(sheet, dynamic_npcs, fmt, seed)
    offers = offer_prompts(sheet, fmt, seed)
    return {"game_id": sheet.game_id, "generator_version": GENERATOR_VERSION, "format": fmt.name, "seed": seed,
            "lore": lore, "spoiler": spoiler, "dynamic": dynamic, "offers": offers, "leak_terms": leak_terms(sheet),
            "counts": {"lore": len(lore), "lore_items": len({r["lore_id"] for r in lore}), "spoiler": len(spoiler), "dynamic": len(dynamic),
                       "offers": len(offers), "leak_terms": len(leak_terms(sheet))}}


def all_rows(bundle: dict) -> list[dict]:
    return [*bundle["lore"], *bundle["spoiler"], *bundle["dynamic"], *bundle.get("offers", [])]


def prompt_rows(bundle: dict, fmt_name: str | None = None) -> list[dict]:
    """The bundle's prompts as the training and export scripts answer them ({id, prompt, max_new} and nothing else), so
    no answer key travels to a container. A bundle is rendered for one chat format; `fmt_name` refuses another."""
    if fmt_name and bundle.get("format") and bundle["format"] != fmt_name:
        raise ValueError(f"recall bundle rendered for chat format {bundle['format']!r}, not {fmt_name!r}: rebuild it with `inscribe plan`")
    return [{"id": r["id"], "prompt": r["prompt"], "max_new": r["max_new"]} for r in all_rows(bundle)]


def load_prompt_rows(path: str | pathlib.Path, fmt_name: str | None = None) -> list[dict]:
    """`prompt_rows` of a recall.json on disk: what the training and export steps read."""
    return prompt_rows(json.loads(pathlib.Path(path).read_text()), fmt_name)


# --- scoring ----------------------------------------------------------------------------------------------------
def clean_reply(text: str) -> str:
    cut = len(text)
    for m in STOP_MARKERS:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def hit(reply: str, keys: list[str]) -> bool:
    """Whole-word match of any key, as the deterministic checks match; multi-word keys (exact items) match whole."""
    return any(says(reply, k) for k in keys)


def wilson_upper(k: int, n: int, z: float = 1.96) -> float | None:
    """95% upper bound of a proportion (Wilson score interval); reported even when k is zero."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(min(1.0, (centre + half) / denom), 4)


def score(bundle: dict, replies: list[dict]) -> dict:
    """Score one set of replies ([{"id", "text"}]) against the bundle's keys."""
    by_id = {r["id"]: clean_reply(r.get("text") or "") for r in replies}
    out: dict = {"answered": sum(1 for r in all_rows(bundle) if r["id"] in by_id), "prompts": len(all_rows(bundle))}
    lore = [r for r in bundle["lore"] if r["id"] in by_id]
    hits = [(r, hit(by_id[r["id"]], r["expect_any"])) for r in lore]
    items: dict[str, list[bool]] = {}
    for r, h in hits:
        items.setdefault(r["lore_id"], []).append(h)
    by_imp = {}
    for imp in ("high", "normal", "low"):
        rows = [(r, h) for r, h in hits if r["importance"] == imp]
        ids = {r["lore_id"] for r, _ in rows}
        by_imp[imp] = {"prompts": len(rows), "prompt_hits": sum(h for _, h in rows), "items": len(ids),
                       "items_recalled": sum(1 for i in ids if any(items[i]))}
    exact = [(r, h) for r, h in hits if r["exact"]]
    out["lore"] = {"prompts": len(hits), "prompt_hits": sum(h for _, h in hits), "items": len(items),
                   "items_recalled": sum(1 for v in items.values() if any(v)), "by_importance": by_imp,
                   "exact": {"prompts": len(exact), "prompt_hits": sum(h for _, h in exact)},
                   "misses": [r["id"] for r, h in hits if not h][:40]}
    sp = [r for r in bundle["spoiler"] if r["id"] in by_id]
    told = [r["id"] for r in sp if hit(by_id[r["id"]], r["must_not"])]
    out["spoiler"] = {"prompts": len(sp), "told_early": len(told), "upper_bound_95": wilson_upper(len(told), len(sp)), "ids": told[:40]}
    dyn = [r for r in bundle["dynamic"] if r["id"] in by_id]
    d_hits = [r["id"] for r in dyn if hit(by_id[r["id"]], r["expect_any"])]
    d_leaks = [r["id"] for r in dyn if hit(by_id[r["id"]], r["must_not"])]
    out["dynamic"] = {"prompts": len(dyn), "lore_hits": len(d_hits), "secret_leaks": len(d_leaks), "leak_ids": d_leaks[:40],
                      "by_npc": {npc: {"prompts": sum(1 for r in dyn if r["npc"] == npc), "hits": sum(1 for r in dyn if r["npc"] == npc and r["id"] in set(d_hits))}
                                 for npc in sorted({r["npc"] for r in dyn})}}
    off = [r for r in bundle.get("offers", []) if r["id"] in by_id]
    o_right = {r["id"] for r in off if hit(by_id[r["id"]], r["expect_any"])}
    o_wrong = [r["id"] for r in off if r["id"] not in o_right and hit(by_id[r["id"]], r["must_not"])]
    out["offers"] = {"prompts": len(off), "right_price": len(o_right), "wrong_price": len(o_wrong), "wrong_ids": o_wrong[:40]}
    return out


def score_file(bundle_path: str | pathlib.Path, replies_path: str | pathlib.Path) -> dict:
    """Score a run: a training result (retention_before = the base model, retention_after = the game LoRA), or a
    JSONL of {id, text} rows (an export probe's replies)."""
    bundle = json.loads(pathlib.Path(bundle_path).read_text())
    path = pathlib.Path(replies_path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return {"replies": str(path), "scored": score(bundle, rows)}
    run = json.loads(path.read_text())
    out = {"replies": str(path)}
    for label, key in (("before", "retention_before"), ("after", "retention_after")):
        if key in run:
            out[label] = score(bundle, run[key])
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m inscribe.plan.recall <recall.json> <run.json or replies.jsonl>")
        return 2
    report = score_file(argv[0], argv[1])
    out = pathlib.Path(argv[1]).with_name(pathlib.Path(argv[1]).stem + "-recall.json")
    out.write_text(json.dumps(report, indent=1) + "\n")
    for label in ("before", "after", "scored"):
        if label in report:
            r = report[label]
            print(f"{label}: lore {r['lore']['items_recalled']} of {r['lore']['items']} items ({r['lore']['prompt_hits']} of {r['lore']['prompts']} prompts), "
                  f"exact {r['lore']['exact']['prompt_hits']} of {r['lore']['exact']['prompts']}; spoiler told early {r['spoiler']['told_early']} of "
                  f"{r['spoiler']['prompts']} (upper bound {r['spoiler']['upper_bound_95']}); dynamic NPC lore {r['dynamic']['lore_hits']} of "
                  f"{r['dynamic']['prompts']}, secret leaks {r['dynamic']['secret_leaks']}; offers priced right {r['offers']['right_price']} of "
                  f"{r['offers']['prompts']}, wrong price {r['offers']['wrong_price']}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
