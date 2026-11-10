"""What a trained model must keep: 78 short questions the base model answers before training and the LoRA answers after.

Every question is one player line to a fixed everyday character (a smith, a shop clerk, a museum guide), rendered with
the same template as training, answered greedily, and scored by key words on the spoken line (whole words, common
endings allowed), the way a player would hear it.

Sets
- knowhow (29, gated): what anyone in that role knows. Bronze is copper and tin, bees make honey from nectar, four
  loaves at three coins is twelve.
- comprehension (12, gated): reading and following the player. "Just the number", a quantity, a word's meaning, what
  was asked a moment ago, and asking back when the line is ambiguous.
- realworld (19, gated for the larger model, reported for the smaller): real-world facts asked of present-day
  characters: the Mona Lisa, Everest, the moon landing.
- setting_flip (18, reported): the same real-world questions asked of a character in a closed fictional world. A hit
  is a reply with none of the real-world key words: she has never heard of Paris.

The rule. Against the base model in the same container, a run fails a gated set if it loses more than about a tenth of
it net (knowhow 3, comprehension 2, realworld 2), or if the loss is significant (exact McNemar, p < 0.05).

    rows = retention_prompts(fmt)                                   # {id, set, prompt, max_new, expect_any}
    before, after = score(base_replies, rows), score(lora_replies, rows)
    v = verdict(before, after, tier="fast")                         # per-set lines and an overall pass
"""
from __future__ import annotations

import math
import re

from inscribe.runtime.actor_context import ActorContext, ActorIdentity, BoundaryView, OutputPolicy, SettingView, Turn
from inscribe.runtime.template import ChatFormat, render_prompt

MAX_NEW = 60
STOP_MARKERS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|end_of_text|>", "<think>", "\n", "<turn|>", "<|turn>", "<eos>", "<|channel>")


def _p(text: str, mode: str = "typed") -> Turn:
    return Turn(speaker="player", text=text, mode=mode)


def _npc(npc_id: str, name: str, role: str, voice: str, kind: str, technology: str, location: str) -> ActorContext:
    return ActorContext(world_id="retention", session_id="retention", state_revision=0,
                        actor=ActorIdentity(id=npc_id, name=name, role=role, voice=voice), location_name=location,
                        setting=SettingView(knowledge_policy=kind, technology=technology, magic="absent"),
                        output_policy=OutputPolicy(response_style="explanatory"))


NPCS = {
    "smith": _npc("npc_brann", "Brann", "blacksmith", "gruff, patient with apprentices", "historical", "preindustrial", "the smithy"),
    "clerk": _npc("npc_priya", "Priya", "phone repair shop clerk", "friendly, quick, plain-spoken", "contemporary", "modern", "the repair counter"),
    "guide": _npc("npc_theo", "Theo", "museum guide", "warm, precise, fond of dates", "contemporary", "modern", "the history museum"),
    "ranger": _npc("npc_wren", "Wren", "park ranger", "calm, outdoorsy, safety-minded", "contemporary", "modern", "the trailhead"),
    "trader": _npc("npc_odo", "Odo", "market trader", "cheerful, quick with numbers", "historical", "preindustrial", "the market square"),
    "engineer": _npc("npc_ines", "Ines", "ship's engineer", "practical, dry humour", "contemporary", "modern", "the engine room"),
}

# (id, topic, character, player line, typed or voice, key words: any one is a hit)
KNOWLEDGE: list[tuple[str, str, str, str, str, tuple[str, ...]]] = [
    ("k1", "trades", "smith", "how do you harden a steel blade", "voice", ("quench", "temper")),
    ("k2", "phone", "clerk", "My phone won't charge. What should I check first?", "typed", ("cable", "charger", "port", "outlet", "socket", "plug", "lint")),
    ("k3", "computer", "clerk", "whats the difference between ram and storage", "voice", ("memory", "temporary", "permanent", "turned off", "switched off")),
    ("k4", "history", "guide", "Who was the first person to walk on the moon?", "typed", ("armstrong",)),
    ("k5", "history", "guide", "what year did the second world war end", "voice", ("1945",)),
    ("k6", "weather", "ranger", "those tall dark clouds building up, what do they mean", "voice", ("storm", "thunder", "lightning", "rain")),
    ("k7", "money", "trader", "Bread is three coins a loaf. What do I owe you for four loaves?", "typed", ("12", "twelve")),
    ("k8", "physics", "engineer", "Why does ice float on water?", "typed", ("dense", "density", "lighter", "expand")),
]

# (id, topic, character, player line, key words)
HELDOUT: list[tuple[str, str, str, str, tuple[str, ...]]] = [
    ("h1", "history", "guide", "Who painted the Mona Lisa?", ("leonardo", "da vinci", "vinci")),
    ("h2", "history", "guide", "What were the pyramids at Giza built for?", ("tomb", "pharaoh", "burial", "buried", "grave")),
    ("h3", "history", "guide", "Who was the first president of the United States?", ("washington",)),
    ("h4", "history", "guide", "What did the Romans build to carry water into their cities?", ("aqueduct",)),
    ("h5", "history", "guide", "Who wrote Romeo and Juliet?", ("shakespeare",)),
    ("h6", "history", "guide", "Who invented the light bulb?", ("edison", "swan")),
    ("h7", "history", "guide", "What language did the ancient Romans speak?", ("latin",)),
    ("h8", "history", "guide", "Who was the first emperor of Rome?", ("augustus", "octavian")),
    ("g1", "geography", "guide", "What's the largest ocean?", ("pacific",)),
    ("g2", "geography", "engineer", "What's the capital of France?", ("paris",)),
    ("g3", "geography", "ranger", "What's the tallest mountain in the world?", ("everest",)),
    ("g4", "geography", "clerk", "Which is the longest river in the world?", ("nile", "amazon")),
    ("g5", "geography", "engineer", "How many continents are there?", ("seven", "7")),
    ("g6", "geography", "trader", "Which is bigger, a sea or an ocean?", ("ocean",)),
    ("g7", "geography", "ranger", "What's the biggest hot desert?", ("sahara",)),
    ("g8", "geography", "guide", "Which country is Tokyo in?", ("japan",)),
    ("s1", "science", "engineer", "Why does a compass needle point north?", ("magnet", "magnetic", "field")),
    ("s2", "science", "engineer", "What's water made of?", ("hydrogen", "oxygen", "h2o")),
    ("s3", "science", "ranger", "Why do leaves change colour in autumn?", ("chlorophyll", "green", "sunlight", "daylight", "cold", "colder")),
    ("s4", "science", "engineer", "At what temperature does water boil?", ("100", "hundred", "212")),
    ("s5", "science", "clerk", "Why is the sky blue?", ("scatter", "light", "wavelength", "sunlight")),
    ("s6", "science", "engineer", "What gas do plants give off?", ("oxygen",)),
    ("s7", "science", "ranger", "How do bees make honey?", ("nectar", "flower")),
    ("s8", "science", "smith", "What's bronze made of?", ("copper", "tin")),
    ("b1", "body", "clerk", "How many bones are in the human body?", ("206", "two hundred and six", "two hundred six")),
    ("b2", "body", "guide", "What does the heart do?", ("pump", "blood")),
    ("b3", "body", "ranger", "Why do we sweat?", ("cool", "heat", "temperature")),
    ("b4", "body", "engineer", "What do your lungs do?", ("breathe", "air", "oxygen")),
    ("b5", "body", "trader", "How many teeth does a grown person have?", ("32", "thirty-two", "thirty two")),
    ("b6", "body", "clerk", "Which vitamin does your skin make from sunlight?", ("vitamin d",)),
    ("b7", "body", "smith", "Why do we shiver when we're cold?", ("warm", "heat", "muscle")),
    ("b8", "body", "guide", "How many hours of sleep does an adult need?", ("seven", "eight", "nine", "7", "8", "9")),
    ("w1", "words", "clerk", "What's the opposite of ancient?", ("modern", "new", "recent", "young")),
    ("w2", "words", "guide", "What do you call a word that means the same as another word?", ("synonym",)),
    ("w3", "words", "trader", "How many letters are in the alphabet?", ("26", "twenty-six", "twenty six")),
    ("w4", "words", "engineer", "What's the plural of mouse?", ("mice",)),
    ("w5", "words", "ranger", "What do you call a baby deer?", ("fawn",)),
    ("w6", "words", "smith", "What do you call a baby horse?", ("foal", "colt", "filly")),
    ("w7", "words", "trader", "How many days are in a leap year?", ("366", "three hundred and sixty-six", "three hundred sixty-six", "three hundred and sixty six")),
    ("w8", "words", "guide", "What's a group of wolves called?", ("pack",)),
]

# Which question belongs to which set, by content: g6 is know-how, g5 is a real-world fact.
REALWORLD_IDS = {"k2", "k3", "k4", "k5", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "g1", "g2", "g3", "g4", "g5", "g7", "g8"}
KNOWHOW_IDS = {"k1", "k6", "k7", "k8", "g6"} | {f"{t}{i}" for t in "sbw" for i in range(1, 9)}
# The flip asks the real-world questions of a closed-fictional character. g5 is left out (a made-up world may well
# have seven continents); k2 and k3 are kept (a phone or a computer must mean nothing to her).
FLIP_IDS = sorted(REALWORLD_IDS - {"g5"})

# (id, kind, character, dialogue, key words). kind: "any" (any key word is a hit) or "clarify" (a question back).
COMPREHENSION: list[tuple[str, str, str, list, tuple[str, ...]]] = [
    ("c1", "any", "clerk", ["Just the number: how many legs does a horse have?"], ("4", "four")),
    ("c2", "any", "guide", ["Yes or no: is a week longer than a day?"], ("yes",)),
    ("c3", "any", "trader", ["Three sacks of flour at two coins each, what's that come to?"], ("6", "six")),
    ("c4", "any", "trader", ["What's half of twenty?"], ("10", "ten")),
    ("c5", "any", "smith", ["I had a dozen nails and used four. How many are left?"], ("8", "eight")),
    ("c6", "any", "ranger", ["What does reluctant mean?"], ("unwilling", "hesitant", "not keen", "don't want", "doesn't want", "not want", "slow to")),
    ("c7", "any", "engineer", ["Give me another word for big."], ("large", "huge", "great", "vast", "enormous", "massive")),
    ("c8", "clarify", "clerk", ["Which one should I take?"], ("which", "what", "mean", "one", "options", "choices")),
    ("c9", "any", "guide", ["Who built the pyramids?", ("actor", "The pharaohs of old Egypt, as tombs for themselves."),
                            "Sorry, what did I just ask you?"], ("pyramid",)),
    ("c10", "any", "ranger", ["How many is a couple?"], ("2", "two")),
    ("c11", "any", "smith", ["Slower this time: how many legs on a spider?"], ("8", "eight")),
    ("c12", "any", "engineer", ["If I leave at noon and the trip takes three hours, when do I arrive?"], ("3", "three", "15")),
]
CLARIFY_MARK = "?"

# A closed-fictional character for the setting flip, in a world where nothing from the real world exists except a short list.
FLIP_NPC = ActorContext(
    world_id="retention_closed", session_id="retention", state_revision=0,
    actor=ActorIdentity(id="npc_hald", name="Hald", role="innkeeper", voice="slow, dry, fond of a story"),
    location_name="the Crooked Oar, a tavern by the ferry",
    setting=SettingView(knowledge_policy="closed_fictional", technology="medieval", magic="restricted",
                        boundary=BoundaryView(real_world="none", allowed=["bronze", "iron", "horses", "beer"])),
    world_rules=["Nobody has ever crossed the Grey Sea and come back."],
    output_policy=OutputPolicy(response_style="explanatory"),
)

SETS = {"knowhow": 29, "comprehension": 12, "realworld": 19, "setting_flip": 18}
NET_LOSS_ALLOWED = {"knowhow": 3, "comprehension": 2, "realworld": 2}
SET_NAMES = {"knowhow": "Know-how", "comprehension": "Following the player", "realworld": "Real-world facts", "setting_flip": "Staying inside a closed world"}


def clean_reply(text: str) -> str:
    """The first spoken line: cut at the first stop marker or line break."""
    cut = len(text)
    for m in STOP_MARKERS:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def _forms(word: str) -> list[str]:
    if " " in word or not word.isalpha():
        return [word]
    if word.endswith("e"):
        return [word, word + "s", word + "d", word[:-1] + "ing"]
    return [word, word + "s", word + "es", word + "ed", word + "ing"]


def hit(reply: str, keys) -> bool:
    text = reply.lower()
    return any(re.search(rf"(?<!\w){re.escape(f)}(?!\w)", text) for k in keys for f in _forms(k))


def clarifies(reply: str, keys) -> bool:
    """A question back that names the ambiguity."""
    return CLARIFY_MARK in reply and hit(reply, keys)


def _dialogue(lines: list) -> list[Turn]:
    return [Turn(speaker="actor", text=x[1]) if isinstance(x, tuple) else _p(x) for x in lines]


def _set_of(short_id: str) -> str:
    if short_id in REALWORLD_IDS:
        return "realworld"
    if short_id in KNOWHOW_IDS:
        return "knowhow"
    raise KeyError(short_id)


def retention_prompts(fmt: ChatFormat) -> list[dict]:
    """Every retention question with its set and the player's line, in a fixed order: know.k1 to k8, held.*, comp.*, flip.*."""
    if fmt.thinking:
        raise ValueError(f"{fmt.name} can think; the retention questions are for formats that answer directly")
    know = [{"id": f"know.{kid}", "set": _set_of(kid), "asked": line, "prompt": render_prompt(NPCS[npc], [_p(line, mode)], fmt),
             "max_new": MAX_NEW, "expect_any": list(keys)} for kid, _, npc, line, mode, keys in KNOWLEDGE]
    held = [{"id": f"held.{hid}", "set": _set_of(hid), "asked": line, "prompt": render_prompt(NPCS[npc], [_p(line)], fmt), "max_new": MAX_NEW,
             "expect_any": list(keys)} for hid, _, npc, line, keys in HELDOUT]
    comp = [{"id": f"comp.{cid}", "set": "comprehension", "kind": kind, "asked": lines[-1], "prompt": render_prompt(NPCS[npc], _dialogue(lines), fmt),
             "max_new": MAX_NEW, "expect_any": list(keys)} for cid, kind, npc, lines, keys in COMPREHENSION]
    by_id = {hid: (line, keys) for hid, _, _, line, keys in HELDOUT}
    by_id.update({kid: (line, keys) for kid, _, _, line, _, keys in KNOWLEDGE})
    flip = [{"id": f"flip.{fid}", "set": "setting_flip", "absent": True, "asked": by_id[fid][0], "prompt": render_prompt(FLIP_NPC, [_p(by_id[fid][0])], fmt),
             "max_new": MAX_NEW, "expect_any": list(by_id[fid][1])} for fid in FLIP_IDS]
    return know + held + comp + flip


def prompt_rows(fmt: ChatFormat) -> list[dict]:
    """The questions alone ({id, prompt, max_new}), as a training step answers them. No key travels to a container."""
    return [{"id": r["id"], "prompt": r["prompt"], "max_new": r["max_new"]} for r in retention_prompts(fmt)]


def _hit(prompt: dict, reply: str) -> bool:
    spoken = clean_reply(reply)
    if prompt.get("absent"):
        return bool(spoken.strip()) and not hit(reply, prompt["expect_any"])  # the whole generation, not just the first line
    if prompt.get("kind") == "clarify":
        return clarifies(spoken, prompt["expect_any"])
    return hit(spoken, prompt["expect_any"])


def score(rows: list[dict], prompts: list[dict]) -> dict:
    """Hits for one model's replies ([{id, text}]), by set: {set: {hits, n, by_id}}. Replies to other questions are ignored."""
    by_prompt = {p["id"]: p for p in prompts}
    sets: dict[str, dict[str, bool]] = {name: {} for name in SETS}
    for r in rows:
        p = by_prompt.get(r["id"])
        if p is not None:
            sets[p["set"]][p["id"]] = _hit(p, r.get("text") or r.get("reply") or "")
    return {name: {"hits": sum(v.values()), "n": len(v), "by_id": v} for name, v in sets.items()}


def mcnemar_p(before: dict, after: dict) -> float:
    """Exact two-sided McNemar on paired hits: only the questions that changed count."""
    lost = sum(1 for k, v in before.items() if v and not after.get(k, False))
    gained = sum(1 for k, v in after.items() if v and not before.get(k, False))
    n = lost + gained
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(lost, gained) + 1)) / 2**n
    return min(1.0, 2 * tail)


def tier_of(recipe_name: str) -> str:
    """main (the 4B) or fast (anything smaller). The real-world set is gated for main only."""
    return "main" if "4b" in recipe_name.lower() else "fast"


def verdict(before: dict, after: dict, tier: str = "fast") -> dict:
    """The rule applied to two `score()` results, the base model's and the LoRA's from the same container: one line per
    set, and `passed` for the whole. The setting flip is reported and never fails a run on its own."""
    gated = ["knowhow", "comprehension"] + (["realworld"] if tier == "main" else [])
    out, ok = {}, True
    for name in SETS:
        b, a = before[name], after[name]
        net = a["hits"] - b["hits"]
        p = mcnemar_p(b["by_id"], a["by_id"])
        passed = True if name == "setting_flip" else net >= -NET_LOSS_ALLOWED[name] and not (net < 0 and p < 0.05)
        out[name] = {"before": b["hits"], "after": a["hits"], "n": a["n"], "net": net, "p": round(p, 4), "gated": name in gated, "passed": passed,
                     "lost": sorted(k for k, v in b["by_id"].items() if v and not a["by_id"].get(k, False))}
        if name in gated and not passed:
            ok = False
    out["passed"] = ok
    out["tier"] = tier
    return out
