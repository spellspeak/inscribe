"""The blind style check: does the LoRA sound more like the sheet's character than the base model does?

The training step answers the game's held-out questions before training (the base model) and after (the LoRA), so
every pair is the same character and the same player line in two voices. The strong model reads the sheet's
description of how the character talks and picks the reply that sounds more like it. Voice only, facts ignored, the
order drawn from a seed and undone afterwards.

    pairs = style_pairs(bundle, sheet, before_rows, after_rows)       # up to 80 pairs, seeded
    report = judge(bundle, sheet, before_rows, after_rows, api_key)   # wins for the LoRA, for the base, ties, a sign test
"""
from __future__ import annotations

import math
import random

from pydantic import ValidationError

from inscribe.judges import strong
from inscribe.plan.recall import clean_reply
from inscribe.schemas.game_sheet import GameSheet

OUTPUT_FORECAST_TOKENS = 200
PAIRS = 80
SYSTEM = """You are judging which of two replies sounds more like a given character's way of talking. You will see how the character talks, the line the player said, and two replies. Judge the voice only: word choice, rhythm, tics, address, tone. Ignore whether the facts are right or the question is answered. If neither reply sounds more like the character, say tie."""


def _messages(name: str, voice: str, question: str, a: str, b: str) -> list[dict]:
    user = (f"How {name} talks: {voice}\n\nThe player said: \"{question}\"\n\nReply A: \"{a}\"\nReply B: \"{b}\"\n\n"
            f"Which reply sounds more like {name}? Answer A, B or tie, with one sentence of reason.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def style_pairs(bundle: dict, sheet: GameSheet, before_rows: list[dict], after_rows: list[dict], n: int = PAIRS, seed: int = 0,
                model: str = strong.MODEL) -> list[tuple[dict, dict]]:
    """Up to `n` requests on the core characters' lore and price questions where the two replies differ, drawn by seed.
    Each comes with what was shown: the question, the character, and the base's and the LoRA's reply."""
    voices = {c.id: (c.name, c.voice) for c in sheet.characters if c.voice}
    before = {r["id"]: clean_reply(r.get("text") or "") for r in before_rows}
    after = {r["id"]: clean_reply(r.get("text") or "") for r in after_rows}
    rows = [r for r in [*bundle["lore"], *bundle.get("offers", [])] if r["actor"] in voices and before.get(r["id"]) and after.get(r["id"])
            and before[r["id"]] != after[r["id"]]]
    rng = random.Random(f"style:{bundle['game_id']}:{seed}")
    rows = rng.sample(rows, min(n, len(rows)))
    out = []
    for r in rows:
        name, voice = voices[r["actor"]]
        pair_id = f"style.{r['id']}"
        order = strong.pair_order(pair_id, seed)
        a, b = (before[r["id"]], after[r["id"]]) if order == "ab" else (after[r["id"]], before[r["id"]])
        line = strong.request(f"{pair_id}.{order}", _messages(name, voice, r["question"], a, b), strong.PREFERENCE_SCHEMA, "preference_verdict", model)
        out.append((line, {"id": r["id"], "character": name, "asked": r["question"], "base": before[r["id"]], "lora": after[r["id"]]}))
    return out


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n)


def judge(bundle: dict, sheet: GameSheet, before_rows: list[dict], after_rows: list[dict], api_key: str = "", max_usd: float = 1.0,
          n: int = PAIRS, seed: int = 0, model: str = strong.MODEL, concurrency: int = 8, answer=None) -> dict:
    """Ask the judge about each pair. The first reply of a pair is the base model's, the second the LoRA's; the order
    the judge saw is undone from the request's id. `answer` replaces the paid call (tests)."""
    pairs = style_pairs(bundle, sheet, before_rows, after_rows, n, seed, model)
    lines = [line for line, _ in pairs]
    shown = {line["custom_id"]: what for line, what in pairs}
    forecast = strong.forecast_usd(lines, OUTPUT_FORECAST_TOKENS)
    if forecast > max_usd:
        raise RuntimeError(f"the style check would cost about ${forecast:.2f}, over its cap of ${max_usd:.2f}")
    answer = answer or (lambda ls: strong.answer(ls, api_key, concurrency=concurrency))
    lora, base, ties, errors, usage, samples = 0, 0, 0, [], [], []
    for o in answer(lines):
        p = strong.parse(o)
        usage.append(p.usage)
        if p.error or not p.text:
            errors.append({"id": p.custom_id, "error": p.error})
            continue
        try:
            v = strong.PreferenceVerdict.model_validate_json(p.text)
        except ValidationError as exc:
            errors.append({"id": p.custom_id, "error": f"invalid verdict: {str(exc)[:300]}"})
            continue
        winner = strong.unswap(p.custom_id, v.winner)
        if winner == "2":
            lora += 1
        elif winner == "1":
            base += 1
        else:
            ties += 1
        samples.append({**shown[p.custom_id], "preferred": {"1": "base", "2": "lora", "tie": "tie"}[winner], "reason": v.reason})
    decided = lora + base + ties
    return {"judge": model, "pairs": decided, "lora_wins": lora, "base_wins": base, "ties": ties,
            "lora_rate_with_half_ties": round((lora + ties / 2) / decided, 3) if decided else None, "sign_test_p": round(sign_test_p(lora, base), 4),
            "samples": samples, "errors": errors, "forecast_usd": round(forecast, 3), "cost_usd": round(strong.cost_usd(usage), 3)}
