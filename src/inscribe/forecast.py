"""What a run will take, before any money is spent.

Every number here is an estimate from one measured run of the example game: one teacher run and one training run
per base model, on one H100. A different game, GPU or day will differ. Costs use the hourly prices in the config,
which are estimates and not quotes. The cost of the outside model that checks the teacher's lines is not counted.
"""
from __future__ import annotations

from inscribe.config import Config, Recipe

ACCEPTED_PER_SCENE = 0.74  # 718 lines kept from 974 scenes, four candidates a scene
TEACHER_LOAD_SECONDS = 275  # loading the teacher model before it writes anything
TEACHER_SECONDS_PER_CANDIDATE = 0.08  # 312 seconds for 3,896 candidates
TARGETED_REPLAY = 1700  # extra general examples for the skills a LoRA loses first
EXPORT_HOURS = 0.5  # convert, quantize and answer the held-out questions; an assumption, not yet measured alone


def forecast(counts: dict, config: Config, recipes: list[Recipe]) -> dict:
    scenes = counts["teacher_cases"]
    candidates = scenes * config.teacher.candidates
    accepted = round(scenes * ACCEPTED_PER_SCENE)
    game_examples = accepted + counts["style_cases"]
    replay = config.replay.to_one * game_examples + TARGETED_REPLAY
    examples = game_examples + replay
    prices = config.backend.usd_per_hour
    teacher_hours = (TEACHER_LOAD_SECONDS + candidates * TEACHER_SECONDS_PER_CANDIDATE) / 3600
    steps = [{"step": "teacher", "gpu": "H100", "hours": round(teacher_hours, 2), "usd": _usd(teacher_hours, prices.get("H100"))}]
    for r in recipes:
        epochs = config.train.epochs or r.train.epochs
        hours = examples * epochs * r.measured.train_seconds_per_example_epoch / 3600
        steps.append({"step": f"train {r.name}", "gpu": r.train.gpu, "hours": round(hours, 2), "usd": _usd(hours, prices.get(r.train.gpu)),
                      "epochs": epochs, "peak_vram_gb": r.measured.peak_vram_gb})
        steps.append({"step": f"export {r.name}", "gpu": r.export.gpu, "hours": EXPORT_HOURS, "usd": _usd(EXPORT_HOURS, prices.get(r.export.gpu))})
    known = [s["usd"] for s in steps if s["usd"] is not None]
    return {"scenes": scenes, "candidates": candidates, "accepted_expected": accepted, "style_examples": counts["style_cases"],
            "game_examples_expected": game_examples, "replay_examples": replay, "training_examples_expected": examples,
            "steps": steps, "gpu_usd": round(sum(known), 2) if len(known) == len(steps) else None}


def _usd(hours: float, price: float | None) -> float | None:
    return None if price is None else round(hours * price, 2)
