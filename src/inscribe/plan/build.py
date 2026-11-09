"""The plan step: a checked game sheet in, everything the later steps take out. Free, local and deterministic.

    info = build_plan("sheet/game-sheet.yaml", "build/plan", npcs=load_npcs("npcs.yaml"))

It writes, under the output folder:

- `worlds/<world_id>.yaml`: the world files the scenes name. The base world, a group copy with each character's
  confidants present, and one world per story stage.
- `scenarios.jsonl`: the scenes the teacher model answers (lore, the real-world boundary, the story, secrets and
  ordinary talk, with misheard twins).
- `style-examples.jsonl`: training records built straight from the sheet's speech examples. No model wrote them.
- `recall.json`: held-out questions with their answer keys, for scoring the trained model. Never trained on.
- `imatrix.txt`: the game's own prompts, used to weigh the weights when the model is quantized.
- `plan.yaml`: counts and a hash of every file.

The same sheet, NPCs, seed and package version always give the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random

import yaml

from inscribe.data.examples import code_fingerprint, to_training_example
from inscribe.plan import recall as R
from inscribe.plan import scenarios as S
from inscribe.plan.world import GENERATOR_VERSION, DynamicNPC, game_worlds
from inscribe.runtime.template import DEFAULT_FORMAT, TEMPLATE_VERSION, ChatFormat
from inscribe.schemas.game_sheet import GameSheet

IMATRIX_PROMPTS = 160  # how many of the game's teacher scenes go into the importance-matrix text
MIN_REPLAY_TO_ONE = 3  # below this the model starts to lose skills it had
_PACKAGE = pathlib.Path(__file__).resolve().parents[1]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sheet(path: str | pathlib.Path) -> GameSheet:
    return GameSheet.model_validate(yaml.safe_load(pathlib.Path(path).read_text()))


def source_fingerprint() -> str:
    """A short hash of the code that builds scenes and records, written into each record's provenance."""
    return code_fingerprint([p for d in ("plan", "scenarios", "data") for p in (_PACKAGE / d).rglob("*.py")])


def style_examples(cases: list[dict], code_hash: str) -> list[dict]:
    """Training records for the style cases, through the same function that turns an accepted teacher line into a
    record. Provenance names the sheet as the teacher, so nothing pretends a model wrote these lines."""
    out = []
    for c in cases:
        if not S.is_style(c):
            continue
        world_id = c["student_input_ref"].removeprefix("worldgen:").split("#")[0]
        row = {"case": c, "world_id": world_id, "generator_version": GENERATOR_VERSION, "prompt_hash": "none",
               "candidate": 0, "finish_reason": "stop", "text": S.style_reply(c),
               "teacher": {"id": "game-sheet", "revision": "game-sheet-0.2", "prompt_version": "none", "sampling": {}}}
        ex = to_training_example(row, S.style_reply(c), 0.0, code_hash)
        ex["example_id"] = "game_" + c["case_id"]
        ex["validation"]["oracle_consistency"] = "speech example from the confirmed sheet; no judge"
        out.append(ex)
    return out


def imatrix_text(cases: list[dict], worlds, fmt: ChatFormat = DEFAULT_FORMAT, n: int = IMATRIX_PROMPTS, seed: int = 0) -> str:
    """The game's own prompts for the importance matrix: a seeded sample of the teacher cases plus every style case."""
    rng = random.Random(f"{GENERATOR_VERSION}:imatrix:{seed}")
    teacher = [c for c in cases if not S.is_style(c)]
    chosen = rng.sample(teacher, min(n, len(teacher))) + [c for c in cases if S.is_style(c)]
    return "\n\n".join(S.resolve_game(c, worlds).prompt(fmt) for c in chosen)


def _by_generator(cases: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cases:
        g = c["generator_like"] + ("+noisy" if c.get("noise_twin_of") else "")
        out[g] = out.get(g, 0) + 1
    return dict(sorted(out.items()))


def plan_counts(sheet: GameSheet, npcs: tuple[DynamicNPC, ...] | list[DynamicNPC] = (), fmt: ChatFormat = DEFAULT_FORMAT, seed: int = 0) -> dict:
    """Everything `build_plan` would write, in memory, with its counts. Nothing touches the disk."""
    worlds = game_worlds(sheet)
    skipped: dict[str, int] = {}
    cases = S.game_scenarios(sheet, seed, worlds, skipped)
    teacher_cases = [c for c in cases if not S.is_style(c)]
    style = style_examples(cases, source_fingerprint())
    bundle = R.recall_bundle(sheet, tuple(npcs), fmt, seed)
    counts = {"cases": len(cases), "teacher_cases": len(teacher_cases), "style_cases": len(style), "by_kind": S.counts_by_kind(cases),
              "by_generator": _by_generator(cases), "skipped": dict(sorted(skipped.items())), "worlds": sorted(worlds),
              "lore_items": len(sheet.lore_items()),
              "lore_items_asked": len({c["hidden_fact_ids"][0] for c in cases if c["generator_like"] == "LW-01"}),
              "recall": bundle["counts"]}
    return {"worlds": worlds, "cases": cases, "teacher_cases": teacher_cases, "style": style, "bundle": bundle, "counts": counts}


def lore_coverage(sheet: GameSheet, seed: int = 0) -> list[tuple[str, str | None]]:
    """Every lore item with the reason it gets no scene of its own, or None when it gets one. The reasons are the
    ones `scenarios.lore_cases` counts, in the same order."""
    world = game_worlds(sheet)[sheet.game_id]
    out = []
    for ask in S.lore_plan(sheet, world, seed):
        if S.in_rules(sheet, ask.item):
            why = "lore_in_rules"
        elif ask.pattern == "none" or not ask.asks:
            why = "lore_no_question"
        elif not ask.knowers:
            why = "lore_in_every_prompt"
        elif not [a for a in ask.asks if not S.crosses_firewall(a)]:
            why = "lore_firewall"
        else:
            why = None
        out.append((ask.item.text, why))
    return out


def build_plan(sheet_path: str | pathlib.Path, out_dir: str | pathlib.Path, npcs: tuple[DynamicNPC, ...] | list[DynamicNPC] = (),
               fmt: ChatFormat = DEFAULT_FORMAT, seed: int = 0) -> dict:
    """Build the plan for one game and write it under `out_dir`. Returns the manifest that is written as plan.yaml."""
    sheet_path, out = pathlib.Path(sheet_path), pathlib.Path(out_dir)
    sheet = load_sheet(sheet_path)
    built = plan_counts(sheet, npcs, fmt, seed)
    worlds, cases = built["worlds"], built["cases"]

    out.mkdir(parents=True, exist_ok=True)
    (out / "worlds").mkdir(exist_ok=True)
    for stale in (out / "worlds").glob("*.yaml"):
        if stale.stem not in worlds:
            stale.unlink()
    for wid, world in worlds.items():
        (out / "worlds" / f"{wid}.yaml").write_text(yaml.safe_dump(world.model_dump(exclude_none=True), sort_keys=False, allow_unicode=True, width=120))
    (out / "scenarios.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in built["teacher_cases"]))
    (out / "style-examples.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in built["style"]))
    (out / "recall.json").write_text(json.dumps(built["bundle"], indent=1, ensure_ascii=False) + "\n")
    (out / "imatrix.txt").write_text(imatrix_text(cases, worlds, fmt, IMATRIX_PROMPTS, seed))

    info = {"game_id": sheet.game_id, "generator_version": GENERATOR_VERSION, "template": TEMPLATE_VERSION, "chat_format": fmt.name,
            "seed": seed, "sheet_sha256": sha256(sheet_path), "npcs": [n.id for n in npcs], "code_fingerprint": source_fingerprint(),
            "counts": built["counts"], "files": {}}
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "plan.yaml"):
        info["files"][str(f.relative_to(out))] = {"sha256": sha256(f), "bytes": f.stat().st_size}
    (out / "plan.yaml").write_text("# Built by `inscribe plan` from the game sheet. Safe to delete: the same sheet gives the same files.\n"
                                   + yaml.safe_dump(info, sort_keys=False, allow_unicode=True))
    return info
