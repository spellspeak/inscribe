"""Build a game's training mix: its own rows, and general rows replayed so the model keeps the skills it had.

A LoRA trained on a game's rows alone forgets how to do what the base model could. So most of the mix is replay:

- the game's own rows: the teacher's accepted lines and the sheet's speech examples;
- a uniform sample of the base model's general training rows, `to_one` for each game row;
- a fixed number from each of a few families a LoRA loses first (the real-world boundary, long conversations, who
  speaks next in a group).

The general rows come from the replay pack: the rows already rendered as text, in a fixed order, so the same seed
always draws the same rows. The game's rows are rendered here, against the worlds the plan step wrote.

The step writes `shard-NNN.jsonl` ({id, prompt, target} per line) and `manifest.json` to the output folder.

Job (`job.json` in the run folder):
    chat_format  the base model's chat format name
    seed         the sampling seed
    to_one       replay rows per game row
    targeted     {family: rows}
    pack         where the replay pack is: "store:<folder on the store>" or "hf://<dataset>[@revision]"
    accepted     file on the store with the game's accepted rows (training records), or null
    style        file on the store with the speech-example records, or null
    worlds       folder on the store with the plan's world files
    out          folder on the store to write the mix to
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
from collections import Counter

SHARD_ROWS = 1000
PACK_FILES = ("pool.jsonl", "targeted.jsonl")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fetch_pack(work: pathlib.Path, source: str) -> pathlib.Path:
    """The pack's folder inside the work folder. A Hugging Face dataset is downloaded once into the store's cache."""
    if source.startswith("store:"):
        return work / source.removeprefix("store:").strip("/")
    if source.startswith("hf://"):
        from huggingface_hub import snapshot_download

        repo, _, revision = source.removeprefix("hf://").partition("@")
        return pathlib.Path(snapshot_download(repo, repo_type="dataset", revision=revision or None, cache_dir=str(work / "cache" / "hf"),
                                              allow_patterns=[*PACK_FILES, "MANIFEST.json"]))
    raise ValueError(f"replay source {source!r}: use store:<folder> or hf://<dataset>[@revision]")


def load_pack(folder: pathlib.Path, chat_format: str) -> tuple[list[dict], list[dict], dict]:
    """The pack's two files, checked against its manifest. Order matters: sampling is by position."""
    manifest_path = folder / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no replay pack at {folder}: MANIFEST.json is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest["chat_format"] != chat_format:
        raise ValueError(f"the replay pack is rendered for {manifest['chat_format']}, and this base model uses {chat_format}")
    files = []
    for name in PACK_FILES:
        body = (folder / name).read_bytes()
        got = hashlib.sha256(body).hexdigest()
        if got != manifest["files"][name]["sha256"]:
            raise RuntimeError(f"replay pack file {name} is {got[:16]}, its manifest says {manifest['files'][name]['sha256'][:16]}")
        files.append([json.loads(line) for line in body.decode().splitlines() if line.strip()])
    return files[0], files[1], manifest


def render_game_rows(records: list[dict], worlds: dict, chat_format: str) -> list[dict]:
    """A game's training records as text rows: each character's view is rebuilt from the game's own world file, without
    the facts the record hides from the model, exactly as the teacher saw it."""
    from inscribe.runtime.template import format_by_name, render_prompt, render_target
    from inscribe.scenarios.core import student_view
    from inscribe.schemas.records import TrainingExample

    fmt = format_by_name(chat_format)
    rows = []
    for ex in records:
        world = worlds.get(ex["world_id"])
        if world is None:
            raise KeyError(f"{ex['example_id']}: no world file for {ex['world_id']}; run `inscribe plan` again")
        _, _, view = student_view(world, ex["student_input_ref"], session_id=ex["example_id"], hide_facts=ex.get("hidden_fact_ids") or ())
        rec = TrainingExample.model_validate({**ex, "student_input": json.loads(view.model_dump_json())})
        dialogue, reply = rec.dialogue_and_reply()
        rows.append({"id": rec.example_id, "family": rec.provenance.generator_id,
                     "prompt": render_prompt(rec.student_input, dialogue, fmt), "target": render_target(reply, fmt)})
    return rows


def sample_mix(game_rows: list[dict], pool: list[dict], targeted_pool: list[dict], to_one: int, targeted: dict[str, int], seed: int) -> tuple[list[dict], dict]:
    """The mix, in training order, and how many rows each part gave. Sampling is by position in the pack, so one seed
    always draws the same rows."""
    rows = list(game_rows)
    replay_n = to_one * len(rows)
    if replay_n > len(pool):
        raise ValueError(f"{to_one} to 1 replay of {len(rows)} game rows needs {replay_n} rows, and the pack's pool has {len(pool)}. Lower replay.to_one.")
    rows += random.Random(seed).sample(pool, replay_n)
    taken = {r["id"] for r in rows}
    drawn: dict[str, int] = {}
    for family, n in targeted.items():
        candidates = [r for r in targeted_pool if r["family"] == family and r["id"] not in taken]
        picked = random.Random(f"{seed}:{family}").sample(candidates, min(n, len(candidates)))
        drawn[family] = len(picked)
        rows += picked
    random.Random(seed).shuffle(rows)
    return rows, {"game": len(game_rows), "replay": replay_n, "targeted": drawn}


def write_shards(rows: list[dict], out: pathlib.Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("shard-*.jsonl"):
        stale.unlink()
    shards = []
    for i in range(0, len(rows), SHARD_ROWS):
        chunk = rows[i:i + SHARD_ROWS]
        text = "\n".join(json.dumps({"id": r["id"], "prompt": r["prompt"], "target": r["target"]}, ensure_ascii=False) for r in chunk) + "\n"
        name = f"shard-{i // SHARD_ROWS:03d}.jsonl"
        (out / name).write_text(text)
        shards.append({"file": name, "rows": len(chunk), "sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text.encode())})
    return shards


def run(work: pathlib.Path, run_dir: pathlib.Path, job: dict) -> dict:
    import yaml

    from inscribe.schemas.world import World

    worlds = {p.stem: World.model_validate(yaml.safe_load(p.read_text())) for p in sorted((work / job["worlds"]).glob("*.yaml"))}
    records = read_jsonl(work / job["accepted"]) if job.get("accepted") else []
    style = read_jsonl(work / job["style"]) if job.get("style") else []
    if not records and not style:
        raise ValueError("no game rows: neither accepted lines nor speech examples")
    game_rows = render_game_rows([*records, *style], worlds, job["chat_format"])
    if len({r["id"] for r in game_rows}) != len(game_rows):
        raise ValueError("a game row id is used twice")
    print(f"game rows: {len(records)} accepted lines and {len(style)} speech examples, on {len(worlds)} worlds", flush=True)

    pool, targeted_pool, pack = load_pack(fetch_pack(work, job["pack"]), job["chat_format"])
    print(f"replay pack {pack['version']}: {len(pool)} pool rows, {len(targeted_pool)} targeted rows", flush=True)
    rows, parts = sample_mix(game_rows, pool, targeted_pool, job["to_one"], job["targeted"], job["seed"])
    shards = write_shards(rows, work / job["out"])
    manifest = {"examples": len(rows), "parts": parts, "seed": job["seed"], "to_one": job["to_one"], "chat_format": job["chat_format"],
                "by_family": dict(sorted(Counter(r["family"] for r in rows).items())),
                "pack": {"source": job["pack"], "version": pack["version"], "sha256": {k: pack["files"][k]["sha256"] for k in PACK_FILES}},
                "shards": shards}
    (work / job["out"] / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"mix: {len(rows)} rows in {len(shards)} shards ({parts['game']} game, {parts['replay']} replay, {sum(parts['targeted'].values())} targeted)", flush=True)
    return manifest
