"""The mix step: the same seed draws the same rows, in the same order, and the game's rows render from its worlds."""
import json
import random

import pytest

from inscribe.backends.base import StageSpec, read_result
from inscribe.backends.local import LocalBackend
from inscribe.data.mix import load_pack, render_game_rows, sample_mix, write_shards
from inscribe.images import MIX
from inscribe.plan.build import build_plan
from inscribe.plan.npcs import load_npcs


def _pack(folder, pool_n=40, families=(("RW-01", 12), ("DL-06", 5), ("GS-04", 3))):
    import hashlib

    folder.mkdir(parents=True)
    pool = [{"id": f"p{i}", "family": "ID-01", "prompt": f"prompt {i}", "target": f"target {i}"} for i in range(pool_n)]
    targeted = [{"id": f"t{f}{i}", "family": f, "prompt": f"q {f} {i}", "target": "a"} for f, n in families for i in range(n)]
    manifest = {"version": "test", "chat_format": "gemma4-instruct", "files": {}}
    for name, rows in (("pool.jsonl", pool), ("targeted.jsonl", targeted)):
        body = "".join(json.dumps(r) + "\n" for r in rows).encode()
        (folder / name).write_bytes(body)
        manifest["files"][name] = {"rows": len(rows), "sha256": hashlib.sha256(body).hexdigest()}
    (folder / "MANIFEST.json").write_text(json.dumps(manifest))
    return pool, targeted


def test_sampling_is_the_reference_algorithm():
    game = [{"id": f"g{i}", "family": "LW-01", "prompt": "p", "target": "t"} for i in range(5)]
    pool = [{"id": f"p{i}", "family": "ID-01"} for i in range(100)]
    targeted = [{"id": f"t{i}", "family": "RW-01" if i % 2 else "DL-06"} for i in range(30)]
    rows, parts = sample_mix(game, pool, targeted, 4, {"RW-01": 6, "DL-06": 99}, seed=0)
    # by hand: the game's rows, a seeded sample of the pool, a seeded sample per family, then one seeded shuffle
    want = list(game) + random.Random(0).sample(pool, 20)
    want += random.Random("0:RW-01").sample([r for r in targeted if r["family"] == "RW-01"], 6)
    want += random.Random("0:DL-06").sample([r for r in targeted if r["family"] == "DL-06"], 15)
    random.Random(0).shuffle(want)
    assert rows == want and parts == {"game": 5, "replay": 20, "targeted": {"RW-01": 6, "DL-06": 15}}
    assert sample_mix(game, pool, targeted, 4, {"RW-01": 6}, seed=1)[0] != rows[:31]
    with pytest.raises(ValueError, match="Lower replay.to_one"):
        sample_mix(game, pool[:10], targeted, 4, {}, seed=0)


def test_a_pack_is_checked_against_its_manifest(tmp_path):
    pool, targeted = _pack(tmp_path / "pack")
    got_pool, got_targeted, manifest = load_pack(tmp_path / "pack", "gemma4-instruct")
    assert got_pool == pool and got_targeted == targeted and manifest["version"] == "test"
    with pytest.raises(ValueError, match="rendered for"):
        load_pack(tmp_path / "pack", "another-format")
    (tmp_path / "pack" / "pool.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="its manifest says"):
        load_pack(tmp_path / "pack", "gemma4-instruct")
    with pytest.raises(FileNotFoundError):
        load_pack(tmp_path / "none", "gemma4-instruct")


def test_shards_hold_only_what_training_reads(tmp_path):
    rows = [{"id": f"r{i}", "family": "x", "prompt": "é", "target": "t"} for i in range(2500)]
    (tmp_path / "shard-009.jsonl").write_text("stale\n")
    shards = write_shards(rows, tmp_path)
    assert [s["rows"] for s in shards] == [1000, 1000, 500] and sorted(p.name for p in tmp_path.glob("*.jsonl")) == ["shard-000.jsonl", "shard-001.jsonl", "shard-002.jsonl"]
    first = json.loads((tmp_path / "shard-000.jsonl").read_text().splitlines()[0])
    assert first == {"id": "r0", "prompt": "é", "target": "t"}


def test_the_whole_step_on_the_example_game(tmp_path, examples):
    """Plan the example game, then mix its speech examples with a small pack, through the step runner."""
    game = examples / "brackenhythe"
    backend = LocalBackend(tmp_path / "store")
    store = backend.store()
    build_plan(game / "sheet" / "game-sheet.yaml", tmp_path / "store" / "plan", load_npcs(game / "npcs.yaml"))
    _pack(tmp_path / "store" / "replay" / "test", pool_n=80)
    spec = StageSpec(stage="mix", run_id="game", image=MIX)
    job = {"chat_format": "gemma4-instruct", "seed": 0, "to_one": 4, "targeted": {"RW-01": 10, "GS-04": 9}, "pack": "store:replay/test",
           "accepted": None, "style": "plan/style-examples.jsonl", "worlds": "plan/worlds", "out": "data/mix"}
    store.write_text(f"{spec.run_dir}/job.json", json.dumps(job))
    handle = backend.launch(spec)
    import time
    while backend.status(handle) == "running":
        time.sleep(0.1)
    result = read_result(store, handle)
    assert result["ok"], result.get("error")
    r = result["result"]
    assert r["parts"] == {"game": 13, "replay": 52, "targeted": {"RW-01": 10, "GS-04": 3}} and r["examples"] == 78
    rows = [json.loads(line) for line in store.read_text("data/mix/shard-000.jsonl").splitlines()]
    mine = [x for x in rows if x["id"].startswith("game_")]
    assert len(rows) == 78 and len(mine) == 13
    assert all(x["prompt"].startswith("<|turn>system\n") and x["target"].endswith("<turn|>") for x in mine)
    assert json.loads(store.read_text("data/mix/manifest.json"))["shards"][0]["rows"] == 78


def test_game_rows_need_their_world():
    with pytest.raises(KeyError, match="no world file"):
        render_game_rows([{"example_id": "x", "world_id": "nowhere", "student_input_ref": "worldgen:nowhere#a"}], {}, "gemma4-instruct")
