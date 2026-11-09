"""The plan step: the same sheet always gives the same files, and nothing private is needed to build them."""
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest
import yaml

from inscribe.plan.build import build_plan, lore_coverage, load_sheet
from inscribe.plan.npcs import load_npcs, npc_from_dict, npcs_to_yaml
from inscribe.plan.recall import foundation_vocabulary, prompt_rows, score

FILES = ("scenarios.jsonl", "style-examples.jsonl", "recall.json", "imatrix.txt")


def _digests(folder: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[pathlib.Path, dict]:
    examples = pathlib.Path(__file__).resolve().parents[1] / "examples" / "brackenhythe"
    out = tmp_path_factory.mktemp("plan")
    info = build_plan(examples / "sheet" / "game-sheet.yaml", out, load_npcs(examples / "npcs.yaml"))
    return out, info


def test_the_manifest_lists_every_file_with_its_hash(built):
    out, info = built
    on_disk = _digests(out)
    on_disk.pop("plan.yaml")
    assert {k: v["sha256"] for k, v in info["files"].items()} == on_disk
    assert yaml.safe_load((out / "plan.yaml").read_text())["counts"] == info["counts"]


def test_two_builds_give_the_same_bytes(built, tmp_path, examples):
    out, _ = built
    game = examples / "brackenhythe"
    build_plan(game / "sheet" / "game-sheet.yaml", tmp_path, load_npcs(game / "npcs.yaml"))
    assert _digests(tmp_path) == _digests(out)


def test_a_build_in_a_fresh_process_gives_the_same_bytes(built, tmp_path, examples):
    """Hash randomisation and import order must not reach the files."""
    out, _ = built
    game = examples / "brackenhythe"
    code = ("import sys; from inscribe.plan.build import build_plan; from inscribe.plan.npcs import load_npcs; "
            "build_plan(sys.argv[1], sys.argv[2], load_npcs(sys.argv[3]))")
    subprocess.run([sys.executable, "-c", code, str(game / "sheet" / "game-sheet.yaml"), str(tmp_path), str(game / "npcs.yaml")],
                   check=True, env={"PYTHONHASHSEED": "12345", "PATH": ""})
    assert _digests(tmp_path) == _digests(out)


def test_prompts_are_in_the_base_models_chat_format(built):
    out, info = built
    assert info["chat_format"] == "gemma4-instruct"
    bundle = json.loads((out / "recall.json").read_text())
    assert bundle["format"] == "gemma4-instruct"
    text = (out / "imatrix.txt").read_text() + "".join(r["prompt"] for r in prompt_rows(bundle))
    assert "<|turn>" in text and "<|im_start|>" not in text


def test_no_held_out_question_is_a_training_scene(built):
    out, _ = built
    bundle = json.loads((out / "recall.json").read_text())
    cases = [json.loads(line) for line in (out / "scenarios.jsonl").read_text().splitlines() if line.strip()]
    trained = {c.get("player_utterance") or c["last_turn"]["text"] for c in cases}
    held = {r["question"] for r in bundle["lore"]}
    assert held and not held & trained


def test_only_the_prompts_travel_to_a_container(built):
    out, _ = built
    bundle = json.loads((out / "recall.json").read_text())
    assert all(set(r) == {"id", "prompt", "max_new"} for r in prompt_rows(bundle, "gemma4-instruct"))
    with pytest.raises(ValueError):
        prompt_rows(bundle, "qwen3.5-2b-base")


def test_scoring_reads_the_answer_keys(built):
    out, _ = built
    bundle = json.loads((out / "recall.json").read_text())
    right = [{"id": r["id"], "text": r["expect_any"][0]} for r in bundle["lore"]]
    assert score(bundle, right)["lore"]["items_recalled"] == bundle["counts"]["lore_items"]
    blank = [{"id": r["id"], "text": "Couldn't tell you."} for r in bundle["lore"]]
    assert score(bundle, blank)["lore"]["items_recalled"] == 0
    told = [{"id": r["id"], "text": r["must_not"][0]} for r in bundle["spoiler"] if r["must_not"]]
    assert told and score(bundle, told)["spoiler"]["told_early"] == len(told)
    assert score(bundle, [{"id": r["id"], "text": r["expect_any"][0] + "<turn|>" + " ".join(r["must_not"])} for r in bundle["dynamic"]])["dynamic"]["secret_leaks"] == 0


def test_lore_coverage_agrees_with_the_plan(built, examples):
    _, info = built
    rows = lore_coverage(load_sheet(examples / "brackenhythe" / "sheet" / "game-sheet.yaml"))
    assert len(rows) == info["counts"]["lore_items"]
    assert sum(1 for _, why in rows if why is None) == info["counts"]["lore_items_asked"]


def test_the_common_word_list_ships_with_the_package():
    words = foundation_vocabulary()
    assert len(words) == 6173 and "the" in words and all(w.isalpha() and w.islower() for w in words)


def test_npcs_round_trip_and_are_checked(examples, tmp_path):
    npcs = load_npcs(examples / "brackenhythe" / "npcs.yaml")
    assert len(npcs) == 3
    (tmp_path / "npcs.yaml").write_text(npcs_to_yaml(npcs))
    assert load_npcs(tmp_path / "npcs.yaml") == npcs
    assert load_npcs(tmp_path / "missing.yaml") == ()
    with pytest.raises(ValueError, match="unknown field"):
        npc_from_dict({"id": "npc_x", "name": "X", "role": "r", "voice": "v", "secret": "no"})
    with pytest.raises(ValueError, match="`voice` is required"):
        npc_from_dict({"id": "npc_x", "name": "X", "role": "r"})


def test_an_npc_already_in_the_sheet_is_refused(examples, tmp_path):
    game = examples / "brackenhythe"
    clash = yaml.safe_dump({"npcs": [{"id": "npc_maren_tolliver", "name": "Maren", "role": "r", "voice": "v"}]})
    (tmp_path / "npcs.yaml").write_text(clash)
    with pytest.raises(ValueError, match="already in the world"):
        build_plan(game / "sheet" / "game-sheet.yaml", tmp_path / "out", load_npcs(tmp_path / "npcs.yaml"))
