"""The commands a developer runs first: init, check, plan."""
import pathlib

import yaml
from typer.testing import CliRunner

from inscribe.cli import app

runner = CliRunner()


def _init(tmp_path: pathlib.Path, monkeypatch, name: str = "crooked-lantern") -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", name])
    assert result.exit_code == 0, result.output
    return tmp_path / name


def test_init_then_check_then_plan(tmp_path, monkeypatch):
    root = _init(tmp_path, monkeypatch)
    assert yaml.safe_load((root / "lora.yaml").read_text())["game"] == "crooked_lantern"
    assert (root / ".gitignore").read_text().startswith("build/")
    for name in ("world.md", "characters.md", "story.md", "style.md"):
        assert (root / "lore" / name).exists()

    result = runner.invoke(app, ["check", "--project", str(root)])
    assert result.exit_code == 0, result.output
    assert "The sheet is valid" in result.output
    assert not (root / "build").exists(), "check writes nothing"

    result = runner.invoke(app, ["plan", "--project", str(root)])
    assert result.exit_code == 0, result.output
    plan = root / "build" / "plan"
    for name in ("scenarios.jsonl", "style-examples.jsonl", "recall.json", "imatrix.txt", "plan.yaml"):
        assert (plan / name).stat().st_size > 0
    assert sorted(p.stem for p in (plan / "worlds").glob("*.yaml")) == ["crooked_lantern", "crooked_lantern_group", "crooked_lantern_s1", "crooked_lantern_s2"]


def test_init_refuses_a_folder_with_files(tmp_path, monkeypatch):
    root = _init(tmp_path, monkeypatch)
    result = runner.invoke(app, ["init", root.name])
    assert result.exit_code == 1
    assert "not empty" in result.output


def test_check_explains_a_bad_sheet(tmp_path, monkeypatch):
    root = _init(tmp_path, monkeypatch)
    sheet = yaml.safe_load((root / "sheet" / "game-sheet.yaml").read_text())
    sheet["characters"][0]["voice"] = ""
    sheet["setting"]["magic_policy"] = "plenty"
    (root / "sheet" / "game-sheet.yaml").write_text(yaml.safe_dump(sheet))
    result = runner.invoke(app, ["check", "--project", str(root)])
    assert result.exit_code == 1
    assert "characters.0.voice" in result.output and "setting.magic_policy" in result.output


def test_check_explains_a_bad_config(tmp_path, monkeypatch):
    root = _init(tmp_path, monkeypatch)
    for args, needle in ((["--set", "base=nope"], "unknown base model"), (["--set", "replay.to_one=1"], "replay.to_one"),
                         (["--set", "trian.epochs=2"], "trian"), (["--set", "epochs"], "key=value")):
        result = runner.invoke(app, ["check", "--project", str(root), *args])
        assert result.exit_code == 1, args
        assert needle in result.output, result.output


def test_check_needs_the_sheet_and_config_to_agree(tmp_path, monkeypatch):
    root = _init(tmp_path, monkeypatch)
    result = runner.invoke(app, ["check", "--project", str(root), "--set", "game=another"])
    assert result.exit_code == 1
    assert "Make them the same" in result.output


def test_outside_a_project(tmp_path):
    result = runner.invoke(app, ["check", "--project", str(tmp_path)])
    assert result.exit_code == 1
    assert "inscribe init" in result.output


def test_check_on_the_example_games(examples):
    for game, scenes in (("brackenhythe", 974), ("shalecombe", 893)):
        result = runner.invoke(app, ["check", "--project", str(examples / game)])
        assert result.exit_code == 0, result.output
        assert f"Scenes for the teacher to answer: {scenes}" in result.output
        listing = runner.invoke(app, ["check", "--project", str(examples / game), "--lore"])
        assert listing.exit_code == 0 and "lore items get no scene of their own" in listing.output
