"""lora.yaml and the recipes."""
import pytest

from inscribe.config import Config, ConfigError, apply_overrides, load_config, load_recipe, recipe_names
from inscribe.forecast import forecast


def test_the_recipes_load_and_pin_their_base():
    assert recipe_names() == ["glyph-2b", "sigil-4b"]
    for name in recipe_names():
        r = load_recipe(name)
        assert r.name == name and r.chat_format == "gemma4-instruct"
        assert len(r.base.revision) == 40 and len(r.base.sha256["model.safetensors"]) == 64
        assert (r.train.rank, r.train.alpha, r.train.epochs, r.train.max_length) == (16, 32, 3, 3072)


def test_defaults_and_one_base_as_a_string(tmp_path):
    (tmp_path / "lora.yaml").write_text("game: g\nbase: sigil-4b\n")
    c = load_config(tmp_path / "lora.yaml")
    assert c.base == ["sigil-4b"] and c.replay.to_one == 4 and c.teacher.candidates == 4 and c.backend.name == "modal"


def test_overrides_are_read_as_yaml():
    raw = apply_overrides({"game": "g"}, ["train.epochs=2", "base=[glyph-2b, sigil-4b]", "backend.environment=main"])
    c = Config.model_validate(raw)
    assert c.train.epochs == 2 and c.base == ["glyph-2b", "sigil-4b"] and c.backend.environment == "main"


def test_replay_below_three_to_one_is_refused(tmp_path):
    (tmp_path / "lora.yaml").write_text("game: g\nreplay:\n  to_one: 2\n")
    with pytest.raises(ConfigError, match="replay.to_one"):
        load_config(tmp_path / "lora.yaml")


def test_the_forecast_adds_up():
    config = Config(game="g", base=["glyph-2b", "sigil-4b"])
    fc = forecast({"teacher_cases": 974, "style_cases": 13}, config, [load_recipe(b) for b in config.base])
    assert fc["candidates"] == 974 * 4
    assert fc["training_examples_expected"] == fc["game_examples_expected"] + fc["replay_examples"]
    assert [s["step"] for s in fc["steps"]] == ["teacher", "train glyph-2b", "export glyph-2b", "train sigil-4b", "export sigil-4b"]
    assert fc["gpu_usd"] == pytest.approx(sum(s["usd"] for s in fc["steps"]), abs=0.01)
    unpriced = Config(game="g", backend={"usd_per_hour": {}})
    assert forecast({"teacher_cases": 10, "style_cases": 0}, unpriced, [load_recipe("glyph-2b")])["gpu_usd"] is None
