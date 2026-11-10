"""The parts of the training step that need no GPU: reading rows, token ids, the job a project sends."""
import json

import pytest

from inscribe.config import Config, load_recipe
from inscribe.runs import train_hours, train_job, train_spec
from inscribe.train.lora import read_rows, tokenize


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class _Tok:
    """One id per word; the word END is the reply-end marker."""

    def encode(self, text, add_special_tokens=False):
        return _Enc([9 if w == "END" else len(w) for w in text.split()])


def test_rows_are_read_in_file_then_line_order(tmp_path):
    (tmp_path / "shard-001.jsonl").write_text(json.dumps({"id": "c", "prompt": "p", "target": "t END"}) + "\n")
    (tmp_path / "shard-000.jsonl").write_text("".join(json.dumps({"id": i, "prompt": "p", "target": "t END"}) + "\n" for i in "ab"))
    assert [r["id"] for r in read_rows(tmp_path)] == ["a", "b", "c"]
    with pytest.raises(FileNotFoundError):
        read_rows(tmp_path / "none")
    (tmp_path / "shard-002.jsonl").write_text(json.dumps({"prompt": "p"}) + "\n")
    with pytest.raises(ValueError, match="`prompt` and a `target`"):
        read_rows(tmp_path)


def test_only_the_target_is_learned_and_long_rows_are_left_out():
    rows = [{"prompt": "who are you", "target": "tomas END"}, {"prompt": "a b c d e f g h", "target": "x END"}]
    examples, too_long = tokenize(rows, _Tok(), bos=[1], end_ids=[9], max_len=8)
    assert too_long == 1 and len(examples) == 1
    assert examples[0]["input_ids"] == [1, 3, 3, 3, 5, 9] and examples[0]["labels"] == [-100, -100, -100, -100, 5, 9]
    with pytest.raises(RuntimeError, match="reply-end marker"):
        tokenize([{"prompt": "p", "target": "no end"}], _Tok(), bos=[1], end_ids=[9], max_len=99)


def test_the_job_is_the_recipe_with_the_configs_overrides():
    recipe = load_recipe("glyph-2b")
    job = train_job(Config(game="g"), recipe, "prompts/recall.jsonl", None)
    assert job["lora"] == {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear", "lr": 2e-4, "epochs": 3, "grad_accum": 16,
                           "warmup_steps": 10, "min_lr_ratio": 0.1, "seed": 0, "max_len": 3072}
    assert job["base"]["repo"] == "jptaylor/spellspeak-glyph-2b" and len(job["base"]["sha256"]["model.safetensors"]) == 64
    assert job["data"] == "data/mix" and job["max_steps"] is None
    changed = train_job(Config(game="g", seed=3, train={"epochs": 2, "rank": 8}), recipe, None, 5)
    assert (changed["lora"]["epochs"], changed["lora"]["r"], changed["lora"]["alpha"], changed["lora"]["seed"], changed["max_steps"]) == (2, 8, 16, 3, 5)


def test_the_step_asks_for_the_recipes_machine():
    recipe = load_recipe("sigil-4b")
    full, check = train_spec(recipe, "sigil-4b", None), train_spec(recipe, "sigil-4b-check", 5)
    assert full.gpu == "H100" and full.timeout_s == 24 * 3600 and check.timeout_s == 3600
    assert full.command() == ("python", "-m", "inscribe.stages.run", "train", "--work", "/work", "--run", "train/sigil-4b")
    job = train_job(Config(game="g"), recipe, None, None)
    assert train_hours(recipe, job, 5355, 620) == pytest.approx((420 + 620 + 5355 * 3 * 0.2815) / 3600)
    assert train_hours(recipe, {**job, "max_steps": 5}, 5355, 620) < 0.25


def test_the_export_job_names_the_files_after_the_game_and_the_base():
    from inscribe.export.gguf import calibration_text, quantize_args, readable_config
    from inscribe.runs import export_job, export_spec

    recipe = load_recipe("glyph-2b")
    job = export_job(Config(game="my_game"), recipe, "glyph-2b", "plan/imatrix.txt", "prompts/recall.jsonl")
    assert job["name"] == "my-game-glyph-2b" and job["quant"] == ["Q5_K_M"] and job["embeddings"] == "q4_k" and job["merged"] == "train/glyph-2b/merged"
    assert export_job(Config(game="g", export={"quant": ["Q4_K_M", "Q8_0"]}), recipe, "glyph-2b", None, None)["quant"] == ["Q4_K_M", "Q8_0"]
    spec = export_spec(recipe, "glyph-2b")
    assert spec.gpu == "L4" and spec.image.base.endswith("full-cuda-b11146") and spec.image.clear_entrypoint
    assert quantize_args("q4_k") == ["--tensor-type", "per_layer_token_embd=q4_k", "--token-embedding-type", "q4_k"] and quantize_args(None) == []
    text = calibration_text("gemma4-instruct")
    assert text.count("<|turn>system") >= 160 and "<|im_start|>" not in text
    cfg = readable_config({"sot_token": "<|turn>", "bos_token": "<bos>", "extra_special_tokens": ["<a>", "<gone>"], "model_max_length": 9}, {"<|turn>", "<a>", "<bos>"})
    assert cfg == {"sot_token": "<|turn>", "bos_token": "<bos>", "model_max_length": 9, "additional_special_tokens": ["<|turn>", "<a>"]}
