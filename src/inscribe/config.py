"""A game project's `lora.yaml`, and the recipe for each base model.

    config = load_config("lora.yaml", overrides=["train.epochs=2"])
    recipe = load_recipe("glyph-2b")

The config says what to train and where. A recipe holds the settings one base model was measured with. Anything a
config leaves out comes from the recipe.
"""
from __future__ import annotations

import pathlib
from importlib import resources
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MIN_REPLAY_TO_ONE = 3  # below this the model starts to lose skills it had


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackendConfig(_Strict):
    """Where the GPU steps run."""

    name: Literal["modal"] = "modal"
    environment: str | None = None  # a Modal environment; the workspace default when not set
    app_prefix: str = "inscribe"  # every app and volume this tool makes starts with it
    usd_per_hour: dict[str, float] = {"H100": 4.5, "L4": 0.8}  # estimates for the forecast, not quotes


class LLMConfig(_Strict):
    """The outside model that reads lore into a sheet and judges lines. The key stays on this machine."""

    provider: Literal["openai"] = "openai"
    key_env: str = "OPENAI_API_KEY"


class ReplayConfig(_Strict):
    """General examples mixed in so the model keeps the skills it had."""

    source: str | None = None  # where the replay pack is: a folder, or hf://<dataset>
    to_one: int = Field(4, ge=MIN_REPLAY_TO_ONE)  # replay examples per game example


class TeacherConfig(_Strict):
    candidates: int = Field(4, ge=1, le=8)  # lines the teacher writes per scene; the checks keep the good ones


class TrainConfig(_Strict):
    """Overrides for the recipe's training settings. Leave them out to train as measured."""

    epochs: int | None = Field(None, ge=1, le=10)
    rank: int | None = Field(None, ge=1, le=256)
    learning_rate: float | None = Field(None, gt=0)


class ExportConfig(_Strict):
    quant: list[str] | None = None  # GGUF types to export; the recipe's when not set


class BudgetConfig(_Strict):
    max_usd: float = Field(80, gt=0)  # `run` stops before a step whose forecast would pass this


class Config(_Strict):
    game: str
    base: list[str] = ["glyph-2b"]
    seed: int = 0
    backend: BackendConfig = BackendConfig()
    llm: LLMConfig = LLMConfig()
    replay: ReplayConfig = ReplayConfig()
    teacher: TeacherConfig = TeacherConfig()
    train: TrainConfig = TrainConfig()
    export: ExportConfig = ExportConfig()
    budget: BudgetConfig = BudgetConfig()

    @field_validator("base", mode="before")
    @classmethod
    def _one_or_many(cls, v):
        return [v] if isinstance(v, str) else v

    @field_validator("base")
    @classmethod
    def _known(cls, v):
        known = recipe_names()
        for name in v:
            if name not in known:
                raise ValueError(f"unknown base model {name!r}; one of {', '.join(known)}")
        if len(set(v)) != len(v):
            raise ValueError("a base model is listed twice")
        return v


class BaseModelPin(_Strict):
    repo: str
    revision: str
    tag: str
    sha256: dict[str, str]


class RecipeTrain(_Strict):
    rank: int
    alpha: int
    dropout: float
    target_modules: str
    learning_rate: float
    epochs: int
    batch_size: int
    grad_accumulation: int
    max_length: int
    warmup_steps: int
    gpu: str


class RecipeExport(_Strict):
    quant: list[str]
    embeddings: str
    llama_cpp_build: str
    gpu: str


class RecipeMeasured(_Strict):
    """Numbers from one training run of the example game, used only for the forecast."""

    train_seconds_per_example_epoch: float
    peak_vram_gb: float
    gpu: str


class Recipe(_Strict):
    name: str
    title: str
    base: BaseModelPin
    chat_format: str
    train: RecipeTrain
    export: RecipeExport
    measured: RecipeMeasured


class ConfigError(ValueError):
    """A config or recipe that does not validate, with one line per problem."""


def _explain(where: str, exc: ValidationError) -> ConfigError:
    lines = [f"{where}:"]
    for e in exc.errors():
        path = ".".join(str(p) for p in e["loc"]) or "(top level)"
        lines.append(f"  {path}: {e['msg']}")
    return ConfigError("\n".join(lines))


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """`train.epochs=2` style overrides on a config's raw mapping. Values are read as YAML."""
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set {item!r}: write it as key=value, for example train.epochs=2")
        node = raw
        *parents, leaf = key.strip().split(".")
        for part in parents:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"--set {item!r}: {part!r} is not a section")
            node = child
        node[leaf] = yaml.safe_load(value)
    return raw


def load_config(path: str | pathlib.Path, overrides: list[str] | tuple[str, ...] = ()) -> Config:
    path = pathlib.Path(path)
    if not path.exists():
        raise ConfigError(f"{path} not found. Run `inscribe init <name>` to start a game project.")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping of settings")
    raw = apply_overrides(raw, list(overrides))
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise _explain(str(path), exc) from None


def recipe_names() -> list[str]:
    return sorted(p.name.removesuffix(".yaml") for p in resources.files("inscribe.recipes").iterdir() if p.name.endswith(".yaml"))


def load_recipe(name: str) -> Recipe:
    if name not in recipe_names():
        raise ConfigError(f"unknown base model {name!r}; one of {', '.join(recipe_names())}")
    raw = yaml.safe_load(resources.files("inscribe.recipes").joinpath(f"{name}.yaml").read_text())
    try:
        return Recipe.model_validate(raw)
    except ValidationError as exc:
        raise _explain(f"recipe {name}", exc) from None
