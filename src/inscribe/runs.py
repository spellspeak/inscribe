"""Launching a step from a game project, and finding it again later.

A launched step leaves a handle in `build/runs/<stage>-<run_id>.json`. `status`, `logs` and `stop` read those files,
so they work from another terminal, or the next day.
"""
from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Callable

from inscribe import images
from inscribe.backends.base import Backend, RunHandle, StageSpec, read_result
from inscribe.config import Config, Recipe
from inscribe.project import Project

DATA_AT = "data/mix"  # the training rows, on the store
BASE_SECONDS = 420  # fetching and loading a base model that is already on the store; first time is longer
SECONDS_PER_QUESTION = 0.5  # one greedy reply on an H100, from the reference run


def make_backend(config: Config, project_id: str) -> Backend:
    if config.backend.name == "modal":
        from inscribe.backends.modal_backend import ModalBackend

        return ModalBackend(project_id, environment=config.backend.environment, app_prefix=config.backend.app_prefix,
                            usd_per_hour=config.backend.usd_per_hour)
    raise ValueError(f"unknown backend {config.backend.name!r}")


def runs_dir(project: Project) -> pathlib.Path:
    return project.build_dir / "runs"


def save_handle(project: Project, handle: RunHandle) -> pathlib.Path:
    path = runs_dir(project) / f"{handle.stage}-{handle.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(handle.to_json())
    return path


def load_handles(project: Project) -> list[RunHandle]:
    """Every launched step of this project, oldest first."""
    folder = runs_dir(project)
    handles = [RunHandle.from_json(p.read_text()) for p in folder.glob("*.json")] if folder.exists() else []
    return sorted(handles, key=lambda h: h.started)


def find_handle(project: Project, name: str | None) -> RunHandle | None:
    """The step called `name` (`train-glyph-2b`, or just `train` or `glyph-2b`), or the latest when no name is given."""
    handles = load_handles(project)
    if name:
        handles = [h for h in handles if name in (f"{h.stage}-{h.run_id}", h.stage, h.run_id)]
    return handles[-1] if handles else None


def train_job(config: Config, recipe: Recipe, prompts: str | None, max_steps: int | None) -> dict:
    """The training step's parameters: the recipe's settings, with the config's overrides."""
    t = recipe.train
    if t.batch_size != 1:
        raise ValueError("the trainer takes one example at a time; use grad_accumulation for a larger effective batch")
    rank = config.train.rank or t.rank
    return {"base": {"repo": recipe.base.repo, "revision": recipe.base.revision, "sha256": dict(recipe.base.sha256)},
            "chat_format": recipe.chat_format,
            "lora": {"r": rank, "alpha": t.alpha if rank == t.rank else 2 * rank, "dropout": t.dropout, "target_modules": t.target_modules,
                     "lr": config.train.learning_rate or t.learning_rate, "epochs": config.train.epochs or t.epochs,
                     "grad_accum": t.grad_accumulation, "warmup_steps": t.warmup_steps, "min_lr_ratio": t.min_lr_ratio,
                     "seed": config.seed, "max_len": t.max_length},
            "data": DATA_AT, "prompts": prompts, "max_steps": max_steps}


def train_spec(recipe: Recipe, run_id: str, max_steps: int | None) -> StageSpec:
    return StageSpec(stage="train", run_id=run_id, image=images.TRAIN, gpu=recipe.train.gpu, min_vram_gb=recipe.measured.peak_vram_gb,
                     cpu=8.0, memory_mb=65536, timeout_s=60 * 60 if max_steps else 24 * 60 * 60)


def train_hours(recipe: Recipe, job: dict, examples: int, questions: int) -> float:
    """An estimate of the training step's GPU time, from the recipe's one measured run."""
    lora = job["lora"]
    sequences = min(examples * lora["epochs"], (job["max_steps"] or 10**9) * lora["grad_accum"])
    answers = questions * SECONDS_PER_QUESTION * (1 if job["max_steps"] else 2)
    return (BASE_SECONDS + answers + sequences * recipe.measured.train_seconds_per_example_epoch) / 3600


def follow(backend: Backend, handle: RunHandle, echo: Callable[[str], None], poll_s: float = 10.0) -> dict | None:
    """Print the step's output as it comes, then wait for its result. None when it ended without one."""
    for line in backend.logs(handle, follow=True):
        echo(line)
    while backend.status(handle) == "running":
        time.sleep(poll_s)
    result = read_result(backend.store(), handle)
    return result


def count_rows(paths: list[pathlib.Path]) -> int:
    return sum(1 for p in paths for line in p.read_text().splitlines() if line.strip())


def prompt_file(rows: list[dict]) -> str:
    return "".join(json.dumps({"id": r["id"], "prompt": r["prompt"], "max_new": r.get("max_new", 80)}, ensure_ascii=False) + "\n" for r in rows)
