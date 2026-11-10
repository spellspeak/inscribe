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


def train_job(config: Config, recipe: Recipe, prompts: str | None, max_steps: int | None, keep: bool = False) -> dict:
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
            "data": DATA_AT, "prompts": prompts, "max_steps": max_steps, "keep": bool(keep and max_steps)}


def train_spec(recipe: Recipe, run_id: str, max_steps: int | None) -> StageSpec:
    return StageSpec(stage="train", run_id=run_id, image=images.TRAIN, gpu=recipe.train.gpu, min_vram_gb=recipe.measured.peak_vram_gb,
                     cpu=8.0, memory_mb=65536, timeout_s=60 * 60 if max_steps else 24 * 60 * 60)


def train_hours(recipe: Recipe, job: dict, examples: int, questions: int) -> float:
    """An estimate of the training step's GPU time, from the recipe's one measured run."""
    lora = job["lora"]
    sequences = min(examples * lora["epochs"], (job["max_steps"] or 10**9) * lora["grad_accum"])
    answers = questions * SECONDS_PER_QUESTION * (1 if job["max_steps"] and not job.get("keep") else 2)
    return (BASE_SECONDS + answers + sequences * recipe.measured.train_seconds_per_example_epoch) / 3600


CANDIDATES_AT = "data/candidates.jsonl"  # the teacher's lines, on the store


def fetch_spec() -> StageSpec:
    return StageSpec(stage="fetch", run_id="teacher", image=images.FETCH, cpu=4.0, memory_mb=8192, timeout_s=2 * 60 * 60)


def teacher_job(config: Config, limit: int | None) -> dict:
    return {"scenarios": "plan/scenarios.jsonl", "worlds": "plan/worlds", "candidates": config.teacher.candidates, "limit": limit, "out": CANDIDATES_AT}


def teacher_spec() -> StageSpec:
    return StageSpec(stage="teacher", run_id="game", image=images.TEACHER, gpu="H100", min_vram_gb=70, cpu=8.0, memory_mb=65536, timeout_s=3 * 60 * 60)


def teacher_hours(scenes: int, candidates: int) -> float:
    """An estimate of the teacher step's GPU time, from one measured run: loading, then the lines."""
    from inscribe.forecast import TEACHER_LOAD_SECONDS, TEACHER_SECONDS_PER_CANDIDATE

    return (TEACHER_LOAD_SECONDS + scenes * candidates * TEACHER_SECONDS_PER_CANDIDATE) / 3600


ACCEPTED_AT = "data/accepted.jsonl"  # the game's accepted teacher lines, on the store


def mix_job(config: Config, recipe: Recipe, accepted: bool) -> dict:
    """The mix step's parameters: the config's replay ratio and source, the recipe's targeted families."""
    return {"chat_format": recipe.chat_format, "seed": config.seed, "to_one": config.replay.to_one, "targeted": dict(recipe.replay.targeted),
            "pack": config.replay.source, "accepted": ACCEPTED_AT if accepted else None, "style": "plan/style-examples.jsonl",
            "worlds": "plan/worlds", "out": DATA_AT}


def mix_spec() -> StageSpec:
    return StageSpec(stage="mix", run_id="game", image=images.MIX, cpu=2.0, memory_mb=8192, timeout_s=60 * 60)


def export_job(config: Config, recipe: Recipe, train_run: str, imatrix: str | None, prompts: str | None) -> dict:
    """The export step's parameters. The files are named after the game and the base model."""
    return {"merged": f"train/{train_run}/merged", "chat_format": recipe.chat_format, "name": f"{config.game}-{recipe.name}".replace("_", "-"),
            "title": f"{config.game} on {recipe.title}", "quant": list(config.export.quant or recipe.export.quant),
            "embeddings": recipe.export.embeddings, "imatrix": imatrix, "prompts": prompts}


def export_spec(recipe: Recipe, run_id: str) -> StageSpec:
    return StageSpec(stage="export", run_id=run_id, image=images.EXPORT, gpu=recipe.export.gpu, cpu=4.0, memory_mb=32768, timeout_s=3 * 60 * 60)


def export_hours(job: dict, questions: int) -> float:
    """An estimate: convert and importance matrix, then quantize and answer per file. Not yet measured alone."""
    return (900 + len(job["quant"]) * (300 + questions * 1.0)) / 3600


def follow(backend: Backend, handle: RunHandle, echo: Callable[[str], None], poll_s: float = 10.0) -> dict | None:
    """Print the step's output as it comes, then wait for its result. None when it ended without one."""
    for line in backend.logs(handle, follow=True):
        echo(line)
    while backend.status(handle) == "running":
        time.sleep(poll_s)
    result = read_result(backend.store(), handle)
    return result


def pull_list(store, base: str, adapter: bool = False) -> list[tuple[str, str]]:
    """What `pull` brings from the store for one base model, as (path on the store, path under out/). The GGUF files
    and each step's result and answers always; the adapter when asked. The merged model stays on the store: it is
    the size of the base model, and the GGUF and the adapter are what a game needs."""
    wanted = []
    for remote in store.list(f"export/{base}"):
        tail = remote.removeprefix(f"export/{base}/")
        if tail.startswith("gguf/"):
            wanted.append((remote, tail.removeprefix("gguf/")))
        elif tail == "result.json" or tail.startswith("answers-"):
            wanted.append((remote, f"{base}/export-{tail}"))
    for remote in store.list(f"train/{base}"):
        tail = remote.removeprefix(f"train/{base}/")
        if tail == "result.json" or tail.startswith("answers-"):
            wanted.append((remote, f"{base}/train-{tail}"))
        elif adapter and tail.startswith("adapter/"):
            wanted.append((remote, f"{base}/{tail}"))
    return wanted


def count_rows(paths: list[pathlib.Path]) -> int:
    return sum(1 for p in paths for line in p.read_text().splitlines() if line.strip())


def prompt_file(rows: list[dict]) -> str:
    return "".join(json.dumps({"id": r["id"], "prompt": r["prompt"], "max_new": r.get("max_new", 80)}, ensure_ascii=False) + "\n" for r in rows)
