"""What a compute backend is: somewhere to run a step, and a store the step and this machine both see.

A step is a command that runs inside an image with a work folder mounted. It reads its inputs from the work folder
and writes its outputs and a `result.json` back to it. A backend only has to build the image, run the command with
the resources asked for, and let this machine put files into the work folder and get them out again. Modal is the
first backend. A machine with Docker and a GPU, or another cloud, fits the same shape.

An image is a recipe, not a name, so each backend can build it its own way.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

WORK = "/work"  # where the store is mounted inside a step's container
Status = Literal["running", "succeeded", "failed", "stopped", "unknown"]


@dataclass(frozen=True)
class PipLayer:
    """Packages installed together, optionally from another index."""

    packages: tuple[str, ...]
    index_url: str | None = None


@dataclass(frozen=True)
class ImageSpec:
    """How to build a step's image. `base` is "debian-slim" or a registry reference."""

    base: str = "debian-slim"
    python: str = "3.12"
    apt: tuple[str, ...] = ()
    pip: tuple[PipLayer, ...] = ()
    run: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    clear_entrypoint: bool = False  # for a registry image that starts its own program


@dataclass(frozen=True)
class StageSpec:
    """One step to run. Its parameters are in `<stage>/<run_id>/job.json` on the store, written before launch."""

    stage: str
    run_id: str
    image: ImageSpec
    gpu: str | None = None
    min_vram_gb: float = 0
    cpu: float = 2.0
    memory_mb: int = 4096
    timeout_s: int = 3600
    env: Mapping[str, str] = field(default_factory=dict)
    secret_env: tuple[str, ...] = ()  # names of environment variables on this machine to pass in; values are never stored

    @property
    def run_dir(self) -> str:
        return f"{self.stage}/{self.run_id}"

    def command(self, work: str = WORK) -> tuple[str, ...]:
        return ("python", "-m", "inscribe.stages.run", self.stage, "--work", work, "--run", self.run_dir)


@dataclass(frozen=True)
class RunHandle:
    """Enough to find a launched step again from another process. Saved as JSON in the project's build folder."""

    backend: str
    stage: str
    run_id: str
    id: str  # the backend's own id for the run
    started: str
    gpu: str | None = None
    where: Mapping[str, str] = field(default_factory=dict)  # backend detail: app, store, environment

    @property
    def run_dir(self) -> str:
        return f"{self.stage}/{self.run_id}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "RunHandle":
        return cls(**json.loads(text))


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store(Protocol):
    """The work folder, seen from this machine. Paths are relative to it and use forward slashes."""

    def put(self, local: pathlib.Path, remote: str) -> None: ...
    def get(self, remote: str, local: pathlib.Path) -> None: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def read_text(self, remote: str) -> str | None: ...
    def write_text(self, remote: str, text: str) -> None: ...


class Backend(Protocol):
    name: str

    def check(self) -> list[str]:
        """Problems that would stop a run, in plain words. Empty when ready."""

    def store(self) -> Store: ...

    def launch(self, spec: StageSpec) -> RunHandle:
        """Start the step and return at once. The step keeps running if this machine goes away."""

    def status(self, handle: RunHandle) -> Status: ...

    def logs(self, handle: RunHandle, follow: bool = False) -> Iterator[str]: ...

    def stop(self, handle: RunHandle) -> None: ...

    def hourly_usd(self, spec: StageSpec) -> float | None:
        """An estimate of what the step's machine costs per hour, or None when not known."""

    def cleanup(self) -> None:
        """Leave nothing running or billing once every step has finished."""


def read_result(store: Store, handle_or_spec) -> dict | None:
    """A finished step's result, or None while it has not written one. This file is what says a step is done."""
    text = store.read_text(f"{handle_or_spec.run_dir}/result.json")
    return json.loads(text) if text else None
