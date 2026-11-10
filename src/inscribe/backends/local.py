"""Run steps on this machine, with this Python and a folder as the store.

The tests use it. It builds no image: whatever a step imports must already be installed here, and a training step
needs a GPU. It is not offered in `lora.yaml` yet.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator

from inscribe.backends.base import RunHandle, StageSpec, Status, now, read_result


class LocalStore:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, remote: str) -> pathlib.Path:
        path = (self.root / remote).resolve()
        if self.root.resolve() not in (path, *path.parents):
            raise ValueError(f"{remote!r} is outside the store")
        return path

    def put(self, local: pathlib.Path, remote: str) -> None:
        dest = self._path(remote)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)

    def get(self, remote: str, local: pathlib.Path) -> None:
        local = pathlib.Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote), local)

    def list(self, prefix: str = "") -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if base.is_file():
            return [prefix]
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file()) if base.exists() else []

    def read_text(self, remote: str) -> str | None:
        path = self._path(remote)
        return path.read_text() if path.is_file() else None

    def write_text(self, remote: str, text: str) -> None:
        path = self._path(remote)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


class LocalBackend:
    name = "local"

    def __init__(self, root: pathlib.Path) -> None:
        self._store = LocalStore(root)
        self._procs: dict[str, subprocess.Popen] = {}

    def check(self) -> list[str]:
        return []

    def store(self) -> LocalStore:
        return self._store

    def launch(self, spec: StageSpec) -> RunHandle:
        missing = [name for name in spec.secret_env if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"{', '.join(missing)} not set in this shell")
        env = {**os.environ, **spec.env}
        proc = subprocess.Popen([sys.executable, *spec.command(str(self._store.root))[1:]], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self._procs[str(proc.pid)] = proc
        return RunHandle(backend=self.name, stage=spec.stage, run_id=spec.run_id, id=str(proc.pid), started=now(), gpu=spec.gpu,
                         where={"store": str(self._store.root)})

    def status(self, handle: RunHandle) -> Status:
        proc = self._procs.get(handle.id)
        code = proc.poll() if proc else None
        if proc is not None and code is None:
            return "running"
        result = read_result(self._store, handle)
        if result is not None:
            return "succeeded" if result.get("ok") else "failed"
        if proc is None:
            return "unknown"
        return "stopped" if code and code < 0 else "failed"

    def logs(self, handle: RunHandle, follow: bool = False) -> Iterator[str]:
        text = self._store.read_text(f"{handle.run_dir}/log.txt") or ""
        yield from text.splitlines()

    def stop(self, handle: RunHandle) -> None:
        proc = self._procs.get(handle.id)
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)

    def hourly_usd(self, spec: StageSpec) -> float | None:
        return 0.0

    def cleanup(self) -> None:
        for proc in self._procs.values():
            if proc.poll() is None:
                proc.terminate()
