"""Modal as a backend: each step is a sandbox, and the store is a volume.

A sandbox takes its image, machine and command when it is made, keeps running if this machine goes away, and can be
found again by its id. The project's app holds its sandboxes and has no functions of its own, so nothing stays warm.
The volume is mounted at /work in every step. Its writes are saved when the step's container ends.

Needs the `modal` package and a Modal token on this machine (`modal token new`, or MODAL_TOKEN_ID and
MODAL_TOKEN_SECRET in the environment).
"""
from __future__ import annotations

import io
import os
import pathlib
from collections.abc import Iterator

from inscribe.backends.base import WORK, ImageSpec, RunHandle, StageSpec, Status, now, read_result

PACKAGE = pathlib.Path(__file__).resolve().parents[1]  # this package's own source, copied into every image
PACKAGE_AT = "/opt/inscribe"


class ModalStore:
    def __init__(self, volume) -> None:
        self.volume = volume

    def put(self, local: pathlib.Path, remote: str) -> None:
        with self.volume.batch_upload(force=True) as batch:
            batch.put_file(str(local), "/" + remote.lstrip("/"))

    def put_many(self, files: list[tuple[pathlib.Path, str]]) -> None:
        with self.volume.batch_upload(force=True) as batch:
            for local, remote in files:
                batch.put_file(str(local), "/" + remote.lstrip("/"))

    def get(self, remote: str, local: pathlib.Path) -> None:
        local = pathlib.Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("wb") as f:
            for chunk in self.volume.read_file(remote.lstrip("/")):
                f.write(chunk)

    def list(self, prefix: str = "") -> list[str]:
        from modal.exception import NotFoundError

        try:
            entries = self.volume.listdir("/" + prefix.strip("/"), recursive=True)
        except (NotFoundError, FileNotFoundError):
            return []
        return sorted(e.path.lstrip("/") for e in entries if e.type.name == "FILE")

    def read_text(self, remote: str) -> str | None:
        from modal.exception import NotFoundError

        try:
            return b"".join(self.volume.read_file(remote.lstrip("/"))).decode()
        except (NotFoundError, FileNotFoundError):
            return None

    def write_text(self, remote: str, text: str) -> None:
        with self.volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(text.encode()), "/" + remote.lstrip("/"))


class ModalBackend:
    name = "modal"

    def __init__(self, project: str, *, environment: str | None = None, app_prefix: str = "inscribe",
                 usd_per_hour: dict[str, float] | None = None) -> None:
        self.environment = environment
        self.app_name = f"{app_prefix}-{project}".replace("_", "-")
        self.volume_name = self.app_name
        self.usd_per_hour = usd_per_hour or {}
        self._volume = None

    def check(self) -> list[str]:
        try:
            import modal  # noqa: F401
        except ImportError:
            return ["The `modal` package is not installed. Install it with: uv pip install modal"]
        try:
            self.volume()
        except Exception as exc:  # noqa: BLE001  (whatever it is, the user needs the reason)
            return [f"Modal is not ready ({type(exc).__name__}: {str(exc)[:200]}). Run `modal token new`, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET."]
        return []

    def volume(self):
        import modal

        if self._volume is None:
            self._volume = modal.Volume.from_name(self.volume_name, create_if_missing=True, environment_name=self.environment, version=2)
            self._volume.hydrate()
        return self._volume

    def store(self) -> ModalStore:
        return ModalStore(self.volume())

    def image(self, spec: ImageSpec):
        import modal

        if spec.base == "debian-slim":
            image = modal.Image.debian_slim(python_version=spec.python)
        else:
            image = modal.Image.from_registry(spec.base, add_python=spec.python)
        if spec.clear_entrypoint:
            image = image.entrypoint([])
        if spec.apt:
            image = image.apt_install(*spec.apt)
        for layer in spec.pip:
            image = image.uv_pip_install(*layer.packages, **({"index_url": layer.index_url} if layer.index_url else {}))
        if spec.run:
            image = image.run_commands(*spec.run)
        image = image.env({**spec.env, "PYTHONPATH": PACKAGE_AT, "PYTHONUNBUFFERED": "1"})
        return image.add_local_dir(str(PACKAGE), f"{PACKAGE_AT}/inscribe", copy=True, ignore=["**/__pycache__", "**/*.pyc"])

    def launch(self, spec: StageSpec) -> RunHandle:
        import modal

        missing = [name for name in spec.secret_env if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"{', '.join(missing)} not set in this shell")
        app = modal.App.lookup(self.app_name, create_if_missing=True, environment_name=self.environment)
        secrets = [modal.Secret.from_dict({name: os.environ[name] for name in spec.secret_env})] if spec.secret_env else []
        with modal.enable_output():
            sandbox = modal.Sandbox.create(
                *spec.command(WORK), app=app, image=self.image(spec.image), volumes={WORK: self.volume()}, env=dict(spec.env) or None,
                secrets=secrets, gpu=spec.gpu, cpu=spec.cpu, memory=spec.memory_mb, timeout=spec.timeout_s, workdir=WORK,
                environment_name=self.environment)
        return RunHandle(backend=self.name, stage=spec.stage, run_id=spec.run_id, id=sandbox.object_id, started=now(), gpu=spec.gpu,
                         where={"app": self.app_name, "store": self.volume_name, "environment": self.environment or ""})

    def _sandbox(self, handle: RunHandle):
        import modal

        return modal.Sandbox.from_id(handle.id)

    def status(self, handle: RunHandle) -> Status:
        try:
            code = self._sandbox(handle).poll()
        except Exception:  # noqa: BLE001  (a sandbox that can no longer be found: the store still has the answer)
            code = -1
        if code is None:
            return "running"
        result = read_result(self.store(), handle)
        if result is not None:
            return "succeeded" if result.get("ok") else "failed"
        return "stopped" if code in (137, 143, -1) else "failed"

    def logs(self, handle: RunHandle, follow: bool = False) -> Iterator[str]:
        """While the step runs and `follow` is set: its output as it comes. Otherwise the log saved on the store."""
        if follow:
            try:
                sandbox = self._sandbox(handle)
                if sandbox.poll() is None:
                    for chunk in sandbox.stdout:
                        yield from chunk.splitlines()
                    return
            except Exception:  # noqa: BLE001  (fall back to the saved log)
                pass
        yield from (self.store().read_text(f"{handle.run_dir}/log.txt") or "").splitlines()

    def stop(self, handle: RunHandle) -> None:
        self._sandbox(handle).terminate()

    def hourly_usd(self, spec: StageSpec) -> float | None:
        if spec.gpu is None:
            return self.usd_per_hour.get("CPU", 0.1)
        return self.usd_per_hour.get(spec.gpu)

    def running(self) -> list[str]:
        """Ids of this project's sandboxes that are still running."""
        import modal

        app = modal.App.lookup(self.app_name, create_if_missing=False, environment_name=self.environment)
        return [sb.object_id for sb in modal.Sandbox.list(app_id=app.app_id) if sb.poll() is None]

    def cleanup(self) -> None:
        """Nothing is left running when no sandbox is: the app has no functions and the volume costs only its storage."""
        for sandbox_id in self.running():
            raise RuntimeError(f"a step is still running ({sandbox_id}); stop it first with `inscribe stop`")
