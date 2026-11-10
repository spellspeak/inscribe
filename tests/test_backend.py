"""The backend interface, on the local backend: launch a step, follow it, read its result, stop one."""
import json
import time

from inscribe.backends.base import RunHandle, StageSpec, read_result
from inscribe.backends.local import LocalBackend
from inscribe.images import SMOKE


def _wait(backend, handle, seconds: float = 30) -> str:
    end = time.time() + seconds
    while backend.status(handle) == "running" and time.time() < end:
        time.sleep(0.1)
    return backend.status(handle)


def _launch(backend, run_id: str, job: dict):
    spec = StageSpec(stage="smoke", run_id=run_id, image=SMOKE)
    backend.store().write_text(f"{spec.run_dir}/job.json", json.dumps(job))
    return spec, backend.launch(spec)


def test_a_step_runs_and_leaves_its_result_log_and_files(tmp_path):
    backend = LocalBackend(tmp_path / "store")
    (tmp_path / "in.txt").write_text("x")
    backend.store().put(tmp_path / "in.txt", "inputs/in.txt")
    spec, handle = _launch(backend, "one", {"say": "hi", "list": "inputs"})
    assert _wait(backend, handle) == "succeeded"
    result = read_result(backend.store(), handle)
    assert result["ok"] and result["result"]["said"] == "hi" and result["result"]["inputs"] == ["inputs/in.txt"]
    assert backend.store().read_text("smoke/one/echo.txt") == "hi\n"
    assert any("smoke: python" in line for line in backend.logs(handle))
    assert set(backend.store().list("smoke/one")) == {"smoke/one/echo.txt", "smoke/one/job.json", "smoke/one/log.txt", "smoke/one/result.json"}
    backend.store().get("smoke/one/echo.txt", tmp_path / "out" / "echo.txt")
    assert (tmp_path / "out" / "echo.txt").read_text() == "hi\n"


def test_a_failing_step_says_why(tmp_path):
    backend = LocalBackend(tmp_path / "store")
    _, handle = _launch(backend, "bad", {"fail": "no such model"})
    assert _wait(backend, handle) == "failed"
    result = read_result(backend.store(), handle)
    assert not result["ok"] and "no such model" in result["error"] and "RuntimeError" in result["traceback"]
    assert any("no such model" in line for line in backend.logs(handle))


def test_a_step_can_be_stopped(tmp_path):
    backend = LocalBackend(tmp_path / "store")
    _, handle = _launch(backend, "slow", {"sleep": 60})
    time.sleep(0.5)
    assert backend.status(handle) == "running"
    backend.stop(handle)
    assert backend.status(handle) == "stopped"
    assert read_result(backend.store(), handle) is None


def test_a_handle_survives_a_round_trip(tmp_path):
    backend = LocalBackend(tmp_path / "store")
    _, handle = _launch(backend, "two", {})
    again = RunHandle.from_json(handle.to_json())
    assert again == handle and _wait(backend, again) == "succeeded"


def test_the_store_refuses_paths_outside_it(tmp_path):
    store = LocalBackend(tmp_path / "store").store()
    try:
        store.write_text("../escape.txt", "x")
    except ValueError:
        pass
    else:
        raise AssertionError("a path outside the store was written")
    assert store.read_text("nothing/here.json") is None and store.list("nothing") == []
