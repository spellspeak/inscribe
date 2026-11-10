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


def test_pull_brings_the_gguf_and_results_and_leaves_the_merged_model(tmp_path):
    from inscribe.runs import pull_list

    store = LocalBackend(tmp_path / "store").store()
    for path in ("export/glyph-2b/gguf/my-game-glyph-2b-Q5_K_M.gguf", "export/glyph-2b/result.json", "export/glyph-2b/answers-Q5_K_M.jsonl",
                 "export/glyph-2b/log.txt", "train/glyph-2b/result.json", "train/glyph-2b/answers-after.jsonl", "train/glyph-2b/merged/model.safetensors",
                 "train/glyph-2b/adapter/adapter_model.safetensors", "train/glyph-2b-check/result.json", "train/sigil-4b/result.json"):
        store.write_text(path, "x")
    plain = dict(pull_list(store, "glyph-2b"))
    assert plain == {"export/glyph-2b/gguf/my-game-glyph-2b-Q5_K_M.gguf": "my-game-glyph-2b-Q5_K_M.gguf",
                     "export/glyph-2b/result.json": "glyph-2b/export-result.json", "export/glyph-2b/answers-Q5_K_M.jsonl": "glyph-2b/export-answers-Q5_K_M.jsonl",
                     "train/glyph-2b/result.json": "glyph-2b/train-result.json", "train/glyph-2b/answers-after.jsonl": "glyph-2b/train-answers-after.jsonl"}
    with_adapter = dict(pull_list(store, "glyph-2b", adapter=True))
    assert with_adapter["train/glyph-2b/adapter/adapter_model.safetensors"] == "glyph-2b/adapter/adapter_model.safetensors"
    assert not any("merged" in k for k in with_adapter) and pull_list(store, "none") == []


def test_an_interrupted_step_says_so(tmp_path, monkeypatch):
    import inscribe.stages.run as runner
    import inscribe.stages.smoke as smoke

    def interrupted(work, run_dir, job):
        raise KeyboardInterrupt

    monkeypatch.setattr(smoke, "run", interrupted)
    (tmp_path / "smoke" / "cut").mkdir(parents=True)
    (tmp_path / "smoke" / "cut" / "job.json").write_text("{}")
    assert runner.main(["smoke", "--work", str(tmp_path), "--run", "smoke/cut"]) == 1
    result = json.loads((tmp_path / "smoke" / "cut" / "result.json").read_text())
    assert not result["ok"] and result["interrupted"] and result["error"].startswith("interrupted") and "again" in result["error"]
