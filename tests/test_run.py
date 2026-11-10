"""`inscribe run`: the order of the steps, what is skipped, and what makes a step run again. Every step is a stand-in."""
import json
import shutil

import pytest
from typer.testing import CliRunner

from inscribe import cli, runs
from inscribe.backends.base import RunHandle, now
from inscribe.backends.local import LocalBackend

runner = CliRunner()
ALL = ["data", "mix", "train glyph-2b", "export glyph-2b", "eval glyph-2b", "train sigil-4b", "export sigil-4b", "eval sigil-4b"]


@pytest.fixture
def game(tmp_path, examples, monkeypatch):
    """The example game with a store in a folder and every paid step replaced by one that leaves the file the real one would."""
    root = tmp_path / "brackenhythe"
    shutil.copytree(examples / "brackenhythe", root, ignore=shutil.ignore_patterns("build", "out"))
    (root / "lora.yaml").write_text((root / "lora.yaml").read_text() + "\nreplay:\n  source: store:replay/0.1\n")
    backend = LocalBackend(tmp_path / "store")
    store, calls = backend.store(), []
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(cli, "_backend", lambda config, proj: backend)

    def data(project, set_, limit, skip_teacher, yes):
        calls.append("data")
        (root / "build/data").mkdir(parents=True, exist_ok=True)
        (root / "build/data/accepted.jsonl").write_text(json.dumps({"lines": calls.count("data")}) + "\n")

    def mix(project, set_, yes):
        calls.append("mix")
        store.write_text("mix/game/result.json", '{"ok": true}')

    def train(project, set_, base, max_steps, keep, prompts, yes, detach):
        calls.append(f"train {base}")
        store.write_text(f"train/{base}/result.json", '{"ok": true}')

    def export(project, set_, base, run, prompts, yes, detach):
        calls.append(f"export {base}")
        store.write_text(f"export/{base}/result.json", '{"ok": true}')

    def eval_(project, set_, base, no_judge, offline, max_usd, yes):
        calls.append(f"eval {base}")
        (root / "out" / base).mkdir(parents=True, exist_ok=True)
        (root / "out" / base / "report.json").write_text("{}")

    for name, fn in (("data", data), ("mix", mix), ("train", train), ("export", export), ("eval_", eval_)):
        monkeypatch.setattr(cli, name, fn)
    return root, calls, backend


def _run(root, *args, **kwargs):
    return runner.invoke(cli.app, ["run", "--project", str(root), *args], **kwargs)


def test_every_step_runs_once_in_order_and_then_nothing_runs(game):
    root, calls, _ = game
    result = _run(root, "--yes")
    assert result.exit_code == 0, result.output
    assert calls == ALL
    assert "train glyph-2b    about $" in result.output and "Budget cap: $80" in result.output and "The report card is out/report.md" in result.output
    calls.clear()
    result = _run(root, "--yes")
    assert result.exit_code == 0 and calls == [] and "Everything is done with these inputs" in result.output


def test_a_new_version_of_the_tool_leaves_finished_steps_finished(game, monkeypatch):
    root, calls, _ = game
    assert _run(root, "--yes").exit_code == 0
    calls.clear()
    before = (root / "build/plan/plan.yaml").read_text()
    # the plan is stamped with a hash of the code that wrote it: another version writes another stamp, and the same scenes
    monkeypatch.setattr("inscribe.plan.build.source_fingerprint", lambda: "anotherbuild")
    result = _run(root, "--yes")
    assert (root / "build/plan/plan.yaml").read_text() != before and "anotherbuild" in (root / "build/plan/style-examples.jsonl").read_text()
    assert result.exit_code == 0 and calls == [] and "Everything is done with these inputs" in result.output, result.output


def test_it_asks_once_before_anything_is_spent(game):
    root, calls, _ = game
    result = _run(root, input="n\n")
    assert result.exit_code == 1 and calls == [] and "Start?" in result.output


def test_a_changed_sheet_runs_everything_again_and_new_lines_run_what_reads_them(game):
    root, calls, _ = game
    assert _run(root, "--yes").exit_code == 0
    # the data step run again by hand leaves different lines: the mix and everything after it are stale, the data step is not
    cli.data(root, [], None, False, True)
    calls.clear()
    result = _run(root, "--yes")
    assert result.exit_code == 0 and calls == ALL[1:], result.output
    assert "data              done with these inputs, skipped" in result.output
    calls.clear()
    sheet = root / "sheet/game-sheet.yaml"
    sheet.write_text(sheet.read_text().replace("A pint of beer", "A pint of small beer"))
    assert _run(root, "--yes").exit_code == 0 and calls == ALL


def test_from_to_and_again(game):
    root, calls, _ = game
    assert _run(root, "--to", "data", "--yes").exit_code == 0 and calls == ["data"]
    calls.clear()
    assert _run(root, "--yes").exit_code == 0 and calls == ALL[1:]
    calls.clear()
    assert _run(root, "--from", "export", "--yes").exit_code == 0 and calls == [], "export and eval are done with these inputs"
    assert _run(root, "--from", "export", "--again", "--yes").exit_code == 0
    assert calls == ["export glyph-2b", "eval glyph-2b", "export sigil-4b", "eval sigil-4b"]
    calls.clear()
    assert _run(root, "--to", "plan").exit_code == 0 and calls == [], "the plan alone is free and asks nothing"
    for args, why in ((("--from", "bake"), "--from is one of"), (("--from", "eval", "--to", "data"), "--from comes after --to")):
        result = _run(root, *args)
        assert result.exit_code == 1 and why in result.output


def test_it_stops_before_spending_past_the_budget_or_without_what_it_needs(game, monkeypatch):
    root, calls, _ = game
    result = _run(root, "--yes", "--set", "budget.max_usd=1")
    assert result.exit_code == 1 and "over the budget in lora.yaml ($1)" in result.output and calls == []
    monkeypatch.delenv("OPENAI_API_KEY")
    result = _run(root, "--yes")
    assert result.exit_code == 1 and "OPENAI_API_KEY is not set" in result.output and calls == []
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    (root / "lora.yaml").write_text((root / "lora.yaml").read_text().replace("source: store:replay/0.1", "source: null"))
    result = _run(root, "--yes")
    assert result.exit_code == 1 and "No replay pack" in result.output and calls == []


def test_a_step_still_running_from_an_earlier_launch_is_followed_not_launched_again(game, monkeypatch):
    root, calls, backend = game
    assert _run(root, "--to", "mix", "--yes").exit_code == 0
    calls.clear()
    # a training step launched earlier with the same job, still running
    config = cli.load_config(root / "lora.yaml")
    job = runs.train_job(config, cli.load_recipe("glyph-2b"), "prompts/held-out.jsonl", None)
    backend.store().write_text("train/glyph-2b/job.json", json.dumps(job))
    handle = RunHandle(backend="local", stage="train", run_id="glyph-2b", id="earlier", started=now())
    runs.save_handle(cli.find_project(root), handle)
    monkeypatch.setattr(backend, "status", lambda h: "running" if h.id == "earlier" else "unknown")

    def follow(b, h, echo, poll_s=10.0):
        backend.store().write_text("train/glyph-2b/result.json", '{"ok": true}')
        return {"ok": True}

    monkeypatch.setattr(runs, "follow", follow)
    monkeypatch.setattr(cli, "_report", lambda handle, result: None)
    result = _run(root, "--to", "train", "--yes", "--set", "base=glyph-2b")
    assert result.exit_code == 0, result.output
    assert "train-glyph-2b is already running from an earlier launch. Following it." in result.output and calls == []
    assert "train glyph-2b    already running from an earlier launch, followed" in result.output and "GPU time for the steps to run: about $0.00" in result.output
    assert backend.store().read_text("train/glyph-2b/inputs.sha256"), "and it is stamped as done"
