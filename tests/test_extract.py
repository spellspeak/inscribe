"""The lore reader, with the strong model replaced by canned answers, and `inscribe chat` up to the point a model loads."""
import copy
import json
import pathlib

import pytest
import yaml
from typer.testing import CliRunner

from inscribe import chat
from inscribe.cli import app
from inscribe.plan.build import load_sheet
from inscribe.plan.npcs import load_npcs
from inscribe.runtime.template import format_by_name
from inscribe.sheets import extract as X

runner = CliRunner()

WORLD = {
    "setting": {"kind": "closed_fictional", "technology": "Late medieval: timber, wool and iron.", "magic_policy": "absent",
                "boundary": {"real_world": "none", "allowed": ["iron", "horses"], "denied": []}},
    "rules": ["Nobody has crossed the Grey Sea and come back."],
    "things": [{"id": "Crooked Lantern", "kind": "structure", "name": "the Crooked Lantern", "inside": None}],
    "extensions": [],
    "lore": [{"text": "The Crooked Lantern was built the year after the flood.", "importance": "high", "exact": False},
             {"text": "A mug of ale is two coppers.", "importance": "normal", "exact": False}],
    "story": [{"stage": "Arrival", "summary": "The player takes a room."}, {"stage": "The storm", "summary": "The road is closed."}],
    "gated_terms": ["the drowned heir"],
    "could_not_place": ["world.md: tone notes for writers"],
}
CHARACTERS = {
    "characters": [{"id": "npc_tomas", "name": "Tomas", "role": "The barkeep.", "location": "crooked-lantern", "voice": "Dry and unhurried. Short sentences.",
                    "values": ["a clean bar"], "goal": None, "likes": [], "dislikes": ["credit"],
                    "relationships": [{"who": "player", "kind": None, "trust": "neutral", "affection": "neutral", "fear": "low", "respect": None,
                                       "familiarity": None, "extra": [], "obligations": [], "opinion": "A stranger, so far."}],
                    "knows": [{"about": "crooked_lantern", "what": "cellar_door", "value": "The cellar door has been nailed shut since the flood,",
                               "how": "witnessed", "secret": {"may_tell": [], "cover_story": "Damp got into the hinges,"}},
                              {"about": "crooked_lantern", "what": "cellar_smell", "value": "The cellar smells of the sea.", "how": "witnessed", "secret": None}],
                    "offers": [{"what": "a mug of ale", "kind": "item", "price": "two coppers", "available": True, "note": None}],
                    "never_discuss": ["his brother"], "backstory": ["Kept the bar for thirty years."]}],
    "extensions": [],
    "could_not_place": ["Tomas: what he says after the storm (live state)"],
}
STYLE = {"speech_examples": [{"character_id": "npc_tomas", "player": "Evening.", "reply": "Evening. Mind the step."}], "could_not_place": ["style.md: the house rules"]}


def _line(cid: str, answer: dict) -> dict:
    return {"custom_id": cid, "response": {"status_code": 200, "body": {
        "status": "completed", "model": "stand-in", "usage": {"input_tokens": 4000, "output_tokens": 2000, "output_tokens_details": {"reasoning_tokens": 500}},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]}]}}}


def canned(answers: dict[str, dict], seen: list | None = None):
    def answer(lines: list[dict]) -> list[dict]:
        if seen is not None:
            seen.extend(lines)
        return [_line(lines[0]["custom_id"], answers[lines[0]["custom_id"]])]

    return answer


@pytest.fixture
def project(tmp_path, monkeypatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "lantern"]).exit_code == 0
    return tmp_path / "lantern"


def test_three_calls_make_a_sheet_that_validates(project):
    seen: list[dict] = []
    answers = {"lantern.world": WORLD, "lantern.characters": CHARACTERS, "lantern.style": STYLE}
    sheet, info = X.Extractor(project / "lore", "lantern", canned(answers, seen)).run()
    assert [line["custom_id"] for line in seen] == ["lantern.world", "lantern.characters", "lantern.style"]
    # each call sees only its own documents: the characters call never sees the story
    docs = [set(json.loads(line["body"]["input"][1]["content"])["documents"]) for line in seen]
    assert docs == [{"world.md", "story.md"}, {"characters.md"}, {"style.md"}]
    assert all(line["body"]["text"]["format"]["strict"] and line["body"]["reasoning"] == {"effort": "medium"} for line in seen)
    tomas = sheet.characters[0]
    assert sheet.things[0].id == "crooked_lantern" and tomas.location == "crooked_lantern", "ids are normalised the same way everywhere"
    assert tomas.knows[0].secret.cover_story == "Damp got into the hinges.", "a cover story cut mid-sentence is closed"
    assert tomas.speech_examples[0].reply == "Evening. Mind the step."
    assert info["counts"]["lore_by_importance"] == {"high": 1, "normal": 1, "low": 0} and info["counts"]["secrets"] == 1 and info["repairs"] == 0
    assert info["to_check"] == ["npc_tomas: `cellar_smell` has no secret block but sits beside a secret about `crooked_lantern`; is it part of that secret?"]
    assert info["cost_usd"] == round(3 * (4000 * 2 + 2000 * 10) / 1e6, 4)
    text = X.sheet_yaml(sheet, info)
    assert load_sheet_text(text).game_id == "lantern" and "Check it by hand" in text
    report = X.report_md(info)
    assert "Items to check first" in report and "tone notes for writers" in report and "live state" in report


def load_sheet_text(text: str):
    from inscribe.schemas.game_sheet import GameSheet

    return GameSheet.model_validate(yaml.safe_load(text))


def test_a_gated_word_in_the_sheet_goes_back_for_one_repair(project):
    leaky = copy.deepcopy(WORLD)
    leaky["lore"].append({"text": "The Drowned Heir sleeps under the cellar.", "importance": "normal", "exact": False})
    seen: list[dict] = []
    answers = {"lantern.world": leaky, "lantern.world.repair": WORLD, "lantern.characters": CHARACTERS, "lantern.style": STYLE}
    sheet, info = X.Extractor(project / "lore", "lantern", canned(answers, seen)).run()
    assert info["repairs"] == 1 and len(sheet.lore) == 2
    repair = json.loads(seen[1]["body"]["input"][1]["content"])
    assert "the drowned heir" in repair["validation_errors"] and repair["your_previous_output"]["lore"][-1]["text"].startswith("The Drowned Heir")

    answers["lantern.world.repair"] = leaky
    with pytest.raises(X.ExtractionError, match="still invalid after the repair pass"):
        X.Extractor(project / "lore", "lantern", canned(answers)).run()


def test_nothing_is_sent_past_the_cap(project):
    seen: list[dict] = []
    ex = X.Extractor(project / "lore", "lantern", canned({}, seen), max_usd=0.01)
    with pytest.raises(X.BudgetError, match="nothing was sent"):
        ex.run()
    assert not seen and ex.spent == 0


def test_the_command_writes_the_sheet_and_leaves_an_edited_one_alone(project, monkeypatch):
    result = runner.invoke(app, ["extract", "--project", str(project), "--dry-run"])
    assert result.exit_code == 0 and "about $" in result.output and "Cap: $3.00" in result.output

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(app, ["extract", "--project", str(project), "--yes"])
    assert result.exit_code == 1 and "OPENAI_API_KEY is not set" in result.output

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    answers = {"lantern.world": WORLD, "lantern.characters": CHARACTERS, "lantern.style": STYLE}
    monkeypatch.setattr(X, "paid_answer", lambda key: canned(answers))
    result = runner.invoke(app, ["extract", "--project", str(project)], input="n\n")
    assert result.exit_code == 1, "it asks before spending"
    result = runner.invoke(app, ["extract", "--project", str(project), "--yes"])
    assert result.exit_code == 0, result.output
    assert "Wrote sheet/game-sheet.yaml and sheet/extract-report.md" in result.output and "Next: inscribe check" in result.output
    assert load_sheet(project / "sheet/game-sheet.yaml").characters[0].name == "Tomas", "the template's sheet is replaced"
    raw = json.loads((project / "build/extract/raw.json").read_text())
    assert set(raw["answers"]) == set(answers) and raw["repairs"] == 0
    assert runner.invoke(app, ["check", "--project", str(project)]).exit_code == 0

    sheet = project / "sheet/game-sheet.yaml"
    sheet.write_text(sheet.read_text().replace("Dry and unhurried.", "Dry, unhurried, fond of a pause."))
    result = runner.invoke(app, ["extract", "--project", str(project), "--yes"])
    assert result.exit_code == 0 and "has your edits, so it was left alone" in result.output
    assert "fond of a pause" in sheet.read_text() and (project / "sheet/game-sheet.extracted.yaml").exists()
    result = runner.invoke(app, ["extract", "--project", str(project), "--yes", "--force"])
    assert result.exit_code == 0 and "fond of a pause" not in sheet.read_text()


def test_rules_that_are_many_or_lean_on_the_line_before_are_flagged(project):
    wordy = copy.deepcopy(WORLD)
    wordy["rules"] = ["Spar must never go in a fire.", "It does not explode."] + [f"Rule number {n} of the town stands." for n in range(30)]
    answers = {"lantern.world": wordy, "lantern.characters": CHARACTERS, "lantern.style": STYLE}
    _, info = X.Extractor(project / "lore", "lantern", canned(answers)).run()
    flags = [f for f in info["to_check"] if f.startswith("rules:")]
    assert len(flags) == 2 and "32 world rules" in flags[0] and "`It does not explode.` leans on the line before it" in flags[1]
    assert "leans on the line before it" in X.report_md(info)


def test_the_templates_run_time_character_is_pointed_out(project, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(X, "paid_answer", lambda key: canned({"lantern.world": WORLD, "lantern.characters": CHARACTERS, "lantern.style": STYLE}))
    result = runner.invoke(app, ["extract", "--project", str(project), "--yes"])
    assert result.exit_code == 0 and "npcs.yaml still holds the template's example character" in result.output
    (project / "npcs.yaml").write_text((project / "npcs.yaml").read_text().replace("location: crooked_lantern", "location: the_moon"))
    result = runner.invoke(app, ["check", "--project", str(project)])
    assert result.exit_code == 1 and "npcs.yaml does not fit the sheet" in result.output and "unknown location the_moon" in result.output


def test_a_missing_lore_file_is_named(project):
    (project / "lore/style.md").unlink()
    result = runner.invoke(app, ["extract", "--project", str(project), "--dry-run"])
    assert result.exit_code == 1 and "style.md" in result.output


# --- chat, up to the point a model loads -----------------------------------------------------------------------
def test_chat_finds_the_character_and_the_file(examples, tmp_path):
    root = examples / "brackenhythe"
    sheet, npcs = load_sheet(root / "sheet/game-sheet.yaml"), load_npcs(root / "npcs.yaml")
    assert chat.who(sheet, npcs, "maren") == "npc_maren_tolliver" and chat.who(sheet, npcs, "Maren Tolliver") == "npc_maren_tolliver"
    assert chat.who(sheet, npcs, npcs[0].id) == npcs[0].id, "a run-time character can be talked to as well"
    with pytest.raises(chat.ChatError, match="no character called"):
        chat.who(sheet, npcs, "nobody")

    with pytest.raises(chat.ChatError, match="inscribe pull"):
        chat.find_gguf(tmp_path, "brackenhythe", "glyph-2b")
    (tmp_path / "brackenhythe-glyph-2b-Q5_K_M.gguf").write_bytes(b"")
    (tmp_path / "brackenhythe-glyph-2b-Q4_K_M.gguf").write_bytes(b"")
    with pytest.raises(chat.ChatError, match="--quant"):
        chat.find_gguf(tmp_path, "brackenhythe", "glyph-2b")
    assert chat.find_gguf(tmp_path, "brackenhythe", "glyph-2b", "Q5_K_M").name == "brackenhythe-glyph-2b-Q5_K_M.gguf"

    fmt = format_by_name("gemma4-instruct")
    talk = chat.Conversation(sheet, npcs, "npc_maren_tolliver", fmt)
    first = talk.prompt("What's a pint cost?")
    assert "Maren Tolliver" in first and first.endswith(fmt.turn_open["assistant"]) and "What's a pint cost?" in first
    later = chat.Conversation(sheet, npcs, "npc_maren_tolliver", fmt, stage=len(sheet.story) - 1)
    assert later.prompt("What's a pint cost?") != first, "the story stage changes what the character knows"
    with pytest.raises(chat.ChatError, match="not found"):
        chat.Server(tmp_path / "x.gguf", binary="no-such-llama-server").__enter__()


def test_the_chat_command_lists_characters_and_asks_for_the_file(examples, tmp_path):
    import shutil

    root = tmp_path / "brackenhythe"
    shutil.copytree(examples / "brackenhythe", root, ignore=shutil.ignore_patterns("build", "out"))
    result = runner.invoke(app, ["chat", "--project", str(root)])
    assert result.exit_code == 0 and "Maren Tolliver  (npc_maren_tolliver)" in result.output and "made up at run time" in result.output
    result = runner.invoke(app, ["chat", "maren", "--project", str(root)])
    assert result.exit_code == 1 and "inscribe pull" in result.output
    result = runner.invoke(app, ["chat", "maren", "--project", str(root), "--stage", "99"])
    assert result.exit_code == 1 and "--stage is 1 to" in result.output
