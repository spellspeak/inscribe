"""The scoring step: retention, the strong model's reads, the report card, and `inscribe eval`. No model is called."""
import hashlib
import json
import shutil

import pytest
from typer.testing import CliRunner

from inscribe import runs
from inscribe.cli import app
from inscribe.config import Config, load_recipe
from inscribe.evaluate import retention, scorecard, spoilers, style
from inscribe.judges import fast, strong
from inscribe.plan import recall
from inscribe.plan.build import build_plan, load_sheet
from inscribe.plan.npcs import load_npcs
from inscribe.runtime.template import format_by_name

runner = CliRunner()
FMT = format_by_name("gemma4-instruct")


@pytest.fixture(scope="module")
def game(tmp_path_factory, request):
    """The example game copied somewhere writable, with its plan built."""
    root = tmp_path_factory.mktemp("game") / "brackenhythe"
    shutil.copytree(request.config.rootpath / "examples" / "brackenhythe", root, ignore=shutil.ignore_patterns("build", "out"))
    build_plan(root / "sheet" / "game-sheet.yaml", root / "build" / "plan", load_npcs(root / "npcs.yaml"), FMT, 0)
    return root


def _retention_reply(p: dict, good: bool) -> str:
    if p.get("absent"):
        return "Never heard of such a thing." if good else f"Ah, {p['expect_any'][0]}."
    if not good:
        return "Could not say."
    return f"{p['expect_any'][0]}?" if p.get("kind") == "clarify" else f"That would be {p['expect_any'][0]}."


def _answers(bundle: dict, knows_lore: bool, lost: tuple[str, ...] = ()) -> list[dict]:
    """Made-up answers: retention right except the ids in `lost`; the game's questions answered with a key word or not at all."""
    rows = [{"id": p["id"], "text": _retention_reply(p, p["id"] not in lost)} for p in retention.retention_prompts(FMT)]
    for r in recall.all_rows(bundle):
        said = f"Aye, {r['expect_any'][0]}, as everyone here knows." if knows_lore and r["expect_any"] else "Couldn't tell you, duck."
        rows.append({"id": r["id"], "text": said})
    return rows


def _stand_in(lines: list[dict]) -> list[dict]:
    """The strong judge, replaced: a verdict drawn from the request's id."""
    out = []
    for line in lines:
        h = int(hashlib.sha256(line["custom_id"].encode()).hexdigest(), 16)
        if line["body"]["text"]["format"]["name"] == "spoiler_verdict":
            text = json.dumps({"reveals_later_stage": h % 9 == 0, "evidence": "names what happens next"})
        else:
            lora_shown_as = "B" if line["custom_id"].endswith(".ab") else "A"
            text = json.dumps({"rubric_version": "judge-rubric-0.1", "winner": lora_shown_as if h % 5 else "tie", "reason": "the cadence"})
        out.append({"custom_id": line["custom_id"], "response": {"status_code": 200, "body": {
            "status": "completed", "model": "stand-in", "usage": {"input_tokens": 1000, "output_tokens": 100},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}}})
    return out


def test_the_retention_questions_and_their_sets():
    rows = retention.retention_prompts(FMT)
    assert len(rows) == 78 and len({r["id"] for r in rows}) == 78
    assert {name: sum(r["set"] == name for r in rows) for name in retention.SETS} == retention.SETS
    assert all(r["prompt"].endswith(FMT.turn_open["assistant"]) and r["asked"] in r["prompt"] for r in rows)
    assert set(retention.prompt_rows(FMT)[0]) == {"id", "prompt", "max_new"}, "no answer key travels to a container"


def test_the_retention_rule():
    rows = retention.retention_prompts(FMT)
    all_right = retention.score([{"id": p["id"], "text": _retention_reply(p, True)} for p in rows], rows)
    assert {k: v["hits"] for k, v in all_right.items()} == retention.SETS
    knowhow = [p["id"] for p in rows if p["set"] == "knowhow"]
    realworld = [p["id"] for p in rows if p["set"] == "realworld"]

    def after(lost):
        return retention.score([{"id": p["id"], "text": _retention_reply(p, p["id"] not in lost)} for p in rows], rows)

    assert retention.verdict(all_right, after(knowhow[:3]))["passed"], "three of 29 is inside the allowance"
    v = retention.verdict(all_right, after(knowhow[:4]))
    assert not v["passed"] and v["knowhow"]["net"] == -4 and v["knowhow"]["lost"] == sorted(knowhow[:4])
    assert retention.verdict(all_right, after(realworld[:5]), tier="fast")["passed"], "real-world facts are reported for the smaller model"
    assert not retention.verdict(all_right, after(realworld[:5]), tier="main")["passed"], "and gated for the larger one"
    assert retention.tier_of("sigil-4b") == "main" and retention.tier_of("glyph-2b") == "fast"
    # a real-world word from a closed-world character is a miss, and so is silence
    flip = next(p for p in rows if p["set"] == "setting_flip")
    assert not retention.score([{"id": flip["id"], "text": f"That is {flip['expect_any'][0]}."}], rows)["setting_flip"]["hits"]
    assert not retention.score([{"id": flip["id"], "text": ""}], rows)["setting_flip"]["hits"]


def test_the_strong_client_reads_answers_and_never_guesses():
    schema = strong.strict_schema(spoilers.SpoilerVerdict)
    assert schema["additionalProperties"] is False and schema["required"] == ["reveals_later_stage", "evidence"]
    assert "description" not in strong.PREFERENCE_SCHEMA, "a docstring would travel in every request"
    ok = _stand_in([strong.request("a.ab", [{"role": "user", "content": "x"}], strong.PREFERENCE_SCHEMA, "preference_verdict")])[0]
    assert strong.parse(ok).error is None and strong.parse(ok).model == "stand-in"
    cut = {"custom_id": "a", "response": {"status_code": 200, "body": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}}}
    refused = {"custom_id": "a", "response": {"status_code": 200, "body": {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}}}
    for line, why in ((cut, "max_output_tokens"), (refused, "refusal"), ({"custom_id": "a", "response": {"status_code": 500, "body": {}}}, "HTTP 500"),
                      ({"custom_id": "a", "error": {"message": "timed out"}}, "timed out")):
        p = strong.parse(line)
        assert p.text is None and why in p.error
    assert strong.pair_order("style.x", 0) == strong.pair_order("style.x", 0)
    assert strong.unswap("p.ab", "A") == "1" and strong.unswap("p.ba", "A") == "2" and strong.unswap("p.ba", "tie") == "tie"
    with pytest.raises(ValueError):
        strong.unswap("p", "A")
    assert Config(game="x").llm.strong_model == strong.MODEL and Config(game="x").llm.fast_model == fast.MODEL


def test_the_judge_sees_the_story_so_far_and_not_the_labels(game):
    bundle = json.loads((game / "build/plan/recall.json").read_text())
    sheet = load_sheet(game / "sheet/game-sheet.yaml")
    before, after = _answers(bundle, False), _answers(bundle, True)
    asks = spoilers.spoiler_requests(bundle, sheet, spoilers.spoken(after))
    assert len(asks) == len(bundle["spoiler"])
    first = asks[0]["body"]["input"][1]["content"]
    assert "Story so far" in first and "Still to come" in first and sheet.story[-1].stage in first
    report = spoilers.judge(bundle, sheet, spoilers.spoken(after), answer=_stand_in)
    assert report["prompts"] == len(asks) and 0 < report["told_early"] < len(asks) and report["told"][0]["asked"]
    with pytest.raises(RuntimeError, match="over its cap"):
        spoilers.judge(bundle, sheet, spoilers.spoken(after), max_usd=0.01, answer=_stand_in)

    pairs = style.style_pairs(bundle, sheet, before, after)
    assert len(pairs) == style.PAIRS
    shown = [line["body"]["input"][1]["content"] for line, _ in pairs]
    assert not any(word in s.lower() for s in shown for word in ("lora", "base model", "trained")), "the judge is never told which is which"
    orders = {line["custom_id"].rsplit(".", 1)[1] for line, _ in pairs}
    assert orders == {"ab", "ba"}, "the order shown is mixed"
    report = style.judge(bundle, sheet, before, after, answer=_stand_in)
    assert report["pairs"] == style.PAIRS and report["base_wins"] == 0 and report["lora_wins"] + report["ties"] == style.PAIRS
    assert report["sign_test_p"] < 0.05 and report["samples"][0]["preferred"] in ("lora", "tie")
    assert style.sign_test_p(5, 5) == 1.0 and style.sign_test_p(0, 0) == 1.0


def test_the_card_marks_each_line_and_says_what_it_cannot(game):
    bundle = json.loads((game / "build/plan/recall.json").read_text())
    sheet = load_sheet(game / "sheet/game-sheet.yaml")
    lost = tuple(p["id"] for p in retention.retention_prompts(FMT) if p["set"] == "knowhow")[:5]
    before, after = _answers(bundle, False), _answers(bundle, True, lost)
    card = scorecard.build("brackenhythe", "glyph-2b", "SpellSpeak Glyph 2B", bundle, sheet, FMT, before, after, {"Q5_K_M": _answers(bundle, True)})
    by_key = {c["key"]: c for c in card["scorecard"]}
    assert card["read_at"] == "Q5_K_M" and by_key["lore"]["met"] is True and by_key["lore"]["value"].startswith("33 of 33")
    assert by_key["retention"]["met"] is False and len(card["retention_lost"]) == 5 and card["retention_lost"][0]["asked"]
    assert by_key["spoilers"]["met"] is None, "without the judge the spoiler line is not judged"
    text = scorecard.report_md([card])
    assert "**not met**" in text and "more replay" in text and "not judged" in text
    for claim in ("General ability is not checked", "other worlds is not checked", "a preference, not a guarantee", "comes from your game"):
        assert claim in text
    assert "| Know-how | 29 of 29 | 24 of 29 | yes |" in text and "| Real-world facts | 19 of 19 | 19 of 19 | no, reported |" in text

    # nothing exported yet: the trained model at full precision stands in; no training answers: retention is not scored
    card = scorecard.build("brackenhythe", "glyph-2b", "SpellSpeak Glyph 2B", bundle, sheet, FMT, before, after, {})
    assert card["read_at"] == "after"
    card = scorecard.build("brackenhythe", "glyph-2b", "SpellSpeak Glyph 2B", bundle, sheet, FMT, None, None, {"Q5_K_M": _answers(bundle, False)})
    assert card["retention"] is None and {c["key"]: c for c in card["scorecard"]}["lore"]["met"] is False
    assert "did not answer the retention questions" in scorecard.report_md([card]) and "inscribe check --lore" in scorecard.report_md([card])


def test_training_answers_the_retention_questions_and_the_games(game):
    bundle = json.loads((game / "build/plan/recall.json").read_text())
    rows = runs.held_out_rows(bundle, load_recipe("glyph-2b").chat_format)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) == 78 + len(recall.all_rows(bundle)) and ids[0] == "know.k1"
    assert all(set(r) == {"id", "prompt", "max_new"} for r in rows)


def _put_answers(game, base="glyph-2b"):
    bundle = json.loads((game / "build/plan/recall.json").read_text())
    folder = game / "out" / base
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train-answers-before", _answers(bundle, False)), ("train-answers-after", _answers(bundle, True)), ("export-answers-Q5_K_M", _answers(bundle, True))):
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_eval_writes_the_card_from_the_answers_in_out(game, monkeypatch):
    _put_answers(game)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(app, ["eval", "--project", str(game), "--offline", "--base", "glyph-2b"])
    assert result.exit_code == 1 and "OPENAI_API_KEY is not set" in result.output and "--no-judge" in result.output

    result = runner.invoke(app, ["eval", "--project", str(game), "--offline", "--no-judge", "--base", "glyph-2b"])
    assert result.exit_code == 0, result.output
    assert "met        Important lore recalled: 33 of 33" in result.output and "not judged Later story told early" in result.output
    page = (game / "out/report.md").read_text()
    assert page.startswith("# brackenhythe: report card") and "SpellSpeak Glyph 2B" in page and "What this card does not tell you" in page

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(strong, "answer", lambda lines, api_key, **kw: _stand_in(lines))
    result = runner.invoke(app, ["eval", "--project", str(game), "--offline", "--base", "glyph-2b"], input="n\n")
    assert result.exit_code == 1 and "about $" in result.output, "it asks before the judge is paid"
    result = runner.invoke(app, ["eval", "--project", str(game), "--offline", "--base", "glyph-2b", "--yes"])
    assert result.exit_code == 0, result.output
    card = json.loads((game / "out/glyph-2b/report.json").read_text())
    assert card["spoilers"]["read_at"] == "Q5_K_M" and card["style"]["pairs"] == 80
    page = (game / "out/report.md").read_text()
    assert "### Voice" in page and "### Later story told early" in page and "Judge: names what happens next" in page


def test_eval_without_answers_says_what_to_run(game, tmp_path):
    empty = tmp_path / "brackenhythe"
    shutil.copytree(game, empty, ignore=shutil.ignore_patterns("out"))
    result = runner.invoke(app, ["eval", "--project", str(empty), "--offline", "--no-judge", "--base", "glyph-2b"])
    assert result.exit_code == 1 and "Run `inscribe train` and `inscribe export` first" in result.output
