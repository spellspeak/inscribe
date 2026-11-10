"""The judge's questions, the fast judge's client, and the checks on the teacher's lines. No network: a stand-in answers."""
import json

import httpx
import pytest
import yaml

from inscribe.data import teacher as T
from inscribe.data import validate as V
from inscribe.judges import fast as F
from inscribe.judges import questions as Q
from inscribe.plan.build import build_plan
from inscribe.plan.npcs import load_npcs
from inscribe.schemas.world import World


@pytest.fixture(scope="module")
def plan(tmp_path_factory, ):
    import pathlib

    game = pathlib.Path(__file__).resolve().parents[1] / "examples" / "brackenhythe"
    out = tmp_path_factory.mktemp("plan")
    build_plan(game / "sheet" / "game-sheet.yaml", out, load_npcs(game / "npcs.yaml"))
    cases = [json.loads(line) for line in (out / "scenarios.jsonl").read_text().splitlines() if line.strip()]
    worlds = {p.stem: World.model_validate(yaml.safe_load(p.read_text())) for p in sorted((out / "worlds").glob("*.yaml"))}
    return cases, worlds


def _rows(cases, texts):
    return [{"case": c, "world_id": c["student_input_ref"].removeprefix("worldgen:").split("#")[0], "candidate": k, "text": t, "finish_reason": "stop",
             "prompt_hash": "0" * 16, "teacher": {"id": "t", "revision": "r", "prompt_version": "p", "sampling": {}}, "generator_version": "g"}
            for c in cases for k, t in enumerate(texts(c))]


class _Judge:
    name = "stand-in"

    def __init__(self, fail=lambda req: False):
        self.fail, self.seen = fail, []

    def answer_all(self, requests):
        self.seen += requests
        out = []
        for r in requests:
            bad = self.fail(r)
            out.append({"item_id": r.item_id, "answers": {q: (0.5 if k.polarity == "permission" else (0.05 if bad else 0.95) if k.polarity == "holds"
                                                              else (0.95 if bad else 0.05)) for q, k in r.key.items()}})
        return out


def test_a_property_becomes_a_question_and_its_answer_is_read_the_right_way_round():
    assert Q.property_question("keeps_it_short") == ("Does `reply` keep it short?", "holds")
    assert Q.property_question("does_not_invent_details") == ("Does `reply` invent details?", "violated")
    assert Q.property_question("may_use_the_cover_story")[1] == "permission"
    key = {"q01": Q.QuestionKey("property", "keeps_it_short", "holds"), "q02": Q.QuestionKey("property", "does_not_invent_details", "violated"),
           "q03": Q.QuestionKey("failure", "leak", "present"), "q04": Q.QuestionKey("property", "may_x", "permission")}
    assert Q.fail_probability({"q01": 0.9, "q02": 0.1, "q03": 0.2, "q04": 0.0}, key) == pytest.approx(0.2)
    assert Q.fail_probability({"q01": 0.3, "q02": 0.1, "q03": 0.2, "q04": 0.0}, key) == pytest.approx(0.7)
    assert Q.top_question({"q01": 0.9, "q02": 0.85, "q03": 0.2, "q04": 1.0}, key) == ("property:does_not_invent_details", pytest.approx(0.85))
    assert Q.flagged_failures({"q01": 0.9, "q02": 0.1, "q03": 0.97, "q04": 0.0}, key, 0.9) == ["leak"]


def test_a_request_holds_the_scene_and_one_question_each(plan):
    cases, worlds = plan
    case = next(c for c in cases if c["generator_like"] == "RW-01")
    rc = V.world_resolver(worlds)(case, case["student_input_ref"].removeprefix("worldgen:").split("#")[0])
    req = Q.build_request(rc, "Never heard of such a thing.", "7")
    assert set(req.body["state"]) >= {"character_sheet", "conversation", "reply", "hidden_truth", "acceptable_alternatives", "world_boundary"}
    kinds = [k.kind for k in req.key.values()]
    assert kinds.count("property") == len(case["expected_properties"]) and "boundary_leak" in {k.name for k in req.key.values()}
    body = F.request_body(req, "judge-model")
    assert body["model"] == "judge-model" and body["input"].startswith("One reply by a game character") and len(body["questions"]) == len(req.key)
    leak = next(q for q in body["questions"] if req.key[q["name"]].name == "leak")
    assert " Yes means: " in leak["instructions"] and " No means: " in leak["instructions"]
    assert F.estimate_tokens(req) > 1000


def test_the_client_retries_adds_up_tokens_and_never_guesses_a_refusal():
    req = Q.JudgeRequest("1", {"state": {"reply": "x"}, "questions": {"q01": {"instructions": "A?"}, "q02": {"instructions": "B?"}}},
                         {"q01": Q.QuestionKey("property", "a", "holds"), "q02": Q.QuestionKey("property", "b", "holds")})
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        assert request.headers["authorization"] == "Bearer k" and json.loads(request.content)["model"] == F.MODEL
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after-ms": "1"}, text="slow down")
        return httpx.Response(200, json={"answers": [{"type": "predicate", "name": "q01", "probability": 0.9}, {"type": "predicate", "name": "q02", "probability": 0.2}],
                                         "usage": {"input_tokens": 1234}})

    judge = F.FastJudge(api_key="k", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    [res] = judge.answer_all([req])
    assert res["answers"] == {"q01": 0.9, "q02": 0.2} and res["attempts"] == 2 and judge.input_tokens == 1234

    refusing = F.FastJudge(api_key="k", sleep=lambda s: None, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
        "answers": [{"type": "predicate", "name": "q01", "probability": 0.9}, {"type": "refusal", "name": "q02"}]})))
    [res] = refusing.answer_all([req])
    assert res["answers"] is None and "q02" in res["error"]

    denied = F.FastJudge(api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no")))
    with pytest.raises(F.JudgeError):
        denied.answer_all([req])
    with pytest.raises(F.JudgeError, match="OPENAI_API_KEY"):
        F.FastJudge(api_key=None).answer_all([req])


def test_the_teacher_is_told_the_hidden_lore_and_the_style(plan):
    cases, worlds = plan
    items = T.teacher_items(cases, worlds)
    assert len(items) == len(cases)
    lore = next(i for i in items if i["case"].get("teacher_only"))
    system = lore["messages"][0]["content"]
    assert lore["case"]["teacher_only"][0] in system and "Write only what" in system and lore["messages"][-1]["role"] == "user"
    assert T.clean_line('  Maren: "Four pips, duck."\nsecond line') == "Four pips, duck."


def test_the_checks_keep_one_good_line_a_scene_and_say_why_the_rest_went(plan):
    cases, worlds = plan
    some = [c for c in cases if c["generator_like"] != "LW-01" and not c.get("must_contain_any")][:30]
    rows = _rows(some, lambda c: [f"Ask at the quay, friend, by door {some.index(c)}.", "", f"*shrugs* Try the stair, number {some.index(c)}."])
    rows[0]["finish_reason"] = "length"
    judge = _Judge(fail=lambda req: int(req.item_id) % 5 == 0)
    out = V.validate(rows, V.world_resolver(worlds), judge, "abc", F.ACCEPT_BELOW)
    assert out["candidates"] == 90 and out["scenes"] == 30
    assert {"cut_off", "empty", "code_check", "judge"} <= set(out["reasons"])
    ids = [e["scenario_id"] for e in out["accepted"]]
    assert len(ids) == len(set(ids)) and 0 < len(ids) < 30
    first = out["accepted"][0]
    assert first["example_id"].startswith("ex_") and first["validation"]["oracle_consistency"].startswith("stand-in fail probability")
    assert all(r["reason"] for r in out["rejected"]) and len(out["accepted"]) + len(out["rejected"]) == 90
    assert V.summary(rows, out)["accepted"] == len(ids)


def test_lore_scenes_are_checked_by_code_only(plan):
    cases, worlds = plan
    lore = [c for c in cases if c["generator_like"] == "LW-01" and c.get("must_contain_any")][:5]
    places = ("quay", "stair", "weir", "gate", "loft")
    rows = _rows(lore, lambda c: [f"{c['must_contain_any'][0]}, as they will tell you down at the {places[lore.index(c)]}."])
    judge = _Judge(fail=lambda req: True)
    out = V.validate(rows, V.world_resolver(worlds), judge, "abc", F.ACCEPT_BELOW, V.CODE_ONLY)
    assert judge.seen == [], "a lore scene never reaches the judge"
    assert len(out["accepted"]) + sum(out["reasons"].values()) == 5 and "judge" not in out["reasons"]
    assert out["accepted"], "a line with the scene's key word passes on code checks alone"
