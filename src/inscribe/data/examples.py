"""Training records: a scenario and an accepted reply become one training example."""
from __future__ import annotations

import hashlib

from inscribe.schemas.records import TrainingExample

CAPABILITY_IDS = {
    "ID-01": "identity.direct_identity", "EP-06": "epistemics.hearsay", "WR-04": "world.local_custom_and_law",
    "ST-03": "state_time.witnessed_transition", "ST-04": "state_time.unwitnessed_transition",
    "SO-05": "social.correcting_what_it_told", "BR-01": "dialogue.short_but_sufficient",
    "ID-02": "identity.occupation_and_expertise", "EP-11": "epistemics.setting_appropriate_latent_knowledge",
    "DL-04": "dialogue.multi_turn_repair",
    "LORE": "identity.embedded_game_lore",  # a game's own lines (its speech examples): never in the student's prompt
    "ST-01": "state_time.current_location", "ST-05": "state_time.relationship_change", "ST-07": "state_time.rejected_action_result",
    "SO-01": "social.feelings_toward_a_known_character", "SO-02": "social.mixed_feelings", "SO-03": "social.obligation_or_debt",
    "SO-04": "social.shared_history_with_the_player", "SO-06": "social.promise_status", "SO-07": "social.stance_toward_a_group",
    "EP-07": "epistemics.known_secret", "RB-05": "robustness.unverified_player_claim",
    "ST-08": "state_time.conflicting_source_freshness", "ST-09": "state_time.status_under_pressure",
    "EP-02": "epistemics.unknown_fact", "DL-05": "dialogue.focus_under_pressure",
    # contract 0.3
    "GS-01": "social.addressed_versus_overheard", "GS-02": "social.secret_with_mixed_audience", "GS-03": "state_time.attribution",
    "RW-01": "world.real_world_boundary", "SP-01": "state_time.story_position",
    "LW-01": "epistemics.lore_in_weights",  # a stable fact of the world the student's prompt does not carry
    "DL-06": "dialogue.long_conversation",  # a turn of a six-to-twelve-turn conversation, its own earlier lines in the history
    # who speaks next in a group
    "GS-04": "social.question_to_the_room", "GS-05": "social.question_passed_on", "GS-06": "social.line_from_another_character",
    "GS-07": "social.secret_comes_out",
}


def _message(turn: dict) -> dict:
    """A turn of the scenario as a training message, so multi-turn scenarios train with their history.
    Contract 0.3: another character's line keeps its name, and any turn keeps who it was said to."""
    role = {"player": "user", "actor": "assistant", "game": "game", "other": "other"}[turn["speaker"]]
    msg = {"role": role, "content": turn["text"], **({"mode": turn.get("mode") or "typed"} if role == "user" else {})}
    if turn.get("name"):
        msg["name"] = turn["name"]
    if turn.get("to"):
        msg["to"] = turn["to"]
    return msg


def _final_message(case: dict) -> dict:
    """The turn the reply answers: the plain player line, or (0.3) the case's last_turn."""
    last = case.get("last_turn")
    if not last:
        return {"role": "user", "content": case["player_utterance"], "mode": case.get("input_mode", "typed")}
    msg = _message({**last, "mode": case.get("input_mode", "typed") if last["speaker"] == "player" else None})
    return msg


def _uses_03(case: dict) -> bool:
    turns = [*case.get("prior_turns", []), *( [case["last_turn"]] if case.get("last_turn") else [])]
    return any(t.get("speaker") == "other" or t.get("to") for t in turns)


def to_training_example(row: dict, reply: str, fail_prob: float, code_hash: str, judge: str = "judge") -> dict:
    case = row["case"]
    gen = case["generator_like"]
    secondary = ["robustness.misheard_name"] if case.get("noise_twin_of") else []
    hidden = list(case.get("hidden_fact_ids") or [])  # the mixture step must rebuild the view without these
    ex = {
        "schema_version": "training-example-0.3" if _uses_03(case) or hidden else "training-example-0.1",
        "example_id": f"ex_{case['case_id']}", "split": "train",
        "world_id": row["world_id"], "scenario_id": case["case_id"], "primary_capability": CAPABILITY_IDS[gen],
        "secondary_capabilities": secondary, "difficulty_level": case.get("difficulty_level", "L2"),
        "student_input_ref": case["student_input_ref"], **({"hidden_fact_ids": hidden} if hidden else {}),
        "messages": [*(_message(t) for t in case.get("prior_turns", [])), _final_message(case),
                     {"role": "assistant", "content": reply}],
        "labels": {"assertion_policy": "; ".join(case["expected_properties"]),
                   "forbidden_fact_ids": case.get("forbidden_claim_ids", [])},
        "provenance": {"generator_id": gen, "generator_version": row["generator_version"], "generator_commit": code_hash,
                       "teacher_id": row["teacher"]["id"], "teacher_revision": row["teacher"]["revision"],
                       "teacher_prompt_hash": row["prompt_hash"], "generation_seed": row["candidate"],
                       "canonical_source_ids": [row["world_id"]]},
        "validation": {"schema": "pass", "oracle_consistency": f"{judge} fail probability {fail_prob:.2f}",
                       "style_review": None, "duplicate_cluster_id": None, "reviewer_status": "automated_accept"},
    }
    return TrainingExample.model_validate(ex).model_dump(by_alias=True)


def code_fingerprint(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]
