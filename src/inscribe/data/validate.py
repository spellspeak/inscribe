"""The checks on the teacher's lines: cheap code checks first, then the judge, then duplicates.

A scene gets several candidate lines. A candidate is dropped when it is empty or cut off, quotes the question back,
fails a code check (a banned or missing word, markup, length), is an exact copy seen twice already, or the judge gives
it a fail probability at or above the acceptance line. Of the candidates left, a scene keeps the one the judge likes
best, and a line too close to one already kept from the same kind of scene is dropped. Every dropped candidate keeps
its reason.

A scene that hides lore from the model's prompt is judged on the view with nothing hidden, and the judge is also given
the scene's teacher-only sentences: the character does know those facts, so a line that states one is supported, not
invented. The code checks always run on the view the model will see.
"""
from __future__ import annotations

import re
from collections import defaultdict

from inscribe.checks import check_reply
from inscribe.data.examples import CAPABILITY_IDS, _final_message, to_training_example
from inscribe.data.teacher import clean_line
from inscribe.judges import questions as Q

CODE_ONLY = frozenset({"LW-01"})  # lore scenes: the lore is in no character's view, so only code can check the line

NEAR_DUP_JACCARD = 0.8


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def shingles(text: str) -> set[str]:
    w = norm(text).split()
    return {" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))}


def echoes_question(case: dict, reply: str) -> bool:
    """The reply opens by quoting the line it answers back ('"What can you tell me about the Compact?" It is the name of
    the agreement...'): a teacher habit on lore and story scenes that a LoRA then learns. A quoted span of three or
    more words, most of them from the line."""
    m = re.match(r"\s*[\"'“‘](.{8,}?)[\"'”’]", reply)
    if not m:
        return False
    quoted, asked = set(norm(m.group(1)).split()), set(norm(_final_message(case)["content"]).split())
    return len(quoted) >= 3 and len(quoted & asked) / len(quoted) >= 0.6


def structural(row: dict, reply: str) -> str | None:
    if not reply:
        return "empty"
    if row["finish_reason"] != "stop":
        return "cut_off"
    if echoes_question(row["case"], reply):
        return "echoes_question"
    return None


def world_resolver(worlds: dict):
    """`resolve(case, world_id)`: the scene resolved on its world file."""
    from inscribe.scenarios.core import resolve_generated

    def resolve(case, world_id):
        if world_id not in worlds:
            raise KeyError(f"no world file for {world_id}; run `inscribe plan` again")
        return resolve_generated(case, worlds[world_id])

    return resolve


def screen(rows: list[dict], resolve) -> tuple[list[dict], list[tuple]]:
    """Everything before the judge: structure, code checks, exact duplicates. Returns (rejected candidates with their
    reasons, pending (row, reply, resolved scene) for the judge)."""
    rejected, pending = [], []
    seen_exact: dict[str, int] = defaultdict(int)
    for row in rows:
        reply = clean_line(row["text"])
        reason = structural(row, reply)
        rc = resolve(row["case"], row["world_id"]) if reason is None else None
        if reason is None:
            failed = [r.name for r in check_reply(rc.case, reply, rc.context) if not r.passed]
            if failed:
                reason = "code_check:" + ",".join(failed)
            elif row["case"].get("hidden_fact_ids"):
                rc = resolve({**row["case"], "hidden_fact_ids": []}, row["world_id"])  # the judge sees what the character knows
        if reason is None:
            key = norm(reply)
            seen_exact[key] += 1
            if seen_exact[key] > 2:
                reason = "exact_duplicate"
        if reason:
            rejected.append({**row, "reply": reply, "reason": reason})
        else:
            pending.append((row, reply, rc))
    return rejected, pending


def judge_requests(pending: list[tuple]) -> list[Q.JudgeRequest]:
    return [Q.build_request(rc, reply, item_id=str(i), extra_sheet="\n".join(f"- {s}" for s in (row["case"].get("teacher_only") or [])))
            for i, (row, reply, rc) in enumerate(pending)]


def validate(rows: list[dict], resolve, judge, code_hash: str, accept_below: float, judge_skip: frozenset = frozenset()) -> dict:
    """rows: teacher candidates. resolve(case, world_id) gives the resolved scene. `judge` has `answer_all(requests)` and
    a `name`. Returns the accepted training records, the rejected candidates with reasons, and counts. Scene kinds in
    `judge_skip` are checked by code only."""
    unknown = sorted({row["case"]["generator_like"] for row in rows} - set(CAPABILITY_IDS))
    if unknown:  # fail before any judge call is paid for
        raise ValueError(f"no capability id for scene kinds {unknown}")
    rejected, pending = screen(rows, resolve)
    code_only = [(row, reply, 0.0) for row, reply, _ in pending if row["case"]["generator_like"] in judge_skip]
    pending = [x for x in pending if x[0]["case"]["generator_like"] not in judge_skip]
    requests = judge_requests(pending)
    answers = {a["item_id"]: a for a in judge.answer_all(requests)}
    judged = list(code_only)
    for i, ((row, reply, _), req) in enumerate(zip(pending, requests)):
        a = answers.get(str(i), {})
        if not a.get("answers"):
            rejected.append({**row, "reply": reply, "reason": "judge_error"})
            continue
        p = Q.fail_probability(a["answers"], req.key)
        if p >= accept_below:
            q, _ = Q.top_question(a["answers"], req.key)
            rejected.append({**row, "reply": reply, "reason": f"judge:{q}", "fail_prob": round(p, 3)})
        else:
            judged.append((row, reply, p))

    best: dict[str, tuple] = {}
    for row, reply, p in judged:  # one line per scene: the judge's favourite
        cid = row["case"]["case_id"]
        if cid not in best or p < best[cid][2]:
            if cid in best:
                rejected.append({**best[cid][0], "reply": best[cid][1], "reason": "other_candidate_preferred"})
            best[cid] = (row, reply, p)
        else:
            rejected.append({**row, "reply": reply, "reason": "other_candidate_preferred"})
    accepted, kept_shingles = [], defaultdict(list)
    for cid in sorted(best):  # then near-duplicates within a kind of scene
        row, reply, p = best[cid]
        gen = row["case"]["generator_like"]
        sh = shingles(reply)
        if any(len(sh & o) / max(1, len(sh | o)) >= NEAR_DUP_JACCARD for o in kept_shingles[gen]):
            rejected.append({**row, "reply": reply, "reason": "near_duplicate"})
            continue
        kept_shingles[gen].append(sh)
        accepted.append(to_training_example(row, reply, p, code_hash, judge=judge.name))
    reasons: dict[str, int] = defaultdict(int)
    for r in rejected:
        reasons[r["reason"].split(":")[0]] += 1
    return {"accepted": accepted, "rejected": rejected, "reasons": dict(sorted(reasons.items())),
            "candidates": len(rows), "scenes": len({r["case"]["case_id"] for r in rows})}


def summary(rows: list[dict], out: dict) -> dict:
    """Counts for a report: what went in and what was kept, by kind of scene."""
    asked, kept = {}, {}
    for r in rows:
        g = r["case"]["generator_like"] + ("+noisy" if r["case"].get("noise_twin_of") else "")
        asked[g] = asked.get(g, 0) + 1
    for ex in out["accepted"]:
        g = ex["provenance"]["generator_id"] + ("+noisy" if ex["secondary_capabilities"] else "")
        kept[g] = kept.get(g, 0) + 1
    return {"candidates": out["candidates"], "scenes": out["scenes"], "accepted": len(out["accepted"]), "reasons": out["reasons"],
            "candidates_by_kind": dict(sorted(asked.items())), "accepted_by_kind": dict(sorted(kept.items()))}
