"""The teacher: a larger open model writes each scene's line, several times over.

The teacher sees exactly what the game model will see, plus a style instruction and what a good reply does in this
scene. It never sees hidden truth: no forbidden facts, no other character's knowledge. The answer key stays with the
scene and the checks, never with the teacher.

As a step it runs on one large GPU: it loads the teacher with vLLM, writes `candidates` lines per scene in one batch,
and saves them all. The checks then keep the best line of each scene, or none.

Job (`job.json` in the run folder):
    scenarios   file on the store with the scenes (the plan's scenarios.jsonl)
    worlds      folder on the store with the plan's world files
    candidates  lines to write per scene
    limit       write for only this many scenes, a seeded sample, or null for all
    out         file on the store to write the candidates to
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
import re
import time

from inscribe.cases import ResolvedCase
from inscribe.judges.questions import name_map
from inscribe.runtime.template import group_scene_from, labelled_line, render_system

TEACHER_ID = "qwen3.8-27b-fp8"
TEACHER_REPO = "Qwen/Qwen3.8-27B-FP8"
TEACHER_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
PROMPT_VERSION = "teacher-prompt-0.4"
SAMPLING = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 100, "seed_base": 0}
STYLE_FAMILY = ("LORE", "style")  # a scene whose reply is the sheet's own line never goes to the teacher

STYLE = """You are writing one line of dialogue for {name}, a character in a video game. Everything above is what {name} knows, believes and feels right now.
Write only what {name} says next, as a single line of spoken dialogue.
- Write the way this particular person talks: their Voice line, their mood, what they like and dislike, and how they feel toward whoever just spoke to them. A friend and a rival do not get the same line.
- Draw on the one trait or feeling that fits this moment. A person is not their list; do not work every listed value or dislike into every line.
- Vary how the line begins. It may open with the thing itself, a name, a question, an order, a half-thought. Do not open with "I", "No", "Aye", "Yeah", "Ah" or "Well" unless that is how this person would start. Never repeat or quote the other person's words back at them.
- Dialect belongs to the setting. No "aye" in a modern town or on a station; no slang from outside this world.
- Use only what {name} knows or believes, with the confidence they have. Never invent facts about people, places, prices or events. A price is given only as it is written above; if no price is written, they do not name one. If they do not know, they say so in their own way.
- Keep every secret. A cover story, if one is given, is used only when the secret itself is asked about; never offered unasked, never as the answer to something else.
- Never say what happens next in the story or what anyone will decide. Asked what comes next, {name} can hope, fear, guess vaguely or refuse, in their own way, without naming an event, a decision, a person's part in it or a place the story has not reached.
- Answer what was asked, then stop. Do not add new details, events, plans or what they are busy doing.
- One or two short sentences, at most 45 words. No actions, emotes, stage directions, lists, narration, or quotation marks around the line.
What a good reply does here:
{goals}"""
TEACHER_ONLY_HEADER = "Things {name} also knows, not written above. Use them as the character would, when they matter; never recite them as a list:"


def _goal(prop: str, names: dict[str, str]) -> str:
    return "- " + " ".join(names.get(w, w) for w in prop.split("_"))


def teacher_messages(rc: ResolvedCase, teacher_only: list[str] | None = None) -> list[dict]:
    """Chat messages for the teacher: the scene as the model will see it, plus the style instruction, then the
    conversation. `teacher_only`: lore this character knows that is left out of the model's prompt on purpose, so the
    model learns to carry it in its weights. The teacher is told it; the model's prompt never has it."""
    names = name_map(rc)
    goals = "\n".join(_goal(p, names) for p in rc.case.expected_properties)
    system = render_system(rc.context)
    if teacher_only:
        system += "\n\n" + TEACHER_ONLY_HEADER.format(name=rc.context.actor.name) + "\n" + "\n".join(f"- {f}" for f in teacher_only)
    system += "\n\n" + STYLE.format(name=rc.context.actor.name, goals=goals)
    msgs = [{"role": "system", "content": system}]
    group_from = group_scene_from(rc.dialogue)  # labelled lines from the turn that makes it a group scene
    for i, t in enumerate(rc.dialogue):
        labelled = group_from is not None and i >= group_from
        if t.speaker in ("player", "other"):
            msgs.append({"role": "user", "content": labelled_line(t) if labelled else t.text})
        elif t.speaker == "actor":
            msgs.append({"role": "assistant", "content": t.text})
        else:  # the teacher's chat template has no mid-conversation system turns
            msgs.append({"role": "user", "content": f"[game update] {t.text}"})
    return msgs


def prompt_hash(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


def clean_line(text: str) -> str:
    """The teacher's reply as one spoken line: first line only, surrounding quotes and a leading name removed."""
    line = text.strip().split("\n", 1)[0].strip()
    line = re.sub(r"^[A-Z][\w' .-]{0,30}:\s+", "", line)  # "Brenna: ..." -> "..."
    if len(line) >= 2 and line[0] in "\"“" and line[-1] in "\"”" and line.count('"') <= 2:
        line = line[1:-1].strip()
    return line


def is_style_case(case: dict) -> bool:
    return (case.get("generator_like"), case.get("family")) == STYLE_FAMILY


def teacher_items(cases: list[dict], worlds: dict) -> list[dict]:
    """One item per scene: the scene, its world and the teacher's messages. Each scene is resolved on the world file its
    reference names, and the teacher is shown the lore the model's prompt hides."""
    from inscribe.scenarios.core import resolve_generated, split_ref

    items = []
    for case in cases:
        if is_style_case(case):
            continue
        world_id = split_ref(case["student_input_ref"])[0]
        if world_id not in worlds:
            raise KeyError(f"{case['case_id']} is on {world_id}, which is not among the plan's worlds {sorted(worlds)}")
        rc = resolve_generated(case, worlds[world_id])
        msgs = teacher_messages(rc, teacher_only=rc.case.teacher_only)
        items.append({"case": case, "world_id": world_id, "messages": msgs, "prompt_hash": prompt_hash(msgs)})
    return items


def run(work: pathlib.Path, run_dir: pathlib.Path, job: dict) -> dict:
    import torch
    import yaml
    from vllm import LLM, SamplingParams

    from inscribe.schemas.world import World

    t0 = time.perf_counter()
    worlds = {p.stem: World.model_validate(yaml.safe_load(p.read_text())) for p in sorted((work / job["worlds"]).glob("*.yaml"))}
    cases = [json.loads(line) for line in (work / job["scenarios"]).read_text().splitlines() if line.strip()]
    items = teacher_items(cases, worlds)
    if job.get("limit"):
        items = random.Random(0).sample(items, min(job["limit"], len(items)))
    built_s = time.perf_counter() - t0
    print(f"scenes: {len(items)} on {len(worlds)} worlds, {job['candidates']} lines each", flush=True)

    llm = LLM(model=TEACHER_REPO, revision=TEACHER_REVISION, max_model_len=8192, gpu_memory_utilization=0.9, kv_cache_dtype="fp8",
              enable_prefix_caching=True, seed=0, max_num_seqs=256)
    load_s = time.perf_counter() - t0 - built_s
    print(f"teacher loaded in {load_s:.0f} s", flush=True)
    params = SamplingParams(n=job["candidates"], temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                            max_tokens=SAMPLING["max_tokens"], stop=["\n"], seed=SAMPLING["seed_base"])
    outs = llm.chat([it["messages"] for it in items], params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    gen_s = time.perf_counter() - t0 - built_s - load_s

    rows, prompt_tokens, output_tokens = [], 0, 0
    for it, out in zip(items, outs):
        prompt_tokens += len(out.prompt_token_ids)
        for k, comp in enumerate(out.outputs):
            output_tokens += len(comp.token_ids)
            rows.append({"case": it["case"], "world_id": it["world_id"], "candidate": k, "text": comp.text, "finish_reason": comp.finish_reason,
                         "prompt_tokens": len(out.prompt_token_ids), "output_tokens": len(comp.token_ids), "prompt_hash": it["prompt_hash"],
                         "teacher": {"id": TEACHER_ID, "repo": TEACHER_REPO, "revision": TEACHER_REVISION, "prompt_version": PROMPT_VERSION,
                                     "sampling": {**SAMPLING, "n": job["candidates"]}},
                         "generator_version": it["case"].get("notes") or "unknown"})
    out_path = work / job["out"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} lines for {len(items)} scenes in {gen_s:.0f} s", flush=True)
    return {"scenes": len(items), "candidates": len(rows), "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
            "build_s": round(built_s, 1), "load_s": round(load_s, 1), "generate_s": round(gen_s, 1), "gpu": torch.cuda.get_device_name(0),
            "teacher": {"repo": TEACHER_REPO, "revision": TEACHER_REVISION, "prompt_version": PROMPT_VERSION}, "out": job["out"],
            "examples": [{"asked": r["case"].get("player_utterance") or (r["case"].get("last_turn") or {}).get("text", ""), "line": r["text"]} for r in rows[:6]]}


def fetch(work: pathlib.Path, run_dir: pathlib.Path, job: dict) -> dict:
    """Download a model into the store's cache on a cheap machine, so the GPU step does not pay for the download."""
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    path = snapshot_download(job["repo"], revision=job["revision"])
    size = sum(p.stat().st_size for p in pathlib.Path(path).rglob("*") if p.is_file())
    print(f"{job['repo']}: {size / 1e9:.1f} GB in the cache", flush=True)
    return {"repo": job["repo"], "revision": job["revision"], "gb": round(size / 1e9, 2), "seconds": round(time.perf_counter() - t0)}
