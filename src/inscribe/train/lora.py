"""Train a game LoRA on a base model. One GPU, one container.

The step, in order:

1. Fetch the base model from Hugging Face at its pinned revision and check its hash.
2. Read the training rows: the exact `prompt` and `target` text of each example.
3. Answer the held-out questions with the base model, before anything is trained.
4. Train the LoRA: one example at a time, gradients accumulated, a cosine learning rate after a short warm-up.
5. Save the adapter, merge it into the base, answer the same questions again, and save the merged model.

With `max_steps` set it stops after that many optimizer steps and saves nothing. That is a cheap check that the
base loads, the data tokenizes and the first losses are what they should be.

Job (`job.json` in the run folder):
    base         {repo, revision, sha256: {file: hash}}
    chat_format  the base model's chat format name
    lora         {r, alpha, dropout, target_modules, lr, epochs, grad_accum, warmup_steps, min_lr_ratio, seed, max_len}
    data         folder on the store with the training rows (*.jsonl)
    prompts      file on the store with the questions ({id, prompt, max_new} per line), or null
    max_steps    stop after this many optimizer steps and save nothing, or null for the whole run
    keep         with max_steps: save, merge and answer again all the same, to rehearse the whole path
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import time

BASE_FILES = ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_rows(folder: pathlib.Path) -> list[dict]:
    """The training rows of every *.jsonl file in the folder, files in name order, rows in file order."""
    files = sorted(folder.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no training rows in {folder}")
    rows = [row for f in files for row in read_jsonl(f)]
    for n, row in enumerate(rows):
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("target"), str):
            raise ValueError(f"training row {n} needs a `prompt` and a `target`")
    return rows


def fetch_base(work: pathlib.Path, base: dict) -> pathlib.Path:
    """The base model's files, downloaded once into the store and checked against the recipe's hashes."""
    from huggingface_hub import snapshot_download

    cache = work / "cache" / "hf"
    folder = pathlib.Path(snapshot_download(base["repo"], revision=base["revision"], allow_patterns=list(BASE_FILES), cache_dir=str(cache)))
    for name, want in base.get("sha256", {}).items():
        mark = cache / f"checked-{want}"
        if mark.exists():
            continue
        got = sha256(folder / name)
        if got != want:
            raise RuntimeError(f"{base['repo']}@{base['revision'][:12]} {name} is {got[:16]}, the recipe says {want[:16]}")
        mark.write_text(f"{base['repo']} {base['revision']} {name}\n")
    return folder


def machine() -> dict:
    """What the step ran on. The same weights give slightly different losses, and sometimes differently worded greedy
    replies, on different machines, so a result says which machine it came from."""
    import subprocess

    import torch

    out = {"cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "driver": None, "cpu": None}
    try:
        out["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                                       text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001  (a detail for the record, never a reason to fail a run)
        pass
    try:
        out["cpu"] = next(line.split(":", 1)[1].strip() for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except Exception:  # noqa: BLE001
        pass
    return out


def tokenize(rows: list[dict], ctok, bos: list[int], end_ids: list[int], max_len: int) -> tuple[list[dict], int]:
    """Each row as token ids: the prompt is context, only the target is learned. Rows over `max_len` are left out."""
    examples, too_long = [], 0
    for n, row in enumerate(rows):
        p = bos + ctok.encode(row["prompt"], add_special_tokens=False).ids
        t = ctok.encode(row["target"], add_special_tokens=False).ids
        if len(p) + len(t) > max_len:
            too_long += 1
            continue
        if not t or t[-len(end_ids):] != end_ids:
            raise RuntimeError(f"training row {row.get('id', n)}: the target does not end with the reply-end marker")
        examples.append({"input_ids": p + t, "labels": [-100] * len(p) + t})
    return examples, too_long


def run(work: pathlib.Path, run_dir: pathlib.Path, job: dict) -> dict:
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from inscribe.runtime.template import format_by_name

    t_start = time.perf_counter()
    cfg = job["lora"]
    fmt = format_by_name(job["chat_format"])
    max_steps = job.get("max_steps")

    base_dir = fetch_base(work, job["base"])
    print(f"base: {job['base']['repo']} at {job['base']['revision'][:12]}, fetched and checked in {time.perf_counter() - t_start:.0f} s", flush=True)
    ctok = Tokenizer.from_file(str(base_dir / "tokenizer.json"))
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    bos = [tok.convert_tokens_to_ids(fmt.bos_token)] if fmt.bos_token else []
    end_ids = ctok.encode(fmt.reply_end, add_special_tokens=False).ids
    stop_ids = [tok(fmt.reply_end, add_special_tokens=False).input_ids[0]]
    if tok.eos_token_id is not None and tok.eos_token_id not in stop_ids:
        stop_ids.append(tok.eos_token_id)

    rows = read_rows(work / job["data"])
    examples, too_long = tokenize(rows, ctok, bos, end_ids, cfg["max_len"])
    supervised = sum(sum(1 for x in e["labels"] if x != -100) for e in examples)
    if supervised == 0:
        raise RuntimeError("no supervised tokens")
    print(f"data: {len(examples)} examples ({too_long} over {cfg['max_len']} tokens left out), {supervised} target tokens", flush=True)
    prompts = read_jsonl(work / job["prompts"]) if job.get("prompts") else []

    def answer(model, label: str) -> list[dict]:
        """Greedy replies to the questions, one at a time."""
        model.eval()
        out = []
        t0 = time.perf_counter()
        for pr in prompts:
            ids = torch.tensor([bos + tok(pr["prompt"], add_special_tokens=False).input_ids]).to("cuda")
            with torch.no_grad():
                seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=pr.get("max_new", 80),
                                     do_sample=False, eos_token_id=stop_ids, pad_token_id=stop_ids[0])
            new = seq[0, ids.shape[1]:].tolist()
            stopped = bool(new) and new[-1] in stop_ids
            out.append({"id": pr["id"], "text": tok.decode(new[:-1] if stopped else new, skip_special_tokens=False),
                        "stopped": stopped, "new_tokens": len(new), "prompt_tokens": int(ids.shape[1])})
        (run_dir / f"answers-{label}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
        if out:
            print(f"answers ({label}): {len(out)} questions in {time.perf_counter() - t0:.0f} s", flush=True)
        return out

    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cuda")
    before = answer(model, "before")

    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    pm = get_peft_model(model, LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                                          target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    order = [i for _ in range(cfg["epochs"]) for i in rng.sample(range(len(examples)), len(examples))]
    total_steps = math.ceil(len(order) / cfg["grad_accum"])

    def lr_at(step: int) -> float:
        if step < cfg["warmup_steps"]:
            return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
        frac = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
        return cfg["lr"] * (cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * frac)))

    pm.train()
    torch.cuda.reset_peak_memory_stats()
    losses, window, step, t_train = [], [], 0, time.perf_counter()
    for n, i in enumerate(order):
        ids = torch.tensor([examples[i]["input_ids"]], device="cuda")
        labels = torch.tensor([examples[i]["labels"]], device="cuda")
        loss = pm(input_ids=ids, labels=labels, use_cache=False).loss / cfg["grad_accum"]
        loss.backward()
        window.append(loss.item() * cfg["grad_accum"])
        if (n + 1) % cfg["grad_accum"] == 0 or n + 1 == len(order):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(sum(window) / len(window), 4))
            window, step = [], step + 1
            if step <= 5 or step % 50 == 0 or step == total_steps:
                print(f"step {step} of {total_steps}: loss {losses[-1]:.4f}, {time.perf_counter() - t_train:.0f} s", flush=True)
            if max_steps and step >= max_steps:
                break
    train_s = time.perf_counter() - t_train
    result = {
        "examples": len(examples), "dropped_too_long": too_long, "supervised_tokens": supervised,
        "total_tokens": sum(len(e["input_ids"]) for e in examples), "steps": step, "total_steps": total_steps, "epochs": cfg["epochs"],
        "lora": cfg, "loss_first": losses[:5], "loss_last": losses[-5:], "loss_curve": losses[:: max(1, len(losses) // 40)],
        "train_s": round(train_s), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "gpu": torch.cuda.get_device_name(0),
        "versions": {"torch": str(torch.__version__), "transformers": transformers.__version__, "peft": peft.__version__},
        "machine": machine(),
        "base": job["base"], "questions": len(prompts), "answers_before": f"{run_dir.name}/answers-before.jsonl" if before else None,
        "stopped_early": bool(max_steps),
    }
    if max_steps and not job.get("keep"):
        result["total_s"] = round(time.perf_counter() - t_start)
        return result

    pm.save_pretrained(run_dir / "adapter")
    merged = pm.merge_and_unload()
    after = answer(merged, "after")
    out_dir = run_dir / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    for name in BASE_FILES:
        if name.startswith(("tokenizer", "chat_template")):
            (out_dir / name).write_bytes((base_dir / name).read_bytes())
    result.update({"adapter": f"{run_dir.name}/adapter", "merged": f"{run_dir.name}/merged",
                   "answers_after": f"{run_dir.name}/answers-after.jsonl" if after else None,
                   "total_s": round(time.perf_counter() - t_start)})
    return result
