"""Export a trained model to GGUF for llama.cpp, and answer the held-out questions with the file a game would ship.

The step, in order:

1. Convert the merged model to a full-precision GGUF.
2. Build an importance matrix from text the model will see in play: the game's own prompts, then prompts from the
   worlds the base model was trained on. It tells the quantizer which weights matter most for this game.
3. Quantize to each type asked for. The embedding tables are half of a Gemma 4 file, so they get their own, smaller type.
4. Keep the quantized files on the store, with their hashes.
5. Start llama-server on each file and answer the questions, greedy, one at a time.

Job (`job.json` in the run folder):
    merged       folder on the store with the merged model (a training run's `merged`)
    chat_format  the base model's chat format name
    name         the stem of the files to write: <name>-<type>.gguf
    title        the model name written into the file
    quant        GGUF types to make, for example ["Q5_K_M"]
    embeddings   the type for the embedding tables, for example "q4_k", or null to leave them to the type's default
    imatrix      file on the store with the game's own prompts, or null for no importance matrix
    prompts      file on the store with the questions ({id, prompt, max_new} per line), or null
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile
import time
from importlib import resources

LLAMA_SOURCE = "/opt/llama.cpp"  # the converter, at the pinned commit
LLAMA_BIN = "/app"  # prebuilt binaries in llama.cpp's official image
CONTEXT = 4096
PORT = 8093
ROLE_TOKENS = ("bos_token", "eos_token", "unk_token", "pad_token", "mask_token", "sep_token", "cls_token")


def quantize_args(embeddings: str | None) -> list[str]:
    """llama-quantize's arguments that give the token embeddings and Gemma 4's per-layer embeddings their own type."""
    return ["--tensor-type", f"per_layer_token_embd={embeddings}", "--token-embedding-type", embeddings] if embeddings else []


def readable_config(cfg: dict, vocab: set[str]) -> dict:
    """Gemma 4's tokenizer config as the converter's older transformers reads it: the named tokens and the extra list
    moved into `additional_special_tokens`. Only tokens already in the vocabulary are named, so no id changes. Without
    this the GGUF treats `<|turn>` as ordinary text and llama.cpp splits it into four tokens in every prompt."""
    named = [v for k, v in cfg.items() if k.endswith("_token") and k not in ROLE_TOKENS and isinstance(v, str)]
    extra = [t for t in cfg.get("extra_special_tokens") or [] if isinstance(t, str)]
    out = {k: v for k, v in cfg.items() if k != "extra_special_tokens"}
    out["additional_special_tokens"] = [t for t in dict.fromkeys([*named, *extra]) if t in vocab]
    return out


def converter_view(src: pathlib.Path) -> pathlib.Path:
    """The model folder as the converter should see it: links to every file, with a readable tokenizer config when the
    model's own is written for a newer transformers."""
    cfg_path = src / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if not isinstance(cfg.get("extra_special_tokens"), list):
        return src
    tok = json.loads((src / "tokenizer.json").read_text())
    vocab = set(tok["model"]["vocab"]) | {t["content"] for t in tok.get("added_tokens", [])}
    view = pathlib.Path(tempfile.mkdtemp(prefix="gguf-view-"))
    for f in src.iterdir():
        if f.name != "tokenizer_config.json":
            (view / f.name).symlink_to(f.resolve())
    (view / "tokenizer_config.json").write_text(json.dumps(readable_config(cfg, vocab), indent=1))
    return view


def calibration_text(chat_format: str) -> str:
    """Prompts from the worlds the base model was trained on, shipped with the package."""
    return resources.files("inscribe.export").joinpath(f"calibration-{chat_format}.txt").read_text()


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], log: list[dict], step: str, timeout: int = 3600) -> None:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    tail = (p.stdout + p.stderr)[-1500:]
    log.append({"step": step, "exit": p.returncode, "seconds": round(time.perf_counter() - t0, 1), "tail": tail})
    print(f"{step}: exit {p.returncode} in {log[-1]['seconds']} s", flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"{step} failed: {tail[-800:]}")


def _gpu_mem_mb() -> int | None:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        return int(q.stdout.split()[0])
    except Exception:  # noqa: BLE001  (a detail for the record)
        return None


def _complete(prompt: str, n_predict: int, stop: str) -> dict:
    import requests

    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": False,
            "stream": True, "return_tokens": True, "stop": [stop]}
    t0 = time.perf_counter()
    first, pieces, final = None, [], {}
    with requests.post(f"http://127.0.0.1:{PORT}/completion", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            chunk = json.loads(line[6:])
            if first is None and (chunk.get("tokens") or chunk.get("content")):
                first = (time.perf_counter() - t0) * 1000
            pieces.append(chunk.get("content") or "")
            if chunk.get("stop"):
                final = chunk
    t = final.get("timings") or {}
    return {"text": "".join(pieces), "stopped": final.get("stop_type") in ("eos", "word"), "new_tokens": t.get("predicted_n"),
            "prompt_tokens": t.get("prompt_n"), "first_token_ms": None if first is None else round(first, 1),
            "tokens_per_second": t.get("predicted_per_second")}


def serve_and_answer(gguf: pathlib.Path, prompts: list[dict], reply_end: str) -> dict:
    """Start llama-server on the file with every layer on the GPU, answer the questions, and stop it."""
    import requests

    log_path = pathlib.Path(tempfile.gettempdir()) / f"{gguf.stem}.server.log"
    cmd = [f"{LLAMA_BIN}/llama-server", "-m", str(gguf), "--host", "127.0.0.1", "--port", str(PORT), "-c", str(CONTEXT), "-np", "1",
           "--no-ui", "-cram", "0", "-ngl", "all"]
    before = _gpu_mem_mb()
    with log_path.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    try:
        t0 = time.perf_counter()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server stopped while loading: {log_path.read_text()[-1500:]}")
            if time.perf_counter() - t0 > 600:
                raise TimeoutError("llama-server was not ready after 600 s")
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        after = _gpu_mem_mb()
        rows = [{"id": p["id"], **_complete(p["prompt"], p.get("max_new", 80), reply_end)} for p in prompts]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    used = None if before is None or after is None else after - before
    return {"rows": rows, "gpu_memory_mb": used, "on_gpu": (used or 0) > 500}


def run(work: pathlib.Path, run_dir: pathlib.Path, job: dict) -> dict:
    from inscribe.runtime.template import format_by_name

    t_start = time.perf_counter()
    fmt = format_by_name(job["chat_format"])
    merged = work / job["merged"]
    if not (merged / "model.safetensors").exists():
        raise FileNotFoundError(f"no merged model at {job['merged']}: run the training step first")
    prompts = [json.loads(line) for line in (work / job["prompts"]).read_text().splitlines() if line.strip()] if job.get("prompts") else []
    log: list[dict] = []
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="export-"))

    full = tmp / f"{job['name']}-bf16.gguf"
    _run(["python", f"{LLAMA_SOURCE}/convert_hf_to_gguf.py", str(converter_view(merged)), "--outtype", "bf16", "--outfile", str(full),
          "--model-name", job["title"]], log, "convert")
    imatrix: list[str] = []
    if job.get("imatrix"):
        calib = tmp / "calibration.txt"
        calib.write_text((work / job["imatrix"]).read_text() + "\n\n" + calibration_text(fmt.name))
        _run([f"{LLAMA_BIN}/llama-imatrix", "-m", str(full), "-f", str(calib), "-o", str(tmp / "imatrix.gguf"), "-ngl", "99", "-c", "512"],
             log, "importance matrix")
        imatrix = ["--imatrix", str(tmp / "imatrix.gguf")]

    out_dir = run_dir / "gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for quant in job["quant"]:
        path = tmp / f"{job['name']}-{quant}.gguf"
        _run([f"{LLAMA_BIN}/llama-quantize", *imatrix, *quantize_args(job.get("embeddings")), str(full), str(path), quant], log, f"quantize {quant}")
        shutil.copy(path, out_dir / path.name)
        files[quant] = {"file": f"{run_dir.name}/gguf/{path.name}", "bytes": path.stat().st_size, "sha256": sha256(path)}
        print(f"{path.name}: {files[quant]['bytes'] / 1e9:.2f} GB", flush=True)
    full.unlink()

    for quant in job["quant"]:
        if not prompts:
            break
        t0 = time.perf_counter()
        answered = serve_and_answer(tmp / f"{job['name']}-{quant}.gguf", prompts, fmt.reply_end)
        (run_dir / f"answers-{quant}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in answered["rows"]))
        files[quant].update({"answers": f"{run_dir.name}/answers-{quant}.jsonl", "gpu_memory_mb": answered["gpu_memory_mb"], "on_gpu": answered["on_gpu"]})
        print(f"answers ({quant}): {len(prompts)} questions in {time.perf_counter() - t0:.0f} s, on the GPU: {answered['on_gpu']}", flush=True)
    return {"files": files, "questions": len(prompts), "embeddings": job.get("embeddings"), "importance_matrix": bool(imatrix),
            "steps": log, "total_s": round(time.perf_counter() - t_start)}
