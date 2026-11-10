"""A step that does almost nothing: it proves a backend can build an image, run a step and keep its files.

It reads a number from its job, writes a file beside its result, and says what machine it ran on.
"""
from __future__ import annotations

import pathlib
import platform
import shutil
import subprocess
import time


def run(work: pathlib.Path, run_dir: pathlib.Path, job: dict) -> dict:
    if job.get("fail"):
        raise RuntimeError(str(job["fail"]))
    time.sleep(float(job.get("sleep", 0)))
    (run_dir / "echo.txt").write_text(f"{job.get('say', 'hello')}\n")
    gpu = None
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or None
    inputs = sorted(str(p.relative_to(work)) for p in (work / job["list"]).rglob("*") if p.is_file()) if job.get("list") else []
    print(f"smoke: python {platform.python_version()}, gpu {gpu}, {len(inputs)} input file(s)")
    return {"python": platform.python_version(), "machine": platform.machine(), "gpu": gpu, "inputs": inputs, "said": job.get("say", "hello")}
