"""What a backend runs inside a step's container.

    python -m inscribe.stages.run <stage> --work /work --run <stage>/<run_id>

It reads `job.json` from the run folder, calls the step, and writes `result.json` beside it: `ok`, the step's own
result or the error, and how long it took. Everything the step prints also goes to `log.txt` in the run folder, so the
log outlives the container. A step is done when `result.json` exists. Nothing here imports the command-line tool.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import sys
import time
import traceback

# stage name -> "module:function". The function takes (work, run_dir, job) and returns a JSON-ready mapping.
STAGES = {
    "smoke": "inscribe.stages.smoke:run",
    "train": "inscribe.train.lora:run",
    "export": "inscribe.export.gguf:run",
    "mix": "inscribe.data.mix:run",
}


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for s in self.streams:
            s.write(text)
            s.flush()
        return len(text)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def isatty(self) -> bool:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one step inside its container.")
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--work", required=True, type=pathlib.Path)
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    run_dir = a.work / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").unlink(missing_ok=True)
    started = time.time()
    with (run_dir / "log.txt").open("w") as log:
        sys.stdout, sys.stderr = _Tee(sys.__stdout__, log), _Tee(sys.__stderr__, log)
        try:
            job = json.loads((run_dir / "job.json").read_text())
            module, _, func = STAGES[a.stage].partition(":")
            result = getattr(importlib.import_module(module), func)(a.work, run_dir, job)
            out = {"ok": True, "stage": a.stage, "result": result}
        except BaseException as exc:  # noqa: BLE001  (the reason must reach result.json whatever it is)
            traceback.print_exc()
            out = {"ok": False, "stage": a.stage, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-6000:]}
        out["seconds"] = round(time.time() - started, 1)
        print(f"[inscribe] {a.stage} {'finished' if out['ok'] else 'FAILED'} in {out['seconds']} s", flush=True)
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, default=str) + "\n")
    tmp.replace(run_dir / "result.json")
    os.sync()  # a mounted store must hold the result before the container goes away
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
