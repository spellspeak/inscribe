"""Talk to one of the game's characters with the file a game ships, on this machine.

    with Server(gguf_path) as server:
        talk = Conversation(sheet, npcs, "npc_maren_tolliver", fmt)
        print(talk.say(server, "What's a pint cost?"))

It needs `llama-server` from llama.cpp (build b11146 or later) on PATH. The prompt is built exactly as in training and
in the checks: the character's view of the world from the sheet, then the conversation so far. Replies are greedy by
default, so a line from the report card comes out the same here.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from inscribe.context.compiler import project_actor_view
from inscribe.plan.world import DynamicNPC, game_world
from inscribe.runtime.actor_context import Turn
from inscribe.runtime.template import ChatFormat, render_prompt
from inscribe.schemas.game_sheet import GameSheet

MAX_NEW = 80
CONTEXT = 4096
LOAD_SECONDS = 600


class ChatError(RuntimeError):
    pass


def find_gguf(out_dir: pathlib.Path, game: str, base: str, quant: str | None = None) -> pathlib.Path:
    """The exported file for this game and base model in out/. With several, `quant` picks one."""
    stem = f"{game}-{base}".replace("_", "-")
    files = sorted(out_dir.glob(f"{stem}-*.gguf"))
    if quant:
        files = [f for f in files if f.name == f"{stem}-{quant}.gguf"]
    if not files:
        raise ChatError(f"no {stem}-{quant or '*'}.gguf in {out_dir}. Run `inscribe pull` to bring the exported file here.")
    if len(files) > 1:
        raise ChatError("more than one file: " + ", ".join(f.name for f in files) + ". Pick one with --quant.")
    return files[0]


def who(sheet: GameSheet, npcs: tuple[DynamicNPC, ...] | list[DynamicNPC], name: str) -> str:
    """A character's id from an id, a name or the start of a name, whatever the case."""
    people = [(c.id, c.name) for c in sheet.characters] + [(n.id, n.name) for n in npcs]
    want = name.strip().lower()
    exact = [i for i, n in people if want in (i.lower(), n.lower())]
    if exact:
        return exact[0]
    close = [i for i, n in people if n.lower().startswith(want) or i.lower().removeprefix("npc_").startswith(want)]
    if len(close) == 1:
        return close[0]
    raise ChatError((f"{name!r} matches {len(close)} characters. " if close else f"no character called {name!r}. ")
                    + "Characters: " + ", ".join(f"{n} ({i})" for i, n in people))


class Server:
    """llama-server on a free local port, loaded with one file, stopped on exit."""

    def __init__(self, gguf: pathlib.Path, binary: str = "llama-server", gpu_layers: str = "all") -> None:
        self.gguf, self.binary, self.gpu_layers = pathlib.Path(gguf), binary, gpu_layers
        self.port = 0
        self._proc: subprocess.Popen | None = None
        self._log = None

    def __enter__(self) -> "Server":
        if shutil.which(self.binary) is None:
            raise ChatError(f"{self.binary} not found. Install llama.cpp (build b11146 or later) and put llama-server on PATH, or pass --server.")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self._log = tempfile.TemporaryFile()
        # --lazy-mode on reads the model's per-layer table from the file a row at a time, instead of holding it all in memory.
        self._proc = subprocess.Popen([self.binary, "-m", str(self.gguf), "--host", "127.0.0.1", "--port", str(self.port), "-c", str(CONTEXT), "-np", "1",
                                       "--no-ui", "-cram", "0", "-ngl", self.gpu_layers, "--lazy-mode", "on"], stdout=self._log, stderr=subprocess.STDOUT)
        for _ in range(LOAD_SECONDS):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2).status == 200:
                    return self
            except (urllib.error.URLError, OSError):
                if self._proc.poll() is not None:
                    self._log.seek(0)
                    tail = self._log.read().decode(errors="replace").strip().splitlines()[-6:]
                    self.__exit__(None, None, None)
                    raise ChatError("llama-server stopped while loading. Its build may be too old for this file. It said:\n  " + "\n  ".join(tail)) from None
                time.sleep(1)
        self.__exit__(None, None, None)
        raise ChatError("llama-server did not become ready")

    def __exit__(self, *exc) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait()
        if self._log is not None:
            self._log.close()
        self._proc = self._log = None

    def complete(self, prompt: str, stop: str, temperature: float = 0.0, seed: int = 0) -> str:
        body = {"prompt": prompt, "n_predict": MAX_NEW, "temperature": temperature, "seed": seed, "cache_prompt": True, "stop": [stop, "\n"]}
        if temperature == 0:
            body["top_k"] = 1
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))["content"].strip()


class Conversation:
    """One character's side of a conversation with the player, at one point in the story."""

    def __init__(self, sheet: GameSheet, npcs: tuple[DynamicNPC, ...] | list[DynamicNPC], actor: str, fmt: ChatFormat, stage: int | None = None) -> None:
        world = game_world(sheet, stage, dynamic_npcs=npcs)
        self.view = project_actor_view(world, actor, session_id="chat")
        self.name = self.view.actor.name
        self.fmt = fmt
        self.dialogue: list[Turn] = []

    def prompt(self, line: str) -> str:
        return render_prompt(self.view, [*self.dialogue, Turn(speaker="player", text=line, mode="typed")], self.fmt)

    def say(self, server: Server, line: str, temperature: float = 0.0) -> str:
        reply = server.complete(self.prompt(line), self.fmt.reply_end, temperature)
        self.dialogue += [Turn(speaker="player", text=line, mode="typed"), Turn(speaker="actor", text=reply)]
        return reply
