"""A game project on disk: where its files are.

    my-game/
      lora.yaml                 what to train and where
      lore/                     world.md, characters.md, story.md, style.md: prose, written by the team
      sheet/game-sheet.yaml     the game sheet: what the pipeline reads. Checked by a person
      npcs.yaml                 optional: characters the game makes up at run time, for the checks
      build/                    generated; safe to delete
      out/                      the GGUF, the adapter and the report
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

CONFIG_NAME = "lora.yaml"
LORE_FILES = ("world.md", "characters.md", "story.md", "style.md")


class ProjectError(ValueError):
    pass


@dataclass(frozen=True)
class Project:
    root: pathlib.Path

    @property
    def config_path(self) -> pathlib.Path:
        return self.root / CONFIG_NAME

    @property
    def lore_dir(self) -> pathlib.Path:
        return self.root / "lore"

    @property
    def sheet_path(self) -> pathlib.Path:
        return self.root / "sheet" / "game-sheet.yaml"

    @property
    def npcs_path(self) -> pathlib.Path:
        return self.root / "npcs.yaml"

    @property
    def build_dir(self) -> pathlib.Path:
        return self.root / "build"

    @property
    def plan_dir(self) -> pathlib.Path:
        return self.build_dir / "plan"

    @property
    def out_dir(self) -> pathlib.Path:
        return self.root / "out"


def find_project(start: str | pathlib.Path = ".") -> Project:
    """The project that holds `start`: the nearest folder, going up, with a lora.yaml."""
    here = pathlib.Path(start).resolve()
    for folder in (here, *here.parents):
        if (folder / CONFIG_NAME).is_file():
            return Project(folder)
    raise ProjectError(f"no {CONFIG_NAME} in {here} or above it. Run `inscribe init <name>` to start a game project, "
                       "or pass --project <folder>.")
