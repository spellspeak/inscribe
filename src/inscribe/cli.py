"""The `inscribe` command.

    inscribe init my-game     start a game project from the template
    inscribe check            validate the game sheet, show what would be built, forecast the run. Free
    inscribe plan             build the scenes, the held-out checks and the world files. Free

The steps that spend money (the teacher, training, export and scoring) are not built yet.
"""
from __future__ import annotations

import pathlib
import re
from importlib import resources
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from inscribe import __version__
from inscribe.config import Config, ConfigError, load_config, load_recipe, recipe_names
from inscribe.forecast import forecast
from inscribe.plan.build import build_plan, load_sheet, lore_coverage, plan_counts
from inscribe.plan.npcs import load_npcs
from inscribe.project import CONFIG_NAME, Project, ProjectError, find_project
from inscribe.runtime.template import format_by_name

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Train a game LoRA for SpellSpeak Glyph and Sigil from your lore and character sheets.")

ProjectOpt = Annotated[pathlib.Path, typer.Option("--project", "-p", help="The game project folder. Default: this folder or one above it.")]
SetOpt = Annotated[list[str], typer.Option("--set", help="Override a setting in lora.yaml, for example --set train.epochs=2. Repeatable.")]


# Why a lore item gets no scene of its own, in the planner's words and in plain ones.
LORE_SKIPS = {"lore_no_question": "the planner found no question to ask (often a fragment of the line before)",
              "lore_in_rules": "also a world rule, so every prompt already carries it",
              "lore_in_every_prompt": "already in every character's prompt",
              "lore_firewall": "touches a topic the scenes keep away from"}


def _version(value: bool) -> None:
    if value:
        typer.echo(f"inscribe {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version and exit.")] = False) -> None:
    pass


def _fail(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(1)


def _open(project: pathlib.Path, overrides: list[str]) -> tuple[Project, Config]:
    try:
        proj = find_project(project)
        return proj, load_config(proj.config_path, overrides)
    except (ProjectError, ConfigError) as exc:
        raise _fail(str(exc)) from None


def _load_inputs(proj: Project):
    if not proj.sheet_path.exists():
        raise _fail(f"{proj.sheet_path} not found.\nThe game sheet is what every step reads. Write it by hand (docs/game-sheet.md "
                    "describes each field), starting from the one `inscribe init` makes.")
    try:
        sheet = load_sheet(proj.sheet_path)
    except ValidationError as exc:
        lines = [f"{proj.sheet_path} has {exc.error_count()} problem(s):"]
        for e in exc.errors():
            lines.append(f"  {'.'.join(str(p) for p in e['loc']) or '(top level)'}: {e['msg']}")
        raise _fail("\n".join(lines)) from None
    except yaml.YAMLError as exc:
        raise _fail(f"{proj.sheet_path} is not valid YAML: {exc}") from None
    try:
        npcs = load_npcs(proj.npcs_path)
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        raise _fail(f"{proj.npcs_path}: {exc}") from None
    return sheet, npcs


def _notes(sheet, npcs, counts: dict) -> list[str]:
    notes = []
    if not npcs:
        notes.append("No npcs.yaml, so the checks will not ask run-time characters about the world.")
    if len(sheet.story) < 2:
        notes.append("The story has fewer than two stages, so there are no spoiler checks.")
    if not any(i.importance == "high" for i in sheet.lore_items()):
        notes.append("No lore item is marked `importance: high`. The scorecard's main line counts those.")
    quiet = [c.name for c in sheet.characters if not c.speech_examples]
    if quiet:
        notes.append(f"No speech examples for {', '.join(quiet)}. Their voice then comes only from the teacher's reading of `voice`.")
    not_asked = counts["lore_items"] - counts["lore_items_asked"]
    if not_asked:
        why = "; ".join(f"{v}: {LORE_SKIPS[k]}" for k, v in counts["skipped"].items() if k in LORE_SKIPS)
        notes.append(f"{not_asked} of {counts['lore_items']} lore items get no scene of their own" + (f": {why}." if why else ".")
                     + " `inscribe check --lore` lists them.")
    return notes


def _kinds(by_kind: dict[str, int]) -> str:
    """Scenes by kind, misheard twins counted with their kind."""
    kinds: dict[str, int] = {}
    for k, v in by_kind.items():
        name = k.removesuffix("+noisy")
        if name != "style":
            kinds[name] = kinds.get(name, 0) + v
    twins = sum(v for k, v in by_kind.items() if k.endswith("+noisy"))
    text = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
    return text + (f" ({twins} of them misheard twins)" if twins else "")


def _summary(sheet, npcs, counts: dict) -> list[str]:
    items = sheet.lore_items()
    secrets = sum(1 for c in sheet.characters for k in c.knows if k.secret is not None)
    priced = sum(1 for c in sheet.characters for o in c.offers if o.price)
    speech = sum(len(c.speech_examples) for c in sheet.characters)
    high = sum(1 for i in items if i.importance == "high")
    r = counts["recall"]
    lines = [f"{sheet.game_id} ({sheet.schema_version})",
             f"  {len(sheet.characters)} characters, {len(sheet.things)} places and things, {len(items)} lore items ({high} marked high), "
             f"{secrets} secrets, {priced} priced offers, {speech} speech examples, {len(sheet.story)} story stages",
             f"  {len(npcs)} run-time character{'' if len(npcs) == 1 else 's'} in npcs.yaml", "",
             f"Scenes for the teacher to answer: {counts['teacher_cases']}",
             "  " + _kinds(counts["by_kind"]),
             f"Training lines taken straight from the sheet's speech examples: {counts['style_cases']}",
             f"Held-out questions, never trained on: lore {r['lore']} on {r['lore_items']} items, spoilers {r['spoiler']}, "
             f"run-time characters {r['dynamic']}, prices {r['offers']}"]
    return lines


def _forecast_lines(fc: dict, config: Config) -> list[str]:
    lines = ["Forecast (estimates from one measured run of the example game; prices are the config's, not quotes)",
             f"  the teacher writes {fc['candidates']:,} lines; about {fc['accepted_expected']:,} pass the checks; {fc['style_examples']} more come from the sheet",
             f"  training mix: about {fc['training_examples_expected']:,} examples ({config.replay.to_one} to 1 replay, plus targeted replay)",
             f"  {'step':<18}{'GPU':<7}{'hours':>6}   about"]
    for s in fc["steps"]:
        usd = "not priced" if s["usd"] is None else f"${s['usd']:.2f}"
        lines.append(f"  {s['step']:<18}{s['gpu']:<7}{s['hours']:>6.2f}   {usd}")
    total = "not priced" if fc["gpu_usd"] is None else f"about ${fc['gpu_usd']:.2f}"
    lines.append(f"  GPU total {total}. The outside model that checks the lines is not counted. Budget cap: ${config.budget.max_usd:g}.")
    return lines


@app.command()
def init(name: Annotated[str, typer.Argument(help="The folder to create. Its name becomes the game's id.")],
         base: Annotated[str, typer.Option(help="The model to train on: " + ", ".join(recipe_names()))] = "glyph-2b") -> None:
    """Start a game project from the template: lore files to write, a small game sheet to replace, and lora.yaml."""
    if base not in recipe_names():
        raise _fail(f"unknown base model {base!r}; one of {', '.join(recipe_names())}")
    root = pathlib.Path(name)
    game_id = re.sub(r"[^a-z0-9]+", "_", root.name.lower()).strip("_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", game_id):
        raise _fail(f"{root.name!r} does not make a game id. Start the name with a letter and use letters, digits, - or _.")
    if root.exists() and any(root.iterdir()):
        raise _fail(f"{root} already exists and is not empty.")
    template = resources.files("inscribe.templates").joinpath("game")
    written = []

    def copy(node, dest: pathlib.Path) -> None:
        for child in sorted(node.iterdir(), key=lambda c: c.name):
            target = dest / (".gitignore" if child.name == "gitignore" else child.name)
            if child.is_dir():
                copy(child, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(child.read_text().replace("{{game_id}}", game_id).replace("{{base}}", base))
                written.append(target)

    copy(template, root)
    typer.echo(f"Made {root}/ with {len(written)} files.")
    typer.echo("Next:")
    typer.echo(f"  1. Write your world in {root}/lore/ (four files, prose).")
    typer.echo(f"  2. Put the same facts in {root}/sheet/game-sheet.yaml. It starts as a tiny example you replace.")
    typer.echo(f"  3. cd {root} && inscribe check")


@app.command()
def check(project: ProjectOpt = pathlib.Path("."), set_: SetOpt = [],  # noqa: B006
          lore: Annotated[bool, typer.Option("--lore", help="List the lore items that get no scene of their own, with the reason.")] = False) -> None:
    """Validate the game sheet, show what would be built, and forecast the run. Free: nothing is written or spent."""
    proj, config = _open(project, set_)
    sheet, npcs = _load_inputs(proj)
    if sheet.game_id != config.game:
        raise _fail(f"{CONFIG_NAME} says game: {config.game}, but the sheet's game_id is {sheet.game_id}. Make them the same.")
    recipes = [load_recipe(b) for b in config.base]
    fmt = format_by_name(recipes[0].chat_format)
    try:
        built = plan_counts(sheet, npcs, fmt, config.seed)
    except ValueError as exc:
        raise _fail(f"The sheet validates, but a scene could not be built from it:\n  {exc}") from None
    counts = built["counts"]
    if lore:
        rows = lore_coverage(sheet, config.seed)
        missed = [(text, why) for text, why in rows if why]
        typer.echo(f"{len(missed)} of {len(rows)} lore items get no scene of their own:")
        for text, why in missed:
            typer.echo(f"  - {text}\n      {LORE_SKIPS[why]}")
        return
    for line in _summary(sheet, npcs, counts):
        typer.echo(line)
    notes = _notes(sheet, npcs, counts)
    if notes:
        typer.echo("\nNotes")
        for n in notes:
            typer.echo(f"  - {n}")
    typer.echo("")
    for line in _forecast_lines(forecast(counts, config, recipes), config):
        typer.echo(line)
    typer.echo("\nThe sheet is valid. Next: inscribe plan")


@app.command()
def plan(project: ProjectOpt = pathlib.Path("."), set_: SetOpt = []) -> None:  # noqa: B006
    """Build the scenes, the held-out checks and the world files under build/plan/. Free and repeatable."""
    proj, config = _open(project, set_)
    sheet, npcs = _load_inputs(proj)
    if sheet.game_id != config.game:
        raise _fail(f"{CONFIG_NAME} says game: {config.game}, but the sheet's game_id is {sheet.game_id}. Make them the same.")
    fmt = format_by_name(load_recipe(config.base[0]).chat_format)
    try:
        info = build_plan(proj.sheet_path, proj.plan_dir, npcs, fmt, config.seed)
    except ValueError as exc:
        raise _fail(f"The sheet validates, but a scene could not be built from it:\n  {exc}") from None
    c = info["counts"]
    typer.echo(f"{info['game_id']}: {c['teacher_cases']} scenes for the teacher, {c['style_cases']} lines from the sheet, "
               f"{sum(v for k, v in c['recall'].items() if k in ('lore', 'spoiler', 'dynamic', 'offers'))} held-out questions, {len(c['worlds'])} world files.")
    typer.echo(f"Wrote {proj.plan_dir.relative_to(proj.root)}/ ({len(info['files'])} files). The same sheet always gives the same files.")


if __name__ == "__main__":
    app()
