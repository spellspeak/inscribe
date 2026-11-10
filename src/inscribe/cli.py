"""The `inscribe` command.

    inscribe init my-game     start a game project from the template
    inscribe extract          read the prose in lore/ into a game sheet for you to check. A dollar or two, and asks first
    inscribe check            validate the game sheet, show what would be built, forecast the run. Free
    inscribe plan             build the scenes, the held-out checks and the world files. Free
    inscribe data             the teacher writes each scene's line on the backend, then the lines are checked here. Spends money, and asks first
    inscribe mix              build the training mix on the backend: the game's rows and the replay. Cents
    inscribe train            train the LoRA on the backend. Spends money, and asks first
    inscribe export           make the GGUF file and answer the held-out questions with it. Spends money, and asks first
    inscribe eval             score the trained model and write the report card to out/report.md. Under a dollar, and asks first
    inscribe run              every step from plan to eval, in order. A step that finished with the same inputs is skipped
    inscribe pull             bring the GGUF file and the results to out/
    inscribe chat maren       talk to a character with the exported file, on this machine
    inscribe status           the steps this project has launched
    inscribe logs             a step's output
    inscribe stop             stop a running step
"""
from __future__ import annotations

import json
import os
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
from inscribe import runs
from inscribe.backends.base import read_result
from inscribe.plan.recall import prompt_rows
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


def _plan_error(exc: ValueError) -> str:
    if str(exc).startswith("dynamic NPC"):
        return (f"npcs.yaml does not fit the sheet:\n  {str(exc).removeprefix('dynamic NPC ')}\n"
                "Run-time characters are optional: fix the entry, or delete npcs.yaml.")
    return f"The sheet validates, but a scene could not be built from it:\n  {exc}"


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
    typer.echo(f"  2. cd {root} && inscribe extract   drafts sheet/game-sheet.yaml from your prose. Or write the sheet by hand: it starts as a tiny example.")
    typer.echo("  3. Read the sheet, then: inscribe check")


@app.command()
def extract(project: ProjectOpt = pathlib.Path("."), set_: SetOpt = [],  # noqa: B006
            max_usd: Annotated[float, typer.Option("--max-usd", help="Stop before a call that would take the total past this.")] = 3.0,
            effort: Annotated[str, typer.Option(help="How hard the model thinks: low, medium or high.")] = "medium",
            dry_run: Annotated[bool, typer.Option("--dry-run", help="Show the cost forecast and send nothing.")] = False,
            force: Annotated[bool, typer.Option("--force", help="Replace sheet/game-sheet.yaml even if you have edited it.")] = False,
            yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before spending.")] = False) -> None:
    """Read the prose in lore/ into a game sheet for you to check: three calls to the strong model, each answer validated."""
    from inscribe.sheets import extract as X

    proj, config = _open(project, set_)
    if effort not in ("low", "medium", "high"):
        raise _fail("--effort is one of low, medium, high")
    try:
        docs = X.read_documents(proj.lore_dir)
    except FileNotFoundError as exc:
        raise _fail(f"{exc}. The lore reader needs all four: {', '.join(X.DOCS)}.") from None
    f = X.forecast(docs, config.game)
    typer.echo(f"Reading {sum(f['document_chars'].values()):,} characters of lore for {config.game} with {config.llm.strong_model}.")
    typer.echo(f"  about ${f['usd_first_pass']:.2f} for one pass, up to ${f['usd_with_repairs']:.2f} if every call needs its repair pass (estimates). Cap: ${max_usd:.2f}.")
    if dry_run:
        return
    key = os.environ.get(config.llm.key_env)
    if not key:
        raise _fail(f"{config.llm.key_env} is not set in this shell. The lore reader needs it. It stays on this machine.")
    if not yes and not typer.confirm("Start it?"):
        raise typer.Exit(1)
    ex = X.Extractor(proj.lore_dir, config.game, X.paid_answer(key), max_usd=max_usd, effort=effort, model=config.llm.strong_model)
    try:
        sheet, info = ex.run()
    except (X.ExtractionError, X.BudgetError) as exc:
        paid = ", ".join(f"{c['stage']}{' repair' if c['repair'] else ''} ${c['usd']:.3f}" for c in ex.calls) or "none"
        raise _fail(f"The lore reader failed: {exc}\nCalls already paid: {paid} (${ex.spent:.3f}). Nothing was written.") from None

    target = proj.sheet_path
    if target.exists() and not force and target.read_text() != _template_sheet(config.game):
        target = target.with_name("game-sheet.extracted.yaml")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(X.sheet_yaml(sheet, info))
    report = proj.sheet_path.with_name("extract-report.md")
    report.write_text(X.report_md(info))
    raw = proj.build_dir / "extract" / "raw.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(X.raw_json(info, ex.raw))
    c = info["counts"]
    typer.echo(f"{c['characters']} characters, {c['things']} places and things, {c['lore']} lore items ({c['lore_by_importance']['high']} marked high, {c['exact']} exact), "
               f"{c['secrets']} secrets, {c['offers']} offers, {c['speech_examples']} speech examples, {c['story_stages']} story stages.")
    typer.echo(f"  {info['repairs']} repair pass(es), ${info['cost_usd']:.2f}.")
    typer.echo(f"Wrote {target.relative_to(proj.root)} and {report.relative_to(proj.root)}.")
    if target != proj.sheet_path:
        typer.echo(f"  {proj.sheet_path.relative_to(proj.root)} has your edits, so it was left alone. Compare the two, or pass --force to replace it.")
    flags = info["to_check"]
    typer.echo(f"Read the sheet before you train. The report lists {len(flags)} item(s) to look at first and what the sheet could not hold." if flags
               else "Read the sheet before you train. The report lists what the sheet could not hold.")
    if proj.npcs_path.is_file() and proj.npcs_path.read_text() == _template_text("npcs.yaml", config.game):
        typer.echo("npcs.yaml still holds the template's example character, who lives in the template's world. Write your own run-time "
                   "characters there, or delete the file.")
    typer.echo("Next: inscribe check")


def _template_text(name: str, game_id: str) -> str:
    """A file as `inscribe init` writes it, so an untouched one can be told from one a person has edited."""
    return resources.files("inscribe.templates").joinpath("game", *name.split("/")).read_text().replace("{{game_id}}", game_id)


def _template_sheet(game_id: str) -> str:
    return _template_text("sheet/game-sheet.yaml", game_id)


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
        raise _fail(_plan_error(exc)) from None
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
        raise _fail(_plan_error(exc)) from None
    c = info["counts"]
    typer.echo(f"{info['game_id']}: {c['teacher_cases']} scenes for the teacher, {c['style_cases']} lines from the sheet, "
               f"{sum(v for k, v in c['recall'].items() if k in ('lore', 'spoiler', 'dynamic', 'offers'))} held-out questions, {len(c['worlds'])} world files.")
    typer.echo(f"Wrote {proj.plan_dir.relative_to(proj.root)}/ ({len(info['files'])} files). The same sheet always gives the same files.")


def _backend(config: Config, proj: Project):
    backend = runs.make_backend(config, config.game)
    problems = backend.check()
    if problems:
        raise _fail("\n".join(problems))
    return backend


@app.command()
def data(project: ProjectOpt = pathlib.Path("."), set_: SetOpt = [],  # noqa: B006
         limit: Annotated[int | None, typer.Option(help="Write lines for only this many scenes: a cheap rehearsal.")] = None,
         skip_teacher: Annotated[bool, typer.Option("--skip-teacher", help="Check the lines already written, without running the teacher again.")] = False,
         yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before spending.")] = False) -> None:
    """Make the game's own training lines: a teacher model writes several lines for every scene on the backend, then
    code checks and a judge keep the best line of each scene, or none."""
    import yaml as _yaml

    from inscribe.data import teacher as T
    from inscribe.data import validate as V
    from inscribe.judges import fast as F
    from inscribe.plan.build import source_fingerprint
    from inscribe.schemas.world import World

    proj, config = _open(project, set_)
    if not (proj.plan_dir / "scenarios.jsonl").is_file():
        raise _fail("No build/plan/: run `inscribe plan` first.")
    key = os.environ.get(config.llm.key_env)
    if not key:
        raise _fail(f"{config.llm.key_env} is not set in this shell. The judge that checks the lines needs it. It stays on this machine.")
    scenes = sum(1 for line in (proj.plan_dir / "scenarios.jsonl").read_text().splitlines() if line.strip())
    scenes = min(scenes, limit) if limit else scenes
    out_dir = proj.build_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = out_dir / "candidates.jsonl"

    if not skip_teacher:
        backend = _backend(config, proj)
        store = backend.store()
        spec = runs.teacher_spec()
        hours = runs.teacher_hours(scenes, config.teacher.candidates)
        price = backend.hourly_usd(spec)
        typer.echo(f"Teacher for {config.game}: {scenes:,} scenes, {config.teacher.candidates} lines each, one {spec.gpu}.")
        typer.echo(f"  about {hours * 60:.0f} minutes" + (f", about ${hours * price:.2f} at ${price:g} an hour (an estimate, not a quote)" if price is not None else ""))
        typer.echo("  then the judge checks the lines that are not plain lore, from this machine: about $0.60 for a thousand lines it reads.")
        if not yes and not typer.confirm("Start it?"):
            raise typer.Exit(1)
        files = [(p, f"plan/worlds/{p.name}") for p in sorted((proj.plan_dir / "worlds").glob("*.yaml"))] + [(proj.plan_dir / "scenarios.jsonl", "plan/scenarios.jsonl")]
        store.put_many(files) if hasattr(store, "put_many") else [store.put(p, r) for p, r in files]
        fetch = runs.fetch_spec()
        fetched = read_result(store, fetch)
        if not (fetched and fetched.get("ok") and fetched["result"].get("revision") == T.TEACHER_REVISION):
            typer.echo("Fetching the teacher model into the store's cache, on a CPU. This happens once.")
            store.write_text(f"{fetch.run_dir}/job.json", json.dumps({"repo": T.TEACHER_REPO, "revision": T.TEACHER_REVISION}) + "\n")
            handle = backend.launch(fetch)
            runs.save_handle(proj, handle)
            result = runs.follow(backend, handle, typer.echo)
            if not result or not result.get("ok"):
                _report(handle, result)
                raise typer.Exit(1)
        store.write_text(f"{spec.run_dir}/job.json", json.dumps(runs.teacher_job(config, limit), indent=1) + "\n")
        handle = backend.launch(spec)
        runs.save_handle(proj, handle)
        typer.echo(f"Started teacher-game ({handle.id}).")
        result = runs.follow(backend, handle, typer.echo)
        _report(handle, result)
        if not result or not result.get("ok"):
            raise typer.Exit(1)
        store.get(runs.CANDIDATES_AT, candidates_path)
    elif not candidates_path.is_file():
        raise _fail("No build/data/candidates.jsonl to check. Run `inscribe data` without --skip-teacher.")

    rows = [json.loads(line) for line in candidates_path.read_text().splitlines() if line.strip()]
    worlds = {p.stem: World.model_validate(_yaml.safe_load(p.read_text())) for p in sorted((proj.plan_dir / "worlds").glob("*.yaml"))}
    judge = F.FastJudge(api_key=key, model=config.llm.fast_model)
    typer.echo(f"Checking {len(rows):,} lines: code checks, then the judge ({judge.name}).")
    try:
        out = V.validate(rows, V.world_resolver(worlds), judge, source_fingerprint(), F.ACCEPT_BELOW, V.CODE_ONLY)
    except F.JudgeError as exc:
        raise _fail(f"The judge refused the request: {exc}") from None
    (out_dir / "accepted.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out["accepted"]))
    (out_dir / "rejected.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out["rejected"]))
    report = {**V.summary(rows, out), "judge": {"model": judge.name, "definitions": F.DEFINITIONS_VERSION, "accept_below": F.ACCEPT_BELOW,
                                                  "input_tokens": judge.input_tokens, "usd": round(F.cost_usd(judge.input_tokens), 2)}}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    typer.echo(f"Kept {report['accepted']:,} lines for {report['scenes']:,} scenes ({report['candidates']:,} written).")
    typer.echo("  dropped: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in report["reasons"].items()))
    typer.echo(f"  the judge read {judge.input_tokens:,} tokens, about ${report['judge']['usd']:.2f}.")
    typer.echo("Wrote build/data/accepted.jsonl, rejected.jsonl and report.json. Next: inscribe mix")


@app.command()
def mix(project: ProjectOpt = pathlib.Path("."), set_: SetOpt = [],  # noqa: B006
        yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before spending.")] = False) -> None:
    """Build the training mix on the backend: the game's own rows, and general rows replayed so the model keeps its skills."""
    proj, config = _open(project, set_)
    recipe = load_recipe(config.base[0])
    if not config.replay.source:
        raise _fail("No replay pack: set replay.source in lora.yaml (store:<folder> or hf://<dataset>). The pack is not published yet.")
    if not (proj.plan_dir / "worlds").is_dir():
        raise _fail("No build/plan/: run `inscribe plan` first.")
    backend = _backend(config, proj)
    store = backend.store()
    local = proj.build_dir / "data" / "accepted.jsonl"
    accepted = local.is_file() or bool(store.list(runs.ACCEPTED_AT))
    job = runs.mix_job(config, recipe, accepted)
    spec = runs.mix_spec()
    typer.echo(f"Mix for {config.game}: {'the accepted lines and ' if accepted else ''}the sheet's speech examples, replay {job['to_one']} to 1, "
               f"plus {sum(job['targeted'].values()):,} targeted rows. A few minutes on a CPU, a few cents.")
    if not accepted:
        typer.echo("  No accepted teacher lines yet (the step that writes them is not built), so the game's own rows are only its speech examples.")
    if not yes and not typer.confirm("Start it?"):
        raise typer.Exit(1)
    if local.is_file():
        store.put(local, runs.ACCEPTED_AT)
    files = [(p, f"plan/worlds/{p.name}") for p in sorted((proj.plan_dir / "worlds").glob("*.yaml"))] + [(proj.plan_dir / "style-examples.jsonl", "plan/style-examples.jsonl")]
    store.put_many(files) if hasattr(store, "put_many") else [store.put(p, r) for p, r in files]
    store.write_text(f"{spec.run_dir}/job.json", json.dumps(job, indent=1) + "\n")
    handle = backend.launch(spec)
    runs.save_handle(proj, handle)
    result = runs.follow(backend, handle, typer.echo)
    _report(handle, result)
    if not result or not result.get("ok"):
        raise typer.Exit(1)


@app.command()
def train(project: ProjectOpt = pathlib.Path("."), set_: SetOpt = [],  # noqa: B006
          base: Annotated[str | None, typer.Option(help="Which base model to train. Default: the first in lora.yaml.")] = None,
          max_steps: Annotated[int | None, typer.Option("--max-steps", help="Stop after this many steps and save nothing: a cheap check of the base, the data and the first losses.")] = None,
          keep: Annotated[bool, typer.Option("--keep", help="With --max-steps: save, merge and answer again all the same. A rehearsal of the whole path.")] = False,
          prompts: Annotated[pathlib.Path | None, typer.Option(help="Questions to answer before and after training ({id, prompt, max_new} per line). Default: the retention questions and the plan's held-out questions, which is what `inscribe eval` scores.")] = None,
          yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before spending.")] = False,
          detach: Annotated[bool, typer.Option("--detach", help="Start the step and return. See it later with `inscribe status` and `inscribe logs`.")] = False) -> None:
    """Train the LoRA on the backend: fetch the base model, answer the held-out questions, train, merge, answer again."""
    proj, config = _open(project, set_)
    name = base or config.base[0]
    if name not in config.base:
        raise _fail(f"{name} is not in lora.yaml's base ({', '.join(config.base)}). Add it there, or pass --set base={name}.")
    recipe = load_recipe(name)
    backend = _backend(config, proj)
    store = backend.store()

    local = sorted((proj.build_dir / "data" / "mix").glob("*.jsonl"))
    on_store = [f for f in store.list(runs.DATA_AT) if f.endswith(".jsonl")]
    if local:
        examples = runs.count_rows(local)
    elif on_store:
        examples = None
    else:
        raise _fail("No training rows: nothing in build/data/mix/ here and nothing on the backend's store. The step that makes them is not built yet.")

    if prompts is not None:
        if not prompts.is_file():
            raise _fail(f"{prompts} not found.")
        rows = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
        prompts_at = f"prompts/{prompts.name}"
    elif (proj.plan_dir / "recall.json").is_file():
        rows = runs.held_out_rows(json.loads((proj.plan_dir / "recall.json").read_text()), recipe.chat_format)
        prompts_at = "prompts/held-out.jsonl"
    else:
        raise _fail("No held-out questions: run `inscribe plan` first, or pass --prompts.")

    run_id = f"{name}-check" if max_steps else name
    job = runs.train_job(config, recipe, prompts_at, max_steps, keep)
    spec = runs.train_spec(recipe, run_id, max_steps)
    hours = runs.train_hours(recipe, job, examples or 5000, len(rows))
    price = backend.hourly_usd(spec)
    cost = None if price is None else hours * price
    what = (f"{max_steps} steps as a " + ("rehearsal, everything saved" if job["keep"] else "check, nothing saved")) if max_steps else f"{job['lora']['epochs']} epochs"
    typer.echo(f"Train {recipe.title} for {config.game}: {what}.")
    typer.echo(f"  {f'{examples:,} training rows' if examples is not None else 'the training rows already on the store'}, {len(rows)} questions, one {spec.gpu}")
    typer.echo(f"  about {hours * 60:.0f} minutes" + (f", about ${cost:.2f} at ${price:g} an hour (an estimate, not a quote)" if cost is not None else ", cost not known"))
    if cost is not None and cost > config.budget.max_usd:
        raise _fail(f"That is over the budget in lora.yaml (${config.budget.max_usd:g}). Raise budget.max_usd to go ahead.")
    if not yes and not typer.confirm("Start it?"):
        raise typer.Exit(1)

    if local:
        store.put_many([(p, f"{runs.DATA_AT}/{p.name}") for p in local]) if hasattr(store, "put_many") else [store.put(p, f"{runs.DATA_AT}/{p.name}") for p in local]
    store.write_text(prompts_at, runs.prompt_file(rows))
    store.write_text(f"{spec.run_dir}/job.json", json.dumps(job, indent=1) + "\n")
    handle = backend.launch(spec)
    runs.save_handle(proj, handle)
    typer.echo(f"Started train-{run_id} ({handle.id}).")
    if detach:
        typer.echo("It keeps running if this terminal closes. See it with `inscribe status` and `inscribe logs`.")
        return
    result = runs.follow(backend, handle, typer.echo)
    _report(handle, result)
    if not result or not result.get("ok"):
        raise typer.Exit(1)


@app.command()
def export(project: ProjectOpt = pathlib.Path("."), set_: SetOpt = [],  # noqa: B006
           base: Annotated[str | None, typer.Option(help="Which base model's training run to export. Default: the first in lora.yaml.")] = None,
           run: Annotated[str | None, typer.Option(help="The training run to export, as `inscribe status` names it without `train-`. Default: the base's whole run.")] = None,
           prompts: Annotated[pathlib.Path | None, typer.Option(help="Questions to answer with each file ({id, prompt, max_new} per line). Default: the plan's held-out questions.")] = None,
           yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before spending.")] = False,
           detach: Annotated[bool, typer.Option("--detach", help="Start the step and return.")] = False) -> None:
    """Make the GGUF file a game ships: convert the trained model, quantize it with the game's own prompts in mind, and answer the held-out questions with the result."""
    proj, config = _open(project, set_)
    name = base or config.base[0]
    if name not in config.base:
        raise _fail(f"{name} is not in lora.yaml's base ({', '.join(config.base)}). Add it there, or pass --set base={name}.")
    recipe = load_recipe(name)
    backend = _backend(config, proj)
    store = backend.store()
    run_id = run or name
    trained = read_result(store, type("Run", (), {"run_dir": f"train/{run_id}"}))
    if not trained or not trained.get("ok") or not trained["result"].get("merged"):
        raise _fail(f"No finished training run {run_id} with a merged model on the store. Run `inscribe train` first.")
    if not (proj.plan_dir / "imatrix.txt").is_file():
        raise _fail("No build/plan/imatrix.txt: run `inscribe plan` first.")
    if prompts is not None:
        if not prompts.is_file():
            raise _fail(f"{prompts} not found.")
        rows = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
        prompts_at = f"prompts/{prompts.name}"
    else:
        rows = prompt_rows(json.loads((proj.plan_dir / "recall.json").read_text()), recipe.chat_format)
        prompts_at = "prompts/recall.jsonl"
    job = runs.export_job(config, recipe, run_id, "plan/imatrix.txt", prompts_at)
    spec = runs.export_spec(recipe, run_id)
    hours = runs.export_hours(job, len(rows))
    price = backend.hourly_usd(spec)
    cost = None if price is None else hours * price
    typer.echo(f"Export {recipe.title} for {config.game}: {', '.join(job['quant'])}, embeddings at {job['embeddings']}.")
    typer.echo(f"  {len(rows)} questions per file, one {spec.gpu}")
    typer.echo(f"  about {hours * 60:.0f} minutes" + (f", about ${cost:.2f} at ${price:g} an hour (an estimate, not a quote)" if cost is not None else ", cost not known"))
    if cost is not None and cost > config.budget.max_usd:
        raise _fail(f"That is over the budget in lora.yaml (${config.budget.max_usd:g}). Raise budget.max_usd to go ahead.")
    if not yes and not typer.confirm("Start it?"):
        raise typer.Exit(1)
    store.put(proj.plan_dir / "imatrix.txt", "plan/imatrix.txt")
    store.write_text(prompts_at, runs.prompt_file(rows))
    store.write_text(f"{spec.run_dir}/job.json", json.dumps(job, indent=1) + "\n")
    handle = backend.launch(spec)
    runs.save_handle(proj, handle)
    typer.echo(f"Started export-{run_id} ({handle.id}).")
    if detach:
        typer.echo("It keeps running if this terminal closes. See it with `inscribe status` and `inscribe logs`.")
        return
    result = runs.follow(backend, handle, typer.echo)
    _report(handle, result)
    if not result or not result.get("ok"):
        raise typer.Exit(1)


def _rows(path: pathlib.Path) -> list[dict] | None:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else None


@app.command(name="eval")
def eval_(project: ProjectOpt = pathlib.Path("."), set_: SetOpt = [],  # noqa: B006
          base: Annotated[str | None, typer.Option(help="Which base model's run to score. Default: every base in lora.yaml.")] = None,
          no_judge: Annotated[bool, typer.Option("--no-judge", help="Skip the strong model's spoiler and voice reads. Free, and the card says they were not judged.")] = False,
          offline: Annotated[bool, typer.Option("--offline", help="Score the answers already in out/, without asking the backend for newer ones.")] = False,
          max_usd: Annotated[float, typer.Option("--max-usd", help="A cap for the judge, per base model.")] = 2.0,
          yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before spending.")] = False) -> None:
    """Score the trained model and write the report card: lore recall, prices, run-time characters and what it kept, by
    key words; later story told early and the character's voice, by the strong model."""
    from inscribe.evaluate import scorecard, spoilers, style

    proj, config = _open(project, set_)
    sheet, _ = _load_inputs(proj)
    if not (proj.plan_dir / "recall.json").is_file():
        raise _fail("No build/plan/recall.json: run `inscribe plan` first. It holds the held-out questions and their keys.")
    bundle = json.loads((proj.plan_dir / "recall.json").read_text())
    names = [base] if base else list(config.base)
    for name in names:
        if name not in config.base:
            raise _fail(f"{name} is not in lora.yaml's base ({', '.join(config.base)}).")
    key = None if no_judge else os.environ.get(config.llm.key_env)
    if not no_judge and not key:
        raise _fail(f"{config.llm.key_env} is not set in this shell. The spoiler and voice reads need it. Or pass --no-judge.")
    store = None if offline else _backend(config, proj).store()

    reports = []
    for name in names:
        recipe = load_recipe(name)
        fmt = format_by_name(recipe.chat_format)
        if bundle.get("format") != fmt.name:
            raise _fail(f"build/plan/ was made for chat format {bundle.get('format')}, not {fmt.name}. Run `inscribe plan` again.")
        folder = proj.out_dir / name
        if store is not None:
            for remote, local in runs.small_files(store, name):
                store.get(remote, proj.out_dir / local)
        before, after = _rows(folder / "train-answers-before.jsonl"), _rows(folder / "train-answers-after.jsonl")
        shipped = {p.stem.removeprefix("export-answers-"): _rows(p) for p in sorted(folder.glob("export-answers-*.jsonl"))}
        if not after and not shipped:
            typer.echo(f"{recipe.title}: no answers to score. Run `inscribe train` and `inscribe export` first.")
            continue
        judged: dict = {}
        if key:
            where = next(iter(shipped), None)
            replies = spoilers.spoken(shipped[where] if where else after)
            pairs = style.style_pairs(bundle, sheet, before or [], after or [], model=config.llm.strong_model)
            asks = spoilers.spoiler_requests(bundle, sheet, replies, config.llm.strong_model)
            from inscribe.judges import strong

            cost = strong.forecast_usd(asks, spoilers.OUTPUT_FORECAST_TOKENS) + strong.forecast_usd([line for line, _ in pairs], style.OUTPUT_FORECAST_TOKENS)
            typer.echo(f"{recipe.title}: the judge ({config.llm.strong_model}) reads {len(asks)} replies for spoilers and {len(pairs)} pairs for voice, about ${cost:.2f} (an estimate).")
            if cost > max_usd:
                raise _fail(f"That is over --max-usd {max_usd:g}.")
            if not yes and not typer.confirm("Start it?"):
                raise typer.Exit(1)
            if asks:
                judged["spoilers"] = {"read_at": where or "after", **spoilers.judge(bundle, sheet, replies, key, max_usd, config.llm.strong_model)}
            if pairs:
                judged["style"] = style.judge(bundle, sheet, before, after, key, max_usd, model=config.llm.strong_model)
        rep = scorecard.build(config.game, name, recipe.title, bundle, sheet, fmt, before, after, shipped, judged)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
        reports.append(rep)
        typer.echo(f"{recipe.title}, read at {rep['read_at']}:")
        for c in rep["scorecard"]:
            typer.echo(f"  {'not judged' if c['met'] is None else 'met' if c['met'] else 'NOT MET':<11}{c['item']}: {c['value']}")
        spent = sum(j.get("cost_usd", 0) for j in judged.values())
        if judged:
            typer.echo(f"  the judge cost ${spent:.2f}.")
    if not reports:
        raise typer.Exit(1)
    # one page for every base model that has been scored, this time or before
    done = {r["base"]: r for r in reports}
    for name in config.base:
        earlier = proj.out_dir / name / "report.json"
        if name not in done and earlier.is_file():
            done[name] = json.loads(earlier.read_text())
    page = proj.out_dir / "report.md"
    page.write_text(scorecard.report_md([done[n] for n in config.base if n in done]))
    typer.echo(f"Wrote {page.relative_to(proj.root)}. It says what was not checked too.")


RUN_STEPS = ("plan", "data", "mix", "train", "export", "eval")


def _digest(*parts) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _file_digest(path: pathlib.Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _still_running(proj: Project, backend, stage: str, run_id: str, job: dict):
    """The handle of a step launched earlier with this same job and still running, or None."""
    handle = runs.find_handle(proj, f"{stage}-{run_id}")
    if handle is None or backend.status(handle) != "running":
        return None
    on_store = backend.store().read_text(f"{stage}/{run_id}/job.json")
    return handle if on_store is not None and json.loads(on_store) == job else None


def _adopt(proj: Project, backend, stage: str, run_id: str, job: dict) -> bool:
    """A step launched earlier with this same job and still running is followed, not launched again."""
    handle = _still_running(proj, backend, stage, run_id, job)
    if handle is None:
        return False
    typer.echo(f"{stage}-{run_id} is already running from an earlier launch. Following it.")
    result = runs.follow(backend, handle, typer.echo)
    _report(handle, result)
    if not result or not result.get("ok"):
        raise typer.Exit(1)
    return True


@app.command()
def run(project: ProjectOpt = pathlib.Path("."), set_: SetOpt = [],  # noqa: B006
        from_: Annotated[str | None, typer.Option("--from", help="Start at this step and take the earlier ones as done: " + ", ".join(RUN_STEPS))] = None,
        to: Annotated[str | None, typer.Option("--to", help="Stop after this step.")] = None,
        again: Annotated[bool, typer.Option("--again", help="Run every step, even one that finished with the same inputs.")] = False,
        no_judge: Annotated[bool, typer.Option("--no-judge", help="Score without the strong model's spoiler and voice reads.")] = False,
        yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before spending.")] = False) -> None:
    """Run every step in order: plan, data, mix, then train, export and eval for each base model in lora.yaml. A step that
    already finished with the same inputs is skipped, so the command can be run again after a failure or a change."""
    from inscribe.data import teacher as T
    from inscribe.judges import fast as F

    for name, value in (("--from", from_), ("--to", to)):
        if value is not None and value not in RUN_STEPS:
            raise _fail(f"{name} is one of: {', '.join(RUN_STEPS)}")
    first, last = RUN_STEPS.index(from_ or "plan"), RUN_STEPS.index(to or "eval")
    if first > last:
        raise _fail("--from comes after --to.")
    proj, config = _open(project, set_)
    sheet, npcs = _load_inputs(proj)
    recipes = [load_recipe(b) for b in config.base]
    if not config.replay.source and first <= RUN_STEPS.index("mix") <= last:
        raise _fail("No replay pack: set replay.source in lora.yaml (store:<folder> or hf://<dataset>). The pack is not published yet.")
    # The plan is free and always the same for the same sheet, so it is built first and its hash starts the chain.
    plan(project, set_)
    plan_fp = _file_digest(proj.plan_dir / "plan.yaml")
    if last == 0:
        return
    backend = _backend(config, proj)
    store = backend.store()
    data_dir, accepted = proj.build_dir / "data", proj.build_dir / "data" / "accepted.jsonl"
    bundle = json.loads((proj.plan_dir / "recall.json").read_text())
    jobs = {r.name: (runs.train_job(config, r, "prompts/held-out.jsonl", None), runs.export_job(config, r, r.name, "plan/imatrix.txt", "prompts/recall.jsonl"))
            for r in recipes}

    def hashes() -> dict[str, str]:
        """The hash of what each step reads, from the files as they are now. A later step's hash includes the earlier one's."""
        fp = {"data": _digest("data", plan_fp, config.teacher.candidates, T.TEACHER_REVISION, T.PROMPT_VERSION, config.llm.fast_model, F.DEFINITIONS_VERSION)}
        fp["mix"] = _digest("mix", plan_fp, _file_digest(accepted), runs.mix_job(config, recipes[0], True))
        for r in recipes:
            fp[f"train {r.name}"] = _digest("train", fp["mix"], jobs[r.name][0], runs.held_out_rows(bundle, r.chat_format))
            fp[f"export {r.name}"] = _digest("export", fp[f"train {r.name}"], jobs[r.name][1])
            fp[f"eval {r.name}"] = _digest("eval", fp[f"export {r.name}"], no_judge, config.llm.strong_model)
        return fp

    def stamped(path: pathlib.Path, fp: str) -> bool:
        return path.is_file() and path.read_text().strip() == fp

    def on_store(run_dir: str, fp: str) -> bool:
        result = store.read_text(f"{run_dir}/result.json")
        return bool(result) and bool(json.loads(result).get("ok")) and (store.read_text(f"{run_dir}/inputs.sha256") or "").strip() == fp

    fp = hashes()
    done = {"data": stamped(data_dir / "inputs.sha256", fp["data"]) and accepted.is_file(), "mix": on_store("mix/game", fp["mix"])}
    for r in recipes:
        done[f"train {r.name}"] = on_store(f"train/{r.name}", fp[f"train {r.name}"])
        done[f"export {r.name}"] = on_store(f"export/{r.name}", fp[f"export {r.name}"])
        done[f"eval {r.name}"] = stamped(proj.out_dir / r.name / "inputs.sha256", fp[f"eval {r.name}"]) and (proj.out_dir / r.name / "report.json").is_file()

    # Which steps run. Once one runs, every step after it runs too: its inputs are about to change.
    todo: dict[str, bool] = {}
    stale = again
    for label in fp:
        if first <= RUN_STEPS.index(label.split()[0]) <= last:
            todo[label] = stale or not done[label]
            stale = stale or todo[label]

    fc = forecast(plan_counts(sheet, npcs, format_by_name(recipes[0].chat_format), config.seed)["counts"], config, recipes)
    usd = {step["step"]: step["usd"] for step in fc["steps"]}
    typer.echo("")
    total, priced = 0.0, True
    for label, runs_now in todo.items():
        kind = label.split()[0]
        cost = usd.get("teacher" if kind == "data" else label)
        if not runs_now:
            note = "done with these inputs, skipped"
        elif kind in ("train", "export") and _still_running(proj, backend, kind, label.split()[1], jobs[label.split()[1]][kind == "export"]):
            note = "already running from an earlier launch, followed"
        elif kind == "mix":
            note = "a few cents"
        elif kind == "eval":
            note = "free" if no_judge else "under $1 for the judge"
        else:
            note = "cost not known" if cost is None else f"about ${cost:.2f}" + (", then the judge, about $0.60 a thousand lines it reads" if kind == "data" else "")
            priced = priced and cost is not None
            total += cost or 0.0
        typer.echo(f"  {label:<18}{note}")
    if not any(todo.values()):
        typer.echo("Everything is done with these inputs. `inscribe run --again` runs it all the same.")
        return
    typer.echo(f"GPU time for the steps to run: {'about $' + format(total, '.2f') if priced else 'not known'} (an estimate, not a quote). Budget cap: ${config.budget.max_usd:g}.")
    if total > config.budget.max_usd:
        raise _fail(f"That is over the budget in lora.yaml (${config.budget.max_usd:g}). Raise budget.max_usd to go ahead.")
    needs_key = todo.get("data") or (not no_judge and any(v for k, v in todo.items() if k.startswith("eval")))
    if needs_key and not os.environ.get(config.llm.key_env):
        raise _fail(f"{config.llm.key_env} is not set in this shell. The judge that checks the teacher's lines and reads the trained model needs it.")
    if not yes and not typer.confirm("Start?"):
        raise typer.Exit(1)

    # Each step is stamped with the hash of what it read once it has finished, so the next run can skip it.
    if todo.get("data"):
        data(project, set_, None, False, True)
        fp = hashes()
        (data_dir / "inputs.sha256").write_text(fp["data"] + "\n")
    if todo.get("mix"):
        mix(project, set_, True)
        store.write_text("mix/game/inputs.sha256", fp["mix"] + "\n")
    for r in recipes:
        train_job, export_job = jobs[r.name]
        if todo.get(f"train {r.name}"):
            if not _adopt(proj, backend, "train", r.name, train_job):
                train(project, set_, r.name, None, False, None, True, False)
            store.write_text(f"train/{r.name}/inputs.sha256", fp[f"train {r.name}"] + "\n")
        if todo.get(f"export {r.name}"):
            if not _adopt(proj, backend, "export", r.name, export_job):
                export(project, set_, r.name, None, None, True, False)
            store.write_text(f"export/{r.name}/inputs.sha256", fp[f"export {r.name}"] + "\n")
        if todo.get(f"eval {r.name}"):
            eval_(project, set_, r.name, no_judge, False, 2.0, True)
            (proj.out_dir / r.name / "inputs.sha256").write_text(fp[f"eval {r.name}"] + "\n")
    if last == RUN_STEPS.index("eval"):
        typer.echo("Done. The report card is out/report.md. `inscribe pull` brings the GGUF file, and `inscribe chat` talks to a character with it.")


@app.command()
def chat(character: Annotated[str | None, typer.Argument(help="Who to talk to: an id, a name or the start of one. Leave it out to list them.")] = None,
         project: ProjectOpt = pathlib.Path("."), set_: SetOpt = [],  # noqa: B006
         base: Annotated[str | None, typer.Option(help="Which base model's file. Default: the first in lora.yaml.")] = None,
         quant: Annotated[str | None, typer.Option(help="Which exported file, for example Q5_K_M, when out/ holds several.")] = None,
         stage: Annotated[int | None, typer.Option(help="The story stage the character stands at, counted from 1. Default: before the story starts.")] = None,
         say: Annotated[list[str], typer.Option("--say", help="Say this line and print the reply, without the prompt. Repeatable: the lines form one conversation.")] = [],  # noqa: B006
         temperature: Annotated[float, typer.Option(help="0 is greedy, as the checks are.")] = 0.0,
         server: Annotated[str, typer.Option(help="Path to llama-server.")] = "llama-server",
         gpu_layers: Annotated[str, typer.Option("--gpu-layers", help="llama-server's -ngl: all, or 0 for CPU only.")] = "all") -> None:
    """Talk to one of the game's characters with the exported file, on this machine. Needs llama-server from llama.cpp."""
    from inscribe import chat as C

    proj, config = _open(project, set_)
    sheet, npcs = _load_inputs(proj)
    if character is None:
        typer.echo("Characters:")
        for c in sheet.characters:
            typer.echo(f"  {c.name}  ({c.id})" + (f"  {c.role}" if c.role else ""))
        for n in npcs:
            typer.echo(f"  {n.name}  ({n.id})  made up at run time")
        return
    name = base or config.base[0]
    recipe = load_recipe(name)
    if stage is not None and not 1 <= stage <= len(sheet.story):
        raise _fail(f"--stage is 1 to {len(sheet.story)} for this game.")
    try:
        actor = C.who(sheet, npcs, character)
        gguf = C.find_gguf(proj.out_dir, config.game, recipe.name, quant)
        talk = C.Conversation(sheet, npcs, actor, format_by_name(recipe.chat_format), None if stage is None else stage - 1)
        typer.echo(f"Loading {gguf.name} ...", err=True)
        with C.Server(gguf, server, gpu_layers) as srv:
            if say:
                for line in say:
                    typer.echo(f"Player: {line}\n{talk.name}: {talk.say(srv, line, temperature)}")
                return
            typer.echo(f"Talking to {talk.name}. An empty line or Ctrl-D ends it.", err=True)
            while True:
                try:
                    line = input("You: ").strip()
                except EOFError:
                    break
                if not line:
                    break
                typer.echo(f"{talk.name}: {talk.say(srv, line, temperature)}")
    except C.ChatError as exc:
        raise _fail(str(exc)) from None


def _report(handle, result: dict | None) -> None:
    name = f"{handle.stage}-{handle.run_id}"
    if result is None:
        typer.echo(f"{name}: ended without a result. `inscribe logs {name}` shows what it printed.")
    elif not result.get("ok"):
        typer.echo(f"{name}: failed after {result.get('seconds')} s: {result.get('error')}")
    elif handle.stage == "train":
        r = result["result"]
        typer.echo(f"{name}: {r['steps']} of {r['total_steps']} steps on {r['examples']} examples in {r['train_s']} s, peak {r['peak_gb']} GB on {r['gpu']}.")
        typer.echo(f"  first losses {r['loss_first']}" + ("" if r["stopped_early"] else f", last {r['loss_last']}"))
    elif handle.stage == "teacher":
        r = result["result"]
        typer.echo(f"{name}: {r['candidates']:,} lines for {r['scenes']:,} scenes. Loading took {r['load_s']:.0f} s, writing {r['generate_s']:.0f} s, on {r['gpu']}.")
    elif handle.stage == "mix":
        r = result["result"]
        typer.echo(f"{name}: {r['examples']:,} rows: {r['parts']['game']:,} of the game's, {r['parts']['replay']:,} replayed, {sum(r['parts']['targeted'].values()):,} targeted.")
    elif handle.stage == "export":
        for quant, f in result["result"]["files"].items():
            typer.echo(f"{name}: {f['file'].rsplit('/', 1)[-1]}, {f['bytes'] / 1e9:.2f} GB" + (f", answered on the GPU: {f['on_gpu']}" if "on_gpu" in f else ""))
    else:
        typer.echo(f"{name}: finished in {result.get('seconds')} s.")


@app.command()
def pull(project: ProjectOpt = pathlib.Path("."), set_: SetOpt = [],  # noqa: B006
         base: Annotated[str | None, typer.Option(help="Which base model's files. Default: every base in lora.yaml.")] = None,
         adapter: Annotated[bool, typer.Option("--adapter", help="Bring the LoRA adapter too.")] = False) -> None:
    """Bring the GGUF file, each step's result and its answers from the backend's store to out/."""
    proj, config = _open(project, set_)
    store = _backend(config, proj).store()
    got = 0
    for name in ([base] if base else config.base):
        for remote, local in runs.pull_list(store, name, adapter):
            target = proj.out_dir / local
            store.get(remote, target)
            typer.echo(f"  {target.relative_to(proj.root)}  ({target.stat().st_size / 1e6:,.1f} MB)")
            got += 1
    typer.echo(f"Brought {got} file(s) to out/." if got else "Nothing to bring yet: no finished export or training run on the store.")


@app.command()
def status(project: ProjectOpt = pathlib.Path("."), set_: SetOpt = []) -> None:  # noqa: B006
    """The steps this project has launched, and where each one stands."""
    proj, config = _open(project, set_)
    handles = runs.load_handles(proj)
    if not handles:
        typer.echo("No step has been launched from this project.")
        return
    backend = _backend(config, proj)
    for h in handles:
        state = backend.status(h)
        typer.echo(f"{h.stage}-{h.run_id:<22}{state:<11}started {h.started}  {h.gpu or 'cpu'}  {h.id}")
        if state in ("succeeded", "failed"):
            _report(h, read_result(backend.store(), h))


@app.command()
def logs(run: Annotated[str | None, typer.Argument(help="Which step, for example train-glyph-2b. Default: the latest.")] = None,
         project: ProjectOpt = pathlib.Path("."), set_: SetOpt = [],  # noqa: B006
         follow: Annotated[bool, typer.Option("--follow", "-f", help="Keep printing while the step runs.")] = False) -> None:
    """A step's output."""
    proj, config = _open(project, set_)
    handle = runs.find_handle(proj, run)
    if handle is None:
        raise _fail("No such step. `inscribe status` lists them.")
    for line in _backend(config, proj).logs(handle, follow=follow):
        typer.echo(line)


@app.command()
def stop(run: Annotated[str | None, typer.Argument(help="Which step. Default: every running step of this project.")] = None,
         project: ProjectOpt = pathlib.Path("."), set_: SetOpt = []) -> None:  # noqa: B006
    """Stop a running step. What it had written to the store stays there."""
    proj, config = _open(project, set_)
    backend = _backend(config, proj)
    handles = [h for h in runs.load_handles(proj) if run is None or run in (f"{h.stage}-{h.run_id}", h.stage, h.run_id)]
    stopped = 0
    for h in handles:
        if backend.status(h) == "running":
            backend.stop(h)
            typer.echo(f"Stopped {h.stage}-{h.run_id}.")
            stopped += 1
    if not stopped:
        typer.echo("Nothing was running.")


if __name__ == "__main__":
    app()
