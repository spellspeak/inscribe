"""The `inscribe` command.

    inscribe init my-game     start a game project from the template
    inscribe check            validate the game sheet, show what would be built, forecast the run. Free
    inscribe plan             build the scenes, the held-out checks and the world files. Free
    inscribe mix              build the training mix on the backend: the game's rows and the replay. Cents
    inscribe train            train the LoRA on the backend. Spends money, and asks first
    inscribe export           make the GGUF file and answer the held-out questions with it. Spends money, and asks first
    inscribe pull             bring the GGUF file and the results to out/
    inscribe status           the steps this project has launched
    inscribe logs             a step's output
    inscribe stop             stop a running step

The teacher and scoring steps are not built yet.
"""
from __future__ import annotations

import json
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


def _backend(config: Config, proj: Project):
    backend = runs.make_backend(config, config.game)
    problems = backend.check()
    if problems:
        raise _fail("\n".join(problems))
    return backend


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
          prompts: Annotated[pathlib.Path | None, typer.Option(help="Questions to answer before and after training ({id, prompt, max_new} per line). Default: the plan's held-out questions.")] = None,
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
        rows = prompt_rows(json.loads((proj.plan_dir / "recall.json").read_text()), recipe.chat_format)
        prompts_at = "prompts/recall.jsonl"
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
