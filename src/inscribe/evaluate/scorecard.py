"""The report card for one trained LoRA: what was measured, each line met or not met with its number, and samples.

    report = build(game, recipe_name, title, bundle, sheet, fmt, before, after, shipped, judged)
    text = report_md([report])

`before` and `after` are the training step's answers (the base model, then the LoRA at full precision). `shipped` maps
each exported file's type to the answers that file gave. `judged` holds the strong model's spoiler and style reads, or
nothing when they were skipped.

What the card can and cannot say is part of the card: it is printed on every run.
"""
from __future__ import annotations

from inscribe.evaluate import retention as R
from inscribe.plan import recall
from inscribe.runtime.template import ChatFormat
from inscribe.schemas.game_sheet import GameSheet

IMPORTANT_LORE_LINE = 0.8  # the share of high-importance lore items a LoRA should recall

NOT_CHECKED = (
    "General ability is not checked. A LoRA can pass everything here and still be a little worse than its base model at ordinary "
    "scenes. One of our own runs on the example game was. The retention questions catch a large loss, not a small one.",
    "Whether this game's names turn up in other worlds is not checked.",
    "Lore in the weights is a preference, not a guarantee. A model that recalls nearly every important item still misses some. Keep the canonical lore in your game.",
    "A secret in the weights is not safe from a determined player. The sheet's secrets teach a character to keep them; anything that must not come out before its time belongs to the game at run time.",
    "Live state (who is where, what was sold, what the player did) comes from your game on every turn, never from the LoRA.",
)

IF_NOT_MET = {
    "lore": "Run `inscribe check --lore` to see which items got no scene of their own, and mark what players must hear `importance: high` in the sheet. Then `inscribe plan`, `data` and `train` again.",
    "spoilers": "Read the flagged lines below first: the judge can be wrong. If a later stage's fact is in the sheet's lore or an early stage's summary, move it out. What happens later belongs to the game at run time.",
    "secrets": "Check that the secret has a `secret` block on every knowledge item that carries it, and that nothing in `lore` states it.",
    "style": "Give the character more to go on: a fuller `voice` line and a few `speech_examples` in the sheet.",
    "retention": "The model lost skills it had. Try, in this order: more replay (`replay.to_one`), then fewer epochs (`train.epochs`), then a lower rank (`train.rank`).",
}


def _pct(k: int, n: int) -> str:
    return f"{k} of {n}" + (f" ({round(100 * k / n)}%)" if n else "")


def _by_id(rows: list[dict] | None) -> dict[str, str]:
    return {r["id"]: recall.clean_reply(r.get("text") or "") for r in rows or []}


def build(game: str, recipe_name: str, title: str, bundle: dict, sheet: GameSheet, fmt: ChatFormat, before: list[dict] | None, after: list[dict] | None,
          shipped: dict[str, list[dict]], judged: dict | None = None) -> dict:
    """Score everything that can be scored from the answers at hand, and lay out the card."""
    judged = judged or {}
    out: dict = {"game": game, "base": recipe_name, "title": title, "recall": {}, "files": list(shipped)}
    if before:
        out["recall"]["before"] = recall.score(bundle, before)
    if after:
        out["recall"]["after"] = recall.score(bundle, after)
    for quant, rows in shipped.items():
        out["recall"][quant] = recall.score(bundle, rows)
    # The file a game ships is what counts. Without one, the trained model at full precision stands in.
    where = next(iter(shipped), "after" if after else None)
    out["read_at"] = where
    main = out["recall"].get(where) if where else None

    prompts = R.retention_prompts(fmt)
    ids = {p["id"] for p in prompts}
    out["retention"] = None
    if before and after and ids <= {r["id"] for r in before} and ids <= {r["id"] for r in after}:
        out["retention"] = R.verdict(R.score(before, prompts), R.score(after, prompts), R.tier_of(recipe_name))
        asked, b, a = {p["id"]: p["asked"] for p in prompts}, _by_id(before), _by_id(after)
        out["retention_lost"] = [{"set": n, "asked": asked[i], "base": b[i], "lora": a[i]} for n in R.SETS for i in out["retention"][n]["lost"]]
    out["spoilers"], out["style"] = judged.get("spoilers"), judged.get("style")

    card = []
    if main:
        high = main["lore"]["by_importance"]["high"]
        if high["items"]:
            card.append({"key": "lore", "item": "Important lore recalled", "line": f"at least {round(100 * IMPORTANT_LORE_LINE)}%",
                         "value": _pct(high["items_recalled"], high["items"]), "met": high["items_recalled"] / high["items"] >= IMPORTANT_LORE_LINE})
        else:
            lore = main["lore"]
            card.append({"key": "lore", "item": "Lore recalled (no item is marked `importance: high`)", "line": f"at least {round(100 * IMPORTANT_LORE_LINE)}%",
                         "value": _pct(lore["items_recalled"], lore["items"]), "met": bool(lore["items"]) and lore["items_recalled"] / lore["items"] >= IMPORTANT_LORE_LINE})
        sp = out["spoilers"]
        if sp:
            ub = sp["upper_bound_95"]
            card.append({"key": "spoilers", "item": "Later story told early", "line": "none",
                         "value": f"{sp['told_early']} of {sp['prompts']}" + (f" (the true rate could be as high as {round(100 * ub)}%)" if ub is not None else ""),
                         "met": sp["told_early"] == 0 if sp["prompts"] else None})
        elif main["spoiler"]["prompts"]:
            card.append({"key": "spoilers", "item": "Later story told early", "line": "none",
                         "value": f"{main['spoiler']['told_early']} of {main['spoiler']['prompts']} by key words only, a weak check; the judge did not read them", "met": None})
        dyn = main["dynamic"]
        if dyn["prompts"]:
            card.append({"key": "secrets", "item": "Run-time characters tell no secret", "line": "none told",
                         "value": f"{dyn['secret_leaks']} told in {dyn['prompts']} replies", "met": dyn["secret_leaks"] == 0})
    st = out["style"]
    if st and st["pairs"]:
        card.append({"key": "style", "item": "Sounds more like the character than the base model", "line": "preferred beyond chance",
                     "value": f"preferred in {st['lora_wins']} of {st['pairs']} pairs, the base in {st['base_wins']}, {st['ties']} ties (p = {st['sign_test_p']:.3g})",
                     "met": st["lora_wins"] > st["base_wins"] and st["sign_test_p"] < 0.05})
    if out["retention"]:
        v = out["retention"]
        gated = [name for name in R.SETS if v[name]["gated"]]
        card.append({"key": "retention", "item": "Keeps what the base model could do", "line": "no gated set lost",
                     "value": ", ".join(f"{R.SET_NAMES[n].lower()} {v[n]['before']} to {v[n]['after']} of {v[n]['n']}" for n in gated), "met": v["passed"]})
    out["scorecard"] = card
    out["samples"] = _samples(bundle, main, _by_id(before), _by_id(shipped.get(where) if where in shipped else after), out)
    return out


def _samples(bundle: dict, main: dict | None, base: dict[str, str], lora: dict[str, str], out: dict) -> dict:
    """A few replies to read: lore the LoRA now says, lore it missed, and retention questions it lost."""
    if not main:
        return {}
    rows = {r["id"]: r for r in recall.all_rows(bundle)}
    missed_ids = set(main["lore"]["misses"])
    important = [r for r in bundle["lore"] if r["importance"] == "high" and r["id"] in lora]
    said = [r for r in important if r["id"] not in missed_ids and recall.hit(lora[r["id"]], r["expect_any"])]
    seen, recalled = set(), []
    for r in said:  # one sample per lore item
        if r["lore_id"] not in seen and len(recalled) < 6:
            seen.add(r["lore_id"])
            recalled.append({"asked": r["question"], "base": base.get(r["id"], ""), "lora": lora[r["id"]]})
    items: dict[str, list[dict]] = {}
    for r in important:
        items.setdefault(r["lore_id"], []).append(r)
    missed = [{"asked": rs[0]["question"], "lora": lora[rs[0]["id"]], "wanted_any_of": rs[0]["expect_any"]}
              for rs in items.values() if not any(recall.hit(lora[r["id"]], r["expect_any"]) for r in rs)][:6]
    told = [{"asked": rows[i]["question"], "lora": lora.get(i, "")} for i in main["dynamic"]["leak_ids"][:4] if i in rows]
    wrong = [{"asked": rows[i]["question"], "lora": lora.get(i, ""), "price": rows[i].get("price")} for i in main["offers"]["wrong_ids"][:4] if i in rows]
    return {"recalled": recalled, "missed": missed, "secrets_told": told, "wrong_prices": wrong}


def _mark(met: bool | None) -> str:
    return "not judged" if met is None else "met" if met else "**not met**"


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def report_md(reports: list[dict]) -> str:
    """The card as a page: one section per trained model."""
    game = reports[0]["game"] if reports else "game"
    lines = [f"# {game}: report card", "",
             "One training run per model, so every number here is one run's. Answers are greedy and scored on the spoken line.", ""]
    for rep in reports:
        lines += _section(rep)
    lines += ["## What this card does not tell you", ""] + [f"- {x}" for x in NOT_CHECKED] + [""]
    return "\n".join(lines)


def _section(rep: dict) -> list[str]:
    where = rep.get("read_at")
    at = {"after": "the trained model at full precision (no file exported yet)", None: "nothing yet"}.get(where, f"the {where} file, the one a game ships")
    lines = [f"## {rep['title']}", "", f"Read at {at}.", ""]
    card = rep["scorecard"]
    if not card:
        return lines + ["No answers to score yet. Run `inscribe train` and `inscribe export` first.", ""]
    lines += ["| Check | Line | This run | |", "|---|---|---|---|"]
    lines += [f"| {c['item']} | {c['line']} | {_cell(c['value'])} | {_mark(c['met'])} |" for c in card]
    lines.append("")
    failed = [c for c in card if c["met"] is False]
    if failed:
        lines += ["What to try for the lines not met:", ""] + [f"- {c['item']}: {IF_NOT_MET[c['key']]}" for c in failed] + [""]

    lines += ["### Lore, prices and run-time characters", "", "| | Base model | After training |" + "".join(f" {q} file |" for q in rep["files"]),
              "|---|---|---|" + "---|" * len(rep["files"])]
    cols = [rep["recall"].get(k) for k in ("before", "after", *rep["files"])]

    def row(label: str, f) -> str:
        return f"| {label} | " + " | ".join("not answered" if c is None else f(c) for c in cols) + " |"

    lines += [row("Important lore items recalled", lambda s: f"{s['lore']['by_importance']['high']['items_recalled']} of {s['lore']['by_importance']['high']['items']}"),
              row("All lore items recalled", lambda s: f"{s['lore']['items_recalled']} of {s['lore']['items']}"),
              row("Exact items, word for word (questions)", lambda s: f"{s['lore']['exact']['prompt_hits']} of {s['lore']['exact']['prompts']}"),
              row("Later story told early, by key words", lambda s: f"{s['spoiler']['told_early']} of {s['spoiler']['prompts']}"),
              row("Run-time characters answering lore", lambda s: f"{s['dynamic']['lore_hits']} of {s['dynamic']['prompts']}"),
              row("Run-time characters telling a secret", lambda s: f"{s['dynamic']['secret_leaks']}"),
              row("Prices right", lambda s: f"{s['offers']['right_price']} of {s['offers']['prompts']}"),
              row("Another price given instead", lambda s: f"{s['offers']['wrong_price']}"), ""]

    v = rep.get("retention")
    if v:
        lines += ["### What it kept", "", "| Set | Base model | After training | Counts toward the line |", "|---|---|---|---|"]
        lines += [f"| {R.SET_NAMES[n]} | {v[n]['before']} of {v[n]['n']} | {v[n]['after']} of {v[n]['n']} | {'yes' if v[n]['gated'] else 'no, reported'} |" for n in R.SETS]
        lines.append("")
    else:
        lines += ["### What it kept", "", "Not scored: the training run on the store did not answer the retention questions. `inscribe train` asks them by default.", ""]

    sp = rep.get("spoilers")
    if sp:
        lines += ["### Later story told early", "",
                  f"The judge ({sp['judge']}) read {sp['prompts']} replies to questions about what comes next and flagged {sp['told_early']}. "
                  "Read a flag before you act on it: the judge is sometimes wrong, and has cleared a line on a second read that it flagged on the first.", ""]
        for t in sp["told"]:
            lines += [f"- Asked: {t['asked']}", f"  Reply: {t['reply']}", f"  Judge: {t['evidence']}"]
        if sp["errors"]:
            lines.append(f"- {len(sp['errors'])} replies could not be judged.")
        lines.append("")
    st = rep.get("style")
    if st and st["pairs"]:
        lines += ["### Voice", "",
                  f"The judge ({st['judge']}) saw the base model's reply and the LoRA's to the same line, unlabelled and in a random order, with the sheet's "
                  f"`voice` text. It preferred the LoRA in {st['lora_wins']} of {st['pairs']}, the base in {st['base_wins']}, and called {st['ties']} a tie. "
                  "It judges the voice only, not the facts.", "", "| Character | Player | Base model | LoRA | Preferred |", "|---|---|---|---|---|"]
        shown = [s for s in st["samples"] if s["preferred"] == "lora"][:4] + [s for s in st["samples"] if s["preferred"] != "lora"][:3]
        lines += [f"| {_cell(s['character'])} | {_cell(s['asked'])} | {_cell(s['base'])} | {_cell(s['lora'])} | {s['preferred']} |" for s in shown]
        lines.append("")

    sm = rep.get("samples") or {}
    if sm.get("recalled"):
        lines += ["### Lore it now says", "", "| Player | Base model | This model |", "|---|---|---|"]
        lines += [f"| {_cell(s['asked'])} | {_cell(s['base'])} | {_cell(s['lora'])} |" for s in sm["recalled"]]
        lines.append("")
    if sm.get("missed"):
        lines += ["### Important lore it missed", "", "| Player | This model | A reply that states it uses one of |", "|---|---|---|"]
        lines += [f"| {_cell(s['asked'])} | {_cell(s['lora'])} | {_cell(', '.join(s['wanted_any_of']))} |" for s in sm["missed"]]
        lines.append("")
    for key, title in (("secrets_told", "Replies from run-time characters that use a secret's words"), ("wrong_prices", "Another price given")):
        if sm.get(key):
            lines += [f"### {title}", ""] + [f"- Asked: {s['asked']}\n  Reply: {s['lora']}" for s in sm[key]] + [""]
    lost = rep.get("retention_lost") or []
    if lost:
        lines += ["### Questions it answered before training and not after", "", "For the closed-world set a loss means a real-world word came out.", "",
                  "| Set | Player | Base model | After training |", "|---|---|---|---|"]
        lines += [f"| {R.SET_NAMES[s['set']]} | {_cell(s['asked'])} | {_cell(s['base'])} | {_cell(s['lora'])} |" for s in lost]
        lines.append("")
    return lines
