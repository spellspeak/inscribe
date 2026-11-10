"""The fast judge: one request per reply, a probability for every question.

It is OpenAI's Decisions API. The request carries the scene's state as JSON and one yes-or-no question per expected
property and per failure that applies. The answer is the probability that each question's answer is yes. Input tokens
are billed, output is free.

Asked bare, this judge says yes too easily to a few questions. So each failure question also carries the rubric's own
criteria for yes and for no, a few property questions carry criteria of their own, and the input opens with what each
field of the state is. Nothing is tuned to a scene.

The key comes from the environment and is never printed or stored.
"""
from __future__ import annotations

import email.utils
import json
import random
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from inscribe.judges.questions import FAILURE_QUESTIONS, JudgeRequest, failure_yes_criterion

URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
KEY_ENV = "OPENAI_API_KEY"
USD_PER_M_INPUT = 0.10  # output and cached tokens are free
DEFINITIONS_VERSION = "decisions-definitions-0.5"  # the wording below; a change to it is a change to every decision
# A training line is kept only when its fail probability is below this. The benchmark's own line is laxer: a wrong
# rejection costs a line, and a wrong acceptance teaches the model a mistake.
ACCEPT_BELOW = 0.8
TOKENS_PER_QUESTION = 160  # the API's fixed cost per question, fitted on one probe; for forecasts only
CHARS_PER_TOKEN = 4
RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})

FIELD_GUIDE = (
    "One reply by a game character, to be judged. The state below is JSON; the questions refer to its fields by name in "
    "backticks:\n"
    "- `character_sheet`: what the character was told about itself, its world and the scene. Facts in neither it nor "
    "`conversation` are unknown to the character.\n"
    "- `conversation`: the lines so far, oldest first; each says who spoke, whether the player typed or spoke aloud "
    "(spoken lines may be misheard), and to whom when that matters. The character is answering the last line.\n"
    "- `reply`: the character's next line, the only thing being judged.\n"
    "- `hidden_truth`: true facts the character must not use or reveal.\n"
    "- `acceptable_alternatives`: behaviours this case allows.\n"
    "Other fields, when present, state facts about the scene.\n")
FAILURE_NOTES = {
    "wrong_perspective": "What `acceptable_alternatives` names is allowed in this case and is not this failure. A short reaction to "
                         "an answer someone else gave, such as agreeing with it, is a bystander's reaction, not answering as if asked.",
}
ALLOWANCE_NOTES = {
    "underconfident": "What `acceptable_alternatives` names is allowed in this case and is not this failure.",
}
NO_ALLOWANCES = ["none listed"]  # what the state holds for a scene that lists none
PROPERTY_DEFINITIONS = {
    "shows_no_knowledge_of_": (
        "the reply treats the thing as familiar: it uses, explains or describes it, or says what it is for, how it works "
        "or where to get one.",
        "the reply does not: it asks what the thing is, says the character has never heard of it or knows of none, "
        "guesses wrongly in character, or answers without it. Repeating the player's word to ask about it, or saying "
        "there is none, does not by itself show knowledge."),
    "does_not_answer_as_if_asked": (
        "Does `reply` answer the question another character put to the player, as if this character had been asked it?",
        "the reply gives that answer itself, as if the question were its own to answer.",
        "the reply leaves the question to the one it was put to: it stays out, prompts them to answer, comments, or "
        "speaks of something else. Saying who was asked is not answering."),
    "does_not_answer_for_them": (
        "Does `reply` answer the player's question in place of the character the player asked, giving the answer as its own?",
        "the reply gives the answer itself, as if it were the one asked.",
        "the reply agrees with or confirms what the one asked said, reacts to it, or stays out. Agreeing is not answering for them."),
}


def with_definitions(req: JudgeRequest) -> JudgeRequest:
    """The same request with the rubric's yes and no criteria after each failure question, and after the property
    questions in PROPERTY_DEFINITIONS. Other property questions are the scene's own and have none."""
    questions = {}
    allowances = req.body["state"].get("acceptable_alternatives", NO_ALLOWANCES) not in (NO_ALLOWANCES, [])
    for qid, q in req.body["questions"].items():
        k = req.key.get(qid)
        if k is not None and k.kind == "failure":
            note = FAILURE_NOTES.get(k.name) or (ALLOWANCE_NOTES.get(k.name) if allowances else None)
            q = {**q, "instructions": f"{q['instructions']} Yes means: {failure_yes_criterion(k.name)} "
                                      f"No means: {FAILURE_QUESTIONS[k.name][1]}" + (f" {note}" if note else "")}
        elif k is not None and k.kind == "property":
            crit = next((c for prefix, c in PROPERTY_DEFINITIONS.items() if k.name.startswith(prefix)), None)
            if crit:  # (yes, no), or (a plainer question, yes, no); the polarity stays the property's
                question = crit[0] if len(crit) == 3 else q["instructions"]
                q = {**q, "instructions": f"{question} Yes means: {crit[-2]} No means: {crit[-1]}"}
        questions[qid] = q
    return JudgeRequest(req.item_id, {**req.body, "questions": questions}, req.key)


def request_body(req: JudgeRequest, model: str = MODEL) -> dict:
    """What is sent for one reply: the field guide and the state as sorted JSON, then the questions with their criteria."""
    body = with_definitions(req).body
    return {"model": model,
            "input": FIELD_GUIDE + json.dumps(body["state"], indent=1, sort_keys=True, ensure_ascii=False),
            "questions": [{"type": "predicate", "name": qid, "instructions": q["instructions"]} for qid, q in body["questions"].items()]}


def read_answers(data: dict) -> tuple[dict[str, float], list[str]]:
    """The probabilities by question id, and the questions the judge would not answer."""
    answers, refused = {}, []
    for a in data.get("answers") or []:
        if a.get("type") == "predicate" and isinstance(a.get("probability"), (int, float)):
            answers[a["name"]] = float(a["probability"])
        else:
            refused.append(str(a.get("name")))
    return answers, refused


def estimate_tokens(req: JudgeRequest) -> int:
    body = request_body(req)
    return len(body["input"]) // CHARS_PER_TOKEN + TOKENS_PER_QUESTION * len(body["questions"])


def cost_usd(input_tokens: int) -> float:
    return input_tokens * USD_PER_M_INPUT / 1e6


class JudgeError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _retry_after_seconds(headers) -> float | None:
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return float(ms) / 1000
        except ValueError:
            pass
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, when.timestamp() - time.time())


@dataclass
class FastJudge:
    """Sends judge requests, a bounded number at a time. A request is retried with backoff on a timeout, a connection
    error, 408, 429 or a 5xx. A refused question is never guessed into an answer: that reply gets an error instead."""

    api_key: str | None = None
    model: str = MODEL
    url: str = URL
    concurrency: int = 16
    max_retries: int = 6
    backoff_initial: float = 0.5
    backoff_max: float = 30.0
    backoff_jitter: float = 0.25
    request_timeout: float = 60.0
    transport: object | None = None  # an httpx transport, for tests
    sleep: Callable[[float], None] = time.sleep
    input_tokens: int = 0  # summed from the responses
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def name(self) -> str:
        return self.model

    def _client(self):
        import httpx

        if not self.api_key:
            raise JudgeError(f"no API key: set {KEY_ENV} in this shell")
        return httpx.Client(transport=self.transport, timeout=self.request_timeout,
                            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, 120.0)
        base = min(self.backoff_max, self.backoff_initial * 2 ** attempt)
        return base * (1 - self.backoff_jitter * random.random())

    def answer(self, req: JudgeRequest, client=None) -> tuple[dict[str, float], int]:
        """One reply. Returns (probability by question id, attempts used). Raises JudgeError when it gives up."""
        import httpx

        body = request_body(req, self.model)
        asked = {q["name"] for q in body["questions"]}
        own = client is None
        client = client or self._client()
        try:
            for attempt in range(self.max_retries + 1):
                retry_after = None
                try:
                    r = client.post(self.url, json=body)
                except (httpx.TransportError, httpx.TimeoutException) as exc:
                    err = JudgeError(f"{type(exc).__name__}: {exc}")
                else:
                    if r.status_code == 200:
                        try:
                            data = r.json()
                        except ValueError:
                            data = None
                        if isinstance(data, dict) and isinstance(data.get("answers"), list):
                            answers, refused = read_answers(data)
                            with self._lock:
                                self.input_tokens += int((data.get("usage") or {}).get("input_tokens") or 0)
                            missing = sorted(asked - set(answers))
                            if missing:
                                raise JudgeError(f"no probability for {', '.join(missing)} (refused: {refused})", 200)
                            return answers, attempt + 1
                        err = JudgeError(f"HTTP 200 without answers: {r.text[:300]}", 200)
                    else:
                        err = JudgeError(f"HTTP {r.status_code}: {r.text[:500]}", r.status_code)
                        if r.status_code not in RETRY_STATUSES:
                            raise err
                        retry_after = _retry_after_seconds(r.headers)
                if attempt == self.max_retries:
                    raise err
                self.sleep(self._delay(attempt, retry_after))
            raise AssertionError("unreachable")
        finally:
            if own:
                client.close()

    def answer_all(self, requests: Iterable[JudgeRequest], on_result: Callable[[dict], None] | None = None) -> list[dict]:
        """Every request, at most `concurrency` in flight. One result per request, in input order: its `answers`, or
        None with an `error`. A wrong or missing key stops the run."""
        requests = list(requests)
        client = self._client()

        def one(req: JudgeRequest) -> dict:
            t0 = time.perf_counter()
            try:
                answers, attempts = self.answer(req, client)
                res = {"item_id": req.item_id, "answers": answers, "error": None, "attempts": attempts}
            except JudgeError as exc:
                if exc.status in (401, 403):
                    raise
                res = {"item_id": req.item_id, "answers": None, "error": str(exc), "attempts": None}
            res["seconds"] = round(time.perf_counter() - t0, 3)
            if on_result:
                on_result(res)
            return res

        pool = ThreadPoolExecutor(max_workers=self.concurrency)
        try:
            futures = [pool.submit(one, r) for r in requests]
            return [f.result() for f in futures]
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            client.close()
