"""The strong model: it reads prose lore into a game sheet, and it judges spoilers and style on a trained LoRA.

It is a reasoning model reached through OpenAI's Responses API, asked for an answer in a fixed JSON shape. Every call
is paid: $2 per million input tokens and $10 per million output tokens, with reasoning billed as output. The key is
read from the environment on this machine and goes nowhere else.

    line = request("spoiler.s1", messages, strict_schema(Verdict), "spoiler_verdict", model)
    out = answer([line], api_key)                # one answer per request, in order
    parsed = parse(out[0])                       # .text, or .error; never a guess

A request that hits its output ceiling comes back incomplete, and a refusal comes back as a refusal. Both are recorded
as errors and never turned into a verdict.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

MODEL = "gpt-6.1-sol"
KEY_ENV = "OPENAI_API_KEY"
BASE_URL = "https://api.openai.com/v1"
ENDPOINT = "/v1/responses"
REASONING_EFFORT = "low"  # the cheapest this model supports
MAX_OUTPUT_TOKENS = 8_000  # reasoning plus the visible answer; a request that hits it is an error, not a verdict
USD_PER_M_INPUT = 2.0
USD_PER_M_OUTPUT = 10.0
CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4  # per message; a rough allowance for forecasts
RETRY_STATUSES = (429, 500, 502, 503, 504)

_UNSUPPORTED = {"default", "title", "examples"}


def _schema_children(node: dict):
    """(container, key) for every sub-schema of one schema node. Walks by keyword, so a field that happens to be
    called "properties" is not mistaken for the keyword."""
    for kw in ("properties", "$defs", "definitions"):
        for k in node.get(kw, {}) or {}:
            yield node[kw], k
    for kw in ("items", "additionalItems"):
        if isinstance(node.get(kw), dict):
            yield node, kw
    for kw in ("anyOf", "oneOf", "prefixItems"):
        for i in range(len(node.get(kw, []) or [])):
            yield node[kw], i


def strict_schema(model_cls) -> dict:
    """A pydantic model's JSON schema in the subset the API accepts in strict mode: every property required, no extra
    properties, no defaults."""

    def fix(node: dict) -> dict:
        for k in _UNSUPPORTED & set(node):
            del node[k]
        if "const" in node:
            node["enum"] = [node.pop("const")]
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}))
        for container, k in list(_schema_children(node)):
            container[k] = fix(container[k])
        return node

    return fix(copy.deepcopy(model_cls.model_json_schema()))


def body(messages: list[dict], schema: dict, name: str, model: str = MODEL) -> dict:
    return {"model": model, "input": messages,
            "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
            "reasoning": {"effort": REASONING_EFFORT}, "max_output_tokens": MAX_OUTPUT_TOKENS}


def request(custom_id: str, messages: list[dict], schema: dict, name: str, model: str = MODEL) -> dict:
    """One request, with the id its answer comes back under."""
    return {"custom_id": custom_id, "method": "POST", "url": ENDPOINT, "body": body(messages, schema, name, model)}


class PreferenceVerdict(BaseModel):  # which of two replies, shown as A and B. No docstring: it would travel in the request's schema
    model_config = ConfigDict(extra="forbid")
    rubric_version: Literal["judge-rubric-0.1"] = "judge-rubric-0.1"
    winner: Literal["A", "B", "tie"]
    reason: str


PREFERENCE_SCHEMA = strict_schema(PreferenceVerdict)


def pair_order(pair_id: str, seed: int = 0) -> Literal["ab", "ba"]:
    """Which reply is shown first, drawn from the seed and the pair's id: "ab" shows the first reply as A. The order
    goes into the request's id, so an answer can always be mapped back."""
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode()).digest()
    return "ab" if random.Random(digest).random() < 0.5 else "ba"


def unswap(custom_id: str, winner: str) -> Literal["1", "2", "tie"]:
    """The judge's A, B or tie as the caller's first or second reply, from the order in the request's id."""
    order = custom_id.rsplit(".", 1)[-1]
    if order not in ("ab", "ba"):
        raise ValueError(f"{custom_id}: no order suffix")
    if winner == "tie":
        return "tie"
    return "1" if (winner == "A") == (order == "ab") else "2"


@dataclass
class Parsed:
    custom_id: str
    text: str | None
    error: str | None
    usage: dict
    model: str | None = None


def parse(line: dict) -> Parsed:
    """One answer: its text, or why there is none."""
    cid = line.get("custom_id", "")
    if line.get("error"):
        return Parsed(cid, None, f"request error: {json.dumps(line['error'])[:300]}", {})
    resp = line.get("response") or {}
    data = resp.get("body") or {}
    usage = data.get("usage") or {}
    model = data.get("model")
    if resp.get("status_code") != 200:
        return Parsed(cid, None, f"HTTP {resp.get('status_code')}: {json.dumps(data)[:300]}", usage, model)
    if data.get("status") != "completed":
        reason = (data.get("incomplete_details") or {}).get("reason")
        return Parsed(cid, None, f"status {data.get('status')} ({reason})", usage, model)
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "refusal":
                return Parsed(cid, None, f"refusal: {content.get('refusal', '')[:300]}", usage, model)
            if content.get("type") == "output_text":
                return Parsed(cid, content.get("text", ""), None, usage, model)
    return Parsed(cid, None, "no output_text in response", usage, model)


def estimate_input_tokens(line: dict) -> int:
    b = line["body"]
    chars = sum(len(m["content"]) for m in b["input"]) + len(json.dumps(b["text"]["format"]["schema"]))
    return math.ceil(chars / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD_TOKENS * len(b["input"])


def forecast_usd(lines: list[dict], output_tokens_each: int) -> float:
    """What the requests would cost, from their size and an assumed answer length. An estimate."""
    return sum(estimate_input_tokens(x) for x in lines) / 1e6 * USD_PER_M_INPUT + len(lines) * output_tokens_each / 1e6 * USD_PER_M_OUTPUT


def cost_usd(usage_rows: list[dict]) -> float:
    """What the answers cost, from the token counts the API reported."""
    return (sum(u.get("input_tokens", 0) for u in usage_rows) / 1e6 * USD_PER_M_INPUT
            + sum(u.get("output_tokens", 0) for u in usage_rows) / 1e6 * USD_PER_M_OUTPUT)


def answer(lines: list[dict], api_key: str, concurrency: int = 8, timeout: float = 120.0, retries: int = 4) -> list[dict]:
    """Send each request and return its answer, in order. A request that keeps failing comes back as an error line."""
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    if not api_key:
        raise RuntimeError(f"{KEY_ENV} is not set")

    def one(line: dict) -> dict:
        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}) as http:
                    r = http.post(f"{BASE_URL}/responses", json=line["body"])
                if r.status_code in RETRY_STATUSES and attempt < retries:
                    time.sleep(min(30, 2**attempt))
                    continue
                return {"custom_id": line["custom_id"], "response": {"status_code": r.status_code, "body": r.json()}}
            except (httpx.TransportError, ValueError) as exc:
                if attempt >= retries:
                    return {"custom_id": line["custom_id"], "error": {"message": str(exc)[:200]}}
                time.sleep(min(30, 2**attempt))
        return {"custom_id": line["custom_id"], "error": {"message": "retries exhausted"}}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, lines))
