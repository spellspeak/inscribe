"""Deterministic checks: what code can judge reliably without a model.

These catch format and exact-string failures. Paraphrased leaks, stale state,
over-certainty and voice quality need a model to judge, so the
planted-failure report shows which failure types this layer covers.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from inscribe.runtime.actor_context import ActorContext
from inscribe.runtime.output import speakability_issues
from inscribe.schemas.records import EvalCase

ASSISTANT_PATTERNS = [
    r"\bas an ai\b", r"\blanguage model\b", r"\bai assistant\b",
    r"\bhow (?:can|may) i (?:help|assist) you\b", r"\bi'?d be (?:happy|glad) to help\b",
    r"\bis there anything else i can\b", r"\bi hope this helps\b", r"\bgreat question\b",
    r"\bfeel free to ask\b", r"\blet me know if\b",
]
LENGTH_LIMIT_WORDS = {"brief_conversation": 45, "explanatory": 150}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _contains(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)", text.lower()) is not None


def check_reply(case: EvalCase, reply: str, ctx: ActorContext) -> list[CheckResult]:
    results = [CheckResult("speakable_output", not (i := speakability_issues(reply)), "; ".join(i))]
    hits = [p for p in case.must_not_contain if _contains(reply, p)]
    results.append(CheckResult("must_not_contain", not hits, ", ".join(hits)))
    if case.must_contain_any:
        found = [p for p in case.must_contain_any if _contains(reply, p)]
        results.append(CheckResult("must_contain_any", bool(found), "none of: " + ", ".join(case.must_contain_any)))
    assistant = [p for p in ASSISTANT_PATTERNS if re.search(p, reply.lower())]
    results.append(CheckResult("no_assistant_voice", not assistant, ", ".join(assistant)))
    words = len(reply.split())
    limit = LENGTH_LIMIT_WORDS[ctx.output_policy.response_style]
    results.append(CheckResult("length_limit", words <= limit, f"{words} words, limit {limit}"))
    return results


def passes(case: EvalCase, reply: str, ctx: ActorContext) -> bool:
    return all(r.passed for r in check_reply(case, reply, ctx))


def planted_failure_report(resolved_cases) -> dict:
    """How much of the planted-failure set (bad reference replies) this layer catches, by failure type,
    and how often it wrongly flags a good reference reply."""
    caught, total = Counter(), Counter()
    false_alarms: list[tuple[str, str, list[str]]] = []
    by_case = defaultdict(list)
    for rc in resolved_cases:
        refs = rc.case.reference_replies
        if refs is None:
            continue
        for good in refs.good:
            failed = [r.name for r in check_reply(rc.case, good, rc.context) if not r.passed]
            if failed:
                false_alarms.append((rc.case.case_id, good, failed))
        for bad in refs.bad:
            total[bad.failure] += 1
            hit = not passes(rc.case, bad.reply, rc.context)
            caught[bad.failure] += hit
            by_case[rc.case.case_id].append(hit)
    return {
        "by_failure": {f: (caught[f], total[f]) for f in sorted(total)},
        "caught": sum(caught.values()),
        "planted": sum(total.values()),
        "false_alarms": false_alarms,
    }
