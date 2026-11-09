"""A case resolved against its world: exactly what the model would see."""
from __future__ import annotations

from dataclasses import dataclass

from inscribe.runtime.actor_context import ActorContext, Turn
from inscribe.runtime.template import DEFAULT_FORMAT, ChatFormat, render_prompt
from inscribe.schemas.records import EvalCase
from inscribe.schemas.world import World

@dataclass(frozen=True)
class ResolvedCase:
    case: EvalCase
    world: World
    actor_id: str
    context: ActorContext
    dialogue: list[Turn]

    def prompt(self, fmt: ChatFormat = DEFAULT_FORMAT) -> str:
        return render_prompt(self.context, self.dialogue, fmt)
