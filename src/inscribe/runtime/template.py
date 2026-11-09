from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from inscribe.runtime.actor_context import ActionResult, ActorContext, BoundaryView, EventView, PersonView, RelationshipView, Turn
from inscribe.runtime.output import speakability_issues
from inscribe.runtime.vocab import HistoryEvent, HistoryPromise, HistoryTold
TEMPLATE_VERSION = 'template-0.3'
PLAYER_LABEL = 'Player'
GAME_LABEL = '[game update]'
PRIVATE_HEADER = 'You could tell the player, but not while others can hear:'
OUTPUT_RULE = "Say only your character's words, as one line of dialogue. Normal punctuation and quotes are fine. No actions, emotes, stage directions, lists or formatting. Names in what the player says may be misspelled or misheard."
FORBIDDEN_MARKERS = ('<think>', '</think>')

@dataclass(frozen=True)
class ChatFormat:
    name: str
    tokenizer_repo: str
    tokenizer_revision: str
    turn_open: Mapping[str, str]
    turn_close: str
    reply_end: str
    single_token_markers: tuple[str, ...]
    system_role: bool = True
    assistant_prefix: str = ''
    system_suffix: str = ''
    training: bool = True
    thinking: bool = False
    think_open: str = ''
    think_close: str = ''
    think_off: str = ''
    text_markers: tuple[str, ...] = ()
    bos_token: str = ''
    turn_close_by_role: Mapping[str, str] | None = None
    system_prefix: str = ''

    def close(self, role: str) -> str:
        return self.turn_close_by_role[role] if self.turn_close_by_role else self.turn_close
IM_START, IM_END = ('<|im_start|>', '<|im_end|>')
FORMATS: dict[str, ChatFormat] = {'qwen3.5-2b-base': ChatFormat(name='qwen3.5-2b-base', tokenizer_repo='Qwen/Qwen3.5-2B-Base', tokenizer_revision='b1485b2fa6dfa1287294f269f5fb618e03d52d7c', turn_open={r: f'{IM_START}{r}\n' for r in ('system', 'user', 'assistant')}, turn_close=f'{IM_END}\n', reply_end=IM_END, single_token_markers=(IM_START, IM_END)), 'qwen3.5-2b-instruct': ChatFormat(name='qwen3.5-2b-instruct', tokenizer_repo='Qwen/Qwen3.5-2B', tokenizer_revision='15852e8c16360a2fea060d615a32b45270f8a8fc', turn_open={r: f'{IM_START}{r}\n' for r in ('system', 'user', 'assistant')}, turn_close=f'{IM_END}\n', reply_end=IM_END, single_token_markers=(IM_START, IM_END), assistant_prefix='<think>\n\n</think>\n\n', training=False), 'smollm3-instruct': ChatFormat(name='smollm3-instruct', tokenizer_repo='HuggingFaceTB/SmolLM3-3B', tokenizer_revision='a07cc9a04f16550a088caea529712d1d335b0ac1', turn_open={r: f'{IM_START}{r}\n' for r in ('system', 'user', 'assistant')}, turn_close=f'{IM_END}\n', reply_end=IM_END, single_token_markers=(IM_START, IM_END), assistant_prefix='<think>\n\n</think>\n', system_suffix='\n/no_think', training=False), 'smollm3': ChatFormat(name='smollm3', tokenizer_repo='HuggingFaceTB/SmolLM3-3B-Base', tokenizer_revision='d78a42f79198603e614095753484a04c10c2b940', turn_open={r: f'{IM_START}{r}\n' for r in ('system', 'user', 'assistant')}, turn_close=f'{IM_END}\n', reply_end=IM_END, single_token_markers=(IM_START, IM_END))}
EXPERIMENTAL_FORMATS: dict[str, ChatFormat] = {'qwen3.5-2b-base-think': ChatFormat(name='qwen3.5-2b-base-think', tokenizer_repo='Qwen/Qwen3.5-2B-Base', tokenizer_revision='b1485b2fa6dfa1287294f269f5fb618e03d52d7c', turn_open={r: f'{IM_START}{r}\n' for r in ('system', 'user', 'assistant')}, turn_close=f'{IM_END}\n', reply_end=IM_END, single_token_markers=(IM_START, IM_END), thinking=True, think_open='<think>\n', think_close='\n</think>\n\n', text_markers=('<think>', '</think>'), think_off='<think>\n\n</think>\n\n'), 'gemma4': ChatFormat(name='gemma4', tokenizer_repo='google/gemma-4-E2B', tokenizer_revision='d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f', turn_open={'system': '<|turn>system\n', 'user': '<|turn>user\n', 'assistant': '<|turn>model\n'}, turn_close='<turn|>\n', reply_end='<turn|>', single_token_markers=('<|turn>', '<turn|>'), bos_token='<bos>'), 'gemma4-instruct': ChatFormat(name='gemma4-instruct', tokenizer_repo='google/gemma-4-E2B-it', tokenizer_revision='3e22461f65e89153144f8adb70e3b8c2cc9845a7', turn_open={'system': '<|turn>system\n', 'user': '<|turn>user\n', 'assistant': '<|turn>model\n'}, turn_close='<turn|>\n', reply_end='<turn|>', single_token_markers=('<|turn>', '<turn|>'), bos_token='<bos>'), 'ministral3-instruct': ChatFormat(name='ministral3-instruct', tokenizer_repo='mistralai/Ministral-3-3B-Instruct-2512-BF16', tokenizer_revision='b6d637bef2393152b3da2b2fde72eecdee30557e', turn_open={'system': '[SYSTEM_PROMPT]', 'user': '[INST]', 'assistant': ''}, turn_close='', turn_close_by_role={'system': '[/SYSTEM_PROMPT]', 'user': '[/INST]', 'assistant': '</s>'}, reply_end='</s>', single_token_markers=('[SYSTEM_PROMPT]', '[/SYSTEM_PROMPT]', '[INST]', '[/INST]'), bos_token='<s>', training=False), 'llama3-instruct': ChatFormat(name='llama3-instruct', tokenizer_repo='meta-llama/Llama-3.2-3B-Instruct', tokenizer_revision='0cb88a4f764b7a12671c53f0838cd831a0843b95', turn_open={r: f'<|start_header_id|>{r}<|end_header_id|>\n\n' for r in ('system', 'user', 'assistant')}, turn_close='<|eot_id|>', reply_end='<|eot_id|>', single_token_markers=('<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>'), bos_token='<|begin_of_text|>', system_prefix='Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n', training=False)}

def format_by_name(name: str) -> ChatFormat:
    return FORMATS[name] if name in FORMATS else EXPERIMENTAL_FORMATS[name]

def with_thinking(prompts: list[dict], fmt: ChatFormat, thinking: bool, extra_tokens: int=64) -> list[dict]:
    if not fmt.thinking:
        raise ValueError(f'{fmt.name} has no thinking mode')
    return [{**r, 'prompt': r['prompt'] + ('' if thinking else fmt.think_off), **({'max_new': r.get('max_new', 80) + extra_tokens} if thinking else {})} for r in prompts]

def without_thinking(fmt: ChatFormat) -> ChatFormat:
    from dataclasses import replace
    return replace(fmt, thinking=False, think_open='', think_close='', think_off='')
DEFAULT_FORMAT = EXPERIMENTAL_FORMATS['gemma4-instruct'] if 'gemma4-instruct' in EXPERIMENTAL_FORMATS else FORMATS['gemma4-instruct']
TOKENIZER_REPO = DEFAULT_FORMAT.tokenizer_repo
TOKENIZER_REVISION = DEFAULT_FORMAT.tokenizer_revision
_STYLE = {'brief_conversation': 'Keep it brief: one or two short sentences.', 'explanatory': 'Explain as much as the player asked for, in plain spoken sentences.'}

def _words(identifier: str) -> str:
    return identifier.replace('_', ' ')

def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]
_SOURCE = {'witnessed': 'you saw it', 'overheard': 'you overheard it', 'rumour': 'a rumour', 'read': 'you read it', 'inferred': 'you worked it out', 'common_knowledge': 'common knowledge', 'own_action': 'you did it', 'assumed': 'your assumption'}
_FEELING = {'very_negative': 'very bad', 'negative': 'bad', 'positive': 'good', 'very_positive': 'very good'}
_OBLIGATION = {'owes_favour': 'you owe them a favour', 'is_owed_favour': 'they owe you a favour', 'owes_money': 'you owe them money', 'is_owed_money': 'they owe you money', 'promised': 'you promised them', 'was_promised': 'they promised you'}
_PROMISE = {'open': 'not yet kept', 'kept': 'kept', 'broken': 'broken'}
_TASK = {'not_offered': 'not offered yet', 'offered': 'offered', 'active': 'in progress', 'done': 'done', 'failed': 'failed', 'reward_given': 'done and rewarded'}
_SILENT_VERBS = {'changed_state'}
_VERB_TEXT = {'gave_gift': 'gave', 'completed_task': 'completed', 'failed_task': 'failed', 'abandoned_task': 'abandoned', 'broke_promise': 'broke a promise to'}

def _verb(verb: str) -> str:
    return _VERB_TEXT.get(verb, _words(verb))

def _feeling_text(r: RelationshipView, lead: str) -> str:
    parts = [f'trust {_words(r.trust)}', f'affection {_words(r.affection)}', f'fear {_words(r.fear)}']
    if r.respect:
        parts.append(f'respect {_words(r.respect)}')
    if r.familiarity:
        parts.append(f'familiarity {_words(r.familiarity)}')
    parts += [f'{_words(k)} {_words(v)}' for k, v in sorted(r.extra.items())]
    text = f"{lead}{', '.join(parts)}."
    if r.obligations:
        text += ' ' + _cap('; '.join((_OBLIGATION[o.type] + (f': {o.detail}' if o.detail else '') for o in r.obligations))) + '.'
    if r.opinion:
        text += f' Your view: {r.opinion}.'
    return text

def _person_line(p: PersonView) -> str:
    label = p.who + (f' ({_words(p.kind)})' if p.kind else '') + (' (group)' if p.is_group else '')
    return f'- {label}: ' + _feeling_text(p, '')

def event_text(e: EventView) -> str:
    verb = '' if e.verb in _SILENT_VERBS and e.detail else _verb(e.verb)
    words = [x for x in (e.by, verb, e.whom, e.what) if x]
    if e.where:
        words.append(f'at {e.where}')
    if e.detail:
        words.append(e.detail)
    return _cap(' '.join(words))

def _event_line(e: EventView) -> str:
    source = f'{e.told_by} told you' if e.source == 'told_by' else _SOURCE[e.source]
    felt = f', it felt {_FEELING[e.feeling]}' if e.feeling in _FEELING else ''
    when = _words(e.when) + (f', {e.at_time}' if e.at_time else '')
    line = f'- {event_text(e)}, {when} ({source}, {e.confidence}{felt}).'
    if e.cover_story:
        line += f' If asked, say: {e.cover_story}'
    return line

def _history_line(h) -> str:
    if isinstance(h, HistoryTold):
        return f'- You told the player: {h.said} ({_words(h.when)}).'
    actor = 'You' if h.by == 'self' else 'The player'
    if isinstance(h, HistoryPromise):
        return f'- {actor} promised {h.what} ({_words(h.when)}, {_PROMISE[h.status]}).'
    words = [actor, _verb(h.verb)] + [x for x in (h.what, h.detail) if x]
    felt = f', it felt {_FEELING[h.feeling]}' if h.feeling in _FEELING else ''
    return f"- {' '.join(words)}, {_words(h.when)}{felt}."

def _offer_line(o) -> str:
    line = f'- {o.what}' + (f': {o.price}' if o.price else '')
    if not o.available:
        line += ' (not available right now)'
    return line + (f'. {o.note}' if o.note else '')

def _presence(p) -> str:
    detail = 'can hear you' if p.can_hear else 'out of earshot'
    return f"{p.who} ({detail}{(', ' + p.doing if p.doing else '')})"

def _boundary_lines(b: BoundaryView) -> list[str]:
    if b.real_world == 'none':
        lines = ['Nothing from the real world outside this one exists here, and you have never heard of such things.']
        if b.allowed:
            lines.append('Except these, which do exist here: ' + ', '.join(b.allowed) + '.')
    elif b.real_world == 'listed':
        lines = ['Of the real world, only these exist here: ' + ', '.join(b.allowed) + '.']
    else:
        lines = ['The real world is this world.']
    if b.denied:
        lines.append('These do not exist here, and you have never heard of them: ' + ', '.join(b.denied) + '.')
    return lines

def _tell_only(names: list[str], line: str='.') -> str:
    if not names:
        return ''
    return ('' if line.rstrip().endswith(('.', '!', '?')) else '.') + f" {', '.join(names)} may hear this, not the player."

def _player_seen_lines(seen) -> list[str]:
    lines = ['The player as you see them:']
    if seen.known_as:
        lines.append(f'- Known to you as: {seen.known_as}')
    if seen.appearance:
        lines.append(f'- Looks: {seen.appearance}')
    if seen.condition:
        lines.append(f'- Condition: {seen.condition}')
    if seen.carrying:
        lines.append('- Carrying: ' + ', '.join(seen.carrying))
    if seen.reputation:
        lines.append(f'- Reputation with you: {seen.reputation}')
    lines += [f'- Their task: {t.task} ({_TASK[t.status]})' for t in seen.tasks]
    return lines if len(lines) > 1 else []

def game_note_for_action(result: ActionResult) -> str:
    detail = ', '.join((f'{_words(k)} {v}' for k, v in result.details.items()))
    reason = f' ({_words(result.reason)})' if result.reason else ''
    return _cap(f'{_words(result.action)} {result.status}{reason}' + (f': {detail}' if detail else '')) + '.'

def game_note_for_event(event: EventView) -> str:
    return _event_line(event)[2:]

def render_system(ctx: ActorContext) -> str:
    a = ctx.actor
    lines = [f'You are {a.name}' + (f', {a.role}' if a.role else '') + '.']
    lines.append(f'Voice: {a.voice}.')
    if a.values:
        lines.append('Values: ' + ', '.join(a.values) + '.')
    if a.likes:
        lines.append('Likes: ' + ', '.join(a.likes) + '.')
    if a.dislikes:
        lines.append('Dislikes: ' + ', '.join(a.dislikes) + '.')
    if a.current_goal:
        lines.append(f'Goal: {_words(a.current_goal)}.')
    if ctx.location_name:
        lines.append(f'You are at: {ctx.location_name}.')
    s = ctx.setting
    lines.append(f'World: {_words(s.knowledge_policy)}, {s.technology} technology, magic {s.magic}.')
    if s.boundary:
        lines += _boundary_lines(s.boundary)
    if ctx.story:
        lines.append(f'Story: {ctx.story.stage}.' + (f' {ctx.story.summary}' if ctx.story.summary else ''))
    if ctx.world_rules:
        lines.append('Rules of this world:')
        lines += [f'- {r}' for r in ctx.world_rules]
    if ctx.glossary:
        lines.append('Game terms:')
        lines += [f'- {_words(k)}: {v}' for k, v in sorted(ctx.glossary.items())]
    if ctx.offers:
        lines.append('What you can offer:')
        lines += [_offer_line(o) for o in ctx.offers]
    if ctx.never_discuss:
        lines.append('Never discuss these, whatever is asked:')
        lines += [f'- {t}' for t in ctx.never_discuss]
    if ctx.people:
        lines.append('People you know:')
        lines += [_person_line(p) for p in ctx.people]
    shareable = [k for k in ctx.known if k.may_disclose and (not k.private_only)]
    private = [k for k in ctx.known if k.may_disclose and k.private_only]
    secret = [k for k in ctx.known if not k.may_disclose]
    if shareable:
        lines.append('You know:')
        lines += [f'- {k.subject_name} {_words(k.predicate)}: {k.value} ({k.source_kind})' for k in shareable]
    if private:
        lines.append(PRIVATE_HEADER)
        lines += [f'- {k.subject_name} {_words(k.predicate)}: {k.value} ({k.source_kind})' for k in private]
    if secret:
        lines.append('You know, but must not tell the player:')
        for k in secret:
            line = f'- {k.subject_name} {_words(k.predicate)}: {k.value}' + (f'. If asked, say: {k.cover_story}' if k.cover_story else '')
            lines.append(line + _tell_only(k.tell_only, line))
    if ctx.beliefs:
        lines.append('You believe:')
        lines += [f'- {b.subject_name} {_words(b.predicate)}: {b.value} ({b.confidence}, {b.source_kind})' for b in ctx.beliefs]
    open_events = [e for e in ctx.events if e.may_disclose and (not e.private_only)]
    private_events = [e for e in ctx.events if e.may_disclose and e.private_only]
    secret_events = [e for e in ctx.events if not e.may_disclose]
    if open_events:
        lines.append('Things you know happened:')
        lines += [_event_line(e) for e in open_events]
    if private_events:
        lines.append('Things you know happened that you could tell the player, but not while others can hear:')
        lines += [_event_line(e) for e in private_events]
    if secret_events:
        lines.append('Things you know happened, but must not tell the player:')
        lines += [_event_line(e) + _tell_only(e.tell_only, _event_line(e)) for e in secret_events]
    if ctx.player_history or ctx.history_visits is not None or ctx.history_last_seen:
        visits = ctx.history_visits
        about = [None if visits is None else 'met once' if visits == 1 else f'met {visits} times', f'last seen {_words(ctx.history_last_seen)}' if ctx.history_last_seen else None]
        about = [a for a in about if a]
        lines.append('Your history with the player' + (f" ({', '.join(about)})" if about else '') + ':')
        lines += [_history_line(h) for h in ctx.player_history]
    if ctx.player_seen:
        lines += _player_seen_lines(ctx.player_seen)
    r = ctx.relationship_to_player
    lines.append(_feeling_text(r, 'Toward the player: ') + (f' They are your {_words(r.kind)}.' if r.kind else ''))
    if ctx.activity:
        lines.append(f'Right now you are {ctx.activity}.')
    if ctx.time_of_day:
        lines.append(f'It is {ctx.time_of_day}.')
    if ctx.situation:
        lines.append(f'Around you: {ctx.situation}.')
    if ctx.present:
        lines.append('Also here: ' + '; '.join((_presence(p) for p in ctx.present)) + '.')
    if ctx.mood:
        lines.append(f'Your mood: {_words(ctx.mood)}.')
    if ctx.action_results:
        lines.append('What just happened:')
        for res in ctx.action_results:
            detail = ', '.join((f'{_words(k)} {v}' for k, v in res.details.items()))
            reason = f' ({_words(res.reason)})' if res.reason else ''
            lines.append(f'- {_words(res.action)} {res.status}{reason}' + (f': {detail}' if detail else ''))
    lines.append(_STYLE[ctx.output_policy.response_style])
    lines.append(OUTPUT_RULE)
    return '\n'.join(lines)

def _player_line(t: Turn) -> str:
    return t.text

def _addressee(to: str | None) -> str:
    return 'the player' if to == 'player' else to

def labelled_line(t: Turn) -> str:
    who = PLAYER_LABEL if t.speaker == 'player' else t.name
    return f'{who} (to {_addressee(t.to)}): {t.text}' if t.to else f'{who}: {t.text}'

def group_scene_from(dialogue: Sequence[Turn]) -> int | None:
    return next((i for i, t in enumerate(dialogue) if t.speaker == 'other' or t.addressed_elsewhere()), None)

def render_prompt(ctx: ActorContext, dialogue: Sequence[Turn], fmt: ChatFormat=DEFAULT_FORMAT, system_extra: str='', thinking: bool | None=None) -> str:
    if not dialogue or dialogue[-1].speaker == 'actor':
        raise ValueError("dialogue must end with a player, other or game turn, not the character's own line")
    system = render_system(ctx) + (f'\n{system_extra}' if system_extra else '') + fmt.system_suffix
    parts = [f"{fmt.turn_open['system']}{fmt.system_prefix}{system}{fmt.close('system')}"] if fmt.system_role else []
    folded = not fmt.system_role
    group_from = group_scene_from(dialogue)
    for i, t in enumerate(dialogue):
        for marker in fmt.single_token_markers:
            if marker in t.text:
                raise ValueError(f'{t.speaker} turn contains the chat marker {marker}; the harness must strip it')
        labelled = group_from is not None and i >= group_from
        if t.speaker == 'game' and fmt.system_role and (not folded):
            parts.append(f"{fmt.turn_open['system']}{GAME_LABEL} {t.text}{fmt.close('system')}")
        elif t.speaker in ('player', 'game', 'other'):
            if t.speaker == 'game':
                line = f'{GAME_LABEL} {t.text}'
            elif labelled:
                line = labelled_line(t)
            else:
                line = _player_line(t)
            if folded:
                line, folded = (f'{system}\n\n{line}', False)
            parts.append(f"{fmt.turn_open['user']}{line}{fmt.close('user')}")
        else:
            parts.append(f"{fmt.turn_open['assistant']}{t.text}{fmt.close('assistant')}")
    parts.append(fmt.turn_open['assistant'])
    prompt = ''.join(parts)
    for marker in FORBIDDEN_MARKERS:
        if marker in prompt:
            raise ValueError(f'prompt contains {marker}; thinking mode is not allowed')
    if fmt.thinking:
        if thinking is None:
            raise ValueError(f'{fmt.name} can think: say thinking=True or thinking=False')
        return prompt + ('' if thinking else fmt.think_off)
    if thinking:
        raise ValueError(f'{fmt.name} has no thinking mode')
    return prompt + fmt.assistant_prefix
CHAT_TEMPLATE = "{%- for message in messages -%}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- if not (enable_thinking is defined and enable_thinking) -%}{{- '<think>\\n\\n</think>\\n\\n' -}}{%- endif -%}{%- endif -%}"

def chat_messages(ctx: ActorContext, dialogue: Sequence[Turn], fmt: ChatFormat | None=None) -> list[dict[str, str]]:
    fmt = fmt or EXPERIMENTAL_FORMATS['qwen3.5-2b-base-think']
    if not fmt.system_role or fmt.assistant_prefix or fmt.system_suffix:
        raise ValueError(f'{fmt.name}: chat messages cover formats with a system role and no vendor additions')
    if not dialogue or dialogue[-1].speaker == 'actor':
        raise ValueError("dialogue must end with a player, other or game turn, not the character's own line")
    out = [{'role': 'system', 'content': render_system(ctx)}]
    group_from = group_scene_from(dialogue)
    for i, t in enumerate(dialogue):
        for marker in (*fmt.single_token_markers, *FORBIDDEN_MARKERS):
            if marker in t.text:
                raise ValueError(f'{t.speaker} turn contains the marker {marker}; the harness must strip it')
        labelled = group_from is not None and i >= group_from
        if t.speaker == 'game':
            out.append({'role': 'system', 'content': f'{GAME_LABEL} {t.text}'})
        elif t.speaker in ('player', 'other'):
            out.append({'role': 'user', 'content': labelled_line(t) if labelled else _player_line(t)})
        else:
            out.append({'role': 'assistant', 'content': t.text})
    return out

def split_reasoning(text: str) -> tuple[str | None, str]:
    head = text.lstrip()
    if not head.startswith(FORBIDDEN_MARKERS[0]):
        return (None, text)
    body = head[len(FORBIDDEN_MARKERS[0]):]
    end = body.find(FORBIDDEN_MARKERS[1])
    if end == -1:
        return (body.strip(), '')
    return (body[:end].strip(), body[end + len(FORBIDDEN_MARKERS[1]):].lstrip())

def render_target(reply: str, fmt: ChatFormat=DEFAULT_FORMAT) -> str:
    note, spoken = split_reasoning(reply)
    if note is not None:
        if not fmt.thinking:
            raise ValueError(f'{fmt.name} has no thinking mode; the reply opens with a think block')
        if not note or any((m in note for m in (*FORBIDDEN_MARKERS, *fmt.single_token_markers))) or '\n\n' in note:
            raise ValueError('a thinking note must be one short block of plain text')
        reply = spoken
    issues = speakability_issues(reply)
    if issues:
        raise ValueError('training target is not speakable: ' + '; '.join(issues))
    if note is not None:
        return f'{fmt.think_open}{note}{fmt.think_close}{reply}{fmt.reply_end}'
    return f'{reply}{fmt.reply_end}'

def inference_ids(tokenizer, ctx: ActorContext, dialogue: Sequence[Turn], fmt: ChatFormat=DEFAULT_FORMAT, thinking: bool | None=None) -> list[int]:
    ids = tokenizer.encode(render_prompt(ctx, dialogue, fmt, thinking=thinking), add_special_tokens=False).ids
    return ([tokenizer.token_to_id(fmt.bos_token)] if fmt.bos_token else []) + ids

def training_ids(tokenizer, ctx: ActorContext, dialogue: Sequence[Turn], reply: str, fmt: ChatFormat=DEFAULT_FORMAT) -> tuple[list[int], list[int]]:
    thinking = split_reasoning(reply)[0] is not None if fmt.thinking else None
    prompt = inference_ids(tokenizer, ctx, dialogue, fmt, thinking=thinking)
    target = tokenizer.encode(render_target(reply, fmt), add_special_tokens=False).ids
    return (prompt, target)

def load_tokenizer(fmt: ChatFormat=DEFAULT_FORMAT):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(fmt.tokenizer_repo, 'tokenizer.json', revision=fmt.tokenizer_revision)
    return Tokenizer.from_file(path)
