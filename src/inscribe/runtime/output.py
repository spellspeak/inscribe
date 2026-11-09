from __future__ import annotations
import re
ALLOWED_PUNCTUATION = frozenset('.,?!;:\'"‘’“”-–—…()%$£€&')
MAX_CHARS = 600
_LIST_START = re.compile('^\\s*(?:[-–—•]\\s|\\d+[.)]\\s)')
_PARENTHETICAL = re.compile('\\(([^()]*)\\)')

def _looks_like_stage_direction(inner: str) -> bool:
    words = inner.strip(' .!?').split()
    return 1 <= len(words) <= 3 and all((w.isalpha() for w in words))

def speakability_issues(text: str) -> list[str]:
    if not text.strip():
        return ['empty line']
    issues: list[str] = []
    if text != text.strip():
        issues.append('leading or trailing whitespace')
    if '\n' in text or '\r' in text:
        issues.append('line break')
    if _LIST_START.match(text):
        issues.append('starts like a list item')
    if '  ' in text:
        issues.append('double space')
    if len(text) > MAX_CHARS:
        issues.append(f'longer than {MAX_CHARS} characters')
    bad = sorted({c for c in text if not (c.isalpha() or c.isdigit() or c == ' ' or (c in ALLOWED_PUNCTUATION))})
    if bad:
        issues.append('markup or characters not allowed: ' + ' '.join((repr(c) for c in bad)))
    if any((_looks_like_stage_direction(m) for m in _PARENTHETICAL.findall(text))):
        issues.append('looks like a stage direction in parentheses')
    if text.count('(') != text.count(')'):
        issues.append('unbalanced parentheses')
    if text.count('"') % 2 or text.count('“') != text.count('”'):
        issues.append('unbalanced quotes')
    if not any((c.isalpha() for c in text)):
        issues.append('no words')
    return issues

def is_speakable(text: str) -> bool:
    return not speakability_issues(text)
