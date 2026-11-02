"""Tiny text helpers used by the placeholder black boxes and the memory store."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-]*|[㐀-鿿]+")
_CJK = re.compile(r"[㐀-鿿]")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    he her him his i if in into is it its just me my no not of on or our she so some
    than that the their them then there these they this to too up us was we were what
    when where which who will with would you your about also very really thing things
    的 了 是 在 我 你 他 她 它 们 和 也 就 都 而 及 与 着 或 一个 这 那 有 没有 不""".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; CJK runs are split into character bigrams."""
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if _CJK.match(tok):
            tokens.extend(tok[i : i + 2] for i in range(max(1, len(tok) - 1)))
        else:
            tokens.append(tok.strip("'-"))
    return [t for t in tokens if t and t not in STOPWORDS and len(t) > 1]
