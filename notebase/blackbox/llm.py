"""LLM-backed black boxes: Atomizer and Enricher on Claude (default: Haiku 5.5).

Both return structured outputs (``client.messages.parse`` + Pydantic), so
there is no free-text parsing. A write must never fail because of the LLM:
on an API error, a refusal, or unusable output, the box logs a warning and
falls back to its stub for that capture. The raw capture is saved before any
of this runs, so nothing the user wrote is lost.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Literal

from pydantic import BaseModel

from notebase.blackbox.stubs import FrequencyEnricher, ParagraphAtomizer
from notebase.models import Capture, Enrichment, Entity

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-5-5"

# --------------------------------------------------------------------- atomizer

ATOMIZER_SYSTEM = """\
You turn a raw capture from a personal note-taking app into atomic notes. The app \
works like an extension of the user's memory: notes have no folders or titles and \
are found later by association, so each note must stand on its own.

Rules:
- One idea per note. Split when the capture holds several distinct ideas, facts, \
decisions, questions or tasks. Keep a single thought as a single note, and never \
spread one idea across notes.
- Each note must make sense on its own when read months later without the rest of \
the capture. Resolve pronouns and vague references ("it", "that meeting", "he") \
using the capture. If the capture doesn't say what they refer to, leave them as \
written; never invent context.
- Keep the user's words, voice and language. Don't translate, summarize away \
details, add facts, or comment on the content. Keep names, numbers, URLs and \
#hashtags exactly as written.
- Light cleanup only: drop filler words and fix obvious typos.
- Keep the ideas in the order they appear."""


class _AtomizedNotes(BaseModel):
    notes: list[str]


class LLMAtomizer:
    """Split a capture into self-contained atomic notes.

    Because each note is made self-contained here, the Contextualizer can stay
    the identity stub while this box is in use.
    """

    def __init__(
        self,
        client,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        fallback=None,
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.fallback = fallback or ParagraphAtomizer()

    def atomize(self, capture: Capture) -> list[str]:
        text = capture.text.strip()
        if not text:
            return []
        try:
            parsed = _parse(
                self.client,
                model=self.model,
                effort=self.effort,
                system=ATOMIZER_SYSTEM,
                user=f"<capture>\n{text}\n</capture>",
                output_format=_AtomizedNotes,
            )
            notes = [n.strip() for n in parsed.notes if n.strip()]
            if not notes:
                raise ValueError("model returned no notes")
            return notes
        except Exception as e:  # noqa: BLE001 - any failure -> stub, never lose a write
            log.warning("LLM atomizer failed (%s); falling back to paragraph split", e)
            return self.fallback.atomize(capture)


# --------------------------------------------------------------------- enricher

ENRICHER_SYSTEM = """\
You index atomic notes from a personal note-taking app so that the user can find \
them later by association. Notes may be in any language, often Chinese or English.

For each note return:
- keywords: 5-10 words or short phrases the user might type later when trying to \
recall this note. Include the key terms it contains, plus synonyms, broader \
concepts and closely related terms it doesn't contain. If the note is in Chinese, \
add English equivalents of the main terms; if it is in English, add Chinese ones.
- tags: 1-3 broad topics, lowercase, one or two words each (for example "sleep", \
"cooking", "work"). Tags group notes into areas, so they are broader than keywords. \
Reuse a tag from <existing_tags> whenever one fits; create a new tag only when none \
does. Write tags in the language the existing tags use, or English if there are none.
- entities: specific named things the note mentions: people, places, organizations, \
projects, works (books, papers, films), events and URLs. Generic concepts are not \
entities. Use an empty list if there are none.

Return exactly one entry per note, with the note's index."""

EntityType = Literal["person", "place", "organization", "project", "work", "event", "url", "other"]


class _Entity(BaseModel):
    type: EntityType
    name: str


class _NoteIndex(BaseModel):
    index: int
    keywords: list[str]
    tags: list[str]
    entities: list[_Entity]


class _IndexedNotes(BaseModel):
    notes: list[_NoteIndex]


_HASHTAG = re.compile(r"#([\w\-]+)")


class LLMEnricher:
    """Keywords (with synonyms and zh/en equivalents), tags reused from the
    existing vocabulary, and named entities, for all notes of a capture at once."""

    def __init__(
        self,
        client,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        tag_vocabulary: Callable[[], list[str]] = lambda: [],
        max_tags: int = 3,
        fallback=None,
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.tag_vocabulary = tag_vocabulary
        self.max_tags = max_tags
        self.fallback = fallback or FrequencyEnricher()

    def enrich(self, contents: list[str]) -> list[Enrichment]:
        if not contents:
            return []
        try:
            parsed = _parse(
                self.client,
                model=self.model,
                effort=self.effort,
                system=ENRICHER_SYSTEM,
                user=_enricher_prompt(contents, self.tag_vocabulary()),
                output_format=_IndexedNotes,
            )
            by_index = {n.index: n for n in parsed.notes}
            missing = [i for i in range(len(contents)) if i not in by_index]
            if missing:
                raise ValueError(f"no index entry for notes {missing}")
            return [self._to_enrichment(contents[i], by_index[i]) for i in range(len(contents))]
        except Exception as e:  # noqa: BLE001 - any failure -> stub, never lose a write
            log.warning("LLM enricher failed (%s); falling back to frequency stub", e)
            return self.fallback.enrich(contents)

    def _to_enrichment(self, content: str, idx: _NoteIndex) -> Enrichment:
        # Hashtags the user typed are explicit intent: they always become tags.
        hashtags = [normalize_tag(h) for h in _HASHTAG.findall(content)]
        tags = _dedupe(hashtags + [normalize_tag(t) for t in idx.tags])
        tags = tags[: max(self.max_tags, len(hashtags))]
        keywords = _dedupe(k.strip() for k in idx.keywords)
        entities = []
        seen = set()
        for e in idx.entities:
            key = (e.type, e.name.strip().lower())
            if e.name.strip() and key not in seen:
                seen.add(key)
                entities.append(Entity(e.type, e.name.strip()))
        return Enrichment(keywords=keywords, tags=tags, entities=entities)


def _enricher_prompt(contents: list[str], vocabulary: list[str]) -> str:
    tags = ", ".join(vocabulary) if vocabulary else "(none yet)"
    notes = "\n".join(
        f'<note index="{i}">\n{c}\n</note>' for i, c in enumerate(contents)
    )
    return f"<existing_tags>{tags}</existing_tags>\n\n<notes>\n{notes}\n</notes>"


def normalize_tag(tag: str) -> str:
    tag = tag.strip().lstrip("#").lower()
    return re.sub(r"[\s_]+", "-", tag)


def _dedupe(items) -> list[str]:
    out, seen = [], set()
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ----------------------------------------------------------------------- shared


class LLMRefusal(RuntimeError):
    pass


def _parse(client, *, model, effort, system, user, output_format):
    response = client.messages.parse(
        model=model,
        max_tokens=16000,  # thinking counts toward max_tokens on Haiku 5.5
        output_config={"effort": effort},
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=output_format,
    )
    if response.stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        raise LLMRefusal(f"model declined (category={category})")
    if response.stop_reason == "max_tokens":
        raise ValueError("output truncated at max_tokens")
    if response.parsed_output is None:
        raise ValueError("no parsed output")
    return response.parsed_output


def build_client():
    """Anthropic client from the environment.

    Uses ANTHROPIC_API_KEY (or any credential the SDK resolves on its own, such
    as an ``ant auth login`` profile); CLAUDE_API_KEY is accepted as an alias.
    """
    import os

    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_API_KEY"):
        return anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
    return anthropic.Anthropic()
