"""LLM Atomizer / Enricher against a fake client (no network, no credentials)."""

from types import SimpleNamespace

import pytest

from notebase import Notebase
from notebase.blackbox.llm import (
    LLMAtomizer,
    LLMEnricher,
    _AtomizedNotes,
    _Entity,
    _IndexedNotes,
    _NoteIndex,
)
from notebase.blackbox.stubs import HashingEmbedder
from notebase.models import Capture
from notebase.storage import InMemoryNoteStore


class FakeClient:
    """Mimics ``client.messages.parse``; replies are queued per output type."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def ok(parsed):
    return SimpleNamespace(stop_reason="end_turn", parsed_output=parsed, stop_details=None)


def idx(i, keywords=("k",), tags=("t",), entities=()):
    return _NoteIndex(index=i, keywords=list(keywords), tags=list(tags), entities=list(entities))


# --------------------------------------------------------------------- atomizer


def test_atomizer_returns_model_notes_and_sends_a_proper_request():
    client = FakeClient(ok(_AtomizedNotes(notes=["Idea one.", "  ", "Idea two."])))
    notes = LLMAtomizer(client).atomize(Capture("Idea one and idea two"))
    assert notes == ["Idea one.", "Idea two."]  # blanks dropped
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_format"] is _AtomizedNotes
    assert call["output_config"] == {"effort": "medium"}
    assert "<capture>\nIdea one and idea two\n</capture>" in call["messages"][0]["content"]
    assert not {"temperature", "top_p", "top_k"} & call.keys()  # 400 on Haiku 5.5


@pytest.mark.parametrize(
    "reply",
    [
        RuntimeError("connection reset"),
        SimpleNamespace(
            stop_reason="refusal", parsed_output=None, stop_details=SimpleNamespace(category="x")
        ),
        SimpleNamespace(stop_reason="max_tokens", parsed_output=None, stop_details=None),
        ok(_AtomizedNotes(notes=[])),
    ],
    ids=["api-error", "refusal", "truncated", "empty"],
)
def test_atomizer_falls_back_to_paragraph_split(reply, caplog):
    notes = LLMAtomizer(FakeClient(reply)).atomize(Capture("first\n\nsecond"))
    assert notes == ["first", "second"]
    assert "falling back" in caplog.text


def test_atomizer_skips_the_call_for_blank_input():
    client = FakeClient()
    assert LLMAtomizer(client).atomize(Capture("   ")) == []
    assert client.calls == []


# --------------------------------------------------------------------- enricher


def test_enricher_maps_by_index_and_normalizes():
    parsed = _IndexedNotes(
        notes=[  # out of order on purpose
            idx(1, keywords=["pasta", "意大利面", "pasta"], tags=["Cooking"]),
            idx(
                0,
                keywords=["sleep", "睡眠"],
                tags=["Sleep Science", "health"],
                entities=[_Entity(type="person", name="Matthew Walker"), _Entity(type="person", name="matthew walker ")],
            ),
        ]
    )
    client = FakeClient(ok(parsed))
    out = LLMEnricher(client, tag_vocabulary=lambda: ["health", "work"]).enrich(
        ["Walker says sleep matters. #deep_sleep", "Fresh pasta."]
    )
    assert out[0].tags == ["deep-sleep", "sleep-science", "health"]  # hashtag first
    assert [e.name for e in out[0].entities] == ["Matthew Walker"]  # deduped
    assert out[1].keywords == ["pasta", "意大利面"]
    assert out[1].tags == ["cooking"]
    prompt = client.calls[0]["messages"][0]["content"]
    assert "<existing_tags>health, work</existing_tags>" in prompt
    assert '<note index="1">\nFresh pasta.\n</note>' in prompt
    assert client.calls[0]["output_config"] == {"effort": "low"}


def test_enricher_caps_generated_tags_but_keeps_all_hashtags():
    client = FakeClient(ok(_IndexedNotes(notes=[idx(0, tags=["a", "b", "c", "d"])])))
    (e,) = LLMEnricher(client, max_tags=2).enrich(["#x #y #z note"])
    assert e.tags == ["x", "y", "z"]


def test_enricher_falls_back_when_a_note_is_missing(caplog):
    client = FakeClient(ok(_IndexedNotes(notes=[idx(0)])))
    out = LLMEnricher(client).enrich(["sleep memory", "pasta flour"])
    assert len(out) == 2 and "pasta" in out[1].keywords  # stub output
    assert "no index entry" in caplog.text


# ------------------------------------------------------------- whole pipeline


def test_write_with_llm_boxes_end_to_end():
    store = InMemoryNoteStore()
    client = FakeClient(
        ok(_AtomizedNotes(notes=["Deep sleep consolidates memory.", "Coffee late hurts deep sleep."])),
        ok(_IndexedNotes(notes=[idx(0, tags=["sleep"]), idx(1, tags=["sleep", "coffee"])])),
    )
    app = Notebase.create(
        store,
        HashingEmbedder(dim=64),
        atomizer=LLMAtomizer(client),
        enricher=LLMEnricher(client, tag_vocabulary=store.top_tags),
    )
    result = app.write("deep sleep consolidates memory, and coffee late hurts it")
    assert [n.content for n in result.notes] == [
        "Deep sleep consolidates memory.",
        "Coffee late hurts deep sleep.",
    ]
    assert store.top_tags() == ["sleep", "coffee"]
    assert any(l.to == result.notes[0].id for l in result.notes[1].links)
