"""Behaviour of the real embedding model (downloads ~1.2 GB on first run).

    NOTEBASE_TEST_MODEL=1 pytest tests/test_embedding_model.py
"""

import os

import pytest

from notebase import Notebase
from notebase.storage import InMemoryNoteStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("NOTEBASE_TEST_MODEL"), reason="set NOTEBASE_TEST_MODEL=1"
)

NOTES = {
    "sleep": "Deep sleep consolidates memory in the hippocampus.",
    "coffee": "Coffee in the afternoon hurts deep sleep.",
    "names": "I keep forgetting names of people I just met.",
    "srs": "Spaced repetition helps long-term retention.",
    "pasta": "Fresh pasta needs only flour and eggs.",
    "budget": "Quarterly budget review is moved to Friday.",
    "zh_coffee": "下午喝咖啡会影响深度睡眠。",
    "zh_pasta": "意大利面只需要面粉和鸡蛋。",
}


@pytest.fixture(scope="module")
def embedder():
    from notebase.blackbox.embedders import build_embedder

    return build_embedder("qwen3-0.6b")


@pytest.fixture(scope="module")
def world(embedder):
    store = InMemoryNoteStore()
    app = Notebase.create(store, embedder)
    ids = {key: app.write(text).notes[0].id for key, text in NOTES.items()}
    return app, store, ids


def _linked(store, a, b):
    return any(l.to == b for l in store.notes[a].links)


def test_vectors_are_normalized_and_sized(embedder):
    (v,) = embedder.embed_documents(["hello"])
    assert len(v) == embedder.dim == 1024
    assert sum(x * x for x in v) == pytest.approx(1.0, abs=1e-3)


def test_links_follow_meaning_not_words(world):
    _, store, ids = world
    assert _linked(store, ids["coffee"], ids["sleep"])
    assert _linked(store, ids["coffee"], ids["zh_coffee"])  # cross-lingual
    assert _linked(store, ids["pasta"], ids["zh_pasta"])
    assert not _linked(store, ids["pasta"], ids["sleep"])
    assert not _linked(store, ids["budget"], ids["zh_pasta"])


@pytest.mark.parametrize(
    "cue, expected",
    [
        ("how to remember things better", "srs"),
        ("that thing about cooking", "pasta"),
        ("meeting schedule", "budget"),
        ("咖啡", "zh_coffee"),
    ],
)
def test_find_by_meaning(world, cue, expected):
    app, _, ids = world
    result = app.find(cue, k=3)
    assert ids[expected] in result.seeds[:2]


def test_find_spreads_to_associations(world):
    app, _, ids = world
    result = app.find("coffee", k=1)
    assert ids["sleep"] in result.notes  # never mentions coffee
    assert ids["pasta"] not in result.notes


def test_matryoshka_truncation():
    from notebase.blackbox.embedders import build_embedder

    small = build_embedder("qwen3-0.6b", dim=256)
    assert small.dim == 256 and small.name.endswith(":256")
    assert len(small.embed_query("hi")) == 256
