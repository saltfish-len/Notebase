"""Integration tests against a real MongoDB with Vector Search.

    docker compose up -d
    NOTEBASE_TEST_MONGO_URI="mongodb://localhost:27017/?directConnection=true" pytest -m mongo
"""

import os
import time
import uuid

import pytest

from notebase import Notebase

URI = os.environ.get("NOTEBASE_TEST_MONGO_URI")
pytestmark = [
    pytest.mark.mongo,
    pytest.mark.skipif(not URI, reason="NOTEBASE_TEST_MONGO_URI not set"),
]


@pytest.fixture
def store():
    from notebase.storage.mongo import MongoNoteStore

    db = f"notebase_test_{uuid.uuid4().hex[:8]}"
    s = MongoNoteStore(URI, db, embedding_dim=64)
    s.ensure_indexes(wait=True)
    yield s
    s.client.drop_database(db)


def _wait_indexed(store, n, timeout=60):
    """Vector Search indexes are eventually consistent with the collection."""
    probe = [1.0] + [0.0] * (store.embedding_dim - 1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(store.vector_search(probe, n + 5)) >= n:
            return
        time.sleep(0.5)
    raise TimeoutError


def test_write_and_find_roundtrip(store):
    app = Notebase.with_stubs(store, embedding_dim=64)
    sleep = app.write("Deep sleep consolidates memory in the hippocampus.").notes[0]
    _wait_indexed(store, 1)
    coffee = app.write("Coffee in the afternoon hurts deep sleep.").notes[0]
    app.write("Fresh pasta needs flour and eggs.")
    _wait_indexed(store, 3)

    assert any(l.to == sleep.id for l in coffee.links)
    assert any(l.to == coffee.id for l in store.get_notes([sleep.id])[sleep.id].links)

    result = app.find("coffee")
    assert result.seeds[0] == coffee.id
    assert sleep.id in result.notes


def test_mongo_refuses_a_second_embedding_model(store):
    from notebase.blackbox.stubs import HashingEmbedder
    from notebase.storage.base import EmbeddingMismatch

    Notebase.create(store, HashingEmbedder(dim=64))
    with pytest.raises(EmbeddingMismatch, match="hashing-v1:64"):
        Notebase.create(store, HashingEmbedder(dim=32))
