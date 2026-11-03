"""MongoDB-backed store: documents, vectors and graph edges in one database.

Collections
-----------
``captures``  raw input, verbatim (provenance)
``notes``     atomic notes; each document holds its ``embedding`` (vector) and
              its ``links`` (outgoing associative edges, mirrored on the target)
``meta``      ``{_id: "embedding", name, dim}``: which model wrote the vectors

Indexes
-------
* Vector Search index ``note_embedding`` on ``notes.embedding`` (cosine), with
  ``tags`` as a filter field. Used through the ``$vectorSearch`` stage, so
  this needs MongoDB Atlas, the ``mongodb/mongodb-atlas-local`` image (see
  ``docker-compose.yml``) or a Community server with ``mongot``.
* A classic text index on ``content`` + ``keywords`` for keyword search.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Iterable

from pymongo import ASCENDING, TEXT, MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from notebase.models import Capture, Entity, Link, Note, ScoredNote
from notebase.storage.base import EmbeddingMismatch

VECTOR_INDEX = "note_embedding"


class MongoNoteStore:
    def __init__(
        self,
        uri: str,
        database: str = "notebase",
        embedding_dim: int = 256,
        client: MongoClient | None = None,
    ) -> None:
        self.client = client or MongoClient(uri)
        self.db = self.client[database]
        self.captures: Collection = self.db["captures"]
        self.notes: Collection = self.db["notes"]
        self.meta: Collection = self.db["meta"]
        self.embedding_dim = embedding_dim

    # ------------------------------------------------------------------ setup
    def bind_embedding(self, name: str, dim: int) -> None:
        recorded = self.meta.find_one({"_id": "embedding"})
        if recorded is None:
            if self.notes.estimated_document_count():
                raise EmbeddingMismatch(
                    f"database {self.db.name!r} has notes but no recorded embedding model; "
                    "use a fresh NOTEBASE_DB"
                )
            self.meta.insert_one({"_id": "embedding", "name": name, "dim": dim})
        elif (recorded["name"], recorded["dim"]) != (name, dim):
            raise EmbeddingMismatch(
                f"database {self.db.name!r} holds vectors from {recorded['name']} "
                f"({recorded['dim']}-d), not {name} ({dim}-d). Vectors from different "
                "models are not comparable: use another NOTEBASE_DB or re-embed."
            )
        self.embedding_dim = dim

    def ensure_indexes(self, wait: bool = True, timeout_s: float = 120) -> None:
        self.notes.create_index([("content", TEXT), ("keywords", TEXT)], name="note_text")
        self.notes.create_index([("capture_id", ASCENDING)])
        self.notes.create_index([("links.to", ASCENDING)])

        existing = {ix["name"]: ix for ix in self.notes.list_search_indexes()}
        if VECTOR_INDEX in existing:
            fields = existing[VECTOR_INDEX].get("latestDefinition", {}).get("fields", [])
            dims = {f.get("numDimensions") for f in fields if f.get("type") == "vector"}
            if dims and dims != {self.embedding_dim}:
                raise EmbeddingMismatch(
                    f"vector index is {dims.pop()}-d but the embedder is {self.embedding_dim}-d"
                )
        else:
            self.notes.create_search_index(
                SearchIndexModel(
                    name=VECTOR_INDEX,
                    type="vectorSearch",
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": self.embedding_dim,
                                "similarity": "cosine",
                            },
                            {"type": "filter", "path": "tags"},
                        ]
                    },
                )
            )
        if wait:
            self._wait_for_vector_index(timeout_s)

    def _wait_for_vector_index(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for ix in self.notes.list_search_indexes(VECTOR_INDEX):
                if ix.get("queryable"):
                    return
            time.sleep(1)
        raise TimeoutError(f"vector index {VECTOR_INDEX!r} not queryable after {timeout_s}s")

    # ------------------------------------------------------------- write side
    def save_capture(self, capture: Capture) -> None:
        doc = asdict(capture)
        doc["_id"] = doc.pop("id")
        self.captures.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def save_note(self, note: Note) -> None:
        doc = _note_to_doc(note)
        self.notes.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def add_links(self, note_id: str, links: Iterable[Link]) -> None:
        for link in links:
            if link.to == note_id:
                continue
            self._upsert_link(note_id, link)
            self._upsert_link(link.to, Link(note_id, link.weight, link.reason))

    def _upsert_link(self, owner: str, link: Link) -> None:
        pushed = self.notes.update_one(
            {"_id": owner, "links.to": {"$ne": link.to}},
            {"$push": {"links": asdict(link)}},
        )
        if pushed.matched_count == 0:
            self.notes.update_one(
                {"_id": owner},
                {"$max": {"links.$[l].weight": link.weight}},
                array_filters=[{"l.to": link.to}],
            )

    # -------------------------------------------------------------- read side
    def get_notes(self, ids: Iterable[str]) -> dict[str, Note]:
        return {d["_id"]: _doc_to_note(d) for d in self.notes.find({"_id": {"$in": list(ids)}})}

    def vector_search(self, embedding, k, exclude_ids=(), tags=()):
        excluded = list(exclude_ids)
        stage: dict[str, Any] = {
            "index": VECTOR_INDEX,
            "path": "embedding",
            "queryVector": embedding,
            "numCandidates": max(100, (k + len(excluded)) * 10),
            "limit": k + len(excluded),
        }
        tags = list(tags)
        if tags:
            stage["filter"] = {"tags": {"$in": tags}}
        pipeline: list[dict[str, Any]] = [
            {"$vectorSearch": stage},
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
        ]
        if excluded:
            pipeline.append({"$match": {"_id": {"$nin": excluded}}})
        pipeline.append({"$limit": k})
        # Atlas reports cosine as (1 + cos) / 2; the store contract is raw cosine.
        return [
            ScoredNote(_doc_to_note(d), 2.0 * d["_score"] - 1.0)
            for d in self.notes.aggregate(pipeline)
        ]

    def keyword_search(self, terms, k, tags=()):
        terms = list(terms)
        if not terms:
            return []
        query: dict[str, Any] = {"$text": {"$search": " ".join(terms)}}
        tags = list(tags)
        if tags:
            query["tags"] = {"$in": tags}
        cursor = (
            self.notes.find(query, {"_score": {"$meta": "textScore"}})
            .sort([("_score", {"$meta": "textScore"})])
            .limit(k)
        )
        return [ScoredNote(_doc_to_note(d), d["_score"]) for d in cursor]

    def count(self) -> int:
        return self.notes.count_documents({})

    def top_tags(self, limit: int = 200) -> list[str]:
        pipeline = [
            {"$unwind": "$tags"},
            {"$group": {"_id": "$tags", "n": {"$sum": 1}}},
            {"$sort": {"n": -1, "_id": 1}},
            {"$limit": limit},
        ]
        return [d["_id"] for d in self.notes.aggregate(pipeline)]


# ------------------------------------------------------------------ mapping


def _note_to_doc(note: Note) -> dict[str, Any]:
    doc = asdict(note)
    doc["_id"] = doc.pop("id")
    return doc


def _doc_to_note(doc: dict[str, Any]) -> Note:
    return Note(
        id=doc["_id"],
        content=doc["content"],
        capture_id=doc["capture_id"],
        source=doc.get("source", "typed"),
        keywords=doc.get("keywords", []),
        tags=doc.get("tags", []),
        entities=[Entity(**e) for e in doc.get("entities", [])],
        embedding=doc.get("embedding", []),
        links=[Link(**l) for l in doc.get("links", [])],
        created_at=doc["created_at"],
    )
