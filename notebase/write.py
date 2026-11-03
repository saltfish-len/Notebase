"""The *write* pipeline: raw input -> atomic, enriched, embedded, linked notes."""

from __future__ import annotations

from dataclasses import dataclass

from notebase.blackbox import Atomizer, Contextualizer, Embedder, Enricher, Linker
from notebase.models import Capture, Note, WriteResult
from notebase.storage.base import NoteStore


@dataclass
class WritePipeline:
    store: NoteStore
    atomizer: Atomizer
    contextualizer: Contextualizer
    enricher: Enricher
    embedder: Embedder
    linker: Linker

    def write(self, text: str, source: str = "typed") -> WriteResult:
        # 1. capture: keep the raw input verbatim
        capture = Capture(text=text, source=source)
        self.store.save_capture(capture)

        # 2. atomize  3. contextualize
        contents = [
            self.contextualizer.contextualize(fragment, capture)
            for fragment in self.atomizer.atomize(capture)
        ]
        # 4. enrich  5. embed (one batch per capture each)
        enrichments = self.enricher.enrich(contents)
        embeddings = self.embedder.embed_documents(contents)

        notes: list[Note] = []
        for content, enrichment, embedding in zip(contents, enrichments, embeddings):
            note = Note(
                content=content,
                capture_id=capture.id,
                source=source,
                keywords=enrichment.keywords,
                tags=enrichment.tags,
                entities=enrichment.entities,
                embedding=embedding,
            )
            # 6. link (against the store + earlier notes from this same capture)
            links = self.linker.link(note, self.store, siblings=list(notes))
            # 7. store
            self.store.save_note(note)
            self.store.add_links(note.id, links)
            note.links = list(self.store.get_notes([note.id])[note.id].links)
            notes.append(note)

        return WriteResult(capture=capture, notes=notes)
