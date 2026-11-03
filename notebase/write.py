"""The *write* pipeline: raw input -> atomic, enriched, embedded, linked notes."""

from __future__ import annotations

from dataclasses import dataclass

from notebase.blackbox import Atomizer, Contextualizer, Embedder, Enricher, Linker, Normalizer
from notebase.models import Attachment, Capture, Note, WriteResult
from notebase.storage.base import NoteStore


@dataclass
class WritePipeline:
    store: NoteStore
    normalizer: Normalizer
    atomizer: Atomizer
    contextualizer: Contextualizer
    enricher: Enricher
    embedder: Embedder
    linker: Linker

    def write(
        self,
        text: str,
        source: str = "typed",
        attachments: list[Attachment] | None = None,
        fingerprint: str | None = None,
    ) -> WriteResult:
        # 0. the same source was written before (e.g. a file imported twice): do nothing
        if fingerprint and (earlier := self.store.find_capture(fingerprint)):
            return WriteResult(capture=earlier, notes=[], duplicate=True)

        # 1. capture: keep the raw input (typed text + file references) verbatim
        capture = Capture(text=text, source=source, attachments=list(attachments or []))
        self.store.save_capture(capture)

        # 2. normalize: attachments -> text, so every later box works on text
        capture.normalized = self.normalizer.normalize(capture)
        self.store.save_capture(capture)

        # 3. atomize  4. contextualize
        contents = [
            self.contextualizer.contextualize(fragment, capture)
            for fragment in self.atomizer.atomize(capture)
        ]
        # 5. enrich  6. embed (one batch per capture each)
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
            # 7. link (against the store + earlier notes from this same capture)
            links = self.linker.link(note, self.store, siblings=list(notes))
            # 8. store
            self.store.save_note(note)
            self.store.add_links(note.id, links)
            note.links = list(self.store.get_notes([note.id])[note.id].links)
            notes.append(note)

        # 9. mark the capture finished, so a repeat of this source is skipped from now on
        if fingerprint:
            capture.fingerprint = fingerprint
            self.store.save_capture(capture)
        return WriteResult(capture=capture, notes=notes)
