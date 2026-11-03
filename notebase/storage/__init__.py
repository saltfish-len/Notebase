from notebase.storage.base import NoteStore
from notebase.storage.memory import InMemoryNoteStore

__all__ = ["NoteStore", "InMemoryNoteStore", "MongoNoteStore"]


def __getattr__(name: str):
    # Import lazily so the in-memory store works without pymongo installed.
    if name == "MongoNoteStore":
        from notebase.storage.mongo import MongoNoteStore

        return MongoNoteStore
    raise AttributeError(name)
