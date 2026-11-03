from datetime import datetime, timezone

import pytest

from notebase import Notebase
from notebase.importer import WIKILINK_WEIGHT, import_vault, parse_note, vault_files
from notebase.storage import InMemoryNoteStore

SLEEP = """---
tags: [neuroscience, Memory]
created: 2025-03-14
---
Deep sleep replays the day in the hippocampus.

Coffee late in the day hurts [[Deep sleep|deep sleep]]. See ![[brain.png|300]]
"""


@pytest.fixture
def app():
    return Notebase.with_stubs(InMemoryNoteStore(), embedding_dim=128)


@pytest.fixture
def vault(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "workspace.md").write_text("not a note")
    (tmp_path / "ideas").mkdir()
    (tmp_path / "ideas" / "Sleep and memory.md").write_text(SLEEP)
    (tmp_path / "ideas" / "Deep sleep.md").write_text("Slow-wave sleep is the deepest stage.")
    (tmp_path / "Naps help.md").write_text("---\ntags: x\n---\n")
    (tmp_path / "brain.png").write_bytes(b"\x89PNG fake")
    return tmp_path


def test_parse_note_strips_frontmatter_and_keeps_link_text():
    note = parse_note(SLEEP, "Sleep and memory")
    assert note.text.startswith("Sleep and memory\nDeep sleep replays")
    assert "---" not in note.text and "[[" not in note.text and "brain.png" not in note.text
    assert "hurts deep sleep." in note.text
    assert note.tags == ["neuroscience", "memory"]
    assert note.links == [("Deep sleep", "deep sleep")]
    assert note.embeds == ["brain.png"]
    assert note.created == datetime(2025, 3, 14, tzinfo=timezone.utc)


def test_parse_note_does_not_repeat_a_title_heading():
    assert parse_note("# Linear probing\nTrain a linear head.", "Linear probing").text == (
        "# Linear probing\nTrain a linear head."
    )
    assert parse_note("[[Other#Section]]", "T").links == [("Other", "Other")]


def test_vault_files_skips_hidden_folders_and_excludes(vault):
    rels = [p.relative_to(vault).as_posix() for p in vault_files(vault)]
    assert rels == ["Naps help.md", "ideas/Deep sleep.md", "ideas/Sleep and memory.md"]
    assert len(vault_files(vault, ["Naps*"])) == 2
    assert len(vault_files(vault, ["ideas/*"])) == 1


def test_import_writes_tags_dates_attachments_and_wikilinks(app, vault):
    report = import_vault(app, vault)
    assert len(report.written) == 3

    # a file with only a name is a note too: the name is the idea
    (naps,) = app.store.notes_of_capture(app.store.captures_from("obsidian:Naps help.md")[0].id)
    assert naps.content == "Naps help"

    store = app.store
    (capture,) = store.captures_from("obsidian:ideas/Sleep and memory.md")
    assert capture.source == "import" and capture.attachments[0].name == "brain.png"
    sleep_notes = store.notes_of_capture(capture.id)
    assert len(sleep_notes) == 2
    assert all({"neuroscience", "memory"} <= set(n.tags) for n in sleep_notes)
    assert all(n.created_at == datetime(2025, 3, 14, tzinfo=timezone.utc) for n in sleep_notes)

    # the wikilink joins the note that mentions it to the linked file's note, both ways
    (deep,) = store.notes_of_capture(store.captures_from("obsidian:ideas/Deep sleep.md")[0].id)
    coffee = next(n for n in sleep_notes if "Coffee" in n.content)
    # (an existing semantic edge keeps its reason; the wikilink raises its weight)
    assert any(l.to == deep.id and l.weight >= WIKILINK_WEIGHT for l in coffee.links)
    assert any(l.to == coffee.id for l in store.get_notes([deep.id])[deep.id].links)
    # the tag update kept the links mirrored: every edge exists in both directions
    every = {n.id: n for n in store.notes.values()}
    for n in every.values():
        assert all(any(b.to == n.id for b in every[l.to].links) for l in n.links)


def test_reimport_skips_unchanged_and_rewrites_changed(app, vault):
    import_vault(app, vault)
    count = app.store.count()

    again = import_vault(app, vault)
    assert again.written == [] and len(again.unchanged) == 3
    assert app.store.count() == count

    (vault / "ideas" / "Deep sleep.md").write_text("Slow-wave sleep is when growth hormone peaks.")
    third = import_vault(app, vault)
    assert third.written == ["ideas/Deep sleep.md"]
    assert len(app.store.captures_from("obsidian:ideas/Deep sleep.md")) == 2


def test_unresolved_links_are_reported(app, tmp_path):
    (tmp_path / "A.md").write_text("Points to [[Nowhere]] and [[paper.pdf]].")
    report = import_vault(app, tmp_path)
    assert report.unresolved == {"Nowhere"}  # a missing PDF isn't a missing note
    assert report.wikilinks == 0
