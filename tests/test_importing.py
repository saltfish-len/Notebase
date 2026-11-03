from pathlib import Path

import pytest

from notebase import Notebase
from notebase.importing import Vault, plan_import, prepare, render_markdown
from notebase.storage import InMemoryNoteStore


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "board.png").write_bytes(b"\x89PNG board")
    (tmp_path / "loose.jpg").write_bytes(b"jpg")
    return tmp_path


def render(text: str, vault: Path) -> tuple[str, list[Path]]:
    return render_markdown(text, vault / "note.md", Vault([vault]))


@pytest.mark.parametrize(
    "source, shown",
    [
        ("See [[Sleep]].", "See Sleep."),
        ("See [[Sleep|deep sleep]].", "See deep sleep."),
        ("See [[Sleep#Stages]].", "See Sleep > Stages."),
        ("See [[Sleep#Stages#REM|REM]].", "See REM."),
        ("See [[Sleep#Stages#REM]].", "See Sleep > Stages > REM."),
        ("See [[#Stages]].", "See Stages."),
        ("See [[Sleep#^a1b2]].", "See Sleep > ^a1b2."),
        ("| [[Sleep\\|deep sleep]] |", "| deep sleep |"),  # alias pipe escaped inside a table
        ("See [[folder/Sleep]].", "See folder/Sleep."),
        ("![[Other note]]", "Other note"),  # a transcluded note is imported on its own
        ("![[missing.pdf]]", "missing.pdf"),
        ("Read [the notes](Sleep.md) and [this](Sleep%20log.md#Day).", "Read the notes and this."),
        ("[Anthropic](https://www.anthropic.com)", "[Anthropic](https://www.anthropic.com)"),
        ("Before %%hidden\nthought%% after", "Before  after"),
    ],
)
def test_links_render_as_obsidian_shows_them(vault, source, shown):
    assert render(source, vault) == (shown, [])


def test_front_matter_is_dropped(vault):
    text, _ = render("---\ntags: [sleep]\naliases: [rest]\n---\n# Sleep\nBody", vault)
    assert text == "# Sleep\nBody"


def test_code_is_left_as_written(vault):
    source = "Use `[[not a link]]`.\n\n```\n[[also not]] %%kept%%\n```\nBut [[Real|this]]."
    text, _ = render(source, vault)
    assert text == "Use `[[not a link]]`.\n\n```\n[[also not]] %%kept%%\n```\nBut this."


def test_embedded_images_become_attachments(vault):
    text, files = render(
        "Sprint plan:\n![[board.png|300]]\n![alt](assets/board.png)\n![x](loose.jpg)", vault
    )
    assert text == "Sprint plan:\n![[board.png]]\n![[board.png]]\n![[loose.jpg]]"
    assert files == [vault / "assets" / "board.png", vault / "loose.jpg"]


def test_embedded_images_are_not_imported_twice(vault):
    (vault / "plan.md").write_text("Plan\n\n![[board.png]]", encoding="utf-8")
    files, _ = plan_import([str(vault)])
    assert sorted(f.name for f in files) == ["loose.jpg", "plan.md"]
    item = prepare(vault / "plan.md", Vault.for_paths([str(vault)]))
    assert item.text == "# plan\n\nPlan\n\n![[board.png]]" and item.attachments == [vault / "assets" / "board.png"]
    assert item.fingerprint.startswith("file:")


def test_prepare_keeps_an_existing_heading_and_tags(vault):
    (vault / "a.md").write_text("# Coffee\n#drink brew at 93 °C", encoding="utf-8")
    (vault / "b.md").write_text("#drink brew at 93 °C", encoding="utf-8")
    v = Vault.for_paths([str(vault)])
    assert prepare(vault / "a.md", v).text == "# Coffee\n#drink brew at 93 °C"
    assert prepare(vault / "b.md", v).text == "# b\n\n#drink brew at 93 °C"  # a tag isn't a heading


def test_write_with_a_seen_fingerprint_is_a_no_op():
    store = InMemoryNoteStore()
    app = Notebase.with_stubs(store)
    first = app.write("A thought.", fingerprint="file:abc")
    again = app.write("A thought, edited.", fingerprint="file:abc")
    assert again.duplicate and again.capture.id == first.capture.id and again.notes == []
    assert store.count() == 1 and len(store.captures) == 1


def test_fingerprint_is_set_only_after_the_write_finishes():
    store = InMemoryNoteStore()
    app = Notebase.with_stubs(store)

    def boom(*_):
        raise RuntimeError("crash mid-write")

    app._write.embedder.embed_documents = boom
    with pytest.raises(RuntimeError):
        app.write("A thought.", fingerprint="file:abc")
    assert store.find_capture("file:abc") is None  # so a retry writes it


def test_an_embedded_image_is_described_where_it_was_embedded(vault):
    (vault / "plan.md").write_text(
        "# Sprint\n\nThe whiteboard:\n![[board.png]]\n\nShip by Friday.", encoding="utf-8"
    )
    from notebase.models import Attachment

    store = InMemoryNoteStore()
    item = prepare(vault / "plan.md", Vault.for_paths([str(vault)]))
    result = Notebase.with_stubs(store).write(
        item.text, attachments=[Attachment.from_path(a) for a in item.attachments]
    )
    assert result.capture.normalized == (
        "# Sprint\n\nThe whiteboard:\n[image: board (board.png)]\n\nShip by Friday."
    )
