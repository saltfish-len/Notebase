import hashlib

import pytest

from notebase import Notebase
from notebase.blackbox.stubs import FilenameAltTextNormalizer, alt_text_marker
from notebase.models import Attachment, Capture
from notebase.storage import InMemoryNoteStore


def att(name: str, media_type: str) -> Attachment:
    return Attachment(name=name, media_type=media_type)


@pytest.mark.parametrize(
    "attachment, marker",
    [
        (
            att("whiteboard_meeting-notes.jpg", "image/jpeg"),
            "[image: whiteboard meeting notes (whiteboard_meeting-notes.jpg)]",
        ),
        (att("IMG_2031.HEIC", "image/heic"), "[image: IMG 2031 (IMG_2031.HEIC)]"),
        (att("standup.mp3", "audio/mpeg"), "[audio: standup (standup.mp3)]"),
        (att("论文 草稿.pdf", "application/pdf"), "[document: 论文 草稿 (论文 草稿.pdf)]"),
        (att("README", "application/octet-stream"), "[file: README]"),
    ],
)
def test_alt_text_marker_comes_from_the_file_name(attachment, marker):
    assert alt_text_marker(attachment) == marker


def test_text_only_capture_is_unchanged():
    capture = Capture(text="  Deep sleep consolidates memory.  ")
    assert FilenameAltTextNormalizer().normalize(capture) == "Deep sleep consolidates memory."


def test_files_stay_in_the_paragraph_they_were_written_with():
    capture = Capture(
        text="First idea.\n\nThe whiteboard from today's planning meeting.",
        attachments=[att("board.jpg", "image/jpeg"), att("plan.pdf", "application/pdf")],
    )
    assert FilenameAltTextNormalizer().normalize(capture) == (
        "First idea.\n\nThe whiteboard from today's planning meeting.\n"
        "[image: board (board.jpg)]\n[document: plan (plan.pdf)]"
    )


def test_files_without_text_become_one_paragraph_each():
    capture = Capture(text="", attachments=[att("a.png", "image/png"), att("b.mp3", "audio/mpeg")])
    assert FilenameAltTextNormalizer().normalize(capture) == (
        "[image: a (a.png)]\n\n[audio: b (b.mp3)]"
    )


def test_attachment_from_path(tmp_path):
    f = tmp_path / "sunset_at_the_pier.png"
    f.write_bytes(b"\x89PNG fake")
    a = Attachment.from_path(f)
    assert a.name == "sunset_at_the_pier.png" and a.media_type == "image/png"
    assert a.kind == "image" and a.size == 9 and a.path == str(f)
    assert a.sha256 == hashlib.sha256(b"\x89PNG fake").hexdigest()


@pytest.fixture
def app():
    return Notebase.with_stubs(InMemoryNoteStore(), embedding_dim=128)


def test_write_with_a_file_keeps_caption_and_file_in_one_note(app):
    board = att("whiteboard_sprint_plan.jpg", "image/jpeg")
    result = app.write("Sprint plan we agreed on.", attachments=[board])
    assert len(result.notes) == 1
    assert result.notes[0].content == (
        "Sprint plan we agreed on.\n[image: whiteboard sprint plan (whiteboard_sprint_plan.jpg)]"
    )
    # the capture keeps the typed text and file reference verbatim, plus what the boxes saw
    stored = app._write.store.captures[result.capture.id]
    assert stored.text == "Sprint plan we agreed on."
    assert stored.attachments == [board]
    assert stored.normalized == result.notes[0].content


def test_file_only_write_is_findable_by_its_name(app):
    app.write("Fresh pasta needs only flour and eggs.")
    note = app.write(attachments=[att("whiteboard_sprint_plan.jpg", "image/jpeg")]).notes[0]
    assert app.find("whiteboard").seeds[0] == note.id


def test_cli_rejects_a_missing_file(tmp_path, capsys):
    from notebase.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["write", "-a", str(tmp_path / "nope.jpg")])
    assert exc.value.code == 2
    assert "cannot attach" in capsys.readouterr().err
