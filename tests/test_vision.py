"""Image alt text: preparing images for the API, and the VisionNormalizer
against a fake client (no network, no credentials)."""

import base64
import io
from types import SimpleNamespace

import pytest

Image = pytest.importorskip("PIL.Image")

from notebase.blackbox.images import MAX_EDGE, UnsupportedImage, prepare_image  # noqa: E402
from notebase.blackbox.llm import VisionNormalizer, _AltText  # noqa: E402
from notebase.models import Attachment, Capture  # noqa: E402


class FakeClient:
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


def ok(alt):
    return SimpleNamespace(stop_reason="end_turn", parsed_output=_AltText(alt_text=alt), stop_details=None)


def image_file(path, size=(64, 48), mode="RGB", fmt=None):
    Image.new(mode, size, "white").save(path, format=fmt)
    return Attachment.from_path(path)


# ----------------------------------------------------------------- prepare_image


def test_small_supported_image_is_sent_unchanged(tmp_path):
    a = image_file(tmp_path / "small.png")
    media_type, data = prepare_image(a.path, a.media_type)
    assert media_type == "image/png" and data == (tmp_path / "small.png").read_bytes()


def test_large_photo_is_shrunk_to_jpeg(tmp_path):
    a = image_file(tmp_path / "big.png", size=(4000, 3000))
    media_type, data = prepare_image(a.path, a.media_type)
    out = Image.open(io.BytesIO(data))
    assert media_type == "image/jpeg" and out.format == "JPEG"
    assert max(out.size) == MAX_EDGE and out.size[0] / out.size[1] == pytest.approx(4 / 3, 0.01)


def test_large_transparent_image_stays_png(tmp_path):
    a = image_file(tmp_path / "logo.png", size=(3000, 3000), mode="RGBA")
    media_type, _ = prepare_image(a.path, a.media_type)
    assert media_type == "image/png"


def test_unsupported_format_is_converted(tmp_path):
    a = image_file(tmp_path / "scan.bmp")
    assert a.media_type == "image/bmp"
    media_type, data = prepare_image(a.path, a.media_type)
    assert media_type == "image/jpeg" and Image.open(io.BytesIO(data)).format == "JPEG"


def test_heic_from_an_iphone_is_converted(tmp_path):
    pytest.importorskip("pillow_heif").register_heif_opener()
    a = image_file(tmp_path / "IMG_2031.HEIC", fmt="HEIF")
    media_type, data = prepare_image(a.path, "image/heic")
    assert media_type == "image/jpeg" and Image.open(io.BytesIO(data)).size == (64, 48)


def test_garbage_is_rejected(tmp_path):
    f = tmp_path / "broken.jpg"
    f.write_bytes(b"not an image")
    with pytest.raises(UnsupportedImage):
        prepare_image(f, "image/jpeg")


# ------------------------------------------------------------- VisionNormalizer


def test_image_gets_a_described_marker_and_a_proper_request(tmp_path):
    a = image_file(tmp_path / "IMG_2031.png")
    client = FakeClient(ok("Whiteboard listing\nthe sprint plan [v2]: cut export."))
    text = VisionNormalizer(client).normalize(Capture("Sprint plan we agreed on.", attachments=[a]))
    assert text == (
        "Sprint plan we agreed on.\n"
        "[image: Whiteboard listing the sprint plan (v2): cut export. (IMG_2031.png)]"
    )
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5" and call["output_format"] is _AltText
    assert call["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k"} & call.keys()
    image, prompt = call["messages"][0]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
    assert base64.b64decode(image["source"]["data"]) == (tmp_path / "IMG_2031.png").read_bytes()
    assert "<user_text>\nSprint plan we agreed on.\n</user_text>" in prompt["text"]
    assert "File name: IMG_2031.png" in prompt["text"]


def test_other_files_keep_the_file_name_marker_without_a_call():
    client = FakeClient()
    pdf = Attachment(name="paper.pdf", media_type="application/pdf")
    assert VisionNormalizer(client).normalize(Capture("", attachments=[pdf])) == (
        "[document: paper (paper.pdf)]"
    )
    assert client.calls == []


@pytest.mark.parametrize(
    "reply",
    [
        RuntimeError("connection reset"),
        SimpleNamespace(stop_reason="refusal", parsed_output=None, stop_details=None),
        ok("   "),
    ],
    ids=["api-error", "refusal", "empty"],
)
def test_failed_image_falls_back_to_its_file_name(tmp_path, reply, caplog):
    a = image_file(tmp_path / "board.png")
    text = VisionNormalizer(FakeClient(reply)).normalize(Capture("", attachments=[a]))
    assert text == "[image: board (board.png)]"
    assert "using its file name" in caplog.text


def test_one_failed_image_does_not_affect_the_others(tmp_path):
    good = image_file(tmp_path / "a.png")
    missing = Attachment(name="gone.jpg", media_type="image/jpeg", path=str(tmp_path / "gone.jpg"))
    client = FakeClient(ok("A cat on a sofa."))
    text = VisionNormalizer(client).normalize(Capture("Pets", attachments=[missing, good]))
    assert text == "Pets\n[image: gone (gone.jpg)]\n[image: A cat on a sofa. (a.png)]"
    assert len(client.calls) == 1  # the missing file never reached the API
