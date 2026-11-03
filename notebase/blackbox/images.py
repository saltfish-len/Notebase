"""Prepare an image file so the Claude API accepts it.

The API takes JPEG, PNG, GIF and WebP up to 5 MB, and downscales anything
whose long edge is over about 1568 px anyway. Images that are already fine
are sent as they are. Everything else (HEIC from iPhones, TIFF, BMP, very
large photos) is decoded with Pillow, rotated upright from its EXIF data,
shrunk to that size and re-encoded as JPEG (PNG when it has transparency).
"""

from __future__ import annotations

import io
from pathlib import Path

API_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_BYTES = 5 * 1024 * 1024
MAX_EDGE = 1568


class UnsupportedImage(ValueError):
    pass


def prepare_image(path: str | Path, media_type: str) -> tuple[str, bytes]:
    """Return ``(media_type, bytes)`` ready for an ``image`` content block."""
    data = Path(path).read_bytes()
    try:
        from PIL import Image, ImageOps
    except ImportError:  # pragma: no cover - Pillow is a dependency
        if media_type in API_MEDIA_TYPES and len(data) <= MAX_BYTES:
            return media_type, data
        raise UnsupportedImage(f"{media_type} needs Pillow to convert") from None
    _register_heif()

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as e:  # noqa: BLE001 - any decoder error means "can't use it"
        raise UnsupportedImage(f"cannot decode {media_type}: {e}") from e

    small_enough = max(image.size) <= MAX_EDGE and len(data) <= MAX_BYTES
    if media_type in API_MEDIA_TYPES and small_enough:
        return media_type, data

    image = ImageOps.exif_transpose(image)
    image.thumbnail((MAX_EDGE, MAX_EDGE))
    out = io.BytesIO()
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image.save(out, format="PNG", optimize=True)
        return "image/png", out.getvalue()
    image.convert("RGB").save(out, format="JPEG", quality=85)
    return "image/jpeg", out.getvalue()


def _register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return
    register_heif_opener()
