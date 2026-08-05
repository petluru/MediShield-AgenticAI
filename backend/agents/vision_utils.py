"""Shared helpers for the vision-based agents (Classifier, KYC, Claims).

Bug fix (2026-08-05, see IMPLEMENTATION_CHALLENGES.md and
[[medishield-known-bugs]]): the real eval run crashed on 3
`policy_amendments` images with no graceful fallback — 1 genuinely exceeds
Anthropic's ~10MB base64 image limit, 2 are truncated/corrupted PNG files
(confirmed with PIL: "broken PNG file", "Truncated File Read" — dataset
artifacts, not something this code caused). Both failure modes are real
things a production upload endpoint will eventually see again, so
`encode_image` now handles them instead of blowing up the graph node:
oversized-but-valid images are downscaled/re-encoded as JPEG until they
fit; images PIL can't even decode raise `ImageProcessingError`, a typed
exception the pipeline nodes catch to degrade to a low-confidence,
needs-review result instead of crashing (see backend/graph/pipeline.py)."""

import base64
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

# Anthropic rejects a base64-encoded image payload over ~10MB; stay
# comfortably under that rather than probing the exact boundary.
_MAX_BASE64_BYTES = 9_000_000


class ImageProcessingError(Exception):
    """The image at this path can't be safely prepared for the API — either
    PIL can't decode it (corrupted/truncated file) or it's still too large
    to encode even after best-effort downscaling. Callers should treat this
    as a low-confidence, needs-review outcome, not an unhandled crash."""


# Shared sentinel the pipeline nodes stamp onto a degraded agent result and
# the Orchestrator checks for — a single source of truth so the two sides
# can't drift into typo'd strings that silently stop matching each other.
IMAGE_PROCESSING_ERROR_FLAG = "image_processing_error"


def encode_image(path: Path) -> dict:
    """Build an Anthropic/LangChain multimodal image content block for `path`."""
    media_type = _MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ImageProcessingError(f"Unsupported image type: {path.suffix}")

    raw = path.read_bytes()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # force full decode now so truncated files fail here, not later
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageProcessingError(f"Could not decode {path.name}: {exc}") from exc

    data = base64.standard_b64encode(raw).decode("utf-8")
    if len(data) > _MAX_BASE64_BYTES:
        raw, media_type = _shrink_to_fit(img)
        data = base64.standard_b64encode(raw).decode("utf-8")
        if len(data) > _MAX_BASE64_BYTES:
            raise ImageProcessingError(
                f"{path.name} is too large to encode even after downscaling "
                f"({len(data)} base64 bytes)"
            )

    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _shrink_to_fit(img: Image.Image) -> tuple[bytes, str]:
    """Progressively downscale + re-encode as JPEG until the base64 payload
    fits under `_MAX_BASE64_BYTES`. JPEG (not PNG) because these are scanned
    photo-like documents where lossy compression buys far more size
    reduction per quality-loss than PNG's lossless recompression can."""
    img = img.convert("RGB")
    quality = 85
    scale = 1.0
    raw = b""
    for _ in range(6):
        w, h = img.size
        resized = img.resize((max(int(w * scale), 1), max(int(h * scale), 1))) if scale < 1.0 else img
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality)
        raw = buf.getvalue()
        if len(base64.standard_b64encode(raw)) <= _MAX_BASE64_BYTES:
            return raw, "image/jpeg"
        scale *= 0.7
        quality = max(quality - 10, 40)
    return raw, "image/jpeg"
