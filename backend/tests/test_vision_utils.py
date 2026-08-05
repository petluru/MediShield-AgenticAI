import base64
import io

import pytest
from PIL import Image

from backend.agents.vision_utils import IMAGE_PROCESSING_ERROR_FLAG, ImageProcessingError, encode_image


def _write_png(path, size=(50, 50), color=(200, 50, 50)):
    Image.new("RGB", size, color).save(path, format="PNG")


def test_encode_image_returns_a_valid_content_block(tmp_path):
    path = tmp_path / "doc.png"
    _write_png(path)

    block = encode_image(path)

    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(block["source"]["data"])  # round-trips cleanly


def test_encode_image_raises_on_corrupted_file(tmp_path):
    # Mirrors the real dataset bug: a PNG file truncated mid-write.
    path = tmp_path / "corrupted.png"
    good = tmp_path / "good.png"
    _write_png(good)
    raw = good.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])

    with pytest.raises(ImageProcessingError):
        encode_image(path)


def test_encode_image_raises_on_unsupported_extension(tmp_path):
    path = tmp_path / "doc.gif"
    path.write_bytes(b"not-a-real-gif")

    with pytest.raises(ImageProcessingError):
        encode_image(path)


def test_encode_image_shrinks_an_oversized_valid_image(tmp_path):
    # A real 8MB+ PNG (the dataset's actual oversized case) is slow to
    # construct here; a large-enough synthetic image reliably exceeds the
    # base64 limit once saved uncompressed, which is all this test needs.
    path = tmp_path / "huge.png"
    Image.new("RGB", (4000, 4000), (10, 20, 30)).save(path, format="PNG", compress_level=0)
    assert len(base64.standard_b64encode(path.read_bytes())) > 9_000_000

    block = encode_image(path)

    assert len(block["source"]["data"]) <= 9_000_000
    assert block["source"]["media_type"] == "image/jpeg"
    # Still a real, decodable image after the downscale/re-encode.
    Image.open(io.BytesIO(base64.standard_b64decode(block["source"]["data"]))).load()


def test_image_processing_error_flag_is_a_stable_string():
    # Regression guard: pipeline.py and orchestrator.py match on this exact
    # string across two separate files — if it ever changes, both call
    # sites must change together.
    assert IMAGE_PROCESSING_ERROR_FLAG == "image_processing_error"
