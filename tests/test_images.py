from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from PIL import Image

from semhash import SemHash


@pytest.mark.parametrize("backend", ["basic", "usearch"])
@pytest.mark.parametrize("constructor", ["from_records", "from_embeddings"])
@pytest.mark.parametrize("operation", ["self", "cross"])
def test_image_sizes_are_not_exact_duplicates(backend: str, constructor: str, operation: str) -> None:
    """Images with different sizes reach the encoder despite having identical raw bytes."""

    class RowEncoder:
        def encode(self, inputs: Sequence[Any], **kwargs: Any) -> np.ndarray:
            return np.asarray([[np.asarray(image)[0].mean(), np.asarray(image)[-1].mean()] for image in inputs])

    square = Image.frombytes("L", (2, 2), bytes([255, 0, 255, 0]))
    portrait = Image.frombytes("L", (1, 4), square.tobytes())
    records = [{"id": 0, "image": square}, {"id": 1, "image": portrait}, {"id": 2, "image": square.copy()}]
    model = RowEncoder()
    reference_records = records if operation == "self" else records[:1]
    kwargs: dict[str, Any] = {"columns": ["image"], "model": model, "ann_backend": backend}
    if constructor == "from_embeddings":
        kwargs["embeddings"] = model.encode([record["image"] for record in reference_records])
    semhash = getattr(SemHash, constructor)(records=reference_records, **kwargs)
    result = (
        semhash.self_deduplicate(threshold=0.9)
        if operation == "self"
        else semhash.deduplicate(records[1:], threshold=0.9)
    )
    assert [record["id"] for record in result.selected] == ([0, 1] if operation == "self" else [1])
    assert [(dup.record["id"], dup.duplicate_of["id"], dup.exact) for dup in result.filtered] == [(2, 0, True)]


@pytest.mark.parametrize("backend", ["basic", "usearch"])
@pytest.mark.parametrize("constructor", ["from_records", "from_embeddings"])
@pytest.mark.parametrize("operation", ["self", "cross"])
@pytest.mark.parametrize(
    "difference", ["mode", "palette", "palette-alpha", "transparent-index", "transparent-table", "rgb-transparency"]
)
def test_image_colors_are_not_exact_duplicates(backend: str, constructor: str, operation: str, difference: str) -> None:
    """Color interpretation participates in exact matching, while true copies still match."""

    class ColorEncoder:
        def encode(self, inputs: Sequence[Any], **kwargs: Any) -> np.ndarray:
            return np.asarray([np.asarray(image.convert("RGBA")).reshape(-1) for image in inputs], dtype=np.float32)

    if difference in ("mode", "rgb-transparency"):
        first = Image.new("RGB", (1, 1), (255, 0, 0))
        second = first.copy()
        if difference == "mode":
            second = Image.frombytes("HSV", first.size, first.tobytes())
        else:
            second.info["transparency"] = (255, 0, 0)
    else:
        first = Image.new("P", (1, 1))
        first.putpalette([255, 0, 0, 255], rawmode="RGBA")
        second = first.copy()
        if difference == "palette":
            second.putpalette([0, 255, 0, 255], rawmode="RGBA")
        elif difference == "palette-alpha":
            second.putpalette([255, 0, 0, 0], rawmode="RGBA")
        else:
            second.info["transparency"] = 0 if difference == "transparent-index" else b"\0"

    assert first.tobytes() == second.tobytes()
    copy = first.copy()
    copy.info["comment"] = "Unrelated metadata does not change the pixels."
    records = [{"id": 0, "image": first}, {"id": 1, "image": second}, {"id": 2, "image": copy}]
    model = ColorEncoder()
    reference_records = records if operation == "self" else records[:1]
    kwargs: dict[str, Any] = {"columns": ["image"], "model": model, "ann_backend": backend}
    if constructor == "from_embeddings":
        kwargs["embeddings"] = model.encode([record["image"] for record in reference_records])
    semhash = getattr(SemHash, constructor)(records=reference_records, **kwargs)
    result = (
        semhash.self_deduplicate(threshold=0.95)
        if operation == "self"
        else semhash.deduplicate(records[1:], threshold=0.95)
    )
    assert [record["id"] for record in result.selected] == ([0, 1] if operation == "self" else [1])
    assert [(dup.record["id"], dup.duplicate_of["id"], dup.exact) for dup in result.filtered] == [(2, 0, True)]
