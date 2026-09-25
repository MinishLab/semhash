import tracemalloc
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from semhash import SemHash
from semhash.utils import Encoder


@pytest.fixture
def angular_model() -> Encoder:
    """Encode known angles so similarity thresholds do not depend on a trained model."""

    class AngularEncoder:
        def encode(self, inputs: Sequence[Any] | Any, **kwargs: Any) -> np.ndarray:
            angles = np.deg2rad([{"A": 0, "B": 40, "C": 50}[text] for text in inputs])
            return np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)

    return AngularEncoder()


@pytest.mark.parametrize("backend", ["basic", "usearch"])
def test_cross_dataset_reports_one_canonical(angular_model: Encoder, backend: str) -> None:
    """Report the best reference canonical, not every near match or exact copy."""
    records = [{"id": i, "text": text} for i, text in enumerate("ABB")]
    semhash = SemHash.from_records(records, columns=["text"], model=angular_model, ann_backend=backend)
    query = {"id": 3, "text": "C"}
    result = semhash.deduplicate([query], threshold=0.6)
    assert result.selected == []
    assert len(result.filtered[0].duplicates) == 1
    assert result.filtered[0].duplicates[0][0] == records[1]
    result.rethreshold(0.995)
    assert result.selected == [query]


@pytest.mark.parametrize(
    "texts,threshold,selected,targets",
    [("ABBC", 0.6, [0], [0, 0, 0]), ("ABC", 0.7, [0, 2], [0]), ("ACB", 0.75, [0, 1], [1])],
)
def test_self_canonicals_and_rethreshold(
    angular_model: Encoder, texts: str, threshold: float, selected: list[int], targets: list[int]
) -> None:
    """Preserve complete groups with direct canonical links, including when higher thresholds split them."""
    records = [{"id": i, "text": text, "metadata": [i]} for i, text in enumerate(texts)]
    semhash = SemHash.from_records(records, model=angular_model, columns=["text"], ann_backend="basic")
    result = semhash.self_deduplicate(threshold)
    assert [r["id"] for r in result.selected] == selected
    assert [d.duplicates[0][0]["id"] for d in result.filtered] == targets
    for cutoff in (threshold, 0.95, 0.99):
        result.rethreshold(cutoff)
        assert result == semhash.self_deduplicate(cutoff)
        reconstructed = [r for g in result.selected_with_duplicates for r in [g.record] + [d for d, _ in g.duplicates]]
        assert sorted(reconstructed, key=lambda r: r["id"]) == records
        for duplicate in result.filtered:
            assert len(duplicate.duplicates) == 1
            canonical, score = duplicate.duplicates[0]
            assert canonical in result.selected and score >= cutoff
            vectors = angular_model.encode([duplicate.record["text"], canonical["text"]])
            assert score == pytest.approx(float(vectors[0] @ vectors[1]), abs=1e-6)
            assert duplicate.exact is (duplicate.record["text"] == canonical["text"])


def test_dense_cluster_beyond_neighbor_limit(angular_model: Encoder) -> None:
    """A near-duplicate cluster larger than the ANN neighbor limit keeps a single record."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=16) + rng.normal(scale=0.05, size=(2000, 16))
    semhash = SemHash.from_embeddings(embeddings, [str(i) for i in range(2000)], model=angular_model)
    result = semhash.self_deduplicate(0.9)
    assert result.selected == ["0"]
    result.rethreshold(0.95)
    assert result.selected == ["0"]


def test_large_exact_groups_have_linear_reporting(angular_model: Encoder) -> None:
    """Self/cross results, grouping and rethresholding avoid all-pairs allocation."""
    n = 1000
    semhash = SemHash.from_records(["A"] * n, model=angular_model, ann_backend="basic")
    tracemalloc.start()
    try:
        result = semhash.self_deduplicate()
        cross = semhash.deduplicate(["A"] * n)
        assert len(result.filtered) == n - 1
        assert len(cross.filtered) == n
        assert all(d.duplicates == [("A", 1.0)] for d in result.filtered + cross.filtered)
        assert len(result.selected_with_duplicates[0].duplicates) == n - 1
        result.rethreshold(0.99)
        cross.rethreshold(0.99)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 8 * 1024 * 1024
    finally:
        tracemalloc.stop()
