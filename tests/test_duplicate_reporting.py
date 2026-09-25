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
@pytest.mark.parametrize("texts,query_text,canonical_id", [("AAA", "A", 0), ("AAA", "B", 0), ("AAB", "C", 2)])
@pytest.mark.parametrize("columns", [["text"], ["text", "context"]])
def test_cross_dataset_reports_one_canonical(
    angular_model: Encoder, backend: str, texts: str, query_text: str, canonical_id: int, columns: list[str]
) -> None:
    """Exact and near matches report just the best reference canonical, including its metadata."""
    records = [{"id": i, "text": text, "context": "C", "metadata": [i]} for i, text in enumerate(texts)]
    queries = [{"id": i, "text": query_text, "context": "C"} for i in (3, 4)]
    semhash = SemHash.from_records(records, columns=columns, model=angular_model, ann_backend=backend)
    result = semhash.deduplicate(queries, threshold=0.6)
    assert result.selected == []
    for duplicate, query in zip(result.filtered, queries):
        assert duplicate.record == query
        assert duplicate.exact is (query_text == "A")
        assert len(duplicate.duplicates) == 1
        canonical, score = duplicate.duplicates[0]
        assert canonical == records[canonical_id]
        assert score >= 0.6
    result.rethreshold(0.995)
    assert result.selected == ([] if query_text == "A" else queries)


@pytest.mark.parametrize("backend", ["basic", "usearch"])
@pytest.mark.parametrize("from_embeddings", [False, True])
def test_self_canonicals_preserve_records_and_rethreshold(
    angular_model: Encoder, backend: str, from_embeddings: bool
) -> None:
    """Every occurrence points directly to a kept canonical, including after semantic groups split."""
    records = [{"id": i, "text": text, "metadata": [i]} for i, text in enumerate(["A", "B", "B", "C"])]
    if from_embeddings:
        semhash = SemHash.from_embeddings(
            angular_model.encode([record["text"] for record in records]),
            records,
            model=angular_model,
            columns=["text"],
            ann_backend=backend,
        )
    else:
        semhash = SemHash.from_records(records, model=angular_model, columns=["text"], ann_backend=backend)
    result = semhash.self_deduplicate(0.6)
    assert result.selected == records[:1]
    for threshold in (0.6, 0.95, 0.99):
        result.rethreshold(threshold)
        assert result == semhash.self_deduplicate(threshold)
        assert sorted(result.selected + [d.record for d in result.filtered], key=lambda r: r["id"]) == records
        reconstructed = [r for g in result.selected_with_duplicates for r in [g.record] + [d for d, _ in g.duplicates]]
        assert sorted(reconstructed, key=lambda r: r["id"]) == records
        for duplicate in result.filtered:
            assert len(duplicate.duplicates) == 1
            canonical, score = duplicate.duplicates[0]
            assert canonical in result.selected
            assert score >= threshold
            vectors = angular_model.encode([duplicate.record["text"], canonical["text"]])
            # USEARCH may quantize vectors (e.g. BF16), unlike BASIC's full-precision cosine.
            tolerance = 0.01 if backend == "usearch" else 1e-6
            assert score == pytest.approx(float(vectors[0] @ vectors[1]), abs=tolerance)
            assert duplicate.exact is (duplicate.record["text"] == canonical["text"])
    assert [r["id"] for r in result.selected] == [0, 1, 3]
    assert result.selected_with_duplicates[1].duplicates == [(records[2], 1.0)]


@pytest.mark.parametrize(
    "texts,threshold,selected,links",
    [
        ("ABC", 0.7, ["A", "C"], [("B", "A")]),
        ("ACB", 0.75, ["A", "C"], [("B", "C")]),
        ("ABC", 0.6, ["A"], [("B", "A"), ("C", "A")]),
    ],
)
def test_near_canonical_requires_a_direct_best_match(
    angular_model: Encoder, texts: str, threshold: float, selected: list[str], links: list[tuple[str, str]]
) -> None:
    """Near matches use the best kept canonical, never transitive links through filtered records."""
    semhash = SemHash.from_records(list(texts), model=angular_model, ann_backend="basic")
    result = semhash.self_deduplicate(threshold)
    assert result.selected == selected
    assert all(len(d.duplicates) == 1 for d in result.filtered)
    assert [(d.record, d.duplicates[0][0]) for d in result.filtered] == links


def test_large_exact_groups_have_linear_reporting(angular_model: Encoder) -> None:
    """Self/cross results, grouping, edge inspection and rethresholding avoid all-pairs allocation."""
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
        assert result.get_least_similar_from_duplicates(3) == [("A", "A", 1.0)] * 3
        result.rethreshold(0.99)
        cross.rethreshold(0.99)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 8 * 1024 * 1024
    finally:
        tracemalloc.stop()
