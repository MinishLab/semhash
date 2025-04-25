import pytest

import semhash
import semhash.version
from semhash.datamodels import DeduplicationResult, DuplicateRecord


def test_deduplication_scoring() -> None:
    """Test the deduplication scoring."""
    d = DeduplicationResult(
        ["a", "b", "c"],
        [DuplicateRecord("a", False, [("b", 0.9)]), DuplicateRecord("b", False, [("c", 0.8)])],
        0.8,
    )
    assert d.duplicate_ratio == 0.4


def test_deduplication_scoring_exact() -> None:
    """Test the deduplication scoring."""
    d = DeduplicationResult(
        ["a", "b", "c"],
        [DuplicateRecord("a", True, [("b", 0.9)]), DuplicateRecord("b", False, [("c", 0.8)])],
        0.8,
    )
    assert d.exact_duplicate_ratio == 0.2


def test_deduplication_scoring_exact_empty() -> None:
    """Test the deduplication scoring."""
    d = DeduplicationResult([], [], 0.8)
    assert d.exact_duplicate_ratio == 0.0


def test_deduplication_scoring_empty() -> None:
    """Test the deduplication scoring."""
    d = DeduplicationResult([], [], 0.8)
    assert d.duplicate_ratio == 0.0


def test_rethreshold() -> None:
    """Test rethresholding the duplicates."""
    d = DuplicateRecord("a", False, [("b", 0.9), ("c", 0.8)])
    d._rethreshold(0.85)
    assert d.duplicates == [("b", 0.9)]


def test_rethreshold_empty() -> None:
    """Test rethresholding the duplicates."""
    d = DuplicateRecord("a", False, [])
    d._rethreshold(0.85)
    assert d.duplicates == []


def test_get_least_similar_from_duplicates() -> None:
    """Test getting the least similar duplicates."""
    d = DeduplicationResult(
        ["a", "b", "c"],
        [DuplicateRecord("a", False, [("b", 0.9), ("c", 0.7)]), DuplicateRecord("b", False, [("c", 0.8)])],
        0.8,
    )
    result = d.get_least_similar_from_duplicates(1)
    assert result == [("a", "c", 0.7)]


def test_get_least_similar_from_duplicates_empty() -> None:
    """Test getting the least similar duplicates."""
    d = DeduplicationResult([], [], 0.8)
    assert d.get_least_similar_from_duplicates(1) == []


def test_rethreshold_deduplication_result() -> None:
    """Test rethresholding the duplicates."""
    d = DeduplicationResult(
        ["a", "b", "c"],
        [
            DuplicateRecord("d", False, [("x", 0.9), ("y", 0.8)]),
            DuplicateRecord("e", False, [("z", 0.8)]),
        ],
        0.8,
    )
    d.rethreshold(0.85)
    assert d.filtered == [DuplicateRecord("d", False, [("x", 0.9)])]
    assert d.selected == ["a", "b", "c", "e"]


def test_rethreshold_exception() -> None:
    """Test rethresholding throws an exception."""
    d = DeduplicationResult(
        ["a", "b", "c"],
        [
            DuplicateRecord("d", False, [("x", 0.9), ("y", 0.8)]),
            DuplicateRecord("e", False, [("z", 0.8)]),
        ],
        0.7,
    )
    with pytest.raises(ValueError):
        d.rethreshold(0.6)


def test_deprecation_deduplicated_duplicates() -> None:
    """Test deprecation warnings for deduplicated and duplicates fields."""
    if semhash.version.__version__ < "0.3.0":
        with pytest.warns(DeprecationWarning):
            d = DeduplicationResult(
                deduplicated=["a", "b", "c"],
                duplicates=[
                    DuplicateRecord("d", False, [("x", 0.9), ("y", 0.8)]),
                    DuplicateRecord("e", False, [("z", 0.8)]),
                ],
                threshold=0.8,
            )
    else:
        raise ValueError("deprecate `deduplicated` and `duplicates` fields in `DeduplicationResult`")
    assert d.selected == ["a", "b", "c"]
    assert d.filtered == [
        DuplicateRecord("d", False, [("x", 0.9), ("y", 0.8)]),
        DuplicateRecord("e", False, [("z", 0.8)]),
    ]
