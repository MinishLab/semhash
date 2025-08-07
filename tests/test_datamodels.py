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
        columns=["text"],
    )
    assert d.duplicate_ratio == 0.4


def test_deduplication_scoring_exact() -> None:
    """Test the deduplication scoring."""
    d = DeduplicationResult(
        ["a", "b", "c"],
        [DuplicateRecord("a", True, [("b", 0.9)]), DuplicateRecord("b", False, [("c", 0.8)])],
        0.8,
        columns=["text"],
    )
    assert d.exact_duplicate_ratio == 0.2


def test_deduplication_scoring_exact_empty() -> None:
    """Test the deduplication scoring."""
    d = DeduplicationResult([], [], 0.8, columns=["text"])
    assert d.exact_duplicate_ratio == 0.0


def test_deduplication_scoring_empty() -> None:
    """Test the deduplication scoring."""
    d = DeduplicationResult([], [], 0.8, columns=["text"])
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
        columns=["text"],
    )
    result = d.get_least_similar_from_duplicates(1)
    assert result == [("a", "c", 0.7)]


def test_get_least_similar_from_duplicates_empty() -> None:
    """Test getting the least similar duplicates."""
    d = DeduplicationResult([], [], 0.8, columns=["text"])
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
        columns=["text"],
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
        columns=["text"],
    )
    with pytest.raises(ValueError):
        d.rethreshold(0.6)


def test_deprecation_deduplicated_duplicates() -> None:
    """Test deprecation warnings for deduplicated and duplicates fields."""
    if semhash.version.__version__ < "0.4.0":
        with pytest.warns(DeprecationWarning):
            d = DeduplicationResult(
                deduplicated=["a", "b", "c"],
                duplicates=[
                    DuplicateRecord("d", False, [("x", 0.9), ("y", 0.8)]),
                    DuplicateRecord("e", False, [("z", 0.8)]),
                ],
                threshold=0.8,
                columns=["text"],
            )
    else:
        raise ValueError("deprecate `deduplicated` and `duplicates` fields in `DeduplicationResult`")
    assert d.selected == ["a", "b", "c"]
    assert d.filtered == [
        DuplicateRecord("d", False, [("x", 0.9), ("y", 0.8)]),
        DuplicateRecord("e", False, [("z", 0.8)]),
    ]


def test_selected_with_duplicates_strings() -> None:
    """Test selected_with_duplicates for strings."""
    d = DeduplicationResult(
        selected=["original"],
        filtered=[
            DuplicateRecord("duplicate_1", False, [("original", 0.9)]),
            DuplicateRecord("duplicate_2", False, [("original", 0.8)]),
        ],
        threshold=0.8,
        columns=["text"],
    )

    expected = [("original", [("duplicate_1", 0.9), ("duplicate_2", 0.8)])]
    assert d.selected_with_duplicates == expected


def test_selected_with_duplicates_dicts() -> None:
    """Test selected_with_duplicates for dicts."""
    selected = {"id": 0, "text": "hello"}
    d = DeduplicationResult(
        selected=[selected],
        filtered=[
            DuplicateRecord({"id": 1, "text": "hello"}, True, [(selected, 1.0)]),
            DuplicateRecord({"id": 2, "text": "helllo"}, False, [(selected, 0.1)]),
        ],
        threshold=0.8,
        columns=["text"],
    )

    pairs = d.selected_with_duplicates
    assert len(pairs) == 1
    kept, dups = pairs[0]
    assert kept == selected
    assert {r["id"] for r, _ in dups} == {1, 2}


def test_selected_with_duplicates_multi_column() -> None:
    """Test selected_with_duplicates for multi-columns."""
    selected = {"text": "hello", "text2": "world"}
    d = DeduplicationResult(
        selected=[selected],
        filtered=[
            DuplicateRecord({"text": "hello", "text2": "world"}, True, [(selected, 1.0)]),
            DuplicateRecord({"text": "helllo", "text2": "world"}, False, [(selected, 0.1)]),
        ],
        threshold=0.8,
        columns=["text", "text2"],
    )

    pairs = d.selected_with_duplicates
    assert len(pairs) == 1
    kept, _ = pairs[0]
    assert kept == selected


def test_selected_with_duplicates_unhashable_values() -> None:
    """Test selected_with_duplicates with unhashable values in records."""
    selected = {"a": [1, 2, 3]}  # list -> unhashable value
    filtered = {"a": [1, 2, 3], "flag": True}

    d = DeduplicationResult(
        selected=[selected],
        filtered=[DuplicateRecord(filtered, exact=False, duplicates=[(selected, 1.0)])],
        threshold=0.8,
        columns=["text"],
    )

    pairs = d.selected_with_duplicates
    assert pairs == [(selected, [(filtered, 1.0)])]


def test_selected_with_duplicates_removes_internal_duplicates() -> None:
    """Test that selected_with_duplicates removes internal duplicates that have the same hash."""
    selected = {"id": 0, "text": "hello"}
    filtered = {"id": 1, "text": "hello"}

    d = DeduplicationResult(
        selected=[selected],
        filtered=[
            DuplicateRecord(filtered, exact=False, duplicates=[(selected, 0.95)]),
            DuplicateRecord(filtered, exact=False, duplicates=[(selected, 0.90)]),
        ],
        threshold=0.8,
        columns=["text"],
    )

    selected_with_duplicates = d.selected_with_duplicates

    assert len(selected_with_duplicates) == 1

    selected_record, duplicate_list = selected_with_duplicates[0]
    # Should keep the kept record unchanged
    assert selected_record == selected
    # The duplicate row must appear only once
    assert len(duplicate_list) == 1
    assert duplicate_list[0][0] == filtered
