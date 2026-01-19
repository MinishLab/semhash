import pytest

from semhash.datamodels import DeduplicationResult, DuplicateRecord, FilterResult, SelectedWithDuplicates


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


def test_deduplication_scoring_empty() -> None:
    """Test the deduplication scoring with empty results."""
    d = DeduplicationResult([], [], 0.8, columns=["text"])
    assert d.duplicate_ratio == 0.0
    assert d.exact_duplicate_ratio == 0.0


def test_rethreshold() -> None:
    """Test rethresholding the duplicates, including empty case."""
    d = DuplicateRecord("a", False, [("b", 0.9), ("c", 0.8)])
    d._rethreshold(0.85)
    assert d.duplicates == [("b", 0.9)]

    # Empty case
    d_empty = DuplicateRecord("a", False, [])
    d_empty._rethreshold(0.85)
    assert d_empty.duplicates == []


def test_get_least_similar_from_duplicates() -> None:
    """Test getting the least similar duplicates, including empty case."""
    d = DeduplicationResult(
        ["a", "b", "c"],
        [DuplicateRecord("a", False, [("b", 0.9), ("c", 0.7)]), DuplicateRecord("b", False, [("c", 0.8)])],
        0.8,
    )
    result = d.get_least_similar_from_duplicates(1)
    assert result == [("a", "c", 0.7)]

    # Empty case
    d_empty = DeduplicationResult([], [], 0.8, columns=["text"])
    assert d_empty.get_least_similar_from_duplicates(1) == []


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


def test_selected_with_duplicates_strings() -> None:
    """Test selected_with_duplicates for strings."""
    d = DeduplicationResult(
        selected=["original"],
        filtered=[
            DuplicateRecord("duplicate_1", False, [("original", 0.9)]),
            DuplicateRecord("duplicate_2", False, [("original", 0.8)]),
        ],
        threshold=0.8,
    )

    expected = [
        SelectedWithDuplicates(
            record="original",
            duplicates=[("duplicate_1", 0.9), ("duplicate_2", 0.8)],
        )
    ]
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

    items = d.selected_with_duplicates
    assert len(items) == 1
    kept = items[0].record
    dups = items[0].duplicates
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

    items = d.selected_with_duplicates
    assert len(items) == 1
    kept = items[0].record
    assert kept == selected


def test_selected_with_duplicates_unhashable_values() -> None:
    """Test selected_with_duplicates with unhashable values in records."""
    selected = {"text": "hello", "a": [1, 2, 3]}  # list -> unhashable value
    filtered = {"text": "hello", "a": [1, 2, 3], "flag": True}

    d = DeduplicationResult(
        selected=[selected],
        filtered=[DuplicateRecord(filtered, exact=False, duplicates=[(selected, 1.0)])],
        threshold=0.8,
        columns=["text"],
    )

    items = d.selected_with_duplicates
    assert items == [SelectedWithDuplicates(record=selected, duplicates=[(filtered, 1.0)])]


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

    items = d.selected_with_duplicates
    assert len(items) == 1

    selected_record = items[0].record
    duplicate_list = items[0].duplicates
    # Should keep the kept record unchanged
    assert selected_record == selected
    # The duplicate row must appear only once
    assert len(duplicate_list) == 1
    assert duplicate_list[0][0] == filtered


def test_selected_with_duplicates_caching() -> None:
    """Test that selected_with_duplicates is properly cached."""
    d = DeduplicationResult(
        selected=["original"],
        filtered=[
            DuplicateRecord("duplicate_1", False, [("original", 0.9)]),
            DuplicateRecord("duplicate_2", False, [("original", 0.8)]),
        ],
        threshold=0.8,
    )

    # First access should compute and cache
    result1 = d.selected_with_duplicates
    # Second access should return the cached result (same object)
    result2 = d.selected_with_duplicates
    assert result1 is result2


def test_selected_with_duplicates_cache_invalidation_on_rethreshold() -> None:
    """Test that rethreshold invalidates the selected_with_duplicates cache."""
    d = DeduplicationResult(
        selected=["original"],
        filtered=[
            DuplicateRecord("duplicate_1", False, [("original", 0.9)]),
            DuplicateRecord("duplicate_2", False, [("original", 0.8)]),
            DuplicateRecord("duplicate_3", False, [("original", 0.7)]),
        ],
        threshold=0.7,
    )

    # Access before rethreshold
    result1 = d.selected_with_duplicates
    assert len(result1[0].duplicates) == 3

    # Rethreshold should invalidate cache
    d.rethreshold(0.85)

    # Access after rethreshold should give new result
    result2 = d.selected_with_duplicates
    assert len(result2[0].duplicates) == 1
    assert result2[0].duplicates[0][0] == "duplicate_1"
    # Results should be different objects
    assert result1 is not result2


def test_filter_result_empty() -> None:
    """Test FilterResult ratios with empty lists."""
    result = FilterResult(selected=[], filtered=[])
    assert result.filter_ratio == 0.0
    assert result.selected_ratio == 1.0
