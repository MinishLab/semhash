from semhash.datamodels import DeduplicationResult, DuplicateRecord


def test_deduplication_scoring() -> None:
    """Test the deduplication scoring."""
    d = DeduplicationResult(["a", "b", "c"], [DuplicateRecord("a", ["b"], False), DuplicateRecord("b", ["c"], False)])
    assert d.duplicate_ratio == 0.4


def test_deduplication_scoring_exact() -> None:
    """Test the deduplication scoring."""
    d = DeduplicationResult(["a", "b", "c"], [DuplicateRecord("a", ["b"], True), DuplicateRecord("b", ["c"], False)])
    assert d.exact_duplicate_ratio == 0.2


def test_deduplication_scoring_exact_empty() -> None:
    """Test the deduplication scoring."""
    d = DeduplicationResult([], [])
    assert d.exact_duplicate_ratio == 0.0


def test_deduplication_scoring_empty() -> None:
    """Test the deduplication scoring."""
    d = DeduplicationResult([], [])
    assert d.duplicate_ratio == 0.0
