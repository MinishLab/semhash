import numpy as np
import pytest
from frozendict import frozendict

from semhash.utils import (
    Encoder,
    compute_candidate_limit,
    featurize,
    prepare_records,
    remove_exact_duplicates,
    to_frozendict,
)


def test_to_frozendict() -> None:
    """Test converting dict to frozendict."""
    record = {"a": "1", "b": "2", "c": "3"}
    result = to_frozendict(record, {"a", "c"})
    assert result == frozendict({"a": "1", "c": "3"})
    assert "b" not in result


def test_compute_candidate_limit() -> None:
    """Test candidate limit computation."""
    # Basic case
    assert compute_candidate_limit(1000, 10) == 100
    # Smaller than min_candidates (but max is capped at total)
    assert compute_candidate_limit(50, 10) == 50
    # Larger than max_candidates
    assert compute_candidate_limit(20000, 10) == 1000
    # Selection size larger than fraction
    assert compute_candidate_limit(100, 50) == 100


def test_featurize(model: Encoder) -> None:
    """Test featurizing records."""
    records = [{"text": "hello"}, {"text": "world"}]
    embeddings = featurize(records, ["text"], model)
    assert embeddings.shape == (2, 128)  # Model has 128 dims
    assert isinstance(embeddings, np.ndarray)


def test_remove_exact_duplicates() -> None:
    """Test exact duplicate removal."""
    records = [
        {"text": "hello", "id": "1"},
        {"text": "world", "id": "2"},
        {"text": "hello", "id": "3"},
    ]
    deduplicated, duplicates = remove_exact_duplicates(records, ["text"])

    assert len(deduplicated) == 2
    assert len(duplicates) == 1
    assert duplicates[0][0] == {"text": "hello", "id": "3"}


def test_prepare_records() -> None:
    """Test preparing records."""
    # String records
    records = ["hello", "world"]
    dict_records, columns, was_string = prepare_records(records, None)
    assert was_string is True
    assert columns == ["text"]
    assert dict_records == [{"text": "hello"}, {"text": "world"}]

    # Dict records
    records = [{"text": "hello"}, {"text": "world"}]
    dict_records, columns, was_string = prepare_records(records, ["text"])
    assert was_string is False
    assert columns == ["text"]
    assert dict_records == records

    # Dict records without columns raises ValueError
    records = [{"text": "hello"}]
    with pytest.raises(ValueError, match="Columns must be specified"):
        prepare_records(records, None)


def test_remove_exact_duplicates_with_reference_records() -> None:
    """Test exact duplicate removal with reference_records for cross-dataset filtering."""
    # Build reference buckets (simulates index.items structure)
    reference_records = [
        [{"text": "apple"}],  # bucket 1: apple (1 occurrence)
        [{"text": "banana"}, {"text": "banana"}],  # bucket 2: banana (2 occurrences)
    ]

    # New records to check against reference
    new_records = [
        {"text": "cherry"},  # New (not in reference)
        {"text": "apple"},  # Exact match with reference
        {"text": "date"},  # New (not in reference)
        {"text": "banana"},  # Exact match with reference
    ]

    deduplicated, duplicates = remove_exact_duplicates(new_records, ["text"], reference_records=reference_records)

    # Deduplicated should only contain records NOT in reference
    assert len(deduplicated) == 2
    assert {"text": "cherry"} in deduplicated
    assert {"text": "date"} in deduplicated

    # Duplicates should contain records that match reference
    assert len(duplicates) == 2
    # Each duplicate is (record, reference_bucket)
    dup_records = [d[0] for d in duplicates]
    assert {"text": "apple"} in dup_records
    assert {"text": "banana"} in dup_records
