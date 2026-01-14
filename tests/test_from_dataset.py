"""Tests for SemHash.from_dataset() with HuggingFace datasets."""

import pytest
from conftest import CountingEncoder

from semhash import SemHash
from semhash.utils import Encoder

# Require datasets for this entire test module
datasets = pytest.importorskip("datasets", reason="pip install 'semhash[datasets]'")
Dataset = datasets.Dataset


def test_from_dataset_basic(model: Encoder) -> None:
    """Test from_dataset with a simple HuggingFace dataset."""
    # Create a simple dataset with text column
    ds = Dataset.from_dict({"text": ["apple", "banana", "cherry", "apple"]})

    semhash = SemHash.from_dataset(dataset=ds, columns=["text"], model=model)

    # Should collapse exact duplicate "apple"
    assert len(semhash.index.vectors) == 3
    assert len(semhash.index.items) == 3

    # Verify deduplication works
    result = semhash.self_deduplicate(threshold=0.95)
    assert len(result.selected) <= 3


def test_from_dataset_multicolumn(model: Encoder) -> None:
    """Test from_dataset with multiple columns."""
    ds = Dataset.from_dict(
        {
            "question": ["What is AI?", "What is ML?", "What is AI?"],
            "context": ["AI explanation", "ML explanation", "AI explanation"],
        }
    )

    semhash = SemHash.from_dataset(dataset=ds, columns=["question", "context"], model=model)

    # Should collapse the duplicate row
    assert len(semhash.index.vectors) == 2
    assert len(semhash.index.items) == 2


def test_from_dataset_validation(model: Encoder) -> None:
    """Test from_dataset input validation."""
    ds = Dataset.from_dict({"text": ["apple", "banana"]})

    # Test invalid dataset type (Protocol violation)
    with pytest.raises(TypeError, match="must satisfy DatasetLike"):
        SemHash.from_dataset(dataset={"text": ["apple"]}, columns=["text"], model=model)

    # Test missing column
    with pytest.raises(ValueError, match="not found in dataset"):
        SemHash.from_dataset(dataset=ds, columns=["missing_col"], model=model)

    # Test None value in column
    ds_with_none = Dataset.from_dict({"text": ["apple", None, "banana"]})
    with pytest.raises(ValueError, match="Column 'text' has None at index 1"):
        SemHash.from_dataset(dataset=ds_with_none, columns=["text"], model=model)

    # Test empty dataset
    ds_empty = Dataset.from_dict({"text": []})
    with pytest.raises(ValueError, match="dataset must not be empty"):
        SemHash.from_dataset(dataset=ds_empty, columns=["text"], model=model)


def test_from_dataset_equivalence_to_from_records(model: Encoder) -> None:
    """Test that from_dataset produces same results as from_records for same data."""
    data = [
        {"question": "What is AI?", "answer": "Artificial Intelligence"},
        {"question": "What is ML?", "answer": "Machine Learning"},
    ]
    ds = Dataset.from_dict({k: [d[k] for d in data] for k in data[0].keys()})

    semhash_from_dataset = SemHash.from_dataset(dataset=ds, columns=["question", "answer"], model=model)
    semhash_from_records = SemHash.from_records(records=data, columns=["question", "answer"], model=model)

    # Both should have same number of vectors
    assert semhash_from_dataset.index.vectors.shape == semhash_from_records.index.vectors.shape

    # Both should give same deduplication results
    result1 = semhash_from_dataset.self_deduplicate(threshold=0.95)
    result2 = semhash_from_records.self_deduplicate(threshold=0.95)

    assert len(result1.selected) == len(result2.selected)
    assert len(result1.filtered) == len(result2.filtered)


@pytest.mark.parametrize(
    "data,columns,expected_representatives,expected_calls",
    [
        # Single column: 6 records, 3 unique -> 1 encode call with 3 items
        ({"text": ["apple", "banana", "apple", "cherry", "banana", "apple"]}, ["text"], 3, [3]),
        # Multi column: 4 records, 3 unique -> 2 encode calls (one per column) with 3 items each
        ({"col1": ["a", "b", "a", "c"], "col2": ["x", "y", "x", "z"]}, ["col1", "col2"], 3, [3, 3]),
    ],
    ids=["single_column", "multi_column"],
)
def test_from_dataset_does_not_embed_duplicates(
    counting_encoder: CountingEncoder,
    data: dict[str, list[str]],
    columns: list[str],
    expected_representatives: int,
    expected_calls: list[int],
) -> None:
    """Test that from_dataset only embeds representative records, not duplicates."""
    ds = Dataset.from_dict(data)

    semhash = SemHash.from_dataset(dataset=ds, columns=columns, model=counting_encoder)  # type: ignore[arg-type]

    # Should only have expected number of representatives
    assert semhash.index.vectors.shape[0] == expected_representatives

    # Should have encoded only representatives, with expected call pattern
    assert counting_encoder.encode_calls == expected_calls
    assert counting_encoder.total_encoded == sum(expected_calls)


def test_from_dataset_handles_non_string_values(model: Encoder) -> None:
    """Test that from_dataset handles non-string values (e.g., integers) by converting them."""
    # Dataset with integer values
    ds = Dataset.from_dict({"id": [1, 2, 3, 1]})  # Has a duplicate

    semhash = SemHash.from_dataset(dataset=ds, columns=["id"], model=model)

    # Should have deduplicated the integer '1'
    assert semhash.index.vectors.shape[0] == 3
    assert len(semhash.index.items) == 3

    # Values should be converted to strings
    result = semhash.self_deduplicate(threshold=0.95)
    # Result should be dicts (not strings, since column is 'id' not 'text')
    assert all(isinstance(r, dict) for r in result.selected)
    assert all("id" in r for r in result.selected)


def test_from_dataset_preserves_first_occurrence_order(model: Encoder) -> None:
    """Test that from_dataset preserves the order of first occurrences (deterministic output)."""
    # Create dataset where duplicates appear out of order
    ds = Dataset.from_dict({"text": ["zebra", "apple", "zebra", "banana", "apple", "cherry"]})

    semhash = SemHash.from_dataset(dataset=ds, columns=["text"], model=model)

    # Should have 4 unique items in first-occurrence order: zebra, apple, banana, cherry
    assert len(semhash.index.vectors) == 4

    # Get all items (each is a bucket of duplicates)
    first_occurrences = [item[0]["text"] for item in semhash.index.items]

    # Should preserve first-occurrence order from dataset
    assert first_occurrences == ["zebra", "apple", "banana", "cherry"]


def test_from_dataset_was_string_behavior(model: Encoder) -> None:
    """Test was_string logic: returns strings only for text column with actual string values."""
    # Case 1: text column with strings -> should return strings (matching from_records)
    ds_strings = Dataset.from_dict({"text": ["apple", "banana", "cherry"]})
    semhash_strings = SemHash.from_dataset(dataset=ds_strings, columns=["text"], model=model)
    result_strings = semhash_strings.self_deduplicate(threshold=0.95)
    assert all(isinstance(r, str) for r in result_strings.selected)

    # Verify equivalence with from_records for string case
    semhash_from_records = SemHash.from_records(records=["apple", "banana", "cherry"], model=model)
    result_from_records = semhash_from_records.self_deduplicate(threshold=0.95)
    assert type(result_strings.selected[0]) == type(result_from_records.selected[0])

    # Case 2: text column with integers -> should return dicts (coerced, not true strings)
    ds_ints = Dataset.from_dict({"text": [1, 2, 3]})
    semhash_ints = SemHash.from_dataset(dataset=ds_ints, columns=["text"], model=model)
    result_ints = semhash_ints.self_deduplicate(threshold=0.95)
    assert all(isinstance(r, dict) for r in result_ints.selected)
    assert all("text" in r for r in result_ints.selected)
