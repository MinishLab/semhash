"""Tests for SemHash.from_dataset() with HuggingFace datasets."""

import pytest

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


def test_from_dataset_returns_strings_for_text_column(model: Encoder) -> None:
    """Test that from_dataset returns strings when columns=["text"], matching from_records behavior."""
    ds = Dataset.from_dict({"text": ["apple", "banana", "cherry"]})

    semhash = SemHash.from_dataset(dataset=ds, columns=["text"], model=model)

    # Should return strings (like from_records does)
    result = semhash.self_deduplicate(threshold=0.95)
    assert all(isinstance(r, str) for r in result.selected)

    # Verify equivalence with from_records
    semhash_from_records = SemHash.from_records(records=["apple", "banana", "cherry"], model=model)
    result_from_records = semhash_from_records.self_deduplicate(threshold=0.95)

    # Both should return strings
    assert type(result.selected[0]) == type(result_from_records.selected[0])


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


def test_from_dataset_does_not_embed_duplicates(model: Encoder) -> None:
    """Test that from_dataset only embeds representative records, not duplicates."""
    from typing import Any

    # Create dataset with exact duplicates: 6 records but only 3 unique
    ds = Dataset.from_dict({"text": ["apple", "banana", "apple", "cherry", "banana", "apple"]})

    # Create a counting encoder wrapper
    class CountingEncoder:
        def __init__(self, base_encoder: Any) -> None:
            self.base_encoder = base_encoder
            self.encode_calls: list[int] = []

        def encode(self, sentences: Any, **kwargs: Any) -> Any:
            if isinstance(sentences, str):
                sentences = [sentences]
            self.encode_calls.append(len(sentences))
            return self.base_encoder.encode(sentences, **kwargs)

    counting_encoder = CountingEncoder(model)

    semhash = SemHash.from_dataset(dataset=ds, columns=["text"], model=counting_encoder)  # type: ignore[arg-type]

    # Should only have 3 representatives
    assert semhash.index.vectors.shape[0] == 3

    # Should have encoded exactly 3 records (representatives only), not 6
    assert sum(counting_encoder.encode_calls) == 3
    # Should have been called once with all 3 representatives
    assert len(counting_encoder.encode_calls) == 1
    assert counting_encoder.encode_calls[0] == 3


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


def test_from_dataset_was_string_only_for_actual_strings(model: Encoder) -> None:
    """Test that was_string is only True for text columns with actual string values."""
    # Test 1: text column with strings -> should return strings
    ds_strings = Dataset.from_dict({"text": ["apple", "banana", "cherry"]})
    semhash_strings = SemHash.from_dataset(dataset=ds_strings, columns=["text"], model=model)
    result_strings = semhash_strings.self_deduplicate(threshold=0.95)
    assert all(isinstance(r, str) for r in result_strings.selected)

    # Test 2: text column with integers -> should return dicts (not strings)
    ds_ints = Dataset.from_dict({"text": [1, 2, 3]})
    semhash_ints = SemHash.from_dataset(dataset=ds_ints, columns=["text"], model=model)
    result_ints = semhash_ints.self_deduplicate(threshold=0.95)
    assert all(isinstance(r, dict) for r in result_ints.selected)
    assert all("text" in r for r in result_ints.selected)


def test_from_dataset_multicolumn_does_not_embed_duplicates(model: Encoder) -> None:
    """Test that multi-column from_dataset only embeds representatives (validates per-column encoding)."""
    from typing import Any

    # Create dataset with exact duplicates across multiple columns
    ds = Dataset.from_dict(
        {
            "col1": ["a", "b", "a", "c"],  # "a" appears twice
            "col2": ["x", "y", "x", "z"],  # matching pattern
        }
    )

    # Create a counting encoder wrapper
    class CountingEncoder:
        def __init__(self, base_encoder: Any) -> None:
            self.base_encoder = base_encoder
            self.encode_calls: list[int] = []

        def encode(self, sentences: Any, **kwargs: Any) -> Any:
            if isinstance(sentences, str):
                sentences = [sentences]
            self.encode_calls.append(len(sentences))
            return self.base_encoder.encode(sentences, **kwargs)

    counting_encoder = CountingEncoder(model)

    semhash = SemHash.from_dataset(dataset=ds, columns=["col1", "col2"], model=counting_encoder)  # type: ignore[arg-type]

    # Should have 3 representatives (unique combinations)
    assert semhash.index.vectors.shape[0] == 3

    # featurize() encodes once per column, so we expect 2 calls (for col1 and col2)
    assert len(counting_encoder.encode_calls) == 2

    # Each call should have 3 texts (the representatives)
    assert counting_encoder.encode_calls[0] == 3  # col1
    assert counting_encoder.encode_calls[1] == 3  # col2

    # Total encoded should be 6 (3 representatives × 2 columns)
    assert sum(counting_encoder.encode_calls) == 6
