import pytest

from semhash import SemHash
from semhash.utils import Encoder

# Require datasets for this entire test module
datasets = pytest.importorskip("datasets", reason="pip install 'semhash[datasets]'")
Dataset = datasets.Dataset


def test_from_dataset_basic(model: Encoder) -> None:
    """Test from_dataset works with single and multi-column datasets."""
    # Single column
    ds_single = Dataset.from_dict({"text": ["apple", "banana", "cherry", "apple"]})
    semhash_single = SemHash.from_dataset(dataset=ds_single, columns=["text"], model=model)
    assert len(semhash_single.index.vectors) == 3  # Collapsed duplicate "apple"

    # Multi-column
    ds_multi = Dataset.from_dict(
        {
            "question": ["What is AI?", "What is ML?", "What is AI?"],
            "context": ["AI explanation", "ML explanation", "AI explanation"],
        }
    )
    semhash_multi = SemHash.from_dataset(dataset=ds_multi, columns=["question", "context"], model=model)
    assert len(semhash_multi.index.vectors) == 2  # Collapsed duplicate row


def test_from_dataset_equivalence_to_from_records(model: Encoder) -> None:
    """Test that from_dataset produces identical results to from_records."""
    # Test with strings
    texts = ["apple", "banana", "cherry", "apple"]
    ds_strings = Dataset.from_dict({"text": texts})

    semhash_dataset = SemHash.from_dataset(dataset=ds_strings, columns=["text"], model=model)
    semhash_records = SemHash.from_records(records=texts, model=model)

    assert semhash_dataset.index.vectors.shape == semhash_records.index.vectors.shape
    result_dataset = semhash_dataset.self_deduplicate(threshold=0.9)
    result_records = semhash_records.self_deduplicate(threshold=0.9)
    assert len(result_dataset.selected) == len(result_records.selected)
    assert type(result_dataset.selected[0]) == type(result_records.selected[0])

    # Test with dicts (multi-column)
    data = [
        {"question": "What is AI?", "answer": "Artificial Intelligence"},
        {"question": "What is ML?", "answer": "Machine Learning"},
    ]
    ds_dicts = Dataset.from_dict({k: [d[k] for d in data] for k in data[0].keys()})

    semhash_dataset = SemHash.from_dataset(dataset=ds_dicts, columns=["question", "answer"], model=model)
    semhash_records = SemHash.from_records(records=data, columns=["question", "answer"], model=model)

    assert semhash_dataset.index.vectors.shape == semhash_records.index.vectors.shape


def test_from_dataset_validation(model: Encoder) -> None:
    """Test from_dataset input validation (dataset-specific error cases)."""
    ds = Dataset.from_dict({"text": ["apple", "banana"]})

    # Invalid dataset type (Protocol violation)
    with pytest.raises(TypeError, match="must satisfy DatasetLike"):
        SemHash.from_dataset(dataset={"text": ["apple"]}, columns=["text"], model=model)

    # Missing column
    with pytest.raises(ValueError, match="not found in dataset"):
        SemHash.from_dataset(dataset=ds, columns=["missing_col"], model=model)

    # None value in column
    ds_with_none = Dataset.from_dict({"text": ["apple", None, "banana"]})
    with pytest.raises(ValueError, match="Column 'text' has None at index 1"):
        SemHash.from_dataset(dataset=ds_with_none, columns=["text"], model=model)

    # Empty dataset
    ds_empty = Dataset.from_dict({"text": []})
    with pytest.raises(ValueError, match="dataset must not be empty"):
        SemHash.from_dataset(dataset=ds_empty, columns=["text"], model=model)

    # Column length mismatch (custom DatasetLike with inconsistent columns)
    class BadDataset:
        column_names = ["col1", "col2"]

        def __len__(self) -> int:
            return 3

        def __getitem__(self, key: str) -> list[str]:
            if key == "col1":
                return ["a", "b", "c"]
            return ["x", "y"]  # Wrong length!

    with pytest.raises(ValueError, match="does not match dataset length"):
        SemHash.from_dataset(dataset=BadDataset(), columns=["col1", "col2"], model=model)


def test_from_dataset_was_string_behavior(model: Encoder) -> None:
    """Test was_string logic: returns strings only for text column with actual string values."""
    # text column with strings -> returns strings
    ds_strings = Dataset.from_dict({"text": ["apple", "banana", "cherry"]})
    result_strings = SemHash.from_dataset(dataset=ds_strings, columns=["text"], model=model).self_deduplicate()
    assert all(isinstance(r, str) for r in result_strings.selected)

    # text column with integers -> returns dicts (coerced values aren't "true" strings)
    ds_ints = Dataset.from_dict({"text": [1, 2, 3]})
    result_ints = SemHash.from_dataset(dataset=ds_ints, columns=["text"], model=model).self_deduplicate()
    assert all(isinstance(r, dict) for r in result_ints.selected)

    # non-text column -> returns dicts
    ds_other = Dataset.from_dict({"id": ["a", "b", "c"]})
    result_other = SemHash.from_dataset(dataset=ds_other, columns=["id"], model=model).self_deduplicate()
    assert all(isinstance(r, dict) for r in result_other.selected)
