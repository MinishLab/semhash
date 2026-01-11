import numpy as np
import pytest

from semhash import SemHash
from semhash.datamodels import FilterResult
from semhash.utils import Encoder


def test_single_dataset_deduplication(model: Encoder) -> None:
    """Test single dataset deduplication."""
    # No duplicates
    texts = [
        "It's dangerous to go alone!",
        "The master sword can seal the darkness.",
        "Ganondorf has invaded Hyrule!",
    ]
    semhash = SemHash.from_records(records=texts, model=model)
    deduplicated_texts = semhash.self_deduplicate().selected

    assert deduplicated_texts == texts

    # With duplicates
    texts = [
        "It's dangerous to go alone!",
        "It's dangerous to go alone!",  # Exact duplicate
        "It's not safe to go alone!",  # Semantically similar
    ]
    semhash = SemHash.from_records(records=texts, model=model)
    deduplicated_texts = semhash.self_deduplicate(0.7).selected
    assert deduplicated_texts == ["It's dangerous to go alone!"]


def test_multi_dataset_deduplication(model: Encoder) -> None:
    """Test deduplication across two datasets."""
    # No duplicates
    texts1 = [
        "It's dangerous to go alone!",
        "It's a secret to everybody.",
        "Ganondorf has invaded Hyrule!",
    ]
    texts2 = [
        "Link is the hero of time.",
        "Zelda is the princess of Hyrule.",
        "Ganon is the king of thieves.",
    ]
    semhash = SemHash.from_records(texts1, columns=None, model=model)
    deduplicated_texts = semhash.deduplicate(texts2).selected
    assert deduplicated_texts == texts2

    # With duplicates
    texts2 = [
        "It's dangerous to go alone!",  # Exact duplicate
        "It's risky to go alone!",  # Semantically similar
        "Ganondorf has attacked Hyrule!",  # Semantically similar
    ]
    deduplicated_texts = semhash.deduplicate(texts2, threshold=0.7).selected
    assert deduplicated_texts == []


def test_single_dataset_deduplication_multicolumn(model: Encoder) -> None:
    """Test single dataset deduplication with multi-column records."""
    records = [
        {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},
        {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},  # Exact duplicate
        {
            "question": "Who is the protagonist?",
            "context": "In this story, Link is the hero",
            "answer": "Link",
        },  # Semantically similar
        {"question": "Who is the princess?", "context": "The princess is Zelda", "answer": "Zelda"},
    ]
    semhash = SemHash.from_records(
        records,
        columns=["question", "context", "answer"],
        model=model,
    )
    deduplicated = semhash.self_deduplicate(threshold=0.7)

    assert deduplicated.selected == [
        {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},
        {"question": "Who is the princess?", "context": "The princess is Zelda", "answer": "Zelda"},
    ]


def test_multi_dataset_deduplication_multicolumn(model: Encoder) -> None:
    """Test multi dataset deduplication with multi-column records."""
    train_records = [
        {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},
        {"question": "Who is the princess?", "context": "The princess is Zelda", "answer": "Zelda"},
    ]
    test_records = [
        {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},  # Exact duplicate
        {
            "question": "Who is the princess?",
            "context": "Zelda is the princess",
            "answer": "Zelda",
        },  # Semantically similar
        {"question": "What is the villain's name?", "context": "The villain is Ganon", "answer": "Ganon"},
    ]
    semhash = SemHash.from_records(
        train_records,
        columns=["question", "context", "answer"],
        model=model,
    )
    deduplicated = semhash.deduplicate(test_records).selected
    assert deduplicated == [
        {"question": "What is the villain's name?", "context": "The villain is Ganon", "answer": "Ganon"}
    ]


def test_from_records_without_columns(model: Encoder) -> None:
    """Test fitting without specifying columns."""
    records = [
        {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},
        {"question": "Who is the princess?", "context": "The princess is Zelda", "answer": "Zelda"},
    ]
    with pytest.raises(ValueError):
        SemHash.from_records(records, columns=None, model=model)


def test_deduplicate_with_only_exact_duplicates(model: Encoder) -> None:
    """Test deduplicating with only exact duplicates."""
    texts1 = [
        "It's dangerous to go alone!",
        "It's dangerous to go alone!",
        "It's dangerous to go alone!",
    ]
    texts2 = [
        "It's dangerous to go alone!",
        "It's dangerous to go alone!",
        "It's dangerous to go alone!",
    ]
    semhash = SemHash.from_records(texts1, model=model)
    deduplicated = semhash.self_deduplicate()
    assert deduplicated.selected == ["It's dangerous to go alone!"]

    deduplicated = semhash.deduplicate(texts2)
    assert deduplicated.selected == []


def test_self_find_representative(model: Encoder, train_texts: list[str]) -> None:
    """Test the self_find_representative method."""
    semhash = SemHash.from_records(records=train_texts, model=model)
    result = semhash.self_find_representative(
        candidate_limit=5,
        selection_size=3,
        diversity=0.5,
    )
    assert len(result.selected) == 3, "Expected 3 representatives"
    selected = {r["text"] for r in result.selected}
    assert selected == {
        "blueberry",
        "pineapple",
        "grape",
    }, "Expected representatives to be blueberry, pineapple, and grape"


def test_find_representative(model: Encoder, train_texts: list[str], test_texts: list[str]) -> None:
    """Test the find_representative method."""
    semhash = SemHash.from_records(records=train_texts, model=model)
    result = semhash.find_representative(records=test_texts, candidate_limit=5, selection_size=3, diversity=0.5)
    assert len(result.selected) == 3, "Expected 3 representatives"
    selected = {r["text"] for r in result.selected}
    assert selected == {"grapefruit", "banana", "apple"}, "Expected representatives to be grapefruit, banana, and apple"


def test_filter_outliers(model: Encoder, train_texts: list[str], test_texts: list[str]) -> None:
    """Test the filter_outliers method."""
    semhash = SemHash.from_records(records=train_texts, model=model)
    result = semhash.filter_outliers(records=test_texts, outlier_percentage=0.2)
    assert len(result.filtered) == 2, "Expected 2 outliers"
    assert len(result.selected) == len(test_texts) - 2
    filtered = {r["text"] for r in result.filtered}
    assert filtered == {"motorcycle", "plane"}, "Expected outliers to be motorcycle and plane"

    # Test with outlier_percentage=0.0 (should return no outliers)
    result_zero = semhash.filter_outliers(records=test_texts, outlier_percentage=0.0)
    assert result_zero.filtered == []
    assert len(result_zero.selected) == len(test_texts)


def test_self_filter_outliers(model: Encoder, train_texts: list[str]) -> None:
    """Test the self_filter_outliers method."""
    semhash = SemHash.from_records(records=train_texts, model=model)
    result = semhash.self_filter_outliers(outlier_percentage=0.1)
    assert len(result.filtered) == 2, "Expected 2 outliers"
    assert len(result.selected) == len(train_texts) - 2
    filtered = {r["text"] for r in result.filtered}
    assert filtered == {"car", "bicycle"}, "Expected outliers to be car and bicycle"

    # Test with outlier_percentage=0.0 (should return no outliers)
    result_zero = semhash.self_filter_outliers(outlier_percentage=0.0)
    assert result_zero.filtered == []
    assert len(result_zero.selected) == len(train_texts)


def test__diversify(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the _diversify method."""
    from semhash import semhash

    semhash_instance = SemHash(index=None, model=None, columns=["text"], was_string=True)
    # Prepare a fake ranking with three records
    records = ["a", "b", "c"]
    scores = [3.0, 2.0, 1.0]
    ranking = FilterResult(selected=records, filtered=[], scores_selected=scores, scores_filtered=[])
    # Create dummy embeddings for the records
    embeddings = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    # Monkeypatch featurize to return the dummy embeddings
    monkeypatch.setattr(semhash, "featurize", lambda records, columns, model: embeddings)

    # Test diversity=0.0: pure relevance, should pick top 2 by score
    result_rel = semhash_instance._diversify(ranking, candidate_limit=3, selection_size=2, diversity=0.0)
    assert result_rel.selected == ["a", "b"]

    # Test diversity=1.0: pure diversity, should first pick 'a', then pick most dissimilar: 'c'
    result_div = semhash_instance._diversify(ranking, candidate_limit=3, selection_size=2, diversity=1.0)
    assert result_div.selected == ["a", "c"]

    # Test empty candidates (candidate_limit=0)
    result_empty = semhash_instance._diversify(ranking, candidate_limit=0, selection_size=2, diversity=0.5)
    assert result_empty.selected == []
    assert result_empty.filtered == []
    assert result_empty.scores_selected == []
    assert result_empty.scores_filtered == []


def test_from_embeddings(model: Encoder, train_texts: list[str]) -> None:
    """Test from_embeddings constructor with validation and comparison to from_records."""
    # Test validation: mismatched shapes
    with pytest.raises(ValueError, match="Number of embeddings"):
        wrong_embeddings = model.encode(["apple", "banana"])
        SemHash.from_embeddings(embeddings=wrong_embeddings, records=train_texts, model=model)

    # Test that from_embeddings behaves same as from_records
    semhash_from_records = SemHash.from_records(records=train_texts, model=model)

    embeddings = model.encode(train_texts)
    semhash_from_embeddings = SemHash.from_embeddings(embeddings=embeddings, records=train_texts, model=model)

    # Both should give same deduplication results
    result1 = semhash_from_records.self_deduplicate(threshold=0.95)
    result2 = semhash_from_embeddings.self_deduplicate(threshold=0.95)

    assert len(result1.selected) == len(result2.selected)
    assert len(result1.filtered) == len(result2.filtered)

    # Test that from_embeddings keeps first-occurrence embeddings and drops duplicates
    records = ["apple", "banana", "apple", "cherry"]
    embeddings = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)

    semhash = SemHash.from_embeddings(embeddings=embeddings, records=records, model=model)

    assert semhash.index.vectors.shape == (3, 1)
    # Should keep embeddings at indices 0, 1, 3 (first occurrences of img1, img2, img3)
    assert semhash.index.vectors.tolist() == [[0.0], [1.0], [3.0]]


def test_from_dataset_basic(model: Encoder) -> None:
    """Test from_dataset with a simple HuggingFace dataset."""
    from datasets import Dataset

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
    from datasets import Dataset

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
    from datasets import Dataset

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
    from datasets import Dataset

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
    from datasets import Dataset

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

    from datasets import Dataset

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
    from datasets import Dataset

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
    from datasets import Dataset

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
    from datasets import Dataset

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


def test_from_dataset_with_custom_dataset_like(model: Encoder) -> None:
    """Test that from_dataset works with custom DatasetLike implementations (no HF dependency)."""

    class MiniDataset:
        """Minimal DatasetLike implementation for testing."""

        column_names = ["text"]

        def __init__(self, data: dict[str, list[str]]) -> None:
            self._data = data

        def __len__(self) -> int:
            return len(self._data["text"])

        def __getitem__(self, key: str) -> list[str]:
            return self._data[key]

    # Create custom dataset with duplicates
    ds = MiniDataset({"text": ["apple", "banana", "apple"]})

    semhash = SemHash.from_dataset(ds, columns=["text"], model=model)

    # Should have deduplicated to 2 unique items
    assert len(semhash.index.items) == 2
    assert len(semhash.index.vectors) == 2

    # Should work with deduplication
    result = semhash.self_deduplicate(threshold=0.95)
    assert len(result.selected) == 2


def test_from_dataset_multicolumn_does_not_embed_duplicates(model: Encoder) -> None:
    """Test that multi-column from_dataset only embeds representatives (validates per-column encoding)."""
    from typing import Any

    from datasets import Dataset

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


def test_from_records_coerces_non_string_dict_values(model: Encoder) -> None:
    """Test that from_records coerces non-string dict values to strings."""
    records = [{"id": 1}, {"id": 2}, {"id": 1}]  # Integers, with duplicate
    semhash = SemHash.from_records(records, columns=["id"], model=model)

    # Should have deduplicated to 2 unique items
    assert semhash.index.vectors.shape[0] == 2
    assert len(semhash.index.items) == 2

    # First bucket should have 2 records (id=1 appears twice)
    bucket_sizes = [len(bucket) for bucket in semhash.index.items]
    assert 2 in bucket_sizes


def test_from_records_preserves_first_occurrence_order(model: Encoder) -> None:
    """Test that from_records preserves first-occurrence order (deterministic)."""
    texts = ["zebra", "apple", "zebra", "banana", "apple", "cherry"]
    semhash = SemHash.from_records(texts, model=model)

    # Get first record from each bucket
    firsts = [bucket[0]["text"] for bucket in semhash.index.items]

    # Should be in first-occurrence order
    assert firsts == ["zebra", "apple", "banana", "cherry"]


def test_from_records_rejects_none_in_dict_values(model: Encoder) -> None:
    """Test that from_records rejects None values in dict records."""
    records = [{"text": "apple"}, {"text": None}, {"text": "banana"}]

    with pytest.raises(ValueError, match="has None value"):
        SemHash.from_records(records, columns=["text"], model=model)


def test_deduplicate_coerces_non_string_dict_values(model: Encoder) -> None:
    """Test that deduplicate() coerces non-string dict values (consistent with from_records)."""
    # Build SemHash with string records
    semhash = SemHash.from_records(["1", "2", "3"], model=model)

    # Pass dict records with integer values to deduplicate
    new_records = [{"text": 1}, {"text": 2}, {"text": 4}]  # type: ignore[list-item]

    # Should coerce ints to strings and work without error
    result = semhash.deduplicate(new_records, threshold=0.95)

    # Should have deduplicated (1, 2 already exist)
    assert len(result.filtered) > 0
    assert len(result.selected) > 0


def test_deduplicate_rejects_none_values(model: Encoder) -> None:
    """Test that deduplicate() rejects None values in dict records."""
    semhash = SemHash.from_records(["apple", "banana"], model=model)

    new_records = [{"text": "cherry"}, {"text": None}]  # type: ignore[list-item]

    with pytest.raises(ValueError, match="has None value"):
        semhash.deduplicate(new_records, threshold=0.95)
