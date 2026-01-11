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

    # Test invalid dataset type
    with pytest.raises(TypeError, match="must have 'column_names' and '__len__' attributes"):
        SemHash.from_dataset(dataset={"text": ["apple"]}, columns=["text"], model=model)

    # Test missing column
    with pytest.raises(ValueError, match="not found in dataset"):
        SemHash.from_dataset(dataset=ds, columns=["missing_col"], model=model)


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


def test_from_dataset_does_not_embed_duplicates(model: Encoder, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that from_dataset only embeds representative records, not duplicates."""
    from typing import Any

    from datasets import Dataset

    # Create dataset with exact duplicates
    ds = Dataset.from_dict({"text": ["apple", "banana", "apple", "cherry", "banana", "apple"]})

    # Track how many times encode is called and with how many records
    encode_call_count = 0
    total_encoded = 0

    original_encode = model.encode

    def tracked_encode(texts: Any, **kwargs: Any) -> Any:
        nonlocal encode_call_count, total_encoded
        encode_call_count += 1
        total_encoded += len(texts)
        return original_encode(texts, **kwargs)

    monkeypatch.setattr(model, "encode", tracked_encode)

    semhash = SemHash.from_dataset(dataset=ds, columns=["text"], model=model)

    # Should only have encoded 3 unique records (apple, banana, cherry)
    assert semhash.index.vectors.shape[0] == 3
    # Total encoded should be 3, not 6
    assert total_encoded == 3
    # Should be called once (for the deduplicated records)
    assert encode_call_count == 1
