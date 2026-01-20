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

    # Test with explicit candidate_limit
    result = semhash.self_find_representative(candidate_limit=5, selection_size=3, diversity=0.5)
    assert len(result.selected) == 3, "Expected 3 representatives"
    selected = {r["text"] for r in result.selected}
    assert selected == {"blueberry", "pineapple", "grape"}

    # Test with auto candidate_limit (default)
    result_auto = semhash.self_find_representative(selection_size=3, diversity=0.5)
    assert len(result_auto.selected) == 3


def test_find_representative(model: Encoder, train_texts: list[str], test_texts: list[str]) -> None:
    """Test the find_representative method."""
    semhash = SemHash.from_records(records=train_texts, model=model)

    # Test with explicit candidate_limit
    result = semhash.find_representative(records=test_texts, candidate_limit=5, selection_size=3, diversity=0.5)
    assert len(result.selected) == 3, "Expected 3 representatives"
    selected = {r["text"] for r in result.selected}
    assert selected == {"grapefruit", "banana", "apple"}

    # Test with auto candidate_limit (default)
    result_auto = semhash.find_representative(records=test_texts, selection_size=3, diversity=0.5)
    assert len(result_auto.selected) == 3


def test_filter_outliers(model: Encoder, train_texts: list[str], test_texts: list[str]) -> None:
    """Test the filter_outliers method."""
    semhash = SemHash.from_records(records=train_texts, model=model)
    result = semhash.filter_outliers(records=test_texts, outlier_percentage=0.2)
    assert len(result.filtered) == 2, "Expected 2 outliers"
    assert len(result.selected) == len(test_texts) - 2
    filtered = {r["text"] for r in result.filtered}
    assert filtered == {"motorcycle", "plane"}, "Expected outliers to be motorcycle and plane"

    # Test FilterResult ratio properties
    assert result.filter_ratio == len(result.filtered) / len(test_texts)
    assert result.selected_ratio == len(result.selected) / len(test_texts)
    assert result.filter_ratio + result.selected_ratio == 1.0

    # Test with outlier_percentage=0.0 (should return no outliers)
    result_zero = semhash.filter_outliers(records=test_texts, outlier_percentage=0.0)
    assert result_zero.filtered == []
    assert len(result_zero.selected) == len(test_texts)
    assert result_zero.filter_ratio == 0.0
    assert result_zero.selected_ratio == 1.0

    # Invalid outlier_percentage raises ValueError
    with pytest.raises(ValueError, match="outlier_percentage must be between 0 and 1"):
        semhash.filter_outliers(records=test_texts, outlier_percentage=-0.1)
    with pytest.raises(ValueError, match="outlier_percentage must be between 0 and 1"):
        semhash.filter_outliers(records=test_texts, outlier_percentage=1.5)


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

    # Invalid outlier_percentage raises ValueError
    with pytest.raises(ValueError, match="outlier_percentage must be between 0 and 1"):
        semhash.self_filter_outliers(outlier_percentage=-0.1)
    with pytest.raises(ValueError, match="outlier_percentage must be between 0 and 1"):
        semhash.self_filter_outliers(outlier_percentage=1.5)


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
    # Validation: empty records
    with pytest.raises(ValueError, match="records must not be empty"):
        SemHash.from_embeddings(embeddings=np.array([[]]), records=[], model=model)

    # Validation: non-2D embeddings
    with pytest.raises(ValueError, match="must be a 2D array"):
        SemHash.from_embeddings(embeddings=np.array([1, 2, 3]), records=["a", "b", "c"], model=model)

    # Validation: mismatched shapes
    with pytest.raises(ValueError, match="Number of embeddings"):
        wrong_embeddings = model.encode(["apple", "banana"])
        SemHash.from_embeddings(embeddings=wrong_embeddings, records=train_texts, model=model)

    # Test that from_embeddings behaves same as from_records
    semhash_from_records = SemHash.from_records(records=train_texts, model=model)
    embeddings = model.encode(train_texts)
    semhash_from_embeddings = SemHash.from_embeddings(embeddings=embeddings, records=train_texts, model=model)

    result1 = semhash_from_records.self_deduplicate(threshold=0.95)
    result2 = semhash_from_embeddings.self_deduplicate(threshold=0.95)
    assert len(result1.selected) == len(result2.selected)

    # Test that from_embeddings keeps first-occurrence embeddings and drops duplicates
    records = ["apple", "banana", "apple", "cherry"]
    embeddings = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    semhash = SemHash.from_embeddings(embeddings=embeddings, records=records, model=model)
    assert semhash.index.vectors.shape == (3, 1)
    assert semhash.index.vectors.tolist() == [[0.0], [1.0], [3.0]]


def test_from_records_edge_cases(model: Encoder) -> None:
    """Test from_records edge cases: coercion, order preservation, None rejection."""
    # Coerces non-string dict values to strings
    records = [{"id": 1}, {"id": 2}, {"id": 1}]  # Integers, with duplicate
    semhash = SemHash.from_records(records, columns=["id"], model=model)
    assert semhash.index.vectors.shape[0] == 2  # Deduplicated
    assert 2 in [len(bucket) for bucket in semhash.index.items]  # id=1 bucket has 2

    # Preserves first-occurrence order (deterministic)
    texts = ["zebra", "apple", "zebra", "banana", "apple", "cherry"]
    semhash = SemHash.from_records(texts, model=model)
    firsts = [bucket[0]["text"] for bucket in semhash.index.items]
    assert firsts == ["zebra", "apple", "banana", "cherry"]

    # Rejects None values in dict records
    with pytest.raises(ValueError, match="has None value"):
        SemHash.from_records([{"text": "apple"}, {"text": None}], columns=["text"], model=model)


def test_preserve_non_embedding_fields(model: Encoder) -> None:
    """Test that fields not specified in columns are preserved in results."""
    records = [
        {"id": 0, "text": "triforce", "metadata": "game1"},
        {"id": 1, "text": "master sword", "metadata": "game2"},
        {"id": 2, "text": "hylian shield", "metadata": "game3"},
    ]
    semhash = SemHash.from_records(records, columns=["text"], model=model)

    # Test self_deduplicate preserves non-embedding fields
    result = semhash.self_deduplicate(threshold=0.9)
    assert len(result.selected) == 3, "All records should be unique"

    # All results should have id and metadata fields preserved
    for record in result.selected:
        assert "id" in record, "id field should be preserved"
        assert "text" in record, "text field should be preserved"
        assert "metadata" in record, "metadata field should be preserved"

    # Check specific values are correct
    ids = {r["id"] for r in result.selected}
    assert ids == {0, 1, 2}, "All id values should be preserved"

    metadatas = {r["metadata"] for r in result.selected}
    assert metadatas == {"game1", "game2", "game3"}, "All metadata values should be preserved"

    # Test that cross-dataset deduplication also preserves fields
    new_records = [{"id": 10, "text": "triforce", "metadata": "duplicate"}]
    dup_result = semhash.deduplicate(new_records, threshold=0.9)

    assert len(dup_result.filtered) == 1, "Should detect duplicate"
    assert "id" in dup_result.filtered[0].record, "id should be preserved in filtered records"
    assert dup_result.filtered[0].record["id"] == 10, "Correct id value"


def test_deduplicate_edge_cases(model: Encoder) -> None:
    """Test deduplicate() edge cases: coercion, None rejection, empty records, type mismatches."""
    semhash = SemHash.from_records(["1", "2", "3"], model=model)

    # Coerces non-string dict values
    result = semhash.deduplicate([{"text": 1}, {"text": 4}], threshold=0.95)
    assert len(result.filtered) + len(result.selected) == 2

    # Rejects None values
    with pytest.raises(ValueError, match="has None value"):
        semhash.deduplicate([{"text": "cherry"}, {"text": None}], threshold=0.95)

    # Rejects empty records
    with pytest.raises(ValueError, match="records must not be empty"):
        semhash.deduplicate([], threshold=0.95)

    # Type mismatch: strings passed to dict-based index
    semhash_dict = SemHash.from_records([{"col": "a"}, {"col": "b"}], columns=["col"], model=model)
    with pytest.raises(ValueError, match="Records were not originally strings"):
        semhash_dict.deduplicate(["x", "y"], threshold=0.95)

    # Type mismatch: mixed strings
    with pytest.raises(ValueError, match="Records must be all strings"):
        semhash.deduplicate(["a", {"text": "b"}], threshold=0.95)

    # Type mismatch: mixed dicts
    with pytest.raises(ValueError, match="Records must be all dictionaries"):
        semhash_dict.deduplicate([{"col": "a"}, "b"], threshold=0.95)
