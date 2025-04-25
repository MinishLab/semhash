import numpy as np
import pytest

from semhash import SemHash
from semhash.datamodels import FilterResult
from semhash.utils import Encoder


def test_single_dataset_deduplication(use_ann: bool, model: Encoder) -> None:
    """Test single dataset deduplication."""
    # No duplicates
    texts = [
        "It's dangerous to go alone!",
        "The master sword can seal the darkness.",
        "Ganondorf has invaded Hyrule!",
    ]
    semhash = SemHash.from_records(records=texts, use_ann=use_ann, model=model)
    deduplicated_texts = semhash.self_deduplicate().selected

    assert deduplicated_texts == texts

    # With duplicates
    texts = [
        "It's dangerous to go alone!",
        "It's dangerous to go alone!",  # Exact duplicate
        "It's not safe to go alone!",  # Semantically similar
    ]
    semhash = SemHash.from_records(records=texts, use_ann=use_ann, model=model)
    deduplicated_texts = semhash.self_deduplicate(0.7).selected
    assert deduplicated_texts == ["It's dangerous to go alone!"]


def test_multi_dataset_deduplication(use_ann: bool, model: Encoder) -> None:
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
    semhash = SemHash.from_records(texts1, columns=None, use_ann=use_ann, model=model)
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


def test_single_dataset_deduplication_multicolumn(use_ann: bool, model: Encoder) -> None:
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
        use_ann=use_ann,
        model=model,
    )
    deduplicated = semhash.self_deduplicate(threshold=0.7)

    assert deduplicated.selected == [
        {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},
        {"question": "Who is the princess?", "context": "The princess is Zelda", "answer": "Zelda"},
    ]


def test_multi_dataset_deduplication_multicolumn(use_ann: bool, model: Encoder) -> None:
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
        use_ann=use_ann,
        model=model,
    )
    deduplicated = semhash.deduplicate(test_records).selected
    assert deduplicated == [
        {"question": "What is the villain's name?", "context": "The villain is Ganon", "answer": "Ganon"}
    ]


def test_from_records_without_columns(use_ann: bool, model: Encoder) -> None:
    """Test fitting without specifying columns."""
    records = [
        {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},
        {"question": "Who is the princess?", "context": "The princess is Zelda", "answer": "Zelda"},
    ]
    with pytest.raises(ValueError):
        SemHash.from_records(records, columns=None, use_ann=use_ann, model=model)


def test_deduplicate_with_only_exact_duplicates(use_ann: bool, model: Encoder) -> None:
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
    semhash = SemHash.from_records(texts1, use_ann=use_ann, model=model)
    deduplicated = semhash.self_deduplicate()
    assert deduplicated.selected == ["It's dangerous to go alone!"]

    deduplicated = semhash.deduplicate(texts2)
    assert deduplicated.selected == []


def test_self_find_representative(use_ann: bool, model: Encoder, train_texts: list[str]) -> None:
    """Test the self_find_representative method."""
    semhash = SemHash.from_records(records=train_texts, use_ann=use_ann, model=model)
    result = semhash.self_find_representative(
        candidate_limit=5,
        selection_size=3,
        lambda_param=0.5,
    )
    assert len(result.selected) == 3, "Expected 3 representatives"
    selected = {r["text"] for r in result.selected}
    assert selected == {
        "blueberry",
        "pineapple",
        "grape",
    }, "Expected representatives to be blueberry, pineapple, and grape"


def test_find_representative(use_ann: bool, model: Encoder, train_texts: list[str], test_texts: list[str]) -> None:
    """Test the find_representative method."""
    semhash = SemHash.from_records(records=train_texts, use_ann=use_ann, model=model)
    result = semhash.find_representative(records=test_texts, candidate_limit=5, selection_size=3, lambda_param=0.5)
    assert len(result.selected) == 3, "Expected 3 representatives"
    selected = {r["text"] for r in result.selected}
    assert selected == {"grapefruit", "banana", "apple"}, "Expected representatives to be grapefruit, banana, and apple"


def test_filter_outliers(use_ann: bool, model: Encoder, train_texts: list[str], test_texts: list[str]) -> None:
    """Test the filter_outliers method."""
    semhash = SemHash.from_records(records=train_texts, use_ann=use_ann, model=model)
    result = semhash.filter_outliers(records=test_texts, outlier_percentage=0.2)
    assert len(result.filtered) == 2, "Expected 2 outliers"
    assert len(result.selected) == len(test_texts) - 2
    filtered = {r["text"] for r in result.filtered}
    assert filtered == {"motorcycle", "plane"}, "Expected outliers to be motorcycle and plane"


def test_self_filter_outliers(use_ann: bool, model: Encoder, train_texts: list[str]) -> None:
    """Test the self_filter_outliers method."""
    semhash = SemHash.from_records(records=train_texts, use_ann=use_ann, model=model)
    result = semhash.self_filter_outliers(outlier_percentage=0.1)
    assert len(result.filtered) == 2, "Expected 2 outliers"
    assert len(result.selected) == len(train_texts) - 2
    filtered = {r["text"] for r in result.filtered}
    assert filtered == {"car", "bicycle"}, "Expected outliers to be car and bicycle"


def test__mmr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the _mmr method."""
    # Create a dummy SemHash instance
    semhash = SemHash(index=None, model=None, columns=["text"], was_string=True)  # type: ignore
    # Prepare a fake ranking with three records
    records = ["a", "b", "c"]
    scores = [3.0, 2.0, 1.0]
    ranking = FilterResult(selected=records, filtered=[], scores_selected=scores, scores_filtered=[])
    # Create dummy embeddings for the records
    embeddings = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    # Monkeypatch featurize to return the dummy embeddings
    monkeypatch.setattr(semhash, "_featurize", lambda records, columns, model: embeddings)

    # Test lambda=1.0: pure relevance, should pick top 2 by score
    result_rel = semhash._mmr(ranking, candidate_limit=3, selection_size=2, lambda_param=1.0)
    assert result_rel.selected == ["a", "b"]

    # Test lambda=0.0: pure diversity, should first pick 'a', then pick most dissimilar: 'c'
    result_div = semhash._mmr(ranking, candidate_limit=3, selection_size=2, lambda_param=0.0)
    assert result_div.selected == ["a", "c"]


def test_mmr_invalid_lambda_raises() -> None:
    """Test that invalid lambda values raise ValueError."""
    semhash = SemHash(index=None, model=None, columns=["text"], was_string=True)  # type: ignore
    dummy = FilterResult(selected=["x"], filtered=[], scores_selected=[0.5], scores_filtered=[])
    with pytest.raises(ValueError):
        semhash._mmr(dummy, candidate_limit=1, selection_size=1, lambda_param=-0.1)
    with pytest.raises(ValueError):
        semhash._mmr(dummy, candidate_limit=1, selection_size=1, lambda_param=1.1)
