import pytest

from semhash import SemHash
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
    deduplicated_texts = semhash.self_deduplicate().selected
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
    deduplicated_texts = semhash.deduplicate(texts2).selected
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
    deduplicated = semhash.self_deduplicate()

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


def test_filter_by_entropy(use_ann: bool, model: Encoder) -> None:
    """Test filtering by entropy."""
    texts = [f"Text {i}" for i in range(1000)]
    semhash = SemHash.from_records(texts, use_ann=use_ann, model=model)

    # Test with absolute budget
    filtered = semhash.self_filter_by_entropy(budget=200)
    assert len(filtered.selected) == 200
    assert len(filtered.filtered) == 800
    assert len(filtered.scores_selected) == 200
    assert len(filtered.scores_filtered) == 800

    # Test with percentage budget
    filtered = semhash.self_filter_by_entropy(budget=0.5)
    assert len(filtered.selected) == 500
    assert len(filtered.filtered) == 500


def test_filter_by_entropy_invalid_budget(use_ann: bool, model: Encoder) -> None:
    """Test filtering by entropy with invalid budget."""
    texts = ["Text 1", "Text 2", "Text 3"]
    semhash = SemHash.from_records(texts, use_ann=use_ann, model=model)

    with pytest.raises(ValueError):
        semhash.self_filter_by_entropy(budget=4)  # Budget larger than dataset

    with pytest.raises(ValueError):
        semhash.self_filter_by_entropy(budget=-1)  # Negative budget


def test_filter_by_entropy_string_validation(use_ann: bool, model: Encoder) -> None:
    """Test filtering by entropy with string validation."""
    texts = [f"Text {i}" for i in range(1000)]
    records = [{"text": t} for t in texts]

    # Initialize with strings
    semhash_str = SemHash.from_records(texts, use_ann=use_ann, model=model)

    # Initialize with dicts
    semhash_dict = SemHash.from_records(records, columns=["text"], use_ann=use_ann, model=model)

    # Should work - filtering strings with string-initialized SemHash
    semhash_str.filter_by_entropy(texts, budget=0.5)

    # Should work - filtering dicts with dict-initialized SemHash
    semhash_dict.filter_by_entropy(records, budget=0.5)

    # Should fail - filtering strings with dict-initialized SemHash
    with pytest.raises(ValueError):
        semhash_dict.filter_by_entropy(texts, budget=0.5)
