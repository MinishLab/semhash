import pytest

from semhash import SemHash


def test_single_dataset_deduplication(semhash: SemHash) -> None:
    """Test single dataset deduplication."""
    # No duplicates
    texts = [
        "It's dangerous to go alone!",
        "The master sword can seal the darkness.",
        "Ganondorf has invaded Hyrule!",
    ]
    deduplicated_texts = semhash.fit_deduplicate(texts).deduplicated
    assert deduplicated_texts == texts

    # With duplicates
    texts = [
        "It's dangerous to go alone!",
        "It's dangerous to go alone!",  # Exact duplicate
        "It's not safe to go alone!",  # Semantically similar
    ]
    deduplicated_texts = semhash.fit_deduplicate(texts).deduplicated
    assert deduplicated_texts == ["It's dangerous to go alone!"]


def test_multi_dataset_deduplication(semhash: SemHash) -> None:
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
    semhash.fit(texts1)
    deduplicated_texts = semhash.deduplicate(texts2).deduplicated

    assert deduplicated_texts == texts2

    # # With duplicates
    texts2 = [
        "It's dangerous to go alone!",  # Exact duplicate
        "It's risky to go alone!",  # Semantically similar
        "Ganondorf has attacked Hyrule!",  # Semantically similar
    ]
    deduplicated_texts = semhash.deduplicate(texts2).deduplicated
    assert deduplicated_texts == []


def test_single_dataset_deduplication_multicolumn(semhash: SemHash) -> None:
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

    semhash.columns = ["question", "context", "answer"]
    deduplicated = semhash.fit_deduplicate(records)
    assert deduplicated.deduplicated == [
        {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},
        {"question": "Who is the princess?", "context": "The princess is Zelda", "answer": "Zelda"},
    ]


def test_multi_dataset_deduplication_multicolumn(semhash: SemHash) -> None:
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

    semhash.columns = ["question", "context", "answer"]
    semhash.fit(train_records)
    deduplicated = semhash.deduplicate(test_records).deduplicated
    assert deduplicated == [
        {"question": "What is the villain's name?", "context": "The villain is Ganon", "answer": "Ganon"}
    ]


def test_fit_without_columns(semhash: SemHash) -> None:
    """Test fitting without specifying columns."""
    records = [
        {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},
        {"question": "Who is the princess?", "context": "The princess is Zelda", "answer": "Zelda"},
    ]
    with pytest.raises(ValueError):
        semhash.fit(records)


def test_deduplicate_without_index(semhash: SemHash) -> None:
    """Test deduplicating without fitting."""
    texts = ["It's dangerous to go alone!"]
    with pytest.raises(ValueError):
        semhash.deduplicate(texts)


def test__featurize_without_columns(semhash: SemHash) -> None:
    """Test featurizing without specifying columns."""
    records = [
        {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"},
        {"question": "Who is the princess?", "context": "The princess is Zelda", "answer": "Zelda"},
    ]
    with pytest.raises(ValueError):
        semhash._featurize(records)


def test__unpack_record_without_columns(semhash: SemHash) -> None:
    """Test unpacking records without specifying columns."""
    record = {"question": "What is the hero's name?", "context": "The hero is Link", "answer": "Link"}
    with pytest.raises(ValueError):
        semhash._unpack_record(record)
