import numpy as np

from semhash import SemHash


def test_single_list_deduplication(semhash: SemHash) -> None:
    """Test single input list deduplication."""
    # No duplicates
    texts = [
        "It's dangerous to go alone!",
        "It's a secret to everybody.",
        "Ganondorf has invaded Hyrule!",
    ]
    semhash.fit(records=texts)
    deduplicated_texts = semhash.fit_deduplicate(texts)
    assert deduplicated_texts == texts

    # With duplicates
    texts = [
        "It's dangerous to go alone!",
        "It's dangerous to go alone!",  # Exact duplicate
        "It's risky to go alone!",  # Semantically similar
    ]
    deduplicated_texts = semhash.fit_deduplicate(texts)
    assert deduplicated_texts == ["It's dangerous to go alone!"]


def test_cross_list_deduplication(semhash: SemHash) -> None:
    """Test deduplication across two lists."""
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
    deduplicated_texts = semhash.deduplicate(texts2)

    assert deduplicated_texts == texts2

    # # With duplicates
    texts2 = [
        "It's dangerous to go alone!",  # Exact duplicate
        "It's risky to go alone!",  # Semantically similar
        "Ganondorf has attacked Hyrule!",  # Semantically similar
    ]
    deduplicated_texts = semhash.deduplicate(texts2)
    assert deduplicated_texts == []
