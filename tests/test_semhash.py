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
    deduplicated_indices, duplicate_mapping = semhash.deduplicate(texts)
    np.testing.assert_array_equal(deduplicated_indices, np.array([0, 1, 2]))
    assert duplicate_mapping == {}

    # With duplicates
    texts = [
        "It's dangerous to go alone!",
        "It's dangerous to go alone!",  # Exact duplicate
        "It's risky to go alone!",  # Semantically similar
    ]
    deduplicated_indices, duplicate_mapping = semhash.deduplicate(texts)
    np.testing.assert_array_equal(deduplicated_indices, np.array([0]))
    assert duplicate_mapping == {1: 0, 2: 0}


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
    deduplicated_indices, duplicate_mapping = semhash.deduplicate(texts1, texts2)

    # # With duplicates
    texts2 = [
        "It's dangerous to go alone!",  # Exact duplicate
        "It's risky to go alone!",  # Semantically similar
        "Ganondorf has attacked Hyrule!",  # Semantically similar
    ]
    deduplicated_indices, duplicate_mapping = semhash.deduplicate(texts1, texts2)
    np.testing.assert_array_equal(deduplicated_indices, np.array([]))
    assert duplicate_mapping == {0: 0, 1: 0, 2: 2}
