import numpy as np
import pytest

from semhash import MinHashEncoder, SemHash
from semhash.minhash import unpack_signatures


def _jaccard(a: str, b: str, encoder: MinHashEncoder) -> float:
    """Exact Jaccard similarity between the shingle sets of two texts."""
    x, y = encoder._shingle(a), encoder._shingle(b)
    return len(x & y) / len(x | y)


@pytest.mark.parametrize("overlap", [0, 10, 25, 40, 50])
def test_cosine_estimates_jaccard(overlap: int) -> None:
    """The cosine similarity of two signatures estimates the Jaccard similarity of the shingle sets."""
    encoder = MinHashEncoder(num_perm=1024)
    left = " ".join(f"w{i}" for i in range(50))
    right = " ".join(f"w{i}" for i in list(range(overlap)) + list(range(100, 150 - overlap)))

    signatures = unpack_signatures(encoder.encode([left, right]))
    estimate = float(signatures[0] @ signatures[1]) / encoder.num_perm

    # At 1024 permutations the standard deviation of the estimate is at most 0.031, so this allows about 3 sigma.
    assert estimate == pytest.approx(_jaccard(left, right, encoder), abs=0.1)


def test_encode_is_bit_packed_and_deterministic() -> None:
    """Signatures are one bit per permutation, and identical texts encode identically."""
    encoder = MinHashEncoder(num_perm=64)
    signatures = encoder.encode(["the quick brown fox", "the quick brown fox", "something else entirely"])

    assert signatures.shape == (3, 8)
    assert signatures.dtype == np.uint8
    assert np.array_equal(signatures[0], signatures[1])
    assert not np.array_equal(signatures[0], signatures[2])
    assert set(np.unique(unpack_signatures(signatures))) == {-1.0, 1.0}


@pytest.mark.parametrize(
    "text",
    ["", "short", "a somewhat longer text that has plenty of words in it"],
)
@pytest.mark.parametrize("analyzer", ["word", "char"])
def test_encode_handles_any_length(text: str, analyzer: str) -> None:
    """Texts shorter than the shingle size fall back to the whole text instead of producing no shingles."""
    encoder = MinHashEncoder(num_perm=32, analyzer=analyzer)  # type: ignore[arg-type]
    assert encoder.encode([text]).shape == (1, 4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_perm": 0}, "num_perm must be a positive multiple of 8"),
        ({"num_perm": 12}, "num_perm must be a positive multiple of 8"),
        ({"ngram_size": 0}, "ngram_size must be positive"),
        ({"analyzer": "sentence"}, "analyzer must be"),
    ],
)
def test_invalid_arguments(kwargs: dict, message: str) -> None:
    """Invalid constructor arguments are rejected."""
    with pytest.raises(ValueError, match=message):
        MinHashEncoder(**kwargs)


def test_encode_rejects_non_strings() -> None:
    """Non-text input is rejected with a message pointing at semantic mode."""
    with pytest.raises(TypeError, match="only encodes strings"):
        MinHashEncoder().encode([b"bytes"])


def test_lexical_mode_deduplicates_on_wording() -> None:
    """Lexical mode removes records that share n-grams and keeps records that only share meaning."""
    texts = [
        "the master sword can seal the darkness",
        "the master sword can seal the darkness forever",  # Lexical near-duplicate
        "the legendary blade banishes evil",  # Same meaning, no shared n-grams
    ]
    semhash = SemHash.from_records(records=texts, mode="lexical")

    assert semhash.mode == "lexical"
    assert semhash.self_deduplicate().selected == [texts[0], texts[2]]


def test_default_threshold_depends_on_mode() -> None:
    """Each mode has its own default threshold, since cosine and Jaccard are not on the same scale."""
    lexical = SemHash.from_records(records=["one two three four"], mode="lexical")

    assert lexical.self_deduplicate().threshold == 0.7
    assert lexical.self_deduplicate(threshold=0.95).threshold == 0.95


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "fuzzy"}, "mode must be"),
        ({"mode": "lexical", "model": MinHashEncoder()}, "cannot be combined with a custom model"),
    ],
)
def test_invalid_mode_arguments(kwargs: dict, message: str) -> None:
    """Unknown modes and mode/model combinations that contradict each other are rejected."""
    with pytest.raises(ValueError, match=message):
        SemHash.from_records(records=["a b c"], **kwargs)


def test_binary_index_scores_are_estimated_jaccard() -> None:
    """The Hamming index reports estimated Jaccard, matching what the signatures themselves say."""
    texts = ["the master sword can seal the darkness", "the master sword can seal the darkness forever"]
    semhash = SemHash.from_records(records=texts, mode="lexical")

    assert semhash.index.vectors.dtype == np.uint8
    assert semhash.index.distance_scale == 512 / 2

    signatures = unpack_signatures(semhash.model.encode(texts))  # type: ignore[union-attr]
    expected = float(signatures[0] @ signatures[1]) / 512
    (duplicate,) = semhash.self_deduplicate(threshold=0.5).filtered

    assert duplicate.duplicates[0][1] == pytest.approx(expected, abs=1e-6)
