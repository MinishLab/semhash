from typing import Any, Callable

import numpy as np
import pytest

from semhash import MinHashEncoder, SemHash


def _estimated_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Estimate Jaccard similarity from the Hamming distance between two packed signatures."""
    return 1 - 2 * int(np.unpackbits(a ^ b).sum()) / (a.size * 8)


@pytest.mark.parametrize("overlap", [0, 10, 25, 40, 50])
def test_signatures_estimate_jaccard(overlap: int) -> None:
    """The Hamming distance between two signatures estimates the Jaccard similarity of their n-grams."""
    encoder = MinHashEncoder(num_perm=1024)
    left = " ".join(f"w{i}" for i in range(50))
    right = " ".join(f"w{i}" for i in list(range(overlap)) + list(range(100, 150 - overlap)))
    x, y = encoder._shingle(left), encoder._shingle(right)

    signatures = encoder.encode([left, right])

    # At 1024 permutations the standard deviation of the estimate is at most 0.031, so this allows about 3 sigma.
    assert _estimated_jaccard(*signatures) == pytest.approx(len(x & y) / len(x | y), abs=0.1)


@pytest.mark.parametrize("text", ["", "short", "a somewhat longer text that has plenty of words in it"])
@pytest.mark.parametrize("analyzer", ["word", "char"])
def test_encode_is_packed_and_deterministic(text: str, analyzer: Any) -> None:
    """Texts of any length encode to num_perm bits, and identical texts encode identically."""
    signatures = MinHashEncoder(num_perm=32, analyzer=analyzer).encode([text, text])

    assert signatures.shape == (2, 4)
    assert signatures.dtype == np.uint8
    assert np.array_equal(signatures[0], signatures[1])


@pytest.mark.parametrize(
    ("create", "error", "message"),
    [
        (lambda: MinHashEncoder(num_perm=12), ValueError, "num_perm must be a positive multiple of 8"),
        (lambda: MinHashEncoder(ngram_size=0), ValueError, "ngram_size must be positive"),
        (lambda: MinHashEncoder(analyzer="sentence"), ValueError, "analyzer must be"),
        (lambda: MinHashEncoder().encode([b"bytes"]), TypeError, "only encodes strings"),
        (lambda: SemHash.from_records(["a b c"], mode="fuzzy"), ValueError, "mode must be"),
        (lambda: SemHash.from_records(["a b c"], mode="lexical", model=MinHashEncoder()), ValueError, "cannot be"),
    ],
)
def test_invalid_arguments(create: Callable[[], Any], error: type[Exception], message: str) -> None:
    """Invalid arguments are rejected."""
    with pytest.raises(error, match=message):
        create()


def test_lexical_mode_deduplicates_on_wording() -> None:
    """Lexical mode removes n-gram near-duplicates, keeps paraphrases, and scores with estimated Jaccard."""
    texts = [
        "the master sword can seal the darkness",
        "the master sword can seal the darkness forever",  # Lexical near-duplicate
        "the legendary blade banishes evil",  # Same meaning, no shared n-grams
    ]
    semhash = SemHash.from_records(records=texts, mode="lexical")
    result = semhash.self_deduplicate()

    assert result.threshold == 0.7
    assert result.selected == [texts[0], texts[2]]
    signatures = semhash.model.encode(texts[:2])  # type: ignore[union-attr]
    assert result.filtered[0].duplicates[0][1] == pytest.approx(_estimated_jaccard(*signatures), abs=1e-6)
    assert len(semhash.self_find_representative(selection_size=2).selected) == 2
