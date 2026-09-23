from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal
from zlib import crc32

import numpy as np

# Permutations are `a * h + b` in wrapping uint32 arithmetic, which is a bijection for odd `a` and, unlike a
# modulo in uint64, vectorizes. The minimum is taken over whole values, but its lowest bit only depends on the
# lowest bits of `h`, so the kept bit is the top bit of the minimum after multiplying by this odd constant.
_MIX = np.uint32(0x9E3779B1)

# Signatures are packed in batches, so that encoding a large dataset does not first materialize every
# unpacked signature. At a million records that transient costs more than a gigabyte.
_CHUNK_SIZE = 8192


class MinHashEncoder:
    """Encode text as a 1-bit MinHash signature whose cosine similarity estimates Jaccard."""

    def __init__(
        self,
        num_perm: int = 512,
        ngram_size: int = 3,
        analyzer: Literal["word", "char"] = "word",
        seed: int = 42,
    ) -> None:
        """
        Initialize a MinHashEncoder.

        Each record is shingled into n-grams, reduced to `num_perm` MinHash values, and the lowest bit of
        every MinHash value is kept. The signature is returned bit-packed, so that the Hamming distance `h`
        between two signatures gives an estimate `1 - 2 * h / num_perm` of the Jaccard similarity of the
        shingle sets.

        The standard deviation of the estimate is `sqrt((1 - J**2) / num_perm)` at Jaccard similarity `J`, which
        is at most 0.06 at 256 permutations, 0.04 at 512, and 0.03 at 1024.

        :param num_perm: The number of permutations. Signatures are `num_perm // 8` bytes wide.
        :param ngram_size: The number of words or characters per shingle.
        :param analyzer: Whether to shingle over words or characters.
        :param seed: The seed used to draw the permutation coefficients.
        :raises ValueError: If num_perm is not a positive multiple of 8, ngram_size is not positive, or
            analyzer is unknown.
        """
        if num_perm < 8 or num_perm % 8:
            raise ValueError(f"num_perm must be a positive multiple of 8, got {num_perm}")
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be positive, got {ngram_size}")
        if analyzer not in ("word", "char"):
            raise ValueError(f"analyzer must be 'word' or 'char', got {analyzer!r}")

        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.analyzer = analyzer
        self.seed = seed

        rng = np.random.default_rng(seed)
        self._a = (rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64) | 1).astype(np.uint32)
        self._b = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64).astype(np.uint32)

    def _shingle(self, text: str) -> set[str]:
        """Split a text into its set of n-grams, falling back to the whole text if it is too short."""
        if self.analyzer == "word":
            units: Sequence[str] = text.split()
            join = " ".join
        else:
            units = text
            join = "".join
        if len(units) < self.ngram_size:
            return {text}
        return {join(units[i : i + self.ngram_size]) for i in range(len(units) - self.ngram_size + 1)}

    def _signature(self, text: str) -> np.ndarray:
        """Compute the 1-bit MinHash signature of a single text, as one bit per permutation."""
        shingles = self._shingle(text)
        hashes = np.fromiter(
            (crc32(shingle.encode("utf-8")) for shingle in shingles), dtype=np.uint32, count=len(shingles)
        )
        permuted = np.multiply.outer(hashes, self._a)
        permuted += self._b
        return ((permuted.min(axis=0) * _MIX) >> np.uint32(31)).astype(np.uint8)

    def encode(self, inputs: Sequence[Any] | Any, **kwargs: Any) -> np.ndarray:
        """
        Encode a sequence of texts into 1-bit MinHash signatures.

        :param inputs: The texts to encode.
        :param **kwargs: Ignored, present to satisfy the Encoder protocol.
        :return: A bit-packed uint8 array of shape (len(inputs), num_perm // 8).
        :raises TypeError: If any input is not a string.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        for text in inputs:
            if not isinstance(text, str):
                raise TypeError(
                    f"MinHashEncoder only encodes strings, got {type(text).__name__}. "
                    f"Lexical deduplication is text-only; use mode='semantic' with a multimodal encoder instead."
                )
        signatures = np.empty((len(inputs), self.num_perm // 8), dtype=np.uint8)
        for start in range(0, len(inputs), _CHUNK_SIZE):
            chunk = [self._signature(text) for text in inputs[start : start + _CHUNK_SIZE]]
            signatures[start : start + len(chunk)] = np.packbits(np.stack(chunk), axis=1)
        return signatures


def unpack_signatures(vectors: np.ndarray) -> np.ndarray:
    """
    Unpack bit-packed MinHash signatures into -1 and +1 vectors.

    The cosine similarity between two unpacked signatures is the same estimate of Jaccard similarity that the
    Hamming distance between the packed signatures gives, which lets them be used with code that expects a
    real vector space.

    :param vectors: A bit-packed uint8 array of shape (n_items, n_bits // 8).
    :return: A float32 array of shape (n_items, n_bits) containing -1 and +1.
    """
    return np.unpackbits(vectors, axis=1).astype(np.float32) * 2.0 - 1.0
