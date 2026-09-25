from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal
from zlib import crc32

import numpy as np

# Mixes the minimum before its top bit is kept, since the low bits of `a * h + b` only depend on the low bits of `h`.
_MIX = np.uint32(0x9E3779B1)


class MinHashEncoder:
    """Encode texts as bit-packed 1-bit MinHash signatures, whose Hamming distance estimates Jaccard similarity."""

    def __init__(
        self,
        num_perm: int = 512,
        ngram_size: int = 3,
        analyzer: Literal["word", "char"] = "word",
        seed: int = 42,
    ) -> None:
        """
        Initialize a MinHashEncoder.

        :param num_perm: The number of permutations. Signatures are `num_perm // 8` bytes, and more permutations give
            a more accurate estimate.
        :param ngram_size: The number of words or characters per n-gram.
        :param analyzer: Whether to build n-grams from words or characters.
        :param seed: The random seed for the permutations.
        :raises ValueError: If num_perm is not a positive multiple of 8, ngram_size is not positive, or analyzer is
            unknown.
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

        # Permutations are `a * h + b` in wrapping uint32 arithmetic, a bijection for odd `a` that vectorizes well.
        rng = np.random.default_rng(seed)
        self._a = (rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64) | 1).astype(np.uint32)
        self._b = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64).astype(np.uint32)

    def _shingle(self, text: str) -> set[str]:
        """Split a text into its set of n-grams, or the whole text if it is too short."""
        units: Sequence[str] = text.split() if self.analyzer == "word" else text
        separator = " " if self.analyzer == "word" else ""
        if len(units) < self.ngram_size:
            return {text}
        return {separator.join(units[i : i + self.ngram_size]) for i in range(len(units) - self.ngram_size + 1)}

    def _signature(self, text: str) -> np.ndarray:
        """Compute the bit-packed MinHash signature of a single text."""
        shingles = self._shingle(text)
        hashes = np.fromiter((crc32(s.encode("utf-8")) for s in shingles), dtype=np.uint32, count=len(shingles))
        permuted = np.multiply.outer(hashes, self._a)
        permuted += self._b
        return np.packbits((permuted.min(axis=0) * _MIX) >> np.uint32(31))

    def encode(self, inputs: Sequence[Any] | Any, **kwargs: Any) -> np.ndarray:
        """
        Encode texts into bit-packed MinHash signatures.

        :param inputs: The texts to encode.
        :param **kwargs: Ignored, present to satisfy the Encoder protocol.
        :return: A uint8 array of shape (len(inputs), num_perm // 8).
        :raises TypeError: If any input is not a string.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        signatures = np.empty((len(inputs), self.num_perm // 8), dtype=np.uint8)
        for i, text in enumerate(inputs):
            if not isinstance(text, str):
                raise TypeError(f"MinHashEncoder only encodes strings, got {type(text).__name__}")
            signatures[i] = self._signature(text)
        return signatures
