import imp
import math
from typing import Any, List, Protocol, Sequence, Union

import numpy as np


class Encoder(Protocol):
    """An encoder protocol for SemHash."""

    def encode(
        self,
        sentences: Union[list[str], str, Sequence[str]],
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode a list of sentences into embeddings.

        :param sentences: A list of sentences to encode.
        :param **kwargs: Additional keyword arguments.
        :return: The embeddings of the sentences.
        """
        ...  # pragma: no cover

        
def entropy_from_distances(distances: np.ndarray, base: float = 2) -> float:
    """
    Compute entropy from embedding distances.

    :param distances: List of distances between embeddings
    :param base: Base of logarithm for entropy calculation (default: 2)
        We use base 2 because Entropy measured in bits (binary units), which is the natural unit for information.
    :return: Shannon entropy of the distance distribution
    """
    # Convert cosine distances to similarities
    # Cosine similarity is 1 - cosine distance
    similarities = 1 - np.array(distances)

    # Normalize to create probability distribution
    probabilities = similarities / np.sum(similarities)

    # Compute entropy
    # Handle zero probabilities to avoid log(0)
    entropy_value = -np.sum(np.where(probabilities > 0, probabilities * np.log(probabilities) / math.log(base), 0))

    return entropy_value