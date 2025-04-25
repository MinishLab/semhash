from typing import Any, Protocol, Sequence, Union

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
