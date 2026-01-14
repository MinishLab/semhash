from typing import Any

import pytest
from model2vec import StaticModel

from semhash.utils import Encoder


@pytest.fixture
def model() -> StaticModel:
    """Load a model for testing."""
    return StaticModel.from_pretrained("tests/data/test_model")


@pytest.fixture
def train_texts() -> list[str]:
    """A list of train texts for testing outlier and representative filtering."""
    return [
        "apple",
        "banana",
        "cherry",
        "strawberry",
        "blueberry",
        "raspberry",
        "blackberry",
        "peach",
        "plum",
        "grape",
        "mango",
        "papaya",
        "pineapple",
        "watermelon",
        "orange",
        "lemon",
        "lime",
        "tangerine",
        "car",  # Outlier
        "bicycle",  # Outlier
    ]


@pytest.fixture
def test_texts() -> list[str]:
    """A list of test texts for testing outlier and representative filtering."""
    return [
        "apple",
        "banana",
        "kiwi",
        "fig",
        "apricot",
        "grapefruit",
        "pomegranate",
        "motorcycle",  # Outlier
        "plane",  # Outlier
    ]


class CountingEncoder:
    """Encoder wrapper that counts how many items are encoded. Useful for testing efficiency."""

    def __init__(self, base_encoder: Encoder) -> None:
        """Initialize the counting encoder."""
        self.base_encoder = base_encoder
        self.encode_calls: list[int] = []

    def encode(self, sentences: Any, **kwargs: Any) -> Any:
        """Encode sentences and count the number of items encoded."""
        if isinstance(sentences, str):
            sentences = [sentences]
        self.encode_calls.append(len(sentences))
        return self.base_encoder.encode(sentences, **kwargs)

    @property
    def total_encoded(self) -> int:
        """Total number of items encoded across all calls."""
        return sum(self.encode_calls)

    def reset(self) -> None:
        """Reset the call counter."""
        self.encode_calls = []


@pytest.fixture
def counting_encoder(model: StaticModel) -> CountingEncoder:
    """A counting encoder wrapper for testing embedding efficiency."""
    return CountingEncoder(model)
