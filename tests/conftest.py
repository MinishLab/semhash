import pytest
from model2vec import StaticModel


@pytest.fixture
def model() -> StaticModel:
    """Load a model for testing."""
    return StaticModel.from_pretrained("tests/data/test_model")


@pytest.fixture(params=[True, False], ids=["use_ann=True", "use_ann=False"])
def use_ann(request: pytest.FixtureRequest) -> bool:
    """Whether to use approximate nearest neighbors or not."""
    return request.param
