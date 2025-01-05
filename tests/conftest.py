import pytest
from model2vec import StaticModel

from semhash import SemHash


@pytest.fixture
def model() -> StaticModel:
    """Load a model for testing."""
    model = StaticModel.from_pretrained("tests/data/test_model")
    return model


@pytest.fixture(params=[True, False], ids=["ann=True", "ann=False"])
def semhash(request: pytest.FixtureRequest, model: StaticModel) -> SemHash:
    """Load a SemHash object for testing with parametrized ann."""
    ann = request.param
    return SemHash(model=model, ann=ann)
