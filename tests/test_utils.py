import numpy as np
import pytest
from frozendict import frozendict

from semhash.records import prepare_records, remove_exact_duplicates
from semhash.utils import Encoder, coerce_value, compute_candidate_limit, featurize, make_hashable, to_frozendict


def test_make_hashable() -> None:
    """Test make_hashable with various types."""
    # Fast path: primitives
    assert make_hashable("hello") == "hello"
    assert make_hashable(42) == 42
    assert make_hashable(3.14) == 3.14
    assert make_hashable(True) is True
    assert make_hashable(None) is None

    # Objects with tobytes() (simulate PIL Image or numpy array)
    class MockImage:
        def tobytes(self) -> bytes:
            return b"fake_image_data"

    img = MockImage()
    result = make_hashable(img)
    assert isinstance(result, str)
    assert len(result) == 32  # MD5 hex digest

    # Hashable objects (like tuples)
    assert make_hashable((1, 2, 3)) == (1, 2, 3)

    # Non-hashable fallback to string
    unhashable = {"key": "value"}
    result = make_hashable(unhashable)
    assert result == "{'key': 'value'}"


def test_coerce_value() -> None:
    """Test coerce_value for encoding preparation."""
    # Strings and bytes pass through
    assert coerce_value("hello") == "hello"
    assert coerce_value(b"bytes") == b"bytes"

    # Primitives converted to strings
    assert coerce_value(42) == "42"
    assert coerce_value(3.14) == "3.14"
    assert coerce_value(True) == "True"

    # Complex types pass through unchanged
    class MockImage:
        pass

    img = MockImage()
    assert coerce_value(img) is img


def test_to_frozendict() -> None:
    """Test converting dict to frozendict, including error cases."""
    record = {"a": "1", "b": "2", "c": "3"}

    # Basic case: select subset of columns
    result = to_frozendict(record, {"a", "c"})
    assert result == frozendict({"a": "1", "c": "3"})
    assert "b" not in result

    # Works with Sequence (not just set)
    result = to_frozendict(record, ["a", "b"])
    assert result == frozendict({"a": "1", "b": "2"})

    # Missing column raises ValueError
    with pytest.raises(ValueError, match="Missing column 'missing'"):
        to_frozendict(record, {"a", "missing"})


def test_compute_candidate_limit() -> None:
    """Test candidate limit computation."""
    # Basic case
    assert compute_candidate_limit(1000, 10) == 100
    # Smaller than min_candidates (but max is capped at total)
    assert compute_candidate_limit(50, 10) == 50
    # Larger than max_candidates
    assert compute_candidate_limit(20000, 10) == 1000
    # Selection size larger than fraction
    assert compute_candidate_limit(100, 50) == 100


def test_featurize(model: Encoder) -> None:
    """Test featurizing records, including error cases."""
    records = [{"text": "hello"}, {"text": "world"}]
    embeddings = featurize(records, ["text"], model)
    assert embeddings.shape == (2, 128)  # Model has 128 dims
    assert isinstance(embeddings, np.ndarray)

    # Missing column raises ValueError
    with pytest.raises(ValueError, match="Missing column 'missing'"):
        featurize(records, ["missing"], model)

    # Non-text data with text encoder raises helpful TypeError
    class FakeImage:
        pass

    records_with_images = [{"img": FakeImage()}, {"img": FakeImage()}]
    with pytest.raises(TypeError, match="Failed to encode column 'img'"):
        featurize(records_with_images, ["img"], model)
    with pytest.raises(TypeError, match="data type: FakeImage"):
        featurize(records_with_images, ["img"], model)


def test_remove_exact_duplicates() -> None:
    """Test exact duplicate removal, with and without reference records."""
    # Basic case: remove duplicates within same list
    records = [
        {"text": "hello", "id": "1"},
        {"text": "world", "id": "2"},
        {"text": "hello", "id": "3"},
    ]
    deduplicated, duplicates = remove_exact_duplicates(records, ["text"])
    assert len(deduplicated) == 2
    assert len(duplicates) == 1
    assert duplicates[0][0] == {"text": "hello", "id": "3"}

    # With reference_records: cross-dataset filtering
    reference_records = [
        [{"text": "apple"}],
        [{"text": "banana"}, {"text": "banana"}],
    ]
    new_records = [
        {"text": "cherry"},  # New
        {"text": "apple"},  # Exists in reference
        {"text": "date"},  # New
        {"text": "banana"},  # Exists in reference
    ]
    deduplicated, duplicates = remove_exact_duplicates(new_records, ["text"], reference_records=reference_records)
    assert len(deduplicated) == 2
    assert {"text": "cherry"} in deduplicated
    assert {"text": "date"} in deduplicated
    assert len(duplicates) == 2


def test_prepare_records() -> None:
    """Test preparing records, including validation and edge cases."""
    # String records -> converts to dicts with "text" column
    records = ["hello", "world"]
    dict_records, columns, was_string = prepare_records(records, None)
    assert was_string is True
    assert columns == ["text"]
    assert dict_records == [{"text": "hello"}, {"text": "world"}]

    # Dict records
    records = [{"text": "hello"}, {"text": "world"}]
    dict_records, columns, was_string = prepare_records(records, ["text"])
    assert was_string is False
    assert columns == ["text"]
    assert dict_records == records

    # Dict records without columns raises ValueError
    with pytest.raises(ValueError, match="Columns must be specified"):
        prepare_records([{"text": "hello"}], None)

    # Empty records raises ValueError
    with pytest.raises(ValueError, match="records must not be empty"):
        prepare_records([], None)

    # Mixed types rejected
    with pytest.raises(ValueError, match="All records must be"):
        prepare_records(["a", {"text": "b"}], None)
    with pytest.raises(ValueError, match="All records must be"):
        prepare_records([{"text": "a"}, "b"], ["text"])
