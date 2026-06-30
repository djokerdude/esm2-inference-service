import pytest
from src.model import _validate_sequences, MAX_SEQUENCE_LENGTH


def test_valid_sequence():
    _validate_sequences(["MKTLL"])


def test_valid_batch():
    _validate_sequences(["MKTLL", "MVLSP", "ACDEFGHIKLMNPQRSTVWY"])


def test_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        _validate_sequences([])


def test_empty_string_raises():
    with pytest.raises(ValueError, match="empty"):
        _validate_sequences([""])


def test_invalid_characters_raises():
    with pytest.raises(ValueError, match="invalid"):
        _validate_sequences(["MKTLL123"])


def test_sequence_too_long_raises():
    with pytest.raises(ValueError, match="exceeds"):
        _validate_sequences(["M" * (MAX_SEQUENCE_LENGTH + 1)])


def test_one_invalid_in_batch_raises():
    with pytest.raises(ValueError):
        _validate_sequences(["MKTLL", "INVALID123", "MVLSP"])


def test_lowercase_is_accepted():
    # _validate_sequences uppercases internally, so lowercase is valid at the
    # model layer. The server's field_validator also uppercases, so this is
    # defensive — either layer handles it without raising.
    _validate_sequences(["mktll"])
