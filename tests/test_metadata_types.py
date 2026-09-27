"""Protect optional native metadata values and unsupported legacy field handling."""

from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pyarrow.parquet as pq
import pytest

from iparq.source import optional_column_metadata, read_parquet_metadata


@pytest.mark.parametrize("value", [True, False, None])
def test_optional_legacy_index_flags_preserve_boolean_states(
    value: bool | None,
) -> None:
    """Preserve known true, known false, and unknown legacy index flags."""
    metadata = SimpleNamespace(has_index_page=value)

    assert optional_column_metadata(metadata, "has_index_page") is value


@pytest.mark.parametrize("value", [0, 128, None])
def test_optional_legacy_offsets_preserve_zero_and_missing_values(
    value: int | None,
) -> None:
    """Preserve zero offsets independently from missing legacy page location values."""
    metadata = SimpleNamespace(index_page_offset=value)

    result = optional_column_metadata(metadata, "index_page_offset")
    assert result == value
    assert result is None or type(result) is int


class UnsupportedLegacyMetadata:
    """Represent legacy fields that native readers cannot expose successfully."""

    def __init__(self, error: type[Exception]) -> None:
        """Store the native field access failure simulated by this metadata object."""
        self.error = error

    def __getattr__(self, name: str) -> NoReturn:
        """Raise the simulated native failure whenever an unavailable field is requested."""
        raise self.error(name)


@pytest.mark.parametrize("error", [AttributeError, NotImplementedError])
def test_unsupported_optional_metadata_remains_unknown(error: type[Exception]) -> None:
    """Represent absent or unsupported native legacy fields as unknown metadata."""
    metadata = UnsupportedLegacyMetadata(error)

    assert optional_column_metadata(metadata, "has_index_page") is None
    assert optional_column_metadata(metadata, "index_page_offset") is None


def test_unexpected_native_metadata_failures_are_not_suppressed() -> None:
    """Propagate unexpected native field failures instead of silently losing metadata."""
    metadata = UnsupportedLegacyMetadata(ValueError)

    with pytest.raises(ValueError, match="has_index_page"):
        optional_column_metadata(metadata, "has_index_page")


def test_metadata_reader_returns_native_metadata_and_string_codecs() -> None:
    """Read real fixture metadata through the annotated native Parquet boundary."""
    fixture = Path(__file__).parent / "dummy.parquet"

    metadata, codecs = read_parquet_metadata(str(fixture))

    assert isinstance(metadata, pq.FileMetaData)
    assert metadata.num_rows == 3
    assert codecs == {"SNAPPY"}
    assert isinstance(metadata.row_group(0).column(0), pq.ColumnChunkMetaData)
