"""Preserve metadata rendering and file aggregation across inspection complexity refactors."""

import json
from io import StringIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rich.console import Console
from rich.table import Table
from typer.testing import CliRunner

from iparq import source
from iparq.source import (
    ColumnInfo,
    ParquetColumnInfo,
    RowGroupInfo,
    SortingColumnInfo,
    app,
    print_column_info_table,
    print_min_max_statistics,
    print_storage_details_table,
)


def _capture_tables(monkeypatch: pytest.MonkeyPatch) -> list[Table]:
    """Capture public Rich table output before its terminal rendering occurs."""
    tables: list[Table] = []

    def record_table(*objects: object, **options: object) -> None:
        """Retain each emitted table while rejecting unexpected additional output objects."""
        assert len(objects) == 1
        assert isinstance(objects[0], Table)
        tables.append(objects[0])

    monkeypatch.setattr(source.console, "print", record_table)
    return tables


def _rendered_rows(table: Table) -> list[list[str]]:
    """Read visible table cells from an unconstrained plain terminal rendering."""
    output = StringIO()
    Console(
        file=output,
        width=300,
        force_terminal=False,
        color_system=None,
        legacy_windows=False,
    ).print(table)
    rows = [
        [cell.strip() for cell in line.split("│")[1:-1]]
        for line in output.getvalue().splitlines()
        if "│" in line
    ]
    return rows


def test_overlapping_globs_preserve_first_encounter_order(tmp_path: Path) -> None:
    """Preserve argument order when overlapping wildcard and literal inputs repeat files."""
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(pa.table({"id": [1]}), first)
    pq.write_table(pa.table({"id": [2]}), second)

    result = CliRunner().invoke(
        app,
        [
            "inspect",
            "--format",
            "json",
            str(second),
            str(tmp_path / "*.parquet"),
            str(first),
            str(second),
        ],
    )

    assert result.exit_code == 0
    assert [item["file"] for item in json.loads(result.stdout)] == [
        str(second),
        str(first),
    ]


@pytest.mark.parametrize("failure_first", [True, False])
def test_mixed_json_results_retain_array_and_failure_exit(
    tmp_path: Path, failure_first: bool
) -> None:
    """Keep successful JSON records readable despite failures before or after them."""
    valid = tmp_path / "valid.parquet"
    missing = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"id": [1, 2]}), valid)
    inputs = [str(missing), str(valid)]
    if not failure_first:
        inputs.reverse()

    result = CliRunner().invoke(
        app, ["inspect", "--format", "json", *inputs], env={"COLUMNS": "400"}
    )

    assert result.exit_code == 1
    records = json.loads(result.stdout)
    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["file"] == str(valid)
    assert records[0]["metadata"]["num_rows"] == 2
    assert f"Error processing {missing}" in result.stderr


@pytest.mark.parametrize("file_count", [1, 2])
def test_all_failed_json_inputs_keep_existing_empty_output_conventions(
    tmp_path: Path, file_count: int
) -> None:
    """Keep single failure stdout empty while multiple failures emit an empty array."""
    inputs = [str(tmp_path / f"missing-{index}.parquet") for index in range(file_count)]

    result = CliRunner().invoke(app, ["inspect", "--format", "json", *inputs])

    assert result.exit_code == 1
    assert result.stdout == ("" if file_count == 1 else "[]\n")
    assert result.stderr.count("Error processing") == file_count


def test_unmatched_glob_is_reported_as_literal_input(tmp_path: Path) -> None:
    """Report unmatched wildcard arguments instead of silently discarding failed inputs."""
    pattern = str(tmp_path / "unmatched-*.parquet")

    result = CliRunner().invoke(
        app, ["inspect", "--format", "json", pattern], env={"COLUMNS": "400"}
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert f"Error processing {pattern}" in result.stderr


@pytest.mark.parametrize("write_statistics", [True, False])
def test_statistics_preserve_nulls_and_bounds_across_row_groups(
    tmp_path: Path, write_statistics: bool
) -> None:
    """Preserve per-group bounds, zero null counts, and all-null column statistics."""
    parquet_path = tmp_path / "statistics.parquet"
    table = pa.table(
        {"id": [7, 2, 8, 1], "empty": pa.array([None] * 4, type=pa.int64())}
    )
    pq.write_table(
        table, parquet_path, row_group_size=2, write_statistics=write_statistics
    )

    result = CliRunner().invoke(app, ["inspect", "--format", "json", str(parquet_path)])

    assert result.exit_code == 0
    columns = json.loads(result.stdout)["columns"]
    observed = [
        (
            column["row_group"],
            column["column_name"],
            column["has_min_max"],
            column["min_value"],
            column["max_value"],
            column["null_count"],
            column["statistics_num_values"],
        )
        for column in columns
    ]
    if write_statistics:
        assert observed == [
            (0, "id", True, "2", "7", 0, 2),
            (0, "empty", False, None, None, 2, 0),
            (1, "id", True, "1", "8", 0, 2),
            (1, "empty", False, None, None, 2, 0),
        ]
    else:
        assert observed == [
            (0, "id", False, None, None, None, None),
            (0, "empty", False, None, None, None, None),
            (1, "id", False, None, None, None, None),
            (1, "empty", False, None, None, None, None),
        ]


def test_statistics_update_only_first_matching_model_column(tmp_path: Path) -> None:
    """Apply chunk statistics to the first matching model without updating duplicates."""
    parquet_path = tmp_path / "duplicates.parquet"
    pq.write_table(pa.table({"id": [4, 8]}), parquet_path)
    first = ColumnInfo(
        row_group=0, column_name="id", column_index=0, compression_type="SNAPPY"
    )
    duplicate = first.model_copy()
    columns = ParquetColumnInfo(columns=[first, duplicate])

    print_min_max_statistics(pq.read_metadata(parquet_path), columns)

    assert (first.has_min_max, first.min_value, first.max_value) == (True, "4", "8")
    assert (duplicate.has_min_max, duplicate.min_value, duplicate.max_value) == (
        False,
        None,
        None,
    )


def test_column_summary_preserves_empty_bounds_and_zero_size_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve empty string bounds and established zero-value summary table conventions."""
    tables = _capture_tables(monkeypatch)
    columns = ParquetColumnInfo(
        columns=[
            ColumnInfo(
                row_group=0,
                column_name="empty",
                column_index=0,
                compression_type="SNAPPY",
                has_min_max=True,
                min_value="",
                max_value="0",
                num_values=0,
                total_compressed_size=0,
                total_uncompressed_size=0,
            ),
            ColumnInfo(
                row_group=0,
                column_name="missing",
                column_index=1,
                compression_type="SNAPPY",
                min_value="stale",
                max_value="stale",
            ),
            ColumnInfo(
                row_group=0,
                column_name="compressed",
                column_index=2,
                compression_type="SNAPPY",
                num_values=4,
                total_compressed_size=2,
                total_uncompressed_size=6,
            ),
        ]
    )

    print_column_info_table(columns, show_sizes=True)

    assert len(tables) == 1
    assert _rendered_rows(tables[0]) == [
        ["0", "empty", "0", "SNAPPY", "❌", "", "0", "N/A", "0.0B", "N/A"],
        ["0", "missing", "1", "SNAPPY", "❌", "N/A", "N/A", "N/A", "N/A", "N/A"],
        ["0", "compressed", "2", "SNAPPY", "❌", "N/A", "N/A", "4", "2.0B", "3.0x"],
    ]


@pytest.mark.parametrize(
    ("has_index_page", "indicator"), [(True, "✅"), (False, "—"), (None, "N/A")]
)
def test_storage_tables_preserve_zero_values_and_legacy_index_states(
    monkeypatch: pytest.MonkeyPatch, has_index_page: bool | None, indicator: str
) -> None:
    """Render zero-valued schema, statistics, and offsets independently from unknown indexes."""
    tables = _capture_tables(monkeypatch)
    columns = ParquetColumnInfo(
        columns=[
            ColumnInfo(
                row_group=0,
                column_name="id",
                column_index=0,
                compression_type="SNAPPY",
                type_length=0,
                precision=0,
                scale=0,
                max_definition_level=0,
                max_repetition_level=0,
                has_index_page=has_index_page,
                bloom_filter_length=0,
                null_count=0,
                distinct_count=0,
                statistics_num_values=0,
                geo_statistics={},
                file_offset=0,
                dictionary_page_offset=0,
                data_page_offset=0,
                index_page_offset=0,
                bloom_filter_offset=0,
            )
        ]
    )

    print_storage_details_table(columns, [])

    assert [table.title for table in tables] == [
        "Parquet Row Group Details",
        "Parquet Encoding Details",
        "Parquet Schema Details",
        "Parquet Index and Statistics Details",
        "Parquet Column Chunk Locations",
    ]
    assert _rendered_rows(tables[2]) == [["0", "id", "—", "0", "0", "0", "0", "0"]]
    assert _rendered_rows(tables[3]) == [
        ["0", "id", "—", "—", "—", indicator, "0.0B", "0", "0", "0", "{}"]
    ]
    assert _rendered_rows(tables[4]) == [["0", "id", "0", "0", "0", "0", "0"]]


def test_rich_sort_order_preserves_direction_null_order_and_terminal_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render complete sort declarations safely while preserving empty row-group markers."""
    tables = _capture_tables(monkeypatch)
    row_groups = [
        RowGroupInfo(
            row_group=0,
            num_columns=2,
            num_rows=0,
            total_byte_size=0,
            sorting_columns=[
                SortingColumnInfo(
                    column_index=0,
                    column_name="z\x1b[31m",
                    descending=True,
                    nulls_first=True,
                ),
                SortingColumnInfo(
                    column_index=1,
                    column_name="a",
                    descending=False,
                    nulls_first=False,
                ),
            ],
        ),
        RowGroupInfo(row_group=1, num_columns=2, num_rows=0, total_byte_size=0),
    ]

    print_storage_details_table(ParquetColumnInfo(), row_groups)

    assert _rendered_rows(tables[0]) == [
        ["0", "0", "2", "0.0B", "z\\x1b[31m DESC NULLS FIRST, a ASC NULLS LAST"],
        ["1", "0", "2", "0.0B", "—"],
    ]
