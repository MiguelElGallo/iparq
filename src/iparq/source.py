"""Inspect Parquet metadata and render terminal tables or JSON results."""

import glob
import json
import unicodedata
from enum import Enum
from typing import Literal, overload

import pyarrow.parquet as pq
import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    help="Inspect Parquet files for metadata, compression, and bloom filters"
)
console = Console()
error_console = Console(stderr=True)


class ParquetInspectionError(Exception):
    """Raised when a Parquet file cannot be inspected."""


class OutputFormat(str, Enum):
    """Enum for output format options."""

    RICH = "rich"
    JSON = "json"


class ParquetMetaModel(BaseModel):
    """
    ParquetMetaModel is a data model representing metadata for a Parquet file.

    Attributes:
        created_by (str): The creator of the Parquet file.
        num_columns (int): The number of columns in the Parquet file.
        num_rows (int): The number of rows in the Parquet file.
        num_row_groups (int): The number of row groups in the Parquet file.
        format_version (str): The version of the Parquet format used.
        serialized_size (int): The size of the serialized Parquet file in bytes.
        key_value_metadata_keys (List[str]): Custom metadata keys stored in the footer.
    """

    created_by: str
    num_columns: int
    num_rows: int
    num_row_groups: int
    format_version: str
    serialized_size: int
    key_value_metadata_keys: list[str] = Field(default_factory=list)


class SortingColumnInfo(BaseModel):
    """Sort order declared for a column in a Parquet row group."""

    column_index: int
    column_name: str
    descending: bool
    nulls_first: bool


class RowGroupInfo(BaseModel):
    """Storage metadata for a Parquet row group."""

    row_group: int
    num_columns: int
    num_rows: int
    total_byte_size: int
    sorting_columns: list[SortingColumnInfo] = Field(default_factory=list)


class ColumnInfo(BaseModel):
    """
    ColumnInfo is a data model representing information about a column in a Parquet file.

    Attributes:
        row_group (int): The row group index.
        column_name (str): The name of the column.
        column_index (int): The index of the column.
        compression_type (str): The compression type used for the column.
        physical_type (str): The Parquet physical type of the column.
        logical_type (Optional[str]): The Parquet logical type, when present.
        encodings (List[str]): Encodings used by the column chunk.
        has_bloom_filter (bool): Whether the column has a bloom filter.
        bloom_filter_offset (Optional[int]): Bloom filter offset in the file.
        bloom_filter_length (Optional[int]): Bloom filter size in bytes.
        has_dictionary_page (bool): Whether a dictionary page is present.
        has_column_index (bool): Whether a page-level column index is present.
        has_offset_index (bool): Whether a page-level offset index is present.
        has_index_page (Optional[bool]): Whether a legacy index page is present, or unknown.
        index_page_offset (Optional[int]): Offset of the legacy index page.
        has_min_max (bool): Whether min/max statistics are available.
        min_value (Optional[str]): The minimum value in the column (as string for display).
        max_value (Optional[str]): The maximum value in the column (as string for display).
        null_count (Optional[int]): Number of null values reported in statistics.
        distinct_count (Optional[int]): Distinct values reported in statistics.
        statistics_num_values (Optional[int]): Values represented by the statistics.
        geo_statistics (Optional[dict]): GeoParquet statistics when present.
        is_min_exact (Optional[bool]): Reserved; always None because PyArrow does not expose it.
        is_max_exact (Optional[bool]): Reserved; always None because PyArrow does not expose it.
        is_encrypted (Optional[bool]): Reserved; always None because PyArrow does not expose it.
        num_values (Optional[int]): Number of values in this column chunk.
        total_compressed_size (Optional[int]): Total compressed size in bytes.
        total_uncompressed_size (Optional[int]): Total uncompressed size in bytes.
        file_offset (Optional[int]): Offset of the column chunk in the file.
        file_path (Optional[str]): External file path stored for the column chunk.
        dictionary_page_offset (Optional[int]): Offset of the dictionary page.
        data_page_offset (Optional[int]): Offset of the first data page.
        converted_type (Optional[str]): Legacy Parquet converted type.
        type_length (Optional[int]): Fixed physical type length, when present.
        precision (Optional[int]): Decimal precision, when present.
        scale (Optional[int]): Decimal scale, when present.
        max_definition_level (Optional[int]): Maximum definition level.
        max_repetition_level (Optional[int]): Maximum repetition level.
        has_geospatial_statistics (bool): Whether GeoParquet statistics are present.
    """

    row_group: int
    column_name: str
    column_index: int
    compression_type: str
    physical_type: str = "UNKNOWN"
    logical_type: str | None = None
    encodings: list[str] = Field(default_factory=list)
    has_bloom_filter: bool = False
    bloom_filter_offset: int | None = None
    bloom_filter_length: int | None = None
    has_dictionary_page: bool = False
    has_column_index: bool = False
    has_offset_index: bool = False
    has_index_page: bool | None = None
    index_page_offset: int | None = None
    has_min_max: bool = False
    min_value: str | None = None
    max_value: str | None = None
    null_count: int | None = None
    distinct_count: int | None = None
    statistics_num_values: int | None = None
    geo_statistics: dict[str, object] | None = None
    is_min_exact: bool | None = None
    is_max_exact: bool | None = None
    is_encrypted: bool | None = None
    num_values: int | None = None
    total_compressed_size: int | None = None
    total_uncompressed_size: int | None = None
    file_offset: int | None = None
    file_path: str | None = None
    dictionary_page_offset: int | None = None
    data_page_offset: int | None = None
    converted_type: str | None = None
    type_length: int | None = None
    precision: int | None = None
    scale: int | None = None
    max_definition_level: int | None = None
    max_repetition_level: int | None = None
    has_geospatial_statistics: bool = False


class ParquetColumnInfo(BaseModel):
    """
    ParquetColumnInfo is a data model representing information about all columns in a Parquet file.

    Attributes:
        columns (List[ColumnInfo]): List of column information.
    """

    columns: list[ColumnInfo] = Field(default_factory=list)


def terminal_safe_text(value: object, *, style: str = "") -> Text:
    """Render untrusted data literally and make terminal controls visible."""
    escaped = "".join(
        ascii(character)[1:-1] if unicodedata.category(character) == "Cc" else character
        for character in str(value)
    )
    return Text(escaped, style=style)


def add_terminal_safe_row(table: Table, *values: object) -> None:
    """Add a table row after converting every value at the terminal boundary."""
    table.add_row(*(terminal_safe_text(value) for value in values))


def build_meta_model(parquet_metadata: pq.FileMetaData) -> ParquetMetaModel:
    """Build file metadata with counts, format details, and decoded metadata keys."""
    metadata = parquet_metadata.metadata or {}
    metadata_keys = sorted(key.decode("utf-8", errors="replace") for key in metadata)
    return ParquetMetaModel(
        created_by=parquet_metadata.created_by,
        num_columns=parquet_metadata.num_columns,
        num_rows=parquet_metadata.num_rows,
        num_row_groups=parquet_metadata.num_row_groups,
        format_version=str(parquet_metadata.format_version),
        serialized_size=parquet_metadata.serialized_size,
        key_value_metadata_keys=metadata_keys,
    )


def collect_row_group_info(parquet_metadata: pq.FileMetaData) -> list[RowGroupInfo]:
    """Collect row counts, sizes, and declared sort order for every row group."""
    row_groups: list[RowGroupInfo] = []
    for row_group_index in range(parquet_metadata.num_row_groups):
        row_group = parquet_metadata.row_group(row_group_index)
        sorting_columns = [
            SortingColumnInfo(
                column_index=sorting_column.column_index,
                column_name=parquet_metadata.schema.column(
                    sorting_column.column_index
                ).path,
                descending=sorting_column.descending,
                nulls_first=sorting_column.nulls_first,
            )
            for sorting_column in row_group.sorting_columns
        ]
        row_groups.append(
            RowGroupInfo(
                row_group=row_group_index,
                num_columns=row_group.num_columns,
                num_rows=row_group.num_rows,
                total_byte_size=row_group.total_byte_size,
                sorting_columns=sorting_columns,
            )
        )
    return row_groups


def optional_positive(value: int) -> int | None:
    """Return positive Parquet schema values and normalize sentinels to None."""
    return value if value > 0 else None


@overload
def optional_column_metadata(
    column_chunk: pq.ColumnChunkMetaData, attribute: Literal["has_index_page"]
) -> bool | None:
    """Read the optional boolean flag for a legacy column index page."""
    ...


@overload
def optional_column_metadata(
    column_chunk: pq.ColumnChunkMetaData, attribute: Literal["index_page_offset"]
) -> int | None:
    """Read the optional byte offset for a legacy column index page."""
    ...


def optional_column_metadata(
    column_chunk: pq.ColumnChunkMetaData,
    attribute: Literal["has_index_page", "index_page_offset"],
) -> bool | int | None:
    """Read optional column metadata across PyArrow versions and file formats."""
    try:
        return getattr(column_chunk, attribute)
    except (AttributeError, NotImplementedError):
        return None


def read_parquet_metadata(filename: str) -> tuple[pq.FileMetaData, set[str]]:
    """
    Reads the metadata of a Parquet file and extracts the compression codecs used.

    Args:
        filename (str): The path to the Parquet file.

    Returns:
        tuple: A tuple containing:
            - parquet_metadata (pyarrow.parquet.FileMetaData): The metadata of the Parquet file.
            - compression_codecs (set): A set of compression codecs used in the Parquet file.

    Raises:
        FileNotFoundError: If the file cannot be found or opened.
    """
    compression_codecs: set[str] = set()
    parquet_metadata = pq.ParquetFile(filename).metadata

    for i in range(parquet_metadata.num_row_groups):
        for j in range(parquet_metadata.num_columns):
            compression_codecs.add(parquet_metadata.row_group(i).column(j).compression)

    return parquet_metadata, compression_codecs


def print_parquet_metadata(parquet_metadata: pq.FileMetaData) -> None:
    """
    Print the file metadata model and report unavailable attributes to the console.

    Args:
        parquet_metadata: An object containing metadata of a Parquet file.
                          Expected attributes are:
                          - created_by: The creator of the Parquet file.
                          - num_columns: The number of columns in the Parquet file.
                          - num_rows: The number of rows in the Parquet file.
                          - num_row_groups: The number of row groups in the Parquet file.
                          - format_version: The format version of the Parquet file.
                          - serialized_size: The serialized size of the Parquet file.

    Notes:
        Missing metadata attributes are reported to the console without
        propagating AttributeError.
    """
    try:
        meta = build_meta_model(parquet_metadata)
        console.print(meta)

    except AttributeError as e:
        console.print(f"Error: {e}", style="blink bold red underline on white")
    finally:
        pass


def print_compression_types(
    parquet_metadata: pq.FileMetaData, column_info: ParquetColumnInfo
) -> None:
    """
    Collects compression type information for each column and adds it to the column_info model.

    Args:
        parquet_metadata: The Parquet file metadata.
        column_info: The ParquetColumnInfo model to update.
    """
    for i in range(parquet_metadata.num_row_groups):
        row_group = parquet_metadata.row_group(i)
        for j in range(parquet_metadata.num_columns):
            column_chunk = row_group.column(j)
            schema_column = parquet_metadata.schema.column(j)
            logical_type: str | None = str(schema_column.logical_type)
            if logical_type == "None":
                logical_type = None
            converted_type: str | None = str(schema_column.converted_type)
            if converted_type == "NONE":
                converted_type = None

            column_info.columns.append(
                ColumnInfo(
                    row_group=i,
                    column_name=column_chunk.path_in_schema,
                    column_index=j,
                    compression_type=column_chunk.compression,
                    physical_type=column_chunk.physical_type,
                    logical_type=logical_type,
                    encodings=list(column_chunk.encodings),
                    has_dictionary_page=column_chunk.has_dictionary_page,
                    has_column_index=column_chunk.has_column_index,
                    has_offset_index=column_chunk.has_offset_index,
                    has_index_page=optional_column_metadata(
                        column_chunk, "has_index_page"
                    ),
                    index_page_offset=optional_column_metadata(
                        column_chunk, "index_page_offset"
                    ),
                    num_values=column_chunk.num_values,
                    total_compressed_size=column_chunk.total_compressed_size,
                    total_uncompressed_size=column_chunk.total_uncompressed_size,
                    file_offset=column_chunk.file_offset,
                    file_path=column_chunk.file_path or None,
                    dictionary_page_offset=column_chunk.dictionary_page_offset,
                    data_page_offset=column_chunk.data_page_offset,
                    converted_type=converted_type,
                    type_length=optional_positive(schema_column.length),
                    precision=optional_positive(schema_column.precision),
                    scale=(
                        schema_column.scale if schema_column.precision > 0 else None
                    ),
                    max_definition_level=schema_column.max_definition_level,
                    max_repetition_level=schema_column.max_repetition_level,
                    has_geospatial_statistics=column_chunk.is_geo_stats_set,
                    geo_statistics=(
                        column_chunk.geo_statistics.to_dict()
                        if column_chunk.is_geo_stats_set
                        else None
                    ),
                )
            )


def print_bloom_filter_info(
    parquet_metadata: pq.FileMetaData, column_info: ParquetColumnInfo
) -> None:
    """
    Updates the column_info model with bloom filter information.

    Args:
        parquet_metadata: The Parquet file metadata.
        column_info: The ParquetColumnInfo model to update.
    """
    for i in range(parquet_metadata.num_row_groups):
        row_group = parquet_metadata.row_group(i)

        for j in range(parquet_metadata.num_columns):
            column_chunk = row_group.column(j)

            # Find the corresponding column in our model
            for col in column_info.columns:
                if col.row_group == i and col.column_index == j:
                    col.bloom_filter_offset = column_chunk.bloom_filter_offset
                    col.bloom_filter_length = column_chunk.bloom_filter_length
                    # The offset has existed since Bloom filters were introduced.
                    # Length was added later and can be absent in older valid files.
                    col.has_bloom_filter = col.bloom_filter_offset is not None
                    break


def _apply_column_statistics(column: ColumnInfo, statistics: pq.Statistics) -> None:
    """Copy available statistics while retaining explicit null and zero values."""
    column.has_min_max = statistics.has_min_max
    column.null_count = statistics.null_count if statistics.has_null_count else None
    column.distinct_count = (
        statistics.distinct_count if statistics.has_distinct_count else None
    )
    column.statistics_num_values = statistics.num_values
    if statistics.has_min_max:
        column.min_value = str(statistics.min) if statistics.min is not None else "null"
        column.max_value = str(statistics.max) if statistics.max is not None else "null"


def _update_column_statistics(
    column: ColumnInfo, column_chunk: pq.ColumnChunkMetaData
) -> None:
    """Update one column using only statistics declared by its chunk."""
    if not column_chunk.is_stats_set:
        column.has_min_max = False
        return
    _apply_column_statistics(column, column_chunk.statistics)


def print_min_max_statistics(
    parquet_metadata: pq.FileMetaData, column_info: ParquetColumnInfo
) -> None:
    """
    Updates the column_info model with min/max statistics information.

    Args:
        parquet_metadata: The Parquet file metadata.
        column_info: The ParquetColumnInfo model to update.
    """
    for i in range(parquet_metadata.num_row_groups):
        row_group = parquet_metadata.row_group(i)

        for j in range(parquet_metadata.num_columns):
            column_chunk = row_group.column(j)

            column = next(
                (
                    col
                    for col in column_info.columns
                    if col.row_group == i and col.column_index == j
                ),
                None,
            )
            if column is not None:
                _update_column_statistics(column, column_chunk)


def format_size(size_bytes: int | None) -> str:
    """Format byte counts in 1024-based units and return N/A for missing sizes."""
    if size_bytes is None:
        return "N/A"
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def _column_size_values(column: ColumnInfo) -> list[str]:
    """Format column size cells with the existing zero-value display conventions."""
    ratio = "N/A"
    if column.total_compressed_size and column.total_uncompressed_size:
        ratio = f"{column.total_uncompressed_size / column.total_compressed_size:.1f}x"
    return [
        str(column.num_values) if column.num_values else "N/A",
        format_size(column.total_compressed_size),
        ratio,
    ]


def _column_row_values(column: ColumnInfo, show_sizes: bool) -> list[str]:
    """Prepare a column summary row before terminal-safe rendering is applied."""
    values = [
        str(column.row_group),
        column.column_name,
        str(column.column_index),
        column.compression_type,
        "✅" if column.has_bloom_filter else "❌",
        column.min_value
        if column.has_min_max and column.min_value is not None
        else "N/A",
        column.max_value
        if column.has_min_max and column.max_value is not None
        else "N/A",
    ]
    if show_sizes:
        values.extend(_column_size_values(column))
    return values


def print_column_info_table(
    column_info: ParquetColumnInfo, show_sizes: bool = False
) -> None:
    """
    Prints the column information using a Rich table.

    Args:
        column_info: The ParquetColumnInfo model to display.
        show_sizes: Whether to show compressed/uncompressed size columns.
    """
    table = Table(title="Parquet Column Information")

    # Add table columns
    table.add_column("Row Group", justify="center", style="cyan")
    table.add_column("Column Name", style="green")
    table.add_column("Index", justify="center")
    table.add_column("Compression", style="magenta")
    table.add_column("Bloom", justify="center")
    table.add_column("Min Value", style="yellow")
    table.add_column("Max Value", style="yellow")

    if show_sizes:
        table.add_column("Values", justify="right")
        table.add_column("Compressed", justify="right", style="blue")
        table.add_column("Ratio", justify="right", style="blue")

    for col in column_info.columns:
        add_terminal_safe_row(table, *_column_row_values(col, show_sizes))

    # Print the table
    console.print(table)


def _metadata_display_value(value: object, missing: str) -> object:
    """Substitute missing metadata without replacing present zero or false values."""
    return missing if value is None else value


def _legacy_index_indicator(has_index_page: bool | None) -> str:
    """Distinguish present, absent, and unknown legacy indexes in terminal output."""
    if has_index_page is True:
        return "✅"
    if has_index_page is False:
        return "—"
    return "N/A"


def _build_row_group_details_table(row_groups: list[RowGroupInfo]) -> Table:
    """Build row-group size rows and declared column sort order safely."""
    row_group_table = Table(title="Parquet Row Group Details")
    row_group_table.add_column("RG", justify="center", style="cyan")
    row_group_table.add_column("Rows", justify="right")
    row_group_table.add_column("Columns", justify="right")
    row_group_table.add_column("Uncompressed Size", justify="right", style="blue")
    row_group_table.add_column("Sort Order")

    for row_group in row_groups:
        sort_order = ", ".join(
            f"{column.column_name} "
            f"{'DESC' if column.descending else 'ASC'} "
            f"NULLS {'FIRST' if column.nulls_first else 'LAST'}"
            for column in row_group.sorting_columns
        )
        add_terminal_safe_row(
            row_group_table,
            row_group.row_group,
            row_group.num_rows,
            row_group.num_columns,
            format_size(row_group.total_byte_size),
            sort_order or "—",
        )
    return row_group_table


def _build_encoding_details_table(column_info: ParquetColumnInfo) -> Table:
    """Build physical and logical type rows alongside column encoding choices."""
    encoding_table = Table(title="Parquet Encoding Details")
    encoding_table.add_column("RG", justify="center", style="cyan")
    encoding_table.add_column("Column", style="green")
    encoding_table.add_column("Physical", style="magenta")
    encoding_table.add_column("Logical")
    encoding_table.add_column("Encodings")

    for col in column_info.columns:
        add_terminal_safe_row(
            encoding_table,
            col.row_group,
            col.column_name,
            col.physical_type,
            col.logical_type or "—",
            ", ".join(col.encodings) or "—",
        )
    return encoding_table


def _build_schema_details_table(column_info: ParquetColumnInfo) -> Table:
    """Build schema parameter rows without hiding valid zero-valued field metadata."""
    schema_table = Table(title="Parquet Schema Details")
    schema_table.add_column("RG", justify="center", style="cyan")
    schema_table.add_column("Column", style="green")
    schema_table.add_column("Converted")
    schema_table.add_column("Length", justify="right")
    schema_table.add_column("Precision", justify="right")
    schema_table.add_column("Scale", justify="right")
    schema_table.add_column("Definition", justify="right")
    schema_table.add_column("Repetition", justify="right")

    for col in column_info.columns:
        add_terminal_safe_row(
            schema_table,
            col.row_group,
            col.column_name,
            col.converted_type or "—",
            _metadata_display_value(col.type_length, "—"),
            _metadata_display_value(col.precision, "—"),
            _metadata_display_value(col.scale, "—"),
            _metadata_display_value(col.max_definition_level, "—"),
            _metadata_display_value(col.max_repetition_level, "—"),
        )
    return schema_table


def _build_index_statistics_details_table(column_info: ParquetColumnInfo) -> Table:
    """Build index and statistics rows while preserving unknown and zero values."""
    index_table = Table(title="Parquet Index and Statistics Details")
    index_table.add_column("RG", justify="center", style="cyan")
    index_table.add_column("Column", style="green")
    index_table.add_column("Dictionary", justify="center")
    index_table.add_column("Column Index", justify="center")
    index_table.add_column("Offset Index", justify="center")
    index_table.add_column("Legacy Index", justify="center")
    index_table.add_column("Bloom Size", justify="right")
    index_table.add_column("Nulls", justify="right")
    index_table.add_column("Distinct", justify="right")
    index_table.add_column("Stats Values", justify="right")
    index_table.add_column("Geo Statistics")

    for col in column_info.columns:
        add_terminal_safe_row(
            index_table,
            col.row_group,
            col.column_name,
            "✅" if col.has_dictionary_page else "—",
            "✅" if col.has_column_index else "—",
            "✅" if col.has_offset_index else "—",
            _legacy_index_indicator(col.has_index_page),
            format_size(col.bloom_filter_length),
            _metadata_display_value(col.null_count, "N/A"),
            _metadata_display_value(col.distinct_count, "N/A"),
            _metadata_display_value(col.statistics_num_values, "N/A"),
            (
                json.dumps(col.geo_statistics, sort_keys=True)
                if col.geo_statistics is not None
                else "—"
            ),
        )
    return index_table


def _build_column_chunk_locations_table(column_info: ParquetColumnInfo) -> Table:
    """Build physical chunk and page offset rows without discarding zero offsets."""
    page_table = Table(title="Parquet Column Chunk Locations")
    page_table.add_column("RG", justify="center", style="cyan")
    page_table.add_column("Column", style="green")
    page_table.add_column("Chunk", justify="right")
    page_table.add_column("Dictionary Page", justify="right")
    page_table.add_column("Data Page", justify="right")
    page_table.add_column("Legacy Index", justify="right")
    page_table.add_column("Bloom Filter", justify="right")

    for col in column_info.columns:
        add_terminal_safe_row(
            page_table,
            col.row_group,
            col.column_name,
            _metadata_display_value(col.file_offset, "N/A"),
            _metadata_display_value(col.dictionary_page_offset, "N/A"),
            _metadata_display_value(col.data_page_offset, "N/A"),
            _metadata_display_value(col.index_page_offset, "N/A"),
            _metadata_display_value(col.bloom_filter_offset, "N/A"),
        )
    return page_table


def print_storage_details_table(
    column_info: ParquetColumnInfo, row_groups: list[RowGroupInfo]
) -> None:
    """Print storage-level metadata exposed by PyArrow 25 and later."""
    tables = [
        _build_row_group_details_table(row_groups),
        _build_encoding_details_table(column_info),
        _build_schema_details_table(column_info),
        _build_index_statistics_details_table(column_info),
        _build_column_chunk_locations_table(column_info),
    ]
    for table in tables:
        console.print(table)


def build_json_result(
    meta_model: ParquetMetaModel,
    column_info: ParquetColumnInfo,
    compression_codecs: set[str],
    metadata_only: bool = False,
    row_groups: list[RowGroupInfo] | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable result without writing to stdout."""
    result: dict[str, object] = {"metadata": meta_model.model_dump()}
    if not metadata_only:
        result.update(
            {
                "columns": [column.model_dump() for column in column_info.columns],
                "compression_codecs": sorted(compression_codecs),
                "row_groups": [
                    row_group.model_dump() for row_group in (row_groups or [])
                ],
            }
        )
    return result


def output_json(
    meta_model: ParquetMetaModel,
    column_info: ParquetColumnInfo,
    compression_codecs: set[str],
    metadata_only: bool = False,
) -> None:
    """
    Write the selected Parquet metadata and column details to stdout as JSON.

    Args:
        meta_model: The Parquet metadata model
        column_info: The column information model
        compression_codecs: Set of compression codecs used
        metadata_only: Omit column details and compression codecs when true.
    """
    result = build_json_result(
        meta_model, column_info, compression_codecs, metadata_only=metadata_only
    )
    print(json.dumps(result, indent=2))


def _filter_columns(
    column_info: ParquetColumnInfo,
    column_filter: str | None,
    format: OutputFormat,
) -> None:
    """Apply the requested column filter and route unmatched diagnostics safely."""
    if not column_filter:
        return
    column_info.columns = [
        column for column in column_info.columns if column.column_name == column_filter
    ]
    if not column_info.columns:
        destination = error_console if format == OutputFormat.JSON else console
        destination.print(
            terminal_safe_text(
                f"No columns match the filter: {column_filter}", style="yellow"
            )
        )


def _print_rich_inspection(
    meta_model: ParquetMetaModel,
    column_info: ParquetColumnInfo,
    compression: set[str],
    row_groups: list[RowGroupInfo],
    metadata_only: bool,
    show_sizes: bool,
    show_details: bool,
) -> None:
    """Display requested Rich details after printing the file metadata model."""
    console.print(meta_model)
    if metadata_only:
        return
    print_column_info_table(column_info, show_sizes=show_sizes)
    if show_details:
        print_storage_details_table(column_info, row_groups)
    console.print(terminal_safe_text(f"Compression codecs: {compression}"))


def inspect_single_file(
    filename: str,
    format: OutputFormat,
    metadata_only: bool,
    column_filter: str | None,
    show_sizes: bool = False,
    show_details: bool = False,
) -> dict[str, object] | None:
    """
    Inspect a single Parquet file and display its metadata, compression settings, and bloom filter information.

    Raises:
        ParquetInspectionError: If the file cannot be opened or parsed.
    """
    try:
        parquet_metadata, compression = read_parquet_metadata(filename)
    except FileNotFoundError as error:
        raise ParquetInspectionError(f"Cannot open: {filename}.") from error
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ParquetInspectionError(f"Failed to read metadata: {error}") from error

    # Create metadata model
    meta_model = build_meta_model(parquet_metadata)
    row_groups = collect_row_group_info(parquet_metadata)

    # Create a model to store column information
    column_info = ParquetColumnInfo()

    # Collect information
    print_compression_types(parquet_metadata, column_info)
    print_bloom_filter_info(parquet_metadata, column_info)
    print_min_max_statistics(parquet_metadata, column_info)

    _filter_columns(column_info, column_filter, format)

    # Output based on format selection
    if format == OutputFormat.JSON:
        return build_json_result(
            meta_model,
            column_info,
            compression,
            metadata_only=metadata_only,
            row_groups=row_groups,
        )
    _print_rich_inspection(
        meta_model,
        column_info,
        compression,
        row_groups,
        metadata_only,
        show_sizes,
        show_details,
    )
    return None


def _expand_unique_files(filenames: list[str]) -> list[str]:
    """Expand input patterns and retain each filename in first-seen order."""
    all_files = []
    for pattern in filenames:
        matches = glob.glob(pattern)
        all_files.extend(matches or [pattern])
    return list(dict.fromkeys(all_files))


def _print_file_heading(filename: str, index: int) -> None:
    """Separate multiple Rich results with safe filenames and existing underlines."""
    if index > 0:
        console.print()
    console.print(terminal_safe_text(f"File: {filename}", style="bold blue"))
    console.print("─" * (len(filename) + 6))


def _emit_json_results(json_results: list[dict[str, object]], file_count: int) -> None:
    """Emit successful results using single-object and multiple-input array conventions."""
    if json_results:
        payload: object = json_results[0] if file_count == 1 else json_results
        print(json.dumps(payload, indent=2))
    elif file_count > 1:
        print("[]")


@app.command(name="")
@app.command(name="inspect")
def inspect(
    filenames: list[str] = typer.Argument(
        ..., help="Path(s) or pattern(s) to Parquet files to inspect"
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.RICH, "--format", "-f", help="Output format (rich or json)"
    ),
    metadata_only: bool = typer.Option(
        False,
        "--metadata-only",
        "-m",
        help="Show only file metadata without column details",
    ),
    column_filter: str | None = typer.Option(
        None, "--column", "-c", help="Filter results to show only specific column"
    ),
    show_sizes: bool = typer.Option(
        False,
        "--sizes",
        "-s",
        help="Show column sizes and compression ratios",
    ),
    show_details: bool = typer.Option(
        False,
        "--details",
        "-d",
        help="Show row groups, schema, indexes, page locations, and statistics",
    ),
) -> None:
    """
    Inspect Parquet files and display their metadata, compression settings, and bloom filter information.
    """
    unique_files = _expand_unique_files(filenames)

    # Process each file
    had_errors = False
    json_results: list[dict[str, object]] = []
    for i, filename in enumerate(unique_files):
        # For multiple files, add a header to separate results
        if format == OutputFormat.RICH and len(unique_files) > 1:
            _print_file_heading(filename, i)

        try:
            result = inspect_single_file(
                filename,
                format,
                metadata_only,
                column_filter,
                show_sizes,
                show_details,
            )
            if result is not None:
                if len(unique_files) > 1:
                    result = {"file": filename, **result}
                json_results.append(result)
        except Exception as e:  # noqa: BLE001 - isolate failures across input files
            error_console.print(
                terminal_safe_text(f"Error processing {filename}: {e}", style="red")
            )
            had_errors = True
            continue

    if format == OutputFormat.JSON:
        _emit_json_results(json_results, len(unique_files))

    if had_errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
