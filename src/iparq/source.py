import glob
import json
from enum import Enum
from typing import List, Optional

import pyarrow.parquet as pq
import typer
from pydantic import BaseModel
from rich import print
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Inspect Parquet files for metadata, compression, and bloom filters"
)
console = Console()


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
    """

    created_by: str
    num_columns: int
    num_rows: int
    num_row_groups: int
    format_version: str
    serialized_size: int


class ColumnInfo(BaseModel):
    """
    ColumnInfo is a data model representing information about a column in a Parquet file.

    Attributes:
        row_group (int): The row group index.
        column_name (str): The name of the column.
        column_index (int): The index of the column.
        compression_type (str): The compression type used for the column.
        has_bloom_filter (bool): Whether the column has a bloom filter.
        has_min_max (bool): Whether min/max statistics are available.
        min_value (Optional[str]): The minimum value in the column (as string for display).
        max_value (Optional[str]): The maximum value in the column (as string for display).
        is_min_exact (Optional[bool]): Whether the min value is exact (PyArrow 22+).
        is_max_exact (Optional[bool]): Whether the max value is exact (PyArrow 22+).
        is_encrypted (Optional[bool]): Whether the column is encrypted.
        num_values (Optional[int]): Number of values in this column chunk.
        total_compressed_size (Optional[int]): Total compressed size in bytes.
        total_uncompressed_size (Optional[int]): Total uncompressed size in bytes.
    """

    row_group: int
    column_name: str
    column_index: int
    compression_type: str
    has_bloom_filter: Optional[bool] = False
    has_min_max: Optional[bool] = False
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    is_min_exact: Optional[bool] = None
    is_max_exact: Optional[bool] = None
    is_encrypted: Optional[bool] = None
    num_values: Optional[int] = None
    total_compressed_size: Optional[int] = None
    total_uncompressed_size: Optional[int] = None


class ParquetColumnInfo(BaseModel):
    """
    ParquetColumnInfo is a data model representing information about all columns in a Parquet file.

    Attributes:
        columns (List[ColumnInfo]): List of column information.
    """

    columns: List[ColumnInfo] = []


def read_parquet_metadata(filename: str):
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
    compression_codecs = set([])
    parquet_metadata = pq.ParquetFile(filename).metadata

    for i in range(parquet_metadata.num_row_groups):
        for j in range(parquet_metadata.num_columns):
            compression_codecs.add(parquet_metadata.row_group(i).column(j).compression)

    return parquet_metadata, compression_codecs


def print_parquet_metadata(parquet_metadata):
    """
    Prints the metadata of a Parquet file.

    Args:
        parquet_metadata: An object containing metadata of a Parquet file.
                          Expected attributes are:
                          - created_by: The creator of the Parquet file.
                          - num_columns: The number of columns in the Parquet file.
                          - num_rows: The number of rows in the Parquet file.
                          - num_row_groups: The number of row groups in the Parquet file.
                          - format_version: The format version of the Parquet file.
                          - serialized_size: The serialized size of the Parquet file.

    Raises:
        AttributeError: If the provided parquet_metadata object does not have the expected attributes.
    """
    try:
        meta = ParquetMetaModel(
            created_by=parquet_metadata.created_by,
            num_columns=parquet_metadata.num_columns,
            num_rows=parquet_metadata.num_rows,
            num_row_groups=parquet_metadata.num_row_groups,
            format_version=str(parquet_metadata.format_version),
            serialized_size=parquet_metadata.serialized_size,
        )
        console.print(meta)

    except AttributeError as e:
        console.print(f"Error: {e}", style="blink bold red underline on white")
    finally:
        pass


def print_compression_types(parquet_metadata, column_info: ParquetColumnInfo) -> None:
    """
    Collects compression type information for each column and adds it to the column_info model.

    Args:
        parquet_metadata: The Parquet file metadata.
        column_info: The ParquetColumnInfo model to update.
    """
    try:
        num_row_groups = parquet_metadata.num_row_groups
        num_columns = parquet_metadata.num_columns

        for i in range(num_row_groups):
            row_group = parquet_metadata.row_group(i)
            for j in range(num_columns):
                column_chunk = row_group.column(j)
                compression = column_chunk.compression
                column_name = parquet_metadata.schema.names[j]

                # Get additional column chunk metadata
                num_values = (
                    column_chunk.num_values
                    if hasattr(column_chunk, "num_values")
                    else None
                )
                total_compressed = (
                    column_chunk.total_compressed_size
                    if hasattr(column_chunk, "total_compressed_size")
                    else None
                )
                total_uncompressed = (
                    column_chunk.total_uncompressed_size
                    if hasattr(column_chunk, "total_uncompressed_size")
                    else None
                )
                is_encrypted = (
                    column_chunk.is_crypto_metadata_set()
                    if hasattr(column_chunk, "is_crypto_metadata_set")
                    else None
                )

                # Create or update column info
                column_info.columns.append(
                    ColumnInfo(
                        row_group=i,
                        column_name=column_name,
                        column_index=j,
                        compression_type=compression,
                        num_values=num_values,
                        total_compressed_size=total_compressed,
                        total_uncompressed_size=total_uncompressed,
                        is_encrypted=is_encrypted,
                    )
                )
    except Exception as e:
        console.print(
            f"Error while collecting compression types: {e}",
            style="blink bold red underline on white",
        )


def print_bloom_filter_info(parquet_metadata, column_info: ParquetColumnInfo) -> None:
    """
    Updates the column_info model with bloom filter information.

    Args:
        parquet_metadata: The Parquet file metadata.
        column_info: The ParquetColumnInfo model to update.
    """
    try:
        num_row_groups = parquet_metadata.num_row_groups
        num_columns = parquet_metadata.num_columns

        for i in range(num_row_groups):
            row_group = parquet_metadata.row_group(i)

            for j in range(num_columns):
                column_chunk = row_group.column(j)

                # Find the corresponding column in our model
                for col in column_info.columns:
                    if col.row_group == i and col.column_index == j:
                        # Check if this column has bloom filters
                        has_bloom_filter = (
                            hasattr(column_chunk, "is_stats_set")
                            and column_chunk.is_stats_set
                        )
                        col.has_bloom_filter = has_bloom_filter
                        break
    except Exception as e:
        console.print(
            f"Error while collecting bloom filter information: {e}",
            style="blink bold red underline on white",
        )


def print_min_max_statistics(parquet_metadata, column_info: ParquetColumnInfo) -> None:
    """
    Updates the column_info model with min/max statistics information.

    Args:
        parquet_metadata: The Parquet file metadata.
        column_info: The ParquetColumnInfo model to update.
    """
    try:
        num_row_groups = parquet_metadata.num_row_groups
        num_columns = parquet_metadata.num_columns

        for i in range(num_row_groups):
            row_group = parquet_metadata.row_group(i)

            for j in range(num_columns):
                column_chunk = row_group.column(j)

                # Find the corresponding column in our model
                for col in column_info.columns:
                    if col.row_group == i and col.column_index == j:
                        # Check if this column has statistics
                        if column_chunk.is_stats_set:
                            stats = column_chunk.statistics
                            col.has_min_max = stats.has_min_max

                            if stats.has_min_max:
                                # Convert values to string for display, handling potential None values
                                try:
                                    col.min_value = (
                                        str(stats.min)
                                        if stats.min is not None
                                        else "null"
                                    )
                                    col.max_value = (
                                        str(stats.max)
                                        if stats.max is not None
                                        else "null"
                                    )
                                except Exception:
                                    # Fallback for complex types that might not stringify well
                                    col.min_value = "<unable to display>"
                                    col.max_value = "<unable to display>"

                                # PyArrow 22+ feature: check if min/max values are exact
                                # This helps users understand if statistics can be trusted for query optimization
                                try:
                                    if hasattr(stats, "is_min_value_exact"):
                                        col.is_min_exact = stats.is_min_value_exact
                                    if hasattr(stats, "is_max_value_exact"):
                                        col.is_max_exact = stats.is_max_value_exact
                                except Exception:
                                    pass  # Not available in older PyArrow versions
                        else:
                            col.has_min_max = False
                        break
    except Exception as e:
        console.print(
            f"Error while collecting min/max statistics: {e}",
            style="blink bold red underline on white",
        )


def format_size(size_bytes: Optional[int]) -> str:
    """Format bytes into human-readable size."""
    if size_bytes is None:
        return "N/A"
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


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
    table.add_column("Encrypted", justify="center")
    table.add_column("Min Value", style="yellow")
    table.add_column("Max Value", style="yellow")
    table.add_column(
        "Exact", justify="center", style="dim"
    )  # Shows if min/max are exact

    if show_sizes:
        table.add_column("Values", justify="right")
        table.add_column("Compressed", justify="right", style="blue")
        table.add_column("Ratio", justify="right", style="blue")

    # Add rows to the table
    for col in column_info.columns:
        # Format min/max values for display
        min_display = (
            col.min_value if col.has_min_max and col.min_value is not None else "N/A"
        )
        max_display = (
            col.max_value if col.has_min_max and col.max_value is not None else "N/A"
        )

        # Format exactness indicator (PyArrow 22+ feature)
        exact_display = "N/A"
        if col.is_min_exact is not None and col.is_max_exact is not None:
            if col.is_min_exact and col.is_max_exact:
                exact_display = "✅"
            elif col.is_min_exact or col.is_max_exact:
                exact_display = "~"  # Partially exact
            else:
                exact_display = "❌"

        # Format encryption status
        encrypted_display = "🔒" if col.is_encrypted else "—"

        row_data = [
            str(col.row_group),
            col.column_name,
            str(col.column_index),
            col.compression_type,
            "✅" if col.has_bloom_filter else "❌",
            encrypted_display,
            min_display,
            max_display,
            exact_display,
        ]

        if show_sizes:
            # Calculate compression ratio
            ratio = "N/A"
            if col.total_compressed_size and col.total_uncompressed_size:
                ratio = (
                    f"{col.total_uncompressed_size / col.total_compressed_size:.1f}x"
                )

            row_data.extend(
                [
                    str(col.num_values) if col.num_values else "N/A",
                    format_size(col.total_compressed_size),
                    ratio,
                ]
            )

        table.add_row(*row_data)

    # Print the table
    console.print(table)


def output_json(
    meta_model: ParquetMetaModel,
    column_info: ParquetColumnInfo,
    compression_codecs: set,
) -> None:
    """
    Outputs the parquet information in JSON format.

    Args:
        meta_model: The Parquet metadata model
        column_info: The column information model
        compression_codecs: Set of compression codecs used
    """
    result = {
        "metadata": meta_model.model_dump(),
        "columns": [column.model_dump() for column in column_info.columns],
        "compression_codecs": list(compression_codecs),
    }

    print(json.dumps(result, indent=2))


def inspect_single_file(
    filename: str,
    format: OutputFormat,
    metadata_only: bool,
    column_filter: Optional[str],
    show_sizes: bool = False,
) -> None:
    """
    Inspect a single Parquet file and display its metadata, compression settings, and bloom filter information.

    Raises:
        Exception: If the file cannot be processed.
    """
    try:
        (parquet_metadata, compression) = read_parquet_metadata(filename)
    except FileNotFoundError:
        raise Exception(f"Cannot open: {filename}.")
    except Exception as e:
        raise Exception(f"Failed to read metadata: {e}")

    # Create metadata model
    meta_model = ParquetMetaModel(
        created_by=parquet_metadata.created_by,
        num_columns=parquet_metadata.num_columns,
        num_rows=parquet_metadata.num_rows,
        num_row_groups=parquet_metadata.num_row_groups,
        format_version=str(parquet_metadata.format_version),
        serialized_size=parquet_metadata.serialized_size,
    )

    # Create a model to store column information
    column_info = ParquetColumnInfo()

    # Collect information
    print_compression_types(parquet_metadata, column_info)
    print_bloom_filter_info(parquet_metadata, column_info)
    print_min_max_statistics(parquet_metadata, column_info)

    # Filter columns if requested
    if column_filter:
        column_info.columns = [
            col for col in column_info.columns if col.column_name == column_filter
        ]
        if not column_info.columns:
            console.print(
                f"No columns match the filter: {column_filter}", style="yellow"
            )

    # Output based on format selection
    if format == OutputFormat.JSON:
        output_json(meta_model, column_info, compression)
    else:  # Rich format
        # Print the metadata
        console.print(meta_model)

        # Print column details if not metadata only
        if not metadata_only:
            print_column_info_table(column_info, show_sizes=show_sizes)
            console.print(f"Compression codecs: {compression}")


@app.command(name="")
@app.command(name="inspect")
def inspect(
    filenames: List[str] = typer.Argument(
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
    column_filter: Optional[str] = typer.Option(
        None, "--column", "-c", help="Filter results to show only specific column"
    ),
    show_sizes: bool = typer.Option(
        False,
        "--sizes",
        "-s",
        help="Show column sizes and compression ratios",
    ),
):
    """
    Inspect Parquet files and display their metadata, compression settings, and bloom filter information.
    """
    # Expand glob patterns and collect all matching files
    all_files = []
    for pattern in filenames:
        matches = glob.glob(pattern)
        if matches:
            all_files.extend(matches)
        else:
            # If no matches found, treat as literal filename (for better error reporting)
            all_files.append(pattern)

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file in all_files:
        if file not in seen:
            seen.add(file)
            unique_files.append(file)

    # Process each file
    for i, filename in enumerate(unique_files):
        # For multiple files, add a header to separate results
        if len(unique_files) > 1:
            if i > 0:
                console.print()  # Add blank line between files
            console.print(f"[bold blue]File: {filename}[/bold blue]")
            console.print("─" * (len(filename) + 6))

        try:
            inspect_single_file(
                filename, format, metadata_only, column_filter, show_sizes
            )
        except Exception as e:
            console.print(f"Error processing {filename}: {e}", style="red")
            continue


if __name__ == "__main__":
    app()
