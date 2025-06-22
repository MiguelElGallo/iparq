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
    """

    row_group: int
    column_name: str
    column_index: int
    compression_type: str
    has_bloom_filter: Optional[bool] = False


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

                # Create or update column info
                column_info.columns.append(
                    ColumnInfo(
                        row_group=i,
                        column_name=column_name,
                        column_index=j,
                        compression_type=compression,
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
                        # Check if this column has bloom filters using
                        # the dedicated attribute available in newer
                        # versions of PyArrow. Fallback to ``False`` if
                        # the attribute is not present.
                        has_bloom_filter = (
                            hasattr(column_chunk, "has_bloom_filter")
                            and column_chunk.has_bloom_filter
                        )
                        col.has_bloom_filter = has_bloom_filter
                        break
    except Exception as e:
        console.print(
            f"Error while collecting bloom filter information: {e}",
            style="blink bold red underline on white",
        )


def print_column_info_table(column_info: ParquetColumnInfo) -> None:
    """
    Prints the column information using a Rich table.

    Args:
        column_info: The ParquetColumnInfo model to display.
    """
    table = Table(title="Parquet Column Information")

    # Add table columns
    table.add_column("Row Group", justify="center", style="cyan")
    table.add_column("Column Name", style="green")
    table.add_column("Index", justify="center")
    table.add_column("Compression", style="magenta")
    table.add_column("Bloom Filter", justify="center")

    # Add rows to the table
    for col in column_info.columns:
        table.add_row(
            str(col.row_group),
            col.column_name,
            str(col.column_index),
            col.compression_type,
            "✅" if col.has_bloom_filter else "❌",
        )

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
            print_column_info_table(column_info)
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
            inspect_single_file(filename, format, metadata_only, column_filter)
        except Exception as e:
            console.print(f"Error processing {filename}: {e}", style="red")
            continue


if __name__ == "__main__":
    app()
