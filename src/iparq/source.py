from typing import List, Optional

import pyarrow.parquet as pq
import typer
from pydantic import BaseModel
from rich import print
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


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
    """
    try:
        compression_codecs = set([])
        parquet_metadata = pq.ParquetFile(filename).metadata

        for i in range(parquet_metadata.num_row_groups):
            for j in range(parquet_metadata.num_columns):
                compression_codecs.add(
                    parquet_metadata.row_group(i).column(j).compression
                )

    except FileNotFoundError:
        console.print(
            f"Cannot open: {filename}.", style="blink bold red underline on white"
        )
        exit(1)

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


@app.command()
def main(filename: str):
    """
    Main function to read and print Parquet file metadata.

    Args:
        filename (str): The path to the Parquet file.

    Returns:
        Metadata of the Parquet file and the compression codecs used.
    """
    (parquet_metadata, compression) = read_parquet_metadata(filename)

    print_parquet_metadata(parquet_metadata)

    # Create a model to store column information
    column_info = ParquetColumnInfo()

    # Collect information
    print_compression_types(parquet_metadata, column_info)
    print_bloom_filter_info(parquet_metadata, column_info)

    # Print the information as a table
    print_column_info_table(column_info)

    print(f"Compression codecs: {compression}")


if __name__ == "__main__":
    app()
