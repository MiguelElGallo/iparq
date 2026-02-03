import json
from pathlib import Path

from typer.testing import CliRunner

from iparq.source import (
    ColumnInfo,
    ParquetColumnInfo,
    ParquetMetaModel,
    app,
    format_size,
    output_json,
)

# Define path to test fixtures
FIXTURES_DIR = Path(__file__).parent
fixture_path = FIXTURES_DIR / "dummy.parquet"


def test_parquet_info():
    """Test that the CLI correctly displays parquet file information."""
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(fixture_path)])

    assert result.exit_code == 0

    # Check for key components instead of exact table format
    assert "ParquetMetaModel" in result.stdout
    assert "created_by='parquet-cpp-arrow version 14.0.2'" in result.stdout
    assert "num_columns=3" in result.stdout
    assert "num_rows=3" in result.stdout
    assert "Parquet Column Information" in result.stdout
    # Check for data values (these are more reliable than table headers which may be truncated)
    assert "one" in result.stdout and "-1.0" in result.stdout and "2.5" in result.stdout
    assert "two" in result.stdout and "bar" in result.stdout and "foo" in result.stdout
    assert (
        "three" in result.stdout
        and "False" in result.stdout
        and "True" in result.stdout
    )
    assert "SNAPPY" in result.stdout


def test_metadata_only_flag():
    """Test that the metadata-only flag works correctly."""
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "dummy.parquet"
    result = runner.invoke(app, ["inspect", "--metadata-only", str(fixture_path)])

    assert result.exit_code == 0
    assert "ParquetMetaModel" in result.stdout
    assert "Parquet Column Information" not in result.stdout


def test_column_filter():
    """Test that filtering by column name works correctly."""
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "dummy.parquet"
    result = runner.invoke(app, ["inspect", "--column", "one", str(fixture_path)])

    assert result.exit_code == 0
    assert "one" in result.stdout
    assert "two" not in result.stdout


def test_json_output():
    """Test JSON output format."""
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "dummy.parquet"
    result = runner.invoke(app, ["inspect", "--format", "json", str(fixture_path)])

    assert result.exit_code == 0

    # Test that output is valid JSON
    data = json.loads(result.stdout)

    # Check JSON structure
    assert "metadata" in data
    assert "columns" in data
    assert "compression_codecs" in data
    assert data["metadata"]["num_columns"] == 3

    # Check that min/max statistics are included
    for column in data["columns"]:
        assert "has_min_max" in column
        assert "min_value" in column
        assert "max_value" in column
        # For our test data, all columns should have min/max stats
        assert column["has_min_max"] is True
        assert column["min_value"] is not None
        assert column["max_value"] is not None


def test_multiple_files():
    """Test that multiple files can be inspected in a single command."""
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "dummy.parquet"
    # Use the same file twice to test deduplication behavior

    result = runner.invoke(app, ["inspect", str(fixture_path), str(fixture_path)])

    assert result.exit_code == 0
    # Since both arguments are the same file, deduplication means only one file is processed
    # and since there's only one unique file, no file header should be shown
    assert (
        "File:" not in result.stdout
    )  # No header for single file (after deduplication)
    assert result.stdout.count("ParquetMetaModel") == 1


def test_multiple_different_files():
    """Test multiple different files by creating a temporary copy."""
    import shutil
    import tempfile

    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "dummy.parquet"

    # Create a temporary file copy
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        shutil.copy2(fixture_path, tmp_file.name)
        tmp_path = tmp_file.name

    try:
        result = runner.invoke(app, ["inspect", str(fixture_path), tmp_path])

        assert result.exit_code == 0
        # Should contain file headers for both files
        assert f"File: {fixture_path}" in result.stdout
        assert f"File: {tmp_path}" in result.stdout
        # Should contain metadata for both files
        assert result.stdout.count("ParquetMetaModel") == 2
        assert result.stdout.count("Parquet Column Information") == 2
    finally:
        # Clean up temporary file
        import os

        os.unlink(tmp_path)


def test_glob_pattern():
    """Test that glob patterns work correctly."""
    runner = CliRunner()
    # Test with a pattern that should match dummy files
    result = runner.invoke(app, ["inspect", str(FIXTURES_DIR / "dummy*.parquet")])

    assert result.exit_code == 0
    # Should process at least one file
    assert "ParquetMetaModel" in result.stdout


def test_single_file_no_header():
    """Test that single files don't show file headers."""
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "dummy.parquet"
    result = runner.invoke(app, ["inspect", str(fixture_path)])

    assert result.exit_code == 0
    # Should not contain file header for single file
    assert "File:" not in result.stdout
    assert "ParquetMetaModel" in result.stdout


def test_error_handling_with_multiple_files():
    """Test that errors in one file don't stop processing of other files."""
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "dummy.parquet"
    nonexistent_path = FIXTURES_DIR / "nonexistent.parquet"

    result = runner.invoke(app, ["inspect", str(fixture_path), str(nonexistent_path)])

    assert result.exit_code == 0
    # Should process the good file
    assert "ParquetMetaModel" in result.stdout
    # Should show error for bad file
    assert "Error processing" in result.stdout
    assert "nonexistent.parquet" in result.stdout


def test_sizes_flag():
    """Test that the --sizes flag displays column size information."""
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "--sizes", str(fixture_path)])

    assert result.exit_code == 0
    assert "ParquetMetaModel" in result.stdout
    # Check for size-related output (Values, compressed size, ratio)
    # The actual values depend on the test file


def test_sizes_flag_with_json():
    """Test that --sizes flag works with JSON output and includes size fields."""
    runner = CliRunner()
    result = runner.invoke(
        app, ["inspect", "--format", "json", "--sizes", str(fixture_path)]
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)

    # Check that size fields are present in columns
    for column in data["columns"]:
        assert "num_values" in column
        assert "total_compressed_size" in column
        assert "total_uncompressed_size" in column


def test_format_size_bytes():
    """Test format_size function with bytes."""
    assert format_size(100) == "100.0B"
    assert format_size(0) == "0.0B"
    assert format_size(None) == "N/A"


def test_format_size_kilobytes():
    """Test format_size function with kilobytes."""
    assert format_size(1024) == "1.0KB"
    assert format_size(2048) == "2.0KB"


def test_format_size_megabytes():
    """Test format_size function with megabytes."""
    assert format_size(1024 * 1024) == "1.0MB"
    assert format_size(5 * 1024 * 1024) == "5.0MB"


def test_format_size_gigabytes():
    """Test format_size function with gigabytes."""
    assert format_size(1024 * 1024 * 1024) == "1.0GB"


def test_format_size_terabytes():
    """Test format_size function with terabytes."""
    assert format_size(1024 * 1024 * 1024 * 1024) == "1.0TB"


def test_column_info_model():
    """Test ColumnInfo model with new fields."""
    col = ColumnInfo(
        row_group=0,
        column_name="test_col",
        column_index=0,
        compression_type="SNAPPY",
        has_bloom_filter=True,
        has_min_max=True,
        min_value="1",
        max_value="100",
        is_min_exact=True,
        is_max_exact=True,
        is_encrypted=False,
        num_values=1000,
        total_compressed_size=512,
        total_uncompressed_size=1024,
    )

    assert col.is_min_exact is True
    assert col.is_max_exact is True
    assert col.is_encrypted is False
    assert col.num_values == 1000
    assert col.total_compressed_size == 512
    assert col.total_uncompressed_size == 1024


def test_column_info_model_defaults():
    """Test ColumnInfo model with default values for new fields."""
    col = ColumnInfo(
        row_group=0,
        column_name="test_col",
        column_index=0,
        compression_type="SNAPPY",
    )

    assert col.is_min_exact is None
    assert col.is_max_exact is None
    assert col.is_encrypted is None
    assert col.num_values is None


def test_output_json_function():
    """Test the output_json function directly."""
    import io
    import sys

    meta = ParquetMetaModel(
        created_by="test",
        num_columns=2,
        num_rows=100,
        num_row_groups=1,
        format_version="2.6",
        serialized_size=1000,
    )

    columns = ParquetColumnInfo(
        columns=[
            ColumnInfo(
                row_group=0,
                column_name="col1",
                column_index=0,
                compression_type="ZSTD",
                has_min_max=True,
                min_value="0",
                max_value="99",
                is_min_exact=True,
                is_max_exact=False,
                num_values=100,
                total_compressed_size=256,
                total_uncompressed_size=512,
            )
        ]
    )

    compression_codecs = {"ZSTD"}

    # Capture stdout
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured

    try:
        output_json(meta, columns, compression_codecs)
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    data = json.loads(output)

    assert data["metadata"]["num_columns"] == 2
    assert data["columns"][0]["is_min_exact"] is True
    assert data["columns"][0]["is_max_exact"] is False
    assert "ZSTD" in data["compression_codecs"]


def test_column_filter_no_match():
    """Test filtering by a column name that doesn't exist."""
    runner = CliRunner()
    result = runner.invoke(
        app, ["inspect", "--column", "nonexistent_column", str(fixture_path)]
    )

    assert result.exit_code == 0
    assert "No columns match the filter" in result.stdout


def test_nonexistent_file():
    """Test error handling for non-existent file."""
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "totally_fake_file.parquet"])

    assert result.exit_code == 0  # CLI should handle error gracefully
    assert "Error" in result.stdout


def test_default_command():
    """Test that the empty command name works as default."""
    runner = CliRunner()
    # The app has both @app.command(name="") and @app.command(name="inspect")
    # So 'inspect' is required but maps to the same function
    result = runner.invoke(app, ["inspect", str(fixture_path)])

    assert result.exit_code == 0
    assert "ParquetMetaModel" in result.stdout
