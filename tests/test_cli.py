import json
from pathlib import Path

from typer.testing import CliRunner

from iparq.source import app

# Define path to test fixtures
FIXTURES_DIR = Path(__file__).parent
fixture_path = FIXTURES_DIR / "dummy.parquet"


def test_parquet_info():
    """Test that the CLI correctly displays parquet file information."""
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(fixture_path)])

    assert result.exit_code == 0

    expected_output = """ParquetMetaModel(
    created_by='parquet-cpp-arrow version 14.0.2',
    num_columns=3,
    num_rows=3,
    num_row_groups=1,
    format_version='2.6',
    serialized_size=2223
)
                   Parquet Column Information                   
┏━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Row Group ┃ Column Name ┃ Index ┃ Compression ┃ Bloom Filter ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│     0     │ one         │   0   │ SNAPPY      │      ✅      │
│     0     │ two         │   1   │ SNAPPY      │      ✅      │
│     0     │ three       │   2   │ SNAPPY      │      ✅      │
└───────────┴─────────────┴───────┴─────────────┴──────────────┘
Compression codecs: {'SNAPPY'}"""

    assert expected_output in result.stdout


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


def test_multiple_files():
    """Test that multiple files can be inspected in a single command."""
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "dummy.parquet"
    # Use the same file twice to test deduplication behavior
    
    result = runner.invoke(app, ["inspect", str(fixture_path), str(fixture_path)])
    
    assert result.exit_code == 0
    # Since both arguments are the same file, deduplication means only one file is processed
    # and since there's only one unique file, no file header should be shown
    assert "File:" not in result.stdout  # No header for single file (after deduplication)
    assert result.stdout.count("ParquetMetaModel") == 1


def test_multiple_different_files():
    """Test multiple different files by creating a temporary copy."""
    import shutil
    import tempfile
    
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "dummy.parquet"
    
    # Create a temporary file copy
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
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
