import json
from pathlib import Path

from typer.testing import CliRunner

from iparq.source import app

# Define path to test fixtures
FIXTURES_DIR = Path(__file__).parent


def test_parquet_info():
    """Test that the CLI correctly displays parquet file information."""
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "dummy.parquet"
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
