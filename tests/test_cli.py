from typer.testing import CliRunner

from src.iparq.source import app


def test_empty():
    assert True


def test_parquet_info():
    """Test that the CLI correctly displays parquet file information."""
    runner = CliRunner()
    result = runner.invoke(app, ["dummy.parquet"])

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
