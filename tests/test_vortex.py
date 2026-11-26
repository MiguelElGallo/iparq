"""Exercise native Vortex inspection alongside existing Parquet CLI behavior."""

import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pytest
from typer.testing import CliRunner

from iparq.source import app

PARQUET = Path(__file__).with_name("dummy.parquet")


@pytest.fixture
def vortex_path(tmp_path: Path) -> Path:
    """Write a nested Vortex table containing nullable and non-nullable fields."""
    vx = pytest.importorskip("vortex")
    path = tmp_path / "sample.vortex"
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("tags", pa.list_(pa.string())),
        ]
    )
    vx.io.write(
        pa.table({"id": [1, 2], "tags": [["a"], None]}, schema=schema), str(path)
    )
    return path


def test_vortex_native_metadata_and_columns(vortex_path: Path) -> None:
    """Compare CLI counts, types, and sizes with the actual Vortex file."""
    result = CliRunner().invoke(app, ["inspect", str(vortex_path), "-f", "json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    meta = report["metadata"]
    data = vortex_path.read_bytes()
    assert meta == {
        "file_format": "vortex",
        "format_version": int.from_bytes(data[-8:-6], "little"),
        "num_rows": 2,
        "num_columns": 2,
        "root_dtype": "struct",
        "root_nullable": False,
        "file_size_bytes": len(data),
        "postscript_size_bytes": int.from_bytes(data[-6:-4], "little"),
    }
    assert report["columns"][0] == {
        "column_name": "id",
        "dtype": "i64",
        "nullable": False,
    }
    assert report["columns"][1]["nullable"] is True
    assert "list(" in report["columns"][1]["dtype"]
    assert report["splits"] == [{"start_row": 0, "stop_row": 2}]
    assert "row_groups" not in report


def test_vortex_inspection_uses_only_metadata(
    vortex_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevent row scans and skip split enumeration for metadata-only output."""
    vx = pytest.importorskip("vortex")

    def unexpected(*args: object, **kwargs: object) -> None:
        """Fail immediately when inspection attempts to materialize file rows."""
        pytest.fail("Metadata inspection accessed a data-reading method")

    for method in ("scan", "to_arrow", "to_dataset"):
        monkeypatch.setattr(vx.file.VortexFile, method, unexpected)
    runner = CliRunner()
    assert runner.invoke(app, ["inspect", str(vortex_path), "--details"]).exit_code == 0
    monkeypatch.setattr(vx.file.VortexFile, "splits", unexpected)
    result = runner.invoke(app, ["inspect", str(vortex_path), "-m", "-f", "json"])
    assert result.exit_code == 0, result.output
    assert set(json.loads(result.stdout)) == {"metadata"}


def test_vortex_rich_flags_and_exact_column_filter(vortex_path: Path) -> None:
    """Check Rich output for selected fields, detailed ranges, and physical sizes."""
    runner = CliRunner()
    default = runner.invoke(app, ["inspect", str(vortex_path), "-c", "id"])
    assert default.exit_code == 0
    assert "Vortex Columns" in default.stdout and "tags" not in default.stdout
    assert "Scan Splits" not in default.stdout and "size (bytes)" not in default.stdout
    detailed = runner.invoke(app, ["inspect", str(vortex_path), "--details", "--sizes"])
    assert detailed.exit_code == 0
    assert "Vortex Scan Splits" in detailed.stdout
    assert str(vortex_path.stat().st_size) in detailed.stdout
    filtered = runner.invoke(
        app, ["inspect", str(vortex_path), "-c", "absent", "-f", "json"]
    )
    assert filtered.exit_code == 0
    assert json.loads(filtered.stdout)["columns"] == []
    assert "No columns match the filter: absent" in filtered.stderr


def test_vortex_signature_and_mixed_batch_errors(
    vortex_path: Path, tmp_path: Path
) -> None:
    """Route renamed inputs by magic and preserve successes around corrupt files."""
    vortex = vortex_path.rename(tmp_path / "renamed.parquet")
    parquet = tmp_path / "renamed.vortex"
    shutil.copyfile(PARQUET, parquet)
    broken = tmp_path / "corrupt.VORTEX"
    broken.write_bytes(b"VTXF" + bytes(12))
    result = CliRunner().invoke(
        app,
        [
            "inspect",
            str(vortex),
            str(broken),
            str(parquet),
            str(tmp_path / "*.parquet"),
            "-f",
            "json",
        ],
    )
    assert result.exit_code == 1
    reports = json.loads(result.stdout)
    assert [r["file"] for r in reports] == [str(vortex), str(parquet)]
    assert reports[0]["metadata"]["file_format"] == "vortex"
    assert reports[1]["metadata"]["created_by"]
    assert "Invalid Vortex" in result.stderr


def test_vortex_scalar_empty_and_untrusted_field_names(tmp_path: Path) -> None:
    """Retain unnamed roots, zero rows, and exact names in machine-readable output."""
    vx = pytest.importorskip("vortex")
    scalar = tmp_path / "scalar"
    vx.io.write(vx.array([1, None]), str(scalar))
    name = "[red]tag\x1b"
    empty = tmp_path / "empty.vortex"
    vx.io.write(
        pa.table(
            {name: pa.array([], type=pa.int32()), "": pa.array([], type=pa.bool_())}
        ),
        str(empty),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(scalar), str(empty), "-f", "json"])
    assert result.exit_code == 0, result.output
    first, second = json.loads(result.stdout)
    assert first["columns"] == [
        {"column_name": None, "dtype": "i64?", "nullable": True}
    ]
    assert first["metadata"]["num_columns"] == 1
    assert second["metadata"]["num_rows"] == 0 and second["splits"] == []
    assert [c["column_name"] for c in second["columns"]] == [name, ""]
    rich = runner.invoke(app, ["inspect", str(empty)])
    assert "[red]tag\\x1b" in rich.stdout
    assert "\x1b" not in rich.stdout
    selected = runner.invoke(app, ["inspect", str(empty), "-c", "", "-f", "json"])
    assert [c["column_name"] for c in json.loads(selected.stdout)["columns"]] == [""]


def test_parquet_without_the_vortex_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the base installation usable and explain how to enable Vortex."""
    monkeypatch.setitem(sys.modules, "vortex", None)
    path = tmp_path / "file.vortex"
    path.write_bytes(b"VTXF")
    result = CliRunner().invoke(app, ["inspect", str(path), str(PARQUET), "-f", "json"])
    assert result.exit_code == 1
    assert [r["file"] for r in json.loads(result.stdout)] == [str(PARQUET)]
    expected = "Python 3.11" if sys.version_info < (3, 11) else "iparq[vortex]"
    assert expected in result.stderr
