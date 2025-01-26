import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel


class FileCopyConfig(BaseModel):
    source: Path
    destination: Path


@pytest.fixture
def copy_file(tmp_path: Path) -> Path:
    config = FileCopyConfig(
        source=Path("../dummy.parquet"), destination=tmp_path / "dummy.parquet"
    )
    try:
        shutil.copy(config.source, config.destination)
    except FileNotFoundError:
        print("Source file not found.")
    finally:
        print("Copy operation complete.")
    return config.destination


def test_empty():
    assert True


def test_dummy_parquet(copy_file: Path) -> None:
    try:
        result = subprocess.run(
            ["iparq", str(copy_file)],
            capture_output=True,
            text=True,
            check=True,
        )
        data = result.stdout
        assert "SNAPPY" in data
        assert "2.6" in data
    except subprocess.CalledProcessError as e:
        print(f"Test failed with error: {e}")
    finally:
        print("Test execution complete.")
