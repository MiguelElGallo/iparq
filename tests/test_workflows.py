"""Verify Python test configuration and immutable package publishing dependencies."""

import re
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_pypi_publisher_is_pinned_to_a_commit() -> None:
    """OIDC publishing must not execute a mutable third-party action ref."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/python-publish.yml").read_text()
    match = re.search(r"pypa/gh-action-pypi-publish@([^\s]+)", workflow)

    assert match is not None
    assert re.fullmatch(r"[0-9a-f]{40}", match.group(1))


@pytest.mark.parametrize(
    ("workflow_name", "version_key"),
    [
        ("test.yml", "python-version"),
        ("python-package.yml", "python_version"),
        ("merge.yml", "python_version"),
    ],
)
def test_python_matrix_uses_the_configured_interpreter(
    workflow_name: str, version_key: str
) -> None:
    """Each CI matrix must explicitly select its advertised interpreter for uv."""
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / workflow_name).read_text()

    assert "UV_SYSTEM_PYTHON: 1" in workflow
    assert f"UV_PYTHON: ${{{{ matrix.{version_key} }}}}" in workflow
