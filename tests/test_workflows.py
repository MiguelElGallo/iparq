import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_pypi_publisher_is_pinned_to_a_commit() -> None:
    """OIDC publishing must not execute a mutable third-party action ref."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/python-publish.yml").read_text()
    match = re.search(r"pypa/gh-action-pypi-publish@([^\s]+)", workflow)

    assert match is not None
    assert re.fullmatch(r"[0-9a-f]{40}", match.group(1))


def test_python_matrix_uses_the_configured_interpreter() -> None:
    """uv must use the Python version installed by each matrix job."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/test.yml").read_text()

    assert "UV_SYSTEM_PYTHON: 1" in workflow
