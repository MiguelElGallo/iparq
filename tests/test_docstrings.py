"""Check documentation completeness, summary boundaries, and failed source inspection."""

from pathlib import Path
from typing import NoReturn

import pytest

from scripts.check_docstrings import (
    Policy,
    check_docstrings,
    check_file,
    discover_python_files,
    load_policy,
    summary_counts,
)

REPOSITORY = Path(__file__).resolve().parents[1]
SUMMARY = "Return the supplied value unchanged for testing documentation requirements."


def _write_source(tmp_path: Path, body: str) -> Path:
    """Write a documented test module while preserving deliberately invalid definitions."""
    source = tmp_path / "example.py"
    source.write_text(
        f'"""Document the temporary module containing focused documentation examples."""\n\n{body}\n',
        encoding="utf-8",
    )
    return source


def _policy() -> Policy:
    """Load the unchanged repository settings for every documentation policy assertion."""
    return load_policy(REPOSITORY)


@pytest.mark.parametrize(
    ("summary", "counts"),
    [
        ("alpha bravo cider delta eagle fable grape honey", (8, 40)),
        ("alpha bravo cider delta eagle fable grape hone", (8, 39)),
        ("alphabet bravado cider delta eagle fable grape", (7, 40)),
        ("alpha bravo cider delta\neagle fable grape honey", (8, 40)),
        ("  \n\t  ", (0, 0)),
        (
            "Sum.\n\nArgs:\n    values: Many useful words belong only to details.",
            (1, 4),
        ),
    ],
)
def test_summary_counts_only_first_paragraph(
    summary: str, counts: tuple[int, int]
) -> None:
    """Count summary words independently from whitespace and later documentation details."""
    assert summary_counts(summary) == counts


@pytest.mark.parametrize(
    ("summary", "valid"),
    [
        ("alpha bravo cider delta eagle fable grape honey", True),
        ("alpha bravo cider delta eagle fable grape hone", False),
        ("alphabet bravado citron deluxe eaglet fabled grapefruit", False),
    ],
)
def test_exact_summary_minimums_are_enforced(
    tmp_path: Path, summary: str, valid: bool
) -> None:
    """Accept exact limits while rejecting either independently insufficient summary measurement."""
    source = _write_source(
        tmp_path, f'def identity() -> int:\n    """{summary}"""\n    return 1'
    )
    functions, issues = check_file(source, _policy())
    assert functions == 1
    assert (not issues) is valid
    if not valid:
        assert "DOC002" in issues[0]


@pytest.mark.parametrize(
    ("body", "name", "count"),
    [
        ("def _private() -> None:\n    pass", "_private", 1),
        ("async def asynchronous() -> None:\n    pass", "asynchronous", 1),
        (
            f'def outer() -> None:\n    """{SUMMARY}"""\n    def inner() -> None:\n        pass',
            "inner",
            2,
        ),
        (
            'class Example:\n    """Describe the documented class."""\n    def __init__(self) -> None:\n        pass',
            "__init__",
            1,
        ),
        (
            'class Example:\n    """Describe the documented class."""\n    def __repr__(self) -> str:\n        return "example"',
            "__repr__",
            1,
        ),
        (
            'class Example:\n    """Describe the documented class."""\n    @property\n    def value(self) -> int:\n        return 1',
            "value",
            1,
        ),
        (
            f'class Example:\n    """Describe the documented class."""\n    @property\n    def value(self) -> int:\n        """{SUMMARY}"""\n        return 1\n\n    @value.setter\n    def value(self, value: int) -> None:\n        pass',
            "value",
            2,
        ),
        (
            f'class Example:\n    """Describe the documented class."""\n    @property\n    def value(self) -> int:\n        """{SUMMARY}"""\n        return 1\n\n    @value.deleter\n    def value(self) -> None:\n        pass',
            "value",
            2,
        ),
        (
            "from typing import overload\n\n@overload\ndef overloaded(value: int) -> int:\n    ...",
            "overloaded",
            1,
        ),
        (
            'def whitespace() -> None:\n    """  \n      \n    """\n    pass',
            "whitespace",
            1,
        ),
    ],
)
def test_every_function_kind_requires_its_own_documentation(
    tmp_path: Path, body: str, name: str, count: int
) -> None:
    """Find missing documentation in private, nested, asynchronous, and decorated functions."""
    functions, issues = check_file(_write_source(tmp_path, body), _policy())
    assert functions == count
    assert len(issues) == 1
    assert name in issues[0]
    assert "DOC001" in issues[0]


def test_inherited_method_documentation_is_insufficient(tmp_path: Path) -> None:
    """Require each overriding method to document its own behavior explicitly."""
    body = f'class Base:\n    """Describe the documented parent class."""\n    def value(self) -> int:\n        """{SUMMARY}"""\n        return 1\n\nclass Child(Base):\n    """Describe the documented child class."""\n    def value(self) -> int:\n        return 2'
    functions, issues = check_file(_write_source(tmp_path, body), _policy())
    assert functions == 2
    assert len(issues) == 1
    assert "DOC001" in issues[0]


def test_module_and_class_docstrings_cannot_be_omitted(tmp_path: Path) -> None:
    """Enforce literal module and class documentation alongside function documentation requirements."""
    source = tmp_path / "bare.py"
    source.write_text("class Bare:\n    pass\n", encoding="utf-8")
    functions, issues = check_file(source, _policy())
    assert functions == 0
    assert len(issues) == 2
    assert all("DOC001" in issue for issue in issues)


def test_single_missing_function_never_rounds_to_success(tmp_path: Path) -> None:
    """Reject one undocumented function even when rounded coverage approaches complete documentation."""
    definitions = [
        f'def documented_{number}() -> int:\n    """{SUMMARY}"""\n    return 1\n'
        for number in range(1000)
    ]
    definitions.append("def undocumented() -> int:\n    return 1\n")
    source = _write_source(tmp_path, "\n".join(definitions))
    functions, issues = check_file(source, _policy())
    assert functions == 1001
    assert len(issues) == 1
    assert "undocumented" in issues[0]
    assert check_docstrings([source], _policy()) == 1


@pytest.mark.parametrize("problem", ["syntax", "encoding", "missing", "symlink"])
def test_invalid_or_missing_source_fails_closed(tmp_path: Path, problem: str) -> None:
    """Report inspection failures instead of treating unreadable source as documented code."""
    source = tmp_path / "broken.py"
    if problem == "syntax":
        source.write_text("def invalid(:\n", encoding="utf-8")
    elif problem == "encoding":
        source.write_bytes(b"# coding: utf-8\n\xff\n")
    elif problem == "symlink":
        target = _write_source(
            tmp_path, f'def identity() -> int:\n    """{SUMMARY}"""\n    return 1'
        )
        source.symlink_to(target)
    functions, issues = check_file(source, _policy())
    assert functions == 0
    assert len(issues) == 1
    assert "DOC003" in issues[0]


def test_unreadable_source_reports_inspection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject file permission errors through the same source inspection diagnostic."""
    source = _write_source(tmp_path, "VALUE = 1")

    def deny_read(*arguments: object, **keywords: object) -> NoReturn:
        """Simulate denied source access independently of the current operating system privileges."""
        raise PermissionError("access denied by test")

    monkeypatch.setattr("scripts.check_docstrings.tokenize.open", deny_read)
    functions, issues = check_file(source, _policy())
    assert functions == 0
    assert len(issues) == 1
    assert "DOC003" in issues[0]
    assert "access denied" in issues[0]


def test_discovery_excludes_generated_and_environment_directories(
    tmp_path: Path,
) -> None:
    """Exclude configured dependencies and generated artifacts while retaining new source locations."""
    expected = _write_source(tmp_path, "VALUE = 1")
    for directory in _policy().exclude_dirs:
        excluded = tmp_path / directory
        excluded.mkdir()
        (excluded / "bad.py").write_text("def invalid(:\n", encoding="utf-8")
    assert discover_python_files(tmp_path, _policy()) == [expected]


def test_discovery_requires_existing_nonempty_repository(tmp_path: Path) -> None:
    """Refuse both absent directories and empty Python source inventories explicitly."""
    with pytest.raises(ValueError, match="No Python files found"):
        discover_python_files(tmp_path, _policy())
    with pytest.raises(ValueError, match="does not exist"):
        discover_python_files(tmp_path / "missing", _policy())
