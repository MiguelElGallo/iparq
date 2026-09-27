"""Check literal documentation on every Python definition without percentage rounding."""

import ast
import os
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass(frozen=True)
class Policy:
    """Store shared file exclusions and minimum function summary requirements."""

    exclude_dirs: frozenset[str]
    min_words: int
    min_characters: int


def load_policy(root: Path) -> Policy:
    """Read and validate documentation limits from the repository configuration."""
    with (root / "pyproject.toml").open("rb") as stream:
        settings = tomllib.load(stream)["tool"]["pythonprs"]
    words = settings["docstring-min-words"]
    characters = settings["docstring-min-characters"]
    exclusions = settings["exclude-dirs"]
    if type(words) is not int or type(characters) is not int:
        raise ValueError("Docstring minimums must be positive integers.")
    if words < 1 or characters < 1:
        raise ValueError("Docstring minimums must be positive integers.")
    if not isinstance(exclusions, list) or not all(
        isinstance(name, str) for name in exclusions
    ):
        raise ValueError("exclude-dirs must contain directory names as strings.")
    return Policy(frozenset(exclusions), words, characters)


def _raise_walk_error(error: OSError) -> NoReturn:
    """Stop repository scanning when a directory cannot be read safely."""
    raise error


def discover_python_files(root: Path, policy: Policy) -> list[Path]:
    """Find Python files throughout the repository, including newly added directories."""
    if not root.is_dir():
        raise ValueError(f"Repository directory does not exist: {root}")
    files = []
    for current, directories, names in os.walk(root, onerror=_raise_walk_error):
        directory = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if name not in policy.exclude_dirs and not (directory / name).is_symlink()
        )
        files.extend(directory / name for name in sorted(names) if name.endswith(".py"))
    if not files:
        raise ValueError("No Python files found; refusing to pass an empty scan.")
    return sorted(files)


def summary_counts(docstring: str) -> tuple[int, int]:
    """Count words and non-whitespace characters in the first documentation paragraph."""
    summary = re.split(r"\n\s*\n", docstring.strip(), maxsplit=1)[0]
    words = re.findall(r"\b\w+(?:['’-]\w+)*\b", summary)
    characters = sum(not character.isspace() for character in summary)
    return len(words), characters


def _definition_issue(node: ast.AST, path: Path, policy: Policy) -> str | None:
    """Describe missing documentation or an undersized summary for one definition."""
    if not isinstance(
        node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        return None
    name = getattr(node, "name", "<module>")
    location = f"{path}:{getattr(node, 'lineno', 1)}: {name}"
    docstring = ast.get_docstring(node)
    if not docstring or not docstring.strip():
        return f"{location}: DOC001 missing or blank docstring"
    if isinstance(node, (ast.Module, ast.ClassDef)):
        return None
    words, characters = summary_counts(docstring)
    if words < policy.min_words or characters < policy.min_characters:
        return (
            f"{location}: DOC002 summary has {words} words/{characters} characters; "
            f"requires >= {policy.min_words} words and >= {policy.min_characters} "
            "non-whitespace characters"
        )
    return None


def check_file(path: Path, policy: Policy) -> tuple[int, list[str]]:
    """Inspect every definition and fail clearly on unreadable or invalid source."""
    try:
        if path.is_symlink():
            raise ValueError("Symbolic-link Python files are unsupported.")
        with tokenize.open(path) as stream:
            tree = ast.parse(stream.read(), filename=str(path))
    except (OSError, SyntaxError, UnicodeError, ValueError) as error:
        return 0, [f"{path}: DOC003 cannot inspect Python source: {error}"]
    functions = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(tree)
    )
    issues = [
        issue
        for node in ast.walk(tree)
        if (issue := _definition_issue(node, path, policy)) is not None
    ]
    return functions, issues


def check_docstrings(files: list[Path], policy: Policy) -> int:
    """Report every missing definition and short function summary without rounding."""
    function_count = 0
    issues = []
    for path in files:
        functions, file_issues = check_file(path, policy)
        function_count += functions
        issues.extend(file_issues)
    for issue in issues:
        print(issue, flush=True)
    print(
        f"Documentation: {len(files)} files, {function_count} functions, "
        f"{len(issues)} violations.",
        flush=True,
    )
    return int(bool(issues))
