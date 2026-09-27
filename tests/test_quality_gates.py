"""Verify real tool failures, boundary limits, and complete repository discovery."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.quality import GATES, _run, main

REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _project(tmp_path: Path, fixture: str, target: str = "sample.py") -> Path:
    """Materialize an isolated example using the unchanged production gate configuration."""
    shutil.copyfile(REPOSITORY / "pyproject.toml", tmp_path / "pyproject.toml")
    destination = tmp_path / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = FIXTURES / fixture
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def _run_gate(gate: str, root: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the actual repository runner and capture its blocking diagnostics."""
    return subprocess.run(
        [sys.executable, "-m", "scripts.quality", gate, "--root", str(root)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _output(result: subprocess.CompletedProcess[str]) -> str:
    """Combine standard output and errors for informative failure assertions."""
    return result.stdout + result.stderr


@pytest.mark.parametrize("gate", ["complexity", "ruff", "types", "docstrings"])
def test_good_example_passes_every_gate(tmp_path: Path, gate: str) -> None:
    """Accept the documented good example through each real production gate."""
    shutil.copyfile(REPOSITORY / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copyfile(REPOSITORY / "examples/good.py", tmp_path / "good.py")
    result = _run_gate(gate, tmp_path)
    assert result.returncode == 0, _output(result)


@pytest.mark.parametrize(
    ("fixture", "score", "status"),
    [("cognitive_15.py.txt", 15, 0), ("cognitive_16.py.txt", 16, 1)],
)
def test_cognitive_complexity_boundary(
    tmp_path: Path, fixture: str, score: int, status: int
) -> None:
    """Enforce the cognitive limit inclusively while reporting the measured score."""
    root = _project(tmp_path, fixture)
    result = _run_gate("complexity", root)
    output = _output(result)
    assert result.returncode == status, output
    assert f"cognitive_boundary {score}" in output, output
    assert _run_gate("ruff", root).returncode == 0


@pytest.mark.parametrize(
    ("fixture", "score", "status"),
    [("mccabe_10.py.txt", 10, 0), ("mccabe_11.py.txt", 11, 1)],
)
def test_mccabe_complexity_boundary(
    tmp_path: Path, fixture: str, score: int, status: int
) -> None:
    """Enforce the independent branch limit without rejecting acceptable cognitive scores."""
    root = _project(tmp_path, fixture)
    result = _run_gate("ruff", root)
    output = _output(result)
    assert result.returncode == status, output
    if status:
        assert "C901" in output, output
        assert f"({score} > 10)" in output, output
    assert _run_gate("complexity", root).returncode == 0


@pytest.mark.parametrize(
    ("gate", "fixture", "suppression", "diagnostic"),
    [
        ("complexity", "cognitive_16.py.txt", "complexipy: ignore", "16"),
        ("ruff", "mccabe_11.py.txt", "noqa: C901", "C901"),
        ("types", "bad_return.py.txt", "type: ignore", "bad-return"),
        ("types", "bad_return.py.txt", "pyrefly: ignore", "bad-return"),
    ],
)
def test_inline_suppression_cannot_bypass_gate(
    tmp_path: Path, gate: str, fixture: str, suppression: str, diagnostic: str
) -> None:
    """Reject prohibited code even when an inline comment requests suppression."""
    root = _project(tmp_path, fixture)
    source = root / "sample.py"
    lines = source.read_text(encoding="utf-8").splitlines()
    target = -1 if gate == "types" else 3
    lines[target] = f"{lines[target]}  # {suppression}"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = _run_gate(gate, root)
    assert result.returncode == 1, _output(result)
    assert diagnostic in _output(result), _output(result)


@pytest.mark.parametrize(
    ("fixture", "diagnostics"),
    [
        ("bad_return.py.txt", ["bad-return"]),
        ("bad_assignment.py.txt", ["bad-assignment"]),
        ("bad_argument.py.txt", ["bad-argument-type"]),
        ("bad_attribute.py.txt", ["missing-attribute"]),
        ("bad_operation.py.txt", ["unsupported-operation"]),
        (
            "missing_annotations.py.txt",
            ["implicit-any-parameter", "unannotated-return"],
        ),
        (
            "untyped_operation.py.txt",
            ["unsupported-operation", "unannotated-return"],
        ),
    ],
)
def test_important_type_errors_block_changes(
    tmp_path: Path, fixture: str, diagnostics: list[str]
) -> None:
    """Require meaningful type diagnostics instead of accepting unrelated tool failures."""
    result = _run_gate("types", _project(tmp_path, fixture))
    assert result.returncode == 1, _output(result)
    for diagnostic in diagnostics:
        assert diagnostic in _output(result), _output(result)


def test_ruff_requires_function_annotations(tmp_path: Path) -> None:
    """Reject functions whose missing annotations could hide important type errors."""
    result = _run_gate("ruff", _project(tmp_path, "missing_annotations.py.txt"))
    assert result.returncode == 1, _output(result)
    assert "ANN001" in _output(result), _output(result)
    assert "ANN201" in _output(result), _output(result)


@pytest.mark.parametrize(
    ("fixture", "diagnostic"),
    [("missing_docstring.py.txt", "DOC001"), ("short_docstring.py.txt", "DOC002")],
)
def test_documentation_examples_block_changes(
    tmp_path: Path, fixture: str, diagnostic: str
) -> None:
    """Reject missing and padded summaries through the complete documentation gate."""
    result = _run_gate("docstrings", _project(tmp_path, fixture))
    assert result.returncode == 1, _output(result)
    assert diagnostic in _output(result), _output(result)


@pytest.mark.parametrize(
    "target", ["added.py", "new_directory/added.py", ".hidden/added.py"]
)
@pytest.mark.parametrize("gate", ["complexity", "ruff", "types", "docstrings"])
def test_gate_discovers_python_outside_existing_source_directories(
    tmp_path: Path, target: str, gate: str
) -> None:
    """Discover violating Python files at repository root and in newly created directories."""
    examples = {
        "complexity": ("cognitive_16.py.txt", "16"),
        "ruff": ("mccabe_11.py.txt", "C901"),
        "types": ("bad_return.py.txt", "bad-return"),
        "docstrings": ("missing_docstring.py.txt", "DOC001"),
    }
    fixture, diagnostic = examples[gate]
    result = _run_gate(gate, _project(tmp_path, fixture, target))
    assert result.returncode == 1, _output(result)
    assert diagnostic in _output(result), _output(result)


@pytest.mark.parametrize("gate", ["complexity", "ruff", "types", "docstrings"])
def test_empty_repository_cannot_silently_pass(tmp_path: Path, gate: str) -> None:
    """Reject empty source inventories instead of producing misleading successful checks."""
    shutil.copyfile(REPOSITORY / "pyproject.toml", tmp_path / "pyproject.toml")
    result = _run_gate(gate, tmp_path)
    assert result.returncode == 2, _output(result)
    assert "No Python files found" in _output(result), _output(result)


@pytest.mark.parametrize("name", ["complexipy.toml", ".complexipy.toml"])
def test_alternate_complexity_configuration_is_rejected(
    tmp_path: Path, name: str
) -> None:
    """Prevent alternate tool configuration from silently weakening the reviewed complexity limit."""
    root = _project(tmp_path, "cognitive_16.py.txt")
    (root / name).write_text("max-complexity-allowed = 100\n", encoding="utf-8")
    result = _run_gate("complexity", root)
    assert result.returncode == 2, _output(result)
    assert "Alternate complexipy configuration" in _output(result), _output(result)


@pytest.mark.parametrize("setting", ['exclude = ["sample.py"]', "diff = {}"])
def test_complexity_rejects_partial_scan_modes(tmp_path: Path, setting: str) -> None:
    """Reject diff and exclusion settings that could evade absolute repository enforcement."""
    root = _project(tmp_path, "cognitive_16.py.txt")
    configuration = root / "pyproject.toml"
    content = configuration.read_text(encoding="utf-8")
    configuration.write_text(
        content.replace("[tool.complexipy]\n", f"[tool.complexipy]\n{setting}\n"),
        encoding="utf-8",
    )
    result = _run_gate("complexity", root)
    assert result.returncode == 2, _output(result)
    assert "bypass the absolute repository cap" in _output(result), _output(result)


def test_complexity_uses_reviewed_configuration_threshold(tmp_path: Path) -> None:
    """Honor a deliberately stricter configured cap through the actual tool invocation."""
    root = _project(tmp_path, "cognitive_15.py.txt")
    configuration = root / "pyproject.toml"
    content = configuration.read_text(encoding="utf-8")
    configuration.write_text(
        content.replace("max-complexity-allowed = 15", "max-complexity-allowed = 10"),
        encoding="utf-8",
    )
    result = _run_gate("complexity", root)
    assert result.returncode == 1, _output(result)
    assert "cognitive_boundary" in _output(result), _output(result)
    assert "15" in _output(result), _output(result)


@pytest.mark.parametrize("gate", ["complexity", "ruff", "types", "docstrings"])
def test_ignore_file_cannot_hide_repository_source(tmp_path: Path, gate: str) -> None:
    """Check discovered Python files even when an ignore file lists their paths."""
    examples = {
        "complexity": ("cognitive_16.py.txt", "16"),
        "ruff": ("mccabe_11.py.txt", "C901"),
        "types": ("bad_return.py.txt", "bad-return"),
        "docstrings": ("missing_docstring.py.txt", "DOC001"),
    }
    fixture, diagnostic = examples[gate]
    root = _project(tmp_path, fixture)
    (root / ".gitignore").write_text("sample.py\n", encoding="utf-8")
    result = _run_gate(gate, root)
    assert result.returncode == 1, _output(result)
    assert diagnostic in _output(result), _output(result)


def test_missing_quality_executable_is_a_configuration_failure(tmp_path: Path) -> None:
    """Report an unavailable tool as failure before attempting any subprocess execution."""
    assert _run(["_pythonprs_nonexistent_quality_binary_"], tmp_path) == 2


def test_terminated_quality_tool_cannot_report_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Convert signal termination into failure before aggregating independent gate components."""

    def locate_binary(command: str) -> str:
        """Provide a known executable path for the simulated termination scenario."""
        return "/test/quality-tool"

    def terminated_process(
        command: list[str], *, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        """Simulate a process terminated by an operating system termination signal."""
        return subprocess.CompletedProcess(command, -15)

    monkeypatch.setattr("scripts.quality.shutil.which", locate_binary)
    monkeypatch.setattr("scripts.quality.subprocess.run", terminated_process)
    assert _run(["quality-tool"], tmp_path) == 2


def test_all_gates_report_failures_without_stopping_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run every selected gate and summarize failures despite earlier unsuccessful checks."""
    root = _project(tmp_path, "missing_docstring.py.txt")
    observed: list[str] = []

    def record_gate(gate: str, root: Path, files: list[Path]) -> int:
        """Record each invocation while simulating independent successful and failing gates."""
        observed.append(gate)
        return int(gate in {"complexity", "docstrings"})

    monkeypatch.setattr("scripts.quality.run_gate", record_gate)
    assert main(["all", "--root", str(root)]) == 1
    assert observed == list(GATES)
    output = capsys.readouterr().out
    assert "complexity: FAIL (exit 1)" in output
    assert "tests: PASS (exit 0)" in output
    assert "2/5 gates failed." in output
