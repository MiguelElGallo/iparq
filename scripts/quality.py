"""Run identical blocking quality checks locally and in GitHub Actions."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from scripts.check_docstrings import (
    check_docstrings,
    discover_python_files,
    load_policy,
)

GATES = ("complexity", "ruff", "types", "docstrings", "tests")


def _run(command: list[str], root: Path) -> int:
    """Execute a quality tool and retain its actual blocking exit status."""
    executable = shutil.which(command[0])
    if executable is None:
        print(f"Required executable not found: {command[0]}", file=sys.stderr)
        return 2
    print(f"Running {command[0]} on {root}", flush=True)
    try:
        status = subprocess.run(
            [executable, *command[1:]], cwd=root, check=False
        ).returncode
        return 2 if status < 0 else status
    except OSError as error:
        print(f"Cannot run {command[0]}: {error}", file=sys.stderr)
        return 2


def _complexity_command(root: Path, paths: list[str]) -> list[str]:
    """Enforce the configured absolute cap without alternate configurations or baselines."""
    if any((root / name).exists() for name in ("complexipy.toml", ".complexipy.toml")):
        raise ValueError(
            "Alternate complexipy configuration is unsupported; use pyproject.toml."
        )
    with (root / "pyproject.toml").open("rb") as stream:
        settings = tomllib.load(stream)["tool"]["complexipy"]
    if "diff" in settings or settings.get("exclude"):
        raise ValueError(
            "Complexipy diff modes and exclusions bypass the absolute repository cap."
        )
    limit = settings["max-complexity-allowed"]
    if type(limit) is not int or limit < 0:
        raise ValueError("max-complexity-allowed must be a non-negative integer.")
    return [
        "complexipy",
        "--max-complexity-allowed",
        str(limit),
        "--snapshot-ignore",
        "--snapshot-create=false",
        "--ignore-complexity=false",
        "--no-ignore",
        "--color",
        "no",
        "--plain",
        *paths,
    ]


def gate_commands(gate: str, root: Path, files: list[Path]) -> list[list[str]]:
    """Build tool commands using one explicit file inventory and production configuration."""
    config = str(root / "pyproject.toml")
    paths = [str(path) for path in files]
    if gate == "complexity":
        return [_complexity_command(root, paths)]
    github = os.environ.get("GITHUB_ACTIONS") == "true"
    commands = {
        "ruff": [
            [
                "ruff",
                "check",
                "--config",
                config,
                "--ignore-noqa",
                "--output-format",
                "github" if github else "concise",
                *paths,
            ],
            ["ruff", "format", "--config", config, "--check", *paths],
        ],
        "types": [
            [
                "pyrefly",
                "check",
                "--config",
                config,
                "--output-format",
                "github" if github else "min-text",
                *paths,
            ]
        ],
        "docstrings": [["interrogate", "--config", config, "--no-color", *paths]],
        "tests": [["pytest"]],
    }
    return commands[gate]


def run_gate(gate: str, root: Path, files: list[Path]) -> int:
    """Run all components of a gate even when an earlier component fails."""
    statuses = [_run(command, root) for command in gate_commands(gate, root, files)]
    if gate == "docstrings":
        statuses.append(check_docstrings(files, load_policy(root)))
    return max(statuses)


def main(argv: list[str] | None = None) -> int:
    """Select repository gates, validate their scope, and return a blocking result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=("all", *GATES), nargs="?", default="all")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    try:
        policy = load_policy(root)
        files = discover_python_files(root, policy)
        selected = GATES if arguments.gate == "all" else (arguments.gate,)
        statuses = [run_gate(gate, root, files) for gate in selected]
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Quality configuration or scan error: {error}", file=sys.stderr)
        return 2
    print("\nQuality summary:", flush=True)
    for gate, status in zip(selected, statuses):
        print(
            f"  {gate}: {'PASS' if status == 0 else 'FAIL'} (exit {status})", flush=True
        )
    print(f"{sum(status != 0 for status in statuses)}/{len(selected)} gates failed.")
    return max(statuses)


if __name__ == "__main__":
    raise SystemExit(main())
