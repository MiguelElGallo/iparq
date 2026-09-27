# Python quality baseline — 27 September 2026

Added on branch `mpz/pythonprs-checks`, based on iParq commit
`58cf6a00ba8775f6044ee6a74a1a61a3f766ddfe`.
The checks are adapted from
[pythonprs commit 7d5cdf1755c0aab3fc43c0f66d74322528920527](https://github.com/MiguelElGallo/pythonprs/tree/7d5cdf1755c0aab3fc43c0f66d74322528920527).

**4 of 5 quality gates fail. All 126 tests pass.** These are local results on
macOS with Python 3.13.15; hosted GitHub Actions have not been run. The findings
are in existing files. The new runner, example, and gate tests pass their lint,
formatting, Pyrefly, and ty checks.

## Results

| Gate | Result | Findings |
| --- | --- | --- |
| Cognitive complexity | FAIL | 5 functions exceed the maximum of 15. |
| Ruff lint and formatting | FAIL | 43 lint findings: 7 missing parameter annotations, 35 missing return annotations, and 1 McCabe complexity violation. Formatting passes for all 14 scanned files. |
| Pyrefly types | FAIL | 42 errors: the same 7 missing parameter and 35 missing return annotations. |
| Documentation | FAIL | 41 AST violations: 22 missing or blank docstrings and 19 short function summaries. Interrogate coverage is 84.1% (116/138 documented objects); required coverage is 100%. |
| Tests | PASS | 126 passed: 47 existing tests and 79 added gate tests. No failures or skips. |

Ruff and Pyrefly report the same 42 missing annotations. Do not add their
counts as separate defects. Cognitive and McCabe complexity are independent
measurements; `inspect` violates both limits. Interrogate checks documentation
presence, while the AST checker also checks summary length.

Additional validation passed: mypy (2 source files), ty, actionlint for the new
workflow, locked dependency consistency, and Git whitespace checks.
The complete 126-test suite also passes on Python 3.10.21 in an isolated
environment, including all 79 new tests of the quality gates.

## Complexity findings

All affected functions are in `src/iparq/source.py`.

| Function | Cognitive score | Allowed maximum |
| --- | ---: | ---: |
| `inspect_single_file` | 18 | 15 |
| `print_column_info_table` | 19 | 15 |
| `inspect` | 28 | 15 |
| `print_storage_details_table` | 47 | 15 |
| `print_min_max_statistics` | 49 | 15 |

Ruff also reports McCabe complexity 14 for `inspect`, against a maximum of 10.

## Annotation findings

`src/iparq/source.py` has 11 annotation findings: 7 parameters and 4 returns.
`tests/test_cli.py` has 31 missing return annotations. Both tools report these
same locations. Full diagnostics are in the accompanying output file.

## Documentation findings

Function summaries must contain at least 8 word tokens and 40 non-whitespace
characters in their first paragraph. Every function, module, and class needs
its own nonempty literal docstring, including tests and private functions.

| File | AST violations |
| --- | ---: |
| `src/iparq/__init__.py` | 1 |
| `src/iparq/source.py` | 5 |
| `tests/conftest.py` | 1 |
| `tests/test_agent_plugin.py` | 12 |
| `tests/test_ai_catalog.py` | 6 |
| `tests/test_cli.py` | 15 |
| `tests/test_workflows.py` | 1 |

The direct AST scan inspected 116 functions across 14 Python files. Its 22
missing-docstring findings include 7 modules and 15 functions; 19 further
functions have summaries below one or both minimums.

## Reproduce

```sh
uv sync --all-extras --locked
uv run --no-sync python -m scripts.quality all
```

The expected baseline exit status is **1**. The runner prints each gate's
status and `4/5 gates failed.` It continues through all five gates after a
failure. Run a single gate with `complexity`, `ruff`, `types`, `docstrings`, or
`tests` in place of `all`.

```sh
uv run --no-sync mypy src/iparq --config-file=pyproject.toml
uv run --no-sync ty check
```

The new workflow supplies independent `Quality (complexity)`, `Quality (ruff)`,
`Quality (types)`, `Quality (docstrings)`, and `Quality (tests)` jobs. Each job
fails on violations. GitHub branch rulesets were not changed. Resolve these
baseline findings before making the jobs required for merging.

The runner and tests retain Python 3.10 compatibility through `tomli`; both
Ruff and Pyrefly target Python 3.10. Python 3.13 remains the local default.
Application behavior and the reported existing violations were left unchanged.

## Tool versions

| Tool | Locked version |
| --- | --- |
| complexipy | 8.0.1 |
| Ruff | 0.16.9 |
| Pyrefly | 1.3.1 |
| Interrogate | 1.7.0 |
| pytest | 9.1.1 |
| mypy | 2.3.1 |
| ty | 0.0.84 |

[Full output for all five gates](pythonprs-baseline-2026-09-27.txt).
