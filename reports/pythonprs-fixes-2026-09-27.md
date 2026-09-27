# Python quality fixes — 27 September 2026

Branch: `mpz/pythonprs-checks`. The checks were introduced in `51dc01d`.
The [plan](pythonprs-fix-plan.md) received three independent peer reviews
before implementation. Agents worked in parallel on disjoint files, and each
section received an independent implementation review before its commit.

**All five quality gates pass on Python 3.10.21 and Python 3.13.15.**
Both full suites have 150 passing tests, with no failures or skips. These are
local macOS results; hosted GitHub Actions have not been run.

## Before and after

The [original baseline](pythonprs-baseline-2026-09-27.md) is preserved unchanged.
No check limits, annotation requirements, documentation thresholds, file scope,
or dependency pins were relaxed to resolve the findings.

| Gate | Original baseline | Completed fixes |
| --- | --- | --- |
| Cognitive complexity | 5 functions over the limit of 15 | 0 violations; PASS |
| Ruff lint and formatting | 43 lint findings; formatting already passed | 0 findings; all 16 Python files formatted; PASS |
| Pyrefly types | 42 errors | 0 errors; 1 existing non-blocking warning; PASS |
| Documentation | 41 AST violations; 84.1% coverage (116/138 objects) | 0 AST violations; 100% coverage (178/178 objects); PASS |
| Tests | 126 passed | 150 passed on each Python version; PASS |
| Failed gate groups | 4/5 | 0/5 |

Ruff and Pyrefly reported the same 42 annotation gaps. The fixes address these
once, split between 31 CLI test returns and 11 production annotations. The
remaining Ruff finding was McCabe complexity in `inspect`.

Pyrefly's existing warning is an untyped import of
`jsonschema.validators` in `tests/test_agent_plugin.py`. It was already present
in the baseline and does not fail the configured gate. No ignore was added.

The AST documentation checker now scans 153 functions across 16 Python files.
The higher documented-object count includes the extracted helpers, precise
overloads, and added regression tests.

## Separate fix commits

| Section | Commit | Changes |
| --- | --- | --- |
| Complexity | `4dccd00` — `refactor: reduce inspection and rendering complexity` | Extract coherent metadata, table, filtering, and file-handling helpers; add 14 rendering and CLI regression cases. Also commits the reviewed plan. |
| Ruff | `0c4fa92` — `style: annotate CLI test return types for Ruff` | Add `None` return annotations to 31 CLI tests without changing their bodies. |
| Pyrefly | `bbb9d3f` — `fix: type native Parquet metadata interfaces for Pyrefly` | Annotate production interfaces with public PyArrow metadata classes and precise optional-field overloads; add 10 metadata regression cases. |
| Documentation | `docs: complete required Python docstrings` — the commit containing this report | Resolve all 41 baseline documentation findings, preserve useful detail, and update contributor guidance. |

Tests were already passing, so there is no separate test-failure fix commit.
The 24 added behavior cases belong to the fixes they validate. Together with
the original 47 application tests and 79 gate tests, they bring the suite to 150.

## Complexity changes

| Function | Before | After | Allowed maximum |
| --- | ---: | ---: | ---: |
| `inspect_single_file` | 18 | 3 | 15 |
| `print_column_info_table` | 19 | 2 | 15 |
| `inspect` | 28 | 13 | 15 |
| `print_storage_details_table` | 47 | 1 | 15 |
| `print_min_max_statistics` | 49 | 11 | 15 |

`inspect` McCabe complexity fell from 14 to 8, below the limit of 10.
Every new helper also passes the configured complexity limits.

## Review and behavior validation

The plan reviews identified behavior that needed explicit preservation:
zero-versus-missing values, legacy index tri-state values, the first matching
statistics column, wildcard ordering and deduplication, and per-file error
handling. Those constraints were incorporated before implementation.

Implementation reviews checked the complexity refactor, CLI annotations,
production type interfaces, and documentation independently. The final
documentation review also verified that removing docstrings from the seven
changed Python ASTs produces exactly the same executable trees as `bbb9d3f`,
including signatures and overloads.

Sixteen CLI scenarios preserve stdout, stderr, and exit codes exactly against
the captured baseline. They cover Rich output, sizes, details, metadata-only
output, JSON, filters, duplicate inputs, missing files, mixed success and
failure, and multiple failed JSON inputs. The new regressions also cover
actual Parquet metadata, absent statistics, multiple row groups, nulls, zero
values, terminal escaping, unsupported legacy metadata, and unexpected errors.

Additional validation passes: mypy on both source files, ty, actionlint for the
quality workflow, locked dependency consistency, and Git whitespace checks.
The dependency lock, gate implementation, quality workflow, and historical
baseline reports remain unchanged from `51dc01d`.

## Reproduce

```sh
uv sync --all-extras --locked
uv run --no-sync python -m scripts.quality all
uv run --no-sync mypy src/iparq --config-file=pyproject.toml
uv run --no-sync ty check
actionlint .github/workflows/python-quality.yml
uv lock --check
git diff --check
```

The all-gates runner exits with status **0** and prints `0/5 gates failed.`
The Python 3.10 verification used a separate environment with the same locked
dependencies. [Full output from both Python versions](pythonprs-fixes-2026-09-27.txt)
is saved with workspace paths normalized to `.`.

No push, pull request, merge, release, or GitHub ruleset change was made.

## Plain ASCII table for copying

```text
Check                  Before                 After
---------------------  ---------------------  ----------------------------
Cognitive complexity   5 violations            0 violations - PASS
Ruff                   43 findings             0 findings - PASS
Pyrefly                42 errors               0 errors, 1 warning - PASS
Documentation          41 violations, 84.1%     0 violations, 100% - PASS
Tests                  126 passed              150 passed on 3.10 and 3.13
Failed check groups    4 of 5                  0 of 5
```
