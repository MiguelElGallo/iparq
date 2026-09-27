# iParq 0.8.2 preparation and validation

This report records local release validation before publication. The
[release plan](release-0.8.2-plan.md) received three independent reviews before
implementation, and the CI and metadata changes received independent reviews
before their commits. Hosted CI and publication results are recorded on the
release PR after they finish.

## Results

| Validation | Result |
| --- | --- |
| All five quality gates, Python 3.10.21 | PASS; 152 tests, no failures or skips |
| All five quality gates, Python 3.13.15 | PASS; 152 tests, no failures or skips |
| Ruff lint and formatting | PASS; 16 Python files formatted |
| Pyrefly | PASS; 0 errors, 1 existing non-blocking `jsonschema.validators` stub warning |
| Documentation | PASS; 178/178 objects documented, 153 functions scanned, 0 AST violations |
| mypy and ty | PASS |
| actionlint | PASS for the publishing workflow and all three matrix workflows |
| Lock consistency | PASS; only the editable iParq version changed |
| Release metadata alignment | PASS; all nine version fields report 0.8.2 |
| Wheel and source distribution | PASS; version, Python requirement, dependencies, extras, typing marker, source, and skill contents verified |
| Clean wheel installation | PASS outside the repository; installed version/origin, CLI help, detailed JSON, and missing-file exit behavior verified |
| Release-ref guard | PASS; accepts `refs/tags/v0.8.2`, rejects `main` and `v0.8.1` |
| Whitespace checks | PASS |

The [completed quality-fix report](pythonprs-fixes-2026-09-27.md) remains a
historical snapshot with 150 tests. The release adds two more cases by extending
the existing matrix regression test to all three matrix workflows.

## Additional CI correction

Commit `4aac43f` fixes a pre-existing interpreter-selection problem. A matrix
job with Python 3.10 on PATH and `UV_SYSTEM_PYTHON=1` could still select the
repository's pinned Python 3.13 when synchronizing dependencies. All three
workflows now set [`UV_PYTHON`](https://docs.astral.sh/uv/reference/environment/#uv_python)
to their respective matrix version. The three targeted regression cases fail
before the fix and pass afterward; all four workflow tests pass.

## Publication changes

The manual publishing workflow now verifies that the tag, installed package
metadata, and runtime version agree. It runs the canonical five-gate runner,
retains coverage reports and mypy/ty, and smoke-tests the wheel built in the
release-build job before uploading that same distribution to PyPI. The smoke
environment lives outside the source checkout. Action pins and OIDC permissions
are preserved.

Package, plugin, marketplace, catalog, and website metadata are synchronized to
0.8.2. Canonical and plugin skill copies remain unchanged and byte-identical.
Dependency versions, quality limits, documentation thresholds, and scan scope
remain unchanged. The original baseline and completed-fix reports are preserved.
