# Plan to resolve the Python quality baseline

Branch: `mpz/pythonprs-checks`. Starting commit: `51dc01d`.

## Commit boundaries

1. **Complexity:** extract coherent helpers from the five functions above the
   cognitive limit of 15 and reduce `inspect` below the McCabe limit of 10.
   Preserve table structure, JSON payloads, terminal escaping, optional metadata,
   filtering, file ordering, error messages, and exit codes. Add focused behavior
   tests where the current suite lacks coverage. New helpers receive useful
   docstrings and complete annotations when introduced.
2. **Ruff annotations:** annotate all 31 CLI test returns as `None`. This removes
   the test-side annotation findings reported by Ruff and Pyrefly.
3. **Pyrefly annotations:** add the seven missing production parameter and four
   return annotations with concrete PyArrow types or justified structural
   interfaces. Resolve any additional diagnostics these types reveal. This
   completes the remaining shared Ruff/Pyrefly annotation requirements.
4. **Docstrings:** document the seven modules and fifteen functions with missing
   documentation and improve the nineteen short summaries. Describe actual
   behavior and preserve existing useful detail. Require 100% Interrogate
   coverage and no direct AST violations.

Ruff and Pyrefly report the same 42 annotation gaps, so commits 2 and 3 divide
them between tests and production code rather than duplicating fixes. Tests
already pass; the tests section receives validation and any necessary behavioral
regressions within the corresponding fix commit.

## Peer review and parallel work

Before implementation, independent agents review the complexity decomposition,
type interfaces, and documentation/validation plan. Incorporate their findings
before editing code.

After review, complexity refactoring, CLI test annotations, and documentation in
disjoint files can proceed in parallel. Only one agent owns a file at a time.
Production typing follows the source refactor; source and CLI-test documentation
follow their respective owners. The primary agent stages each section separately
and owns all commits. Review the implementation before each commit.

## Validation

- Capture representative CLI output before refactoring and compare it afterward.
  Include Rich tables, JSON, filters, metadata-only output, multiple files,
  duplicate patterns, and missing-file failures.
- Keep the configured limits, inventory, annotation rules, and documentation
  thresholds unchanged. Do not suppress findings or replace useful types with
  `Any` to satisfy a check.
- Validate each section with its relevant gate and the application tests. Run
  all five quality gates, mypy, ty, actionlint, locked dependency consistency,
  and whitespace checks on the completed changes.
- Run the full suite on Python 3.10 and Python 3.13. Preserve the original
  baseline report and save the final counts and commit mapping separately.

No push, pull request, merge, release, or GitHub ruleset change is included.

## Plan review outcome

Three independent reviews approved the plan before implementation:

- Complexity review required preserving the first matching statistics column,
  zero-versus-missing rendering, legacy-index tri-state values, and per-file
  error handling. Add regressions for wildcard ordering, failed JSON inputs,
  absent statistics, and nullable table cells.
- Typing review recommended PyArrow's public metadata classes and precise
  `Literal` overloads for the optional Boolean and integer legacy fields.
  Preserve the existing missing-attribute/unsupported-field exception handling
  and the legacy Bloom-filter mock. PyArrow's native API currently has no stubs;
  exercise those boundaries with actual Parquet files as well as static checks.
- Documentation review confirmed all 41 findings and approved parallel editing
  of five disjoint files containing 21 findings. Describe actual assertions;
  the empty `conftest.py` must not claim to provide fixtures.
