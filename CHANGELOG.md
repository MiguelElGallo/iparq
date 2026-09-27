# Changelog

## 0.8.2 — 27 September 2026

This maintenance release adds consistent Python quality checks and resolves all
four initially failing check groups. CLI options, output, and error behavior
are preserved.

- Add five independent CI checks for complexity, Ruff, Pyrefly, documentation,
  and tests, with the same runner available locally.
- Refactor five functions to meet the cognitive complexity limit of 15; reduce
  `inspect` McCabe complexity from 14 to 8, below the limit of 10.
- Add the 42 missing annotations reported by Ruff and Pyrefly, including precise
  types for optional Boolean and integer legacy metadata.
- Resolve all 41 documentation findings and reach 100% documentation coverage.
- Add 24 behavior regression cases. Fix the CI matrices to explicitly select
  their advertised Python versions, with regression checks across all three
  workflows. The full suite now has 152 passing tests.
- Synchronize package, plugin, marketplace, discovery catalog, and website
  versions. Publishing now validates the release tag, runs all five quality
  gates, and smoke-tests the exact wheel that will be uploaded.

See the [original findings](reports/pythonprs-baseline-2026-09-27.md),
[fix commits and validation](reports/pythonprs-fixes-2026-09-27.md), and
[release plan](reports/release-0.8.2-plan.md) for details. One existing
non-blocking Pyrefly warning remains for missing `jsonschema.validators` stubs.
