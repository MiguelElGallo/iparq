# iParq 0.8.2 release plan

Publish a maintenance release of the reviewed Python quality fixes to
`MiguelElGallo/iparq`, PyPI's `iparq` project, and the existing `iparq.dev` site.
The latest published release is 0.8.1; the new release is 0.8.2.

1. Synchronize all nine version fields across eight package, lock, plugin,
   marketplace, catalog, and website files. Refresh the catalog timestamp.
   Preserve identical canonical and bundled skill content, which has no version
   fields and remains accurate for the unchanged CLI behavior.
2. Add release notes and current release links. Preserve the historical quality
   baseline and completed-fix reports.
3. Make manual PyPI publishing validate the version tag, all five canonical
   quality gates, mypy, and ty. Preserve coverage reporting and the pinned OIDC
   publisher. Smoke-test the exact wheel produced by the release-build job from
   a clean environment outside the source checkout before uploading it.
4. Run the full quality suite on Python 3.10 and 3.13, actionlint, locked
   dependency consistency, version alignment, and fresh wheel/sdist checks.
   Verify wheel metadata, dependency extras, typing marker, bundled skill, and
   installed CLI output.
5. Push the existing branch and create a PR. Explain every failed check group
   in a separate detailed comment, linking the original findings, fixing
   commits, and validation. Preserve the individual commits in a merge commit.
6. Wait for all PR checks and merge only the exact tested head. Tag the merged
   main revision as v0.8.2, publish the GitHub release, and explicitly dispatch
   the existing publishing workflow at that tag. Creating a GitHub release
   alone does not trigger this repository's PyPI publishing.
7. Verify publishing and Pages workflows, GitHub release/tag provenance, PyPI
   metadata and artifact hashes, fresh public installation, and the deployed
   catalog and website version before reporting the release as complete.

## Review before implementation

Three independent agents approved this plan. Their requirements are included
above: enumerate every version field, retain the versionless skill copies,
enforce the new gates in publishing, correct its misleading trigger comment,
validate the release ref through an environment variable, and test the exact
uploaded wheel outside the repository. No release configuration or code was
edited before these reviews.

The CI review also reproduced a pre-existing matrix problem:
`UV_SYSTEM_PYTHON=1` does not override `.python-version` for `uv sync`.
Add explicit `UV_PYTHON` requests for each advertised matrix version in all
three workflows, with regression cases that fail before the configuration
fix. Commit this correction separately from the release metadata.
