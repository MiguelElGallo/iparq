# Contributing to iparq

Thank you for considering contributing to iparq! We're excited to collaborate with you. Here are some guidelines to help you get started:

## How to Contribute

1. **Fork the repository**: Click the "Fork" button at the top right of this page to create a copy of the repository.
2. **Clone your fork**: Use `git clone <your-fork-url>` to clone your forked repository to your local machine.
3. **Create a branch**: Use `git checkout -b <branch-name>` to create a new branch for your changes.
4. **Make your changes**: Make the necessary changes in your local repository.
5. **Commit your changes**: Use `git commit -m "Description of changes"` to commit your changes.
6. **Push your changes**: Use `git push origin <branch-name>` to push your changes to your forked repository.
7. **Create a pull request**: Go to the original repository and create a pull request from your forked repository.

## Guidelines

- **Code of Conduct**: Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming and friendly environment.
- **Documentation**: Ensure your code changes are well-documented. Update any relevant documentation in the `docs` folder.
- **Tests**: Include tests for your changes to ensure functionality and avoid regressions.
- **Commit Messages**: Write clear and concise commit messages. Follow the format: `type(scope): message`.

## Run the Python quality checks

Install the locked application, test, and quality dependencies with uv:

```sh
uv sync --all-extras --locked
uv run --no-sync python -m scripts.quality all
```

The runner executes every gate even when another gate fails, then prints a
pass/fail summary. Run one gate by replacing `all` with `complexity`, `ruff`,
`types`, `docstrings`, or `tests`.

| Gate | Requirement |
| --- | --- |
| `complexity` | Cognitive complexity at most 15 per function (complexipy). |
| `ruff` | McCabe complexity at most 10, lint, import ordering, explicit parameter and return annotations, and formatting. |
| `types` | Pyrefly default checks, including unannotated function bodies and required annotations. |
| `docstrings` | 100% Interrogate coverage plus direct checks for literal module, class, and function docstrings. Function summaries need at least 8 words and 40 non-whitespace characters in their first paragraph. |
| `tests` | The application tests and tests of the checks themselves must pass. |

All four code gates use the same Python file inventory, including tests,
examples, private and nested functions, and hidden directories. Environment and
generated directories listed in `[tool.pythonprs]` are excluded. Inline ignores,
complexity snapshots, and tool-specific ignore files cannot hide violations.
The existing mypy and ty checks also remain available:

```sh
uv run --no-sync mypy src/iparq --config-file=pyproject.toml
uv run --no-sync ty check
```

The policy, runner, example, fixtures, and gate tests are adapted from
[pythonprs at commit 7d5cdf1](https://github.com/MiguelElGallo/pythonprs/tree/7d5cdf1755c0aab3fc43c0f66d74322528920527).
Ruff and Pyrefly target iParq's supported Python 3.10 syntax; the runner uses
`tomli` on Python 3.10. The local default remains Python 3.13.

The `Python quality` workflow provides five independent `Quality (...)` jobs
for pushes, pull requests, manual runs, and merge groups. Requiring these jobs
before merging is a separate GitHub ruleset setting. The local checks now pass;
repository administrators can select those five jobs as required checks.

The initial counts and full diagnostics remain in the historical
[27 September 2026 baseline report](reports/pythonprs-baseline-2026-09-27.md).
The [completed fixes and validation report](reports/pythonprs-fixes-2026-09-27.md)
records the separate fix commits and passing results on Python 3.10 and 3.13.

## Releases

Keep the version aligned in `pyproject.toml`, `src/iparq/__init__.py`, `uv.lock`,
both plugin manifests, the Copilot marketplace, the discovery catalog, and the
website's structured metadata. Refresh the catalog's `updatedAt` timestamp and
update the changelog and current-release links. Regenerate the lock with
`uv lock` and validate all quality gates before merging the release PR.

Tag the tested merged revision with `v` followed by the package version, then
create its GitHub release. PyPI publishing uses a separate manual workflow;
creating the GitHub release does not trigger it. For 0.8.2, dispatch it with:

```sh
gh workflow run python-publish.yml --repo MiguelElGallo/iparq --ref v0.8.2
```

The workflow rejects a ref that disagrees with the package version. It runs all
five quality gates, mypy, and ty, then smoke-tests the exact built wheel in an
isolated environment before publishing. Verify the workflow result, published
PyPI version, fresh CLI installation, and deployed catalog after release.

## Reporting Issues

If you encounter any issues or bugs, please open an issue in the repository. Provide as much detail as possible, including steps to reproduce the issue and any relevant logs or screenshots.

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).

Thank you for your contributions and support!

Happy coding!
