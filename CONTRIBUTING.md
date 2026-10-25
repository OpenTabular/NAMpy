# Contributing to NAMpy

Contributions to NAMpy can include bug fixes, documentation improvements, tests,
examples, and new modeling capabilities. Small, focused changes are welcome.

For a substantial feature or a change to a public interface, open an
[issue](https://github.com/OpenTabular/NAMpy/issues) first so that the proposed
behavior and scope can be discussed.

## Report a problem

Include a minimal example, the behavior you expected, and what happened instead.
For errors, include the traceback and your NAMpy and Python versions.

For documentation problems, link to the affected page and identify the example,
statement, or output that needs correction.

## Set up a development environment

NAMpy supports Python 3.11 and 3.12. Fork
[OpenTabular/NAMpy](https://github.com/OpenTabular/NAMpy), clone your fork, and
create a virtual environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

Create a branch for your change and keep unrelated formatting or refactoring
out of the patch.

## Find the relevant code

- `nampy/models/` contains the public estimator interfaces.
- `nampy/neural/architectures/` contains neural architectures and shared components.
- `nampy/neural/configs/` contains architecture configurations.
- `nampy/gam/` contains the statistical GAM implementation.
- `docs/` contains the documentation and tutorials.
- `tests/` contains behavioral, numerical, and compatibility tests.

Use NumPy-style docstrings for public Python interfaces.

## Test your change

For a bug fix, add a regression test that reproduces the problem. Start with the
relevant test function or file, then broaden the checks when the change affects
shared behavior. For example:

```bash
pytest tests/neural/test_neural_sklearn_contracts.py -v
```

The Makefile also supports a targeted test selection:

```bash
make test TEST=tests/neural/test_neural_sklearn_contracts.py
```

Run the relevant static checks:

```bash
ruff check nampy tests
mypy nampy
```

`make quality` runs the configured static checks and repository-hygiene check.
It does not run the test suite.

Report the checks you ran in the pull request. The full test suite is not the
default local development step.

## Change statistical GAM behavior

The statistical engine uses mgcv as a behavioral reference and scam for
shape-constrained smooths. Changes should preserve the documented statistical
construction and supported behavior.

For a compatibility fix:

1. Reproduce the discrepancy in a focused test.
2. Identify the responsible NAMpy code and upstream routine.
3. Explain the numerical or statistical difference.
4. Implement the correction and run the relevant comparison tests.

Preserve numerically stable operations and the upstream treatment of indexing,
constraints, and matrix factorizations. Do not silently substitute an approximate
fitting route for an unsupported combination.

Most compatibility tests use committed reference fixtures and do not require R.
Regenerating fixtures requires the recorded upstream versions and an explicit
refresh using `NAMPY_REFRESH_REFERENCE_FIXTURES=1`. Update the fixture provenance
when reference results change.

See the [upstream-reference guide](https://nampy.readthedocs.io/en/latest/development/reference_sources.html)
for the regeneration workflow and the
[compatibility page](https://nampy.readthedocs.io/en/latest/compatibility.html)
for supported behavior and known differences.

## Improve documentation

Explain the modeling question, show the relevant code, and interpret the
resulting output. Retain figures and tables that help readers understand the
fit; omit routine training logs and tables that only report successful internal
checks.

When changing an executable example, rerun it and update its saved outputs and
numerical discussion together.

Install the documentation dependencies and build the site:

```bash
python -m pip install -e ".[dev,docs]"
make docs
```

The documentation build treats warnings as errors. Follow the
[documentation-validation guide](https://nampy.readthedocs.io/en/latest/development/documentation_validation.html)
when changing notebook execution or generated reference material.

## Submit a pull request

Describe the problem and the resulting behavior. Include:

- The tests and checks run.
- Documentation changes needed for public behavior.
- Any compatibility implications or remaining limitations.
- For statistical compatibility fixes, the upstream routine used as a reference.
- For new methods, the relevant research and implementation sources.

Keep the change focused enough that a reviewer can understand and verify it.
