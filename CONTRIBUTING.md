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

Run the same static quality gates used by CI and release verification:

```bash
make quality
```

This runs `make lint` (Ruff over `nampy`, `tests`, and `examples`),
`make type-check`, and `make hygiene`. Type checking enforces the three stable
contract files: `nampy/contracts.py`, `nampy/neural/contracts.py`, and
`nampy/gam/fit/state.py`, with `--follow-imports=skip`. Pre-commit invokes the
same underlying type-checking and hygiene commands directly, so the hooks do
not require Make. Without Make, run `pre-commit run --all-files` for the same
enforced checks. `make quality` does not run the test suite.

For a broader diagnostic of the package-wide annotation backlog, run
`make type-check-all`. That command may report errors outside the enforced CI
scope; it is not an additional acceptance gate.

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

For a fixture refresh, select the recorded source libraries when needed and
run only the affected test. Refresh mode fills missing keys; rebuild mode is
for an intentional, reviewed source rebaseline:

```bash
export MGCV_LIB_PATH="$PWD/.cache/mgcv-lib"
export SCAM_LIB_PATH="$PWD/.cache/scam-lib"
NAMPY_REFRESH_REFERENCE_FIXTURES=1 pytest path/to/test.py::test_name -v
# Only for an intentional baseline replacement:
NAMPY_REBUILD_REFERENCE_FIXTURES=1 pytest path/to/test.py::test_name -v
pytest path/to/test.py::test_name -v
```

The final run uses committed fixtures without refresh flags. Review the source
revision, manifest, fixture diff, and assertions together. R is needed only
when generating R-reference results; neural fixture regeneration requires the
relevant reference checkout. The [fixture manifest](https://github.com/OpenTabular/NAMpy/blob/main/tests/reference_fixtures/manifest.json)
records source versions, namespaces, and the declared mgcv source difference. The
[compatibility page](https://nampy.readthedocs.io/en/latest/compatibility.html)
records supported behavior and intentional restrictions.

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

The documentation build treats warnings as errors. Run checks appropriate to
the change; the following commands cover contracts, example execution, strict
HTML, structural integrity, and browser navigation:

```bash
python -m pytest -q tests/docs tests/neural/test_documented_component_contracts.py
python docs/scripts/check_short_examples.py
python docs/scripts/execute_notebooks.py docs/notebooks/changed_example.ipynb
python -m sphinx -E -a -b html docs docs/_build/html -W --keep-going
python docs/scripts/check_documentation.py
python docs/scripts/check_reference_examples.py
python docs/scripts/check_reference_contracts.py
python docs/scripts/check_sidebar_navigation.py
```

Replace the notebook path with the affected example. With no paths, the
execution script reruns the teaching collection. Development reports can be
rerun explicitly with `docs/development/notebooks/*.ipynb`. Keep assertions
that verify scientific identities or prediction contracts, and inspect the
actual saved outputs; execution alone does not establish that an explanation
or metric label is correct.

Regenerate the relevant figures and reports from their displayed code:

```bash
python docs/scripts/generate_howto_assets.py
python docs/scripts/generate_detailed_assets.py
python docs/scripts/generate_practical_assets.py
python docs/scripts/generate_optimizer_matrix.py
```

Run only the relevant generator. The detailed-assets generator executes its
complete examples, so review both example and documentation outputs. The
homepage showcase code is shared through `docs/_includes/model_showcases.inc`.
Notebook records retain package versions, source revision and content hashes;
retain the corresponding source to reproduce a saved execution. Preserve
meaningful warnings, fitted plots, and scientific tables. Inspect the deployed
footer and outputs after publication to confirm the intended revision is live.


The detailed reference is generated by `docs/_ext/feature_reference.py` and
`public_reference.py`. Edit parameter types and returns in `reference_schema.py`,
standalone recipes in `reference_examples.py`, shared symbol contracts in
`reference_symbols.py`, and optional neural method contracts in
`neural_method_reference.py`. These files live under `docs/_ext/`. Generated
pages identify their source; editing them directly is overwritten at build time.
The example checker executes the exact downloadable scripts and records their
source fingerprints and output schemas. Regenerate the docs after execution to
include the verified output types and shapes. `--resume` reuses unchanged,
successful scripts during iteration; use a full run for final validation.

## Submit a pull request

Describe the problem and the resulting behavior. Include:

- The tests and checks run.
- Documentation changes needed for public behavior.
- Any compatibility implications or remaining limitations.
- For statistical compatibility fixes, the upstream routine used as a reference.
- For new methods, the relevant research and implementation sources.

Keep the change focused enough that a reviewer can understand and verify it.
