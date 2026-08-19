# Contributing to NAMpy

NAMpy accepts bug fixes, tests, documentation, and focused feature work. Python
3.11 and 3.12 are the supported development interpreters.

## Development setup

```bash
git clone https://github.com/OpenTabular/NAMpy.git
cd NAMpy
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all,dev]"
pre-commit install
```

The configured static checks are:

```bash
ruff check nampy tests scripts debug examples
mypy nampy
```

Common Make targets: `make format` (black + isort), `make lint` (ruff),
`make test TEST=...` (targeted slice), `make type-check` (mypy),
`make quality` (all checks), `make docs` / `make docs-serve`, `make build`.
Release steps live in `RELEASE_CHECKLIST.md`.

Do not run Black as a routine validation step. Keep formatting changes scoped to
the files you are deliberately changing.

## Testing

Run the smallest pytest slice that proves the change, in this order:

1. one exact test function;
2. one exact test file;
3. a narrow `-k` expression in one file;
4. neighboring tests only when the change crosses their ownership boundary.

For example:

```bash
pytest tests/neural/test_neural_sklearn_contracts.py::test_name -v
pytest tests/optimization/test_mgcv_outer_optimization_parity.py -k endpoint -v
```

`make test TEST=...` enforces the same targeted-test policy. Do not run the full
suite by default; explain why a broader run is necessary before doing so.

## GAM parity work

`nampy/gam/` is a faithful port of `mgcv`, not a loosely inspired GAM library.
The vendored `mgcv` R/C source is the behavioral specification. For a GAM
change:

1. reproduce the issue in a targeted test;
2. locate the owning Python subsystem;
3. locate the corresponding vendored `mgcv` function;
4. port its control flow, ordering, indexing, and factorization behavior;
5. run the smallest relevant parity slice.

Do not add heuristic parity fallbacks, use matrix inverses where upstream uses a
solve, silently broaden unsupported behavior, or force arbitrary eigenspace
orientation. If a probe cannot be expressed cleanly as a test, retain it as a
small script under `debug/`.

## Pull requests

- Add or update tests for user-visible behavior.
- Update public docs and `CHANGELOG.md` when appropriate.
- Keep refactors separate from parity fixes.
- Report the exact tests and static checks run.
- For GAM fixes, name the upstream `mgcv` file and function used as reference.
- Call out remaining uncertainty or deliberately unsupported surfaces.

Use NumPy-style docstrings for public Python APIs. Public neural models belong in
`nampy/models/`, their PyTorch implementations in `nampy/neural/modules/`, and their
configuration dataclasses in `nampy/neural/configs/`.

Please use the issue tracker for bug reports and include a minimal reproducer,
expected and actual behavior, NAMpy/Python versions, operating system, and the
relevant traceback.
