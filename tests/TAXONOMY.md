# Test Taxonomy

The GAM test suite is intentionally overlapping. The goal is fast subset runs and fast failure localization, not zero duplication.

## Directory Layout
- `tests/parity/`: end-to-end mgcv parity matrix, output parity, additional scenarios, known gaps, and isolated failing/warning slices
- `tests/smooths/`: smooth constructor, basis, penalty, `pc=`, linked-`id=`, and linked-`id` trace coverage
- `tests/optimization/`: score-history traces, full outer optimization objects, lifecycle trace-plus-final-fit parity, inner traces, score-gamma checks, backend selection, and Gaussian smoothness post-processing
- `tests/families/`: general-family and GAMLSS-specific derivative/value coverage
- `tests/diagnostics/`: diagnostics such as `k_check`
- `tests/regressions/`: targeted regression tests for previously fixed bugs and test-suite structure contracts
- `tests/`: shared helpers, marker inference, taxonomy registry, static reference fixtures, and parity-generation R scripts

## Taxonomy Axes
- `smooth_<name>`: `cr`, `cs`, `cc`, `ps`, `tp`, `ts`, `te`, `ti`, `fs`, `sz`, `re`
- `family_<name>`: `gaussian`, `binomial`, `poisson`, `gamma`, `negbin`, `gaulss`, `gammals`, `general`
- `method_<name>`: `fixed`, `reml`, `ml`, `laml`, `gcv`, `ubre`
- `link_<name>`: `identity`, `log`, `inverse`, `logit`, `probit`, `cloglog`, `cauchit`, `sqrt`
- `optimizer_<name>`: `newton`, `bfgs`, `efs`, `optim`
- `select_true` / `select_false`
- `surface_<name>`: `snapshot`, `output`, `smoothcon`, `trace`, `kcheck`, `derivatives`, `regression`, `backend`
- `status_<name>`: `stable`, `known_gap`, `failing_or_warning`, `regression`

## Common Subset Runs
```bash
pytest tests/smooths -v
pytest tests/optimization -v
pytest tests/families -v
pytest -m "smooth_ps"
pytest -m "family_negbin and select_true"
pytest -m "surface_trace or surface_derivatives"
pytest -m "status_known_gap"
pytest -m "smooth_fs and family_gaussian and surface_snapshot"
```

## Conventions
- Smooth-first slicing is the primary workflow.
- Prefer directory-scoped runs first when you know the subsystem, then narrow further with `-m`, `-k`, or an exact test node.
- Family and method marks are additive; a test may carry several marks.
- Overlap is expected when it improves triage.
- `status_failing_or_warning` and `status_known_gap` are intended to be easy to exclude from normal parity sweeps.
- Canonical file/mark ownership lives in `tests/_taxonomy_registry.py`.

## Owner-Level Coverage
- Owner-level localization lives alongside the broader parity suites; use [SUBSYSTEM_COVERAGE.md](/home/ad32/projects/package/NAMpy/tests/SUBSYSTEM_COVERAGE.md) as the primary map.
- Prefer adding one direct owner contract test before expanding end-to-end scenario matrices.
- New owner-level tests for routing, covariance/post-fit, parity tooling, and general-family postprocessing should usually land in:
  - `tests/optimization/test_gam_fit_backend_owner_contracts.py`
  - `tests/optimization/test_gam_owner_routing_objective_contracts.py`
  - `tests/optimization/test_gam_covariance_owner_contracts.py`
  - `tests/optimization/test_gam_postfit_owner_contracts.py`
  - `tests/diagnostics/test_gam_diagnostics_owner_contracts.py`
  - `tests/parity/test_gam_parity_owner_contracts.py`
  - `tests/families/test_gam_general_family_owner_contracts.py`
