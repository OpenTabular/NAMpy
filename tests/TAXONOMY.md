# Test Taxonomy

The GAM test suite is intentionally overlapping. The goal is fast subset runs and fast failure localization, not zero duplication.

## Directory Layout
- `tests/parity/`: end-to-end mgcv parity matrix, output parity, additional scenarios, known gaps, and isolated failing/warning slices
- `tests/smooths/`: smooth constructor, basis, penalty, `pc=`, and linked-`id=` coverage
- `tests/optimization/`: outer optimization, Newton/trace parity, score-gamma checks, backend selection, and Gaussian smoothness post-processing
- `tests/families/`: general-family and GAMLSS-specific derivative/value coverage
- `tests/diagnostics/`: diagnostics such as `k_check`
- `tests/regressions/`: targeted regression tests for previously fixed bugs
- `tests/`: shared helpers, marker inference, cache, and parity R scripts

## Taxonomy Axes
- `smooth_<name>`: `cr`, `cs`, `cc`, `ps`, `tp`, `ts`, `te`, `ti`, `t2`, `gp`, `fs`, `sz`, `mrf`, `re`
- `family_<name>`: `gaussian`, `binomial`, `poisson`, `gamma`, `negbin`, `gaulss`, `gammals`, `gevlss`, `shashlss`, `ziplss`, `general`
- `method_<name>`: `fixed`, `reml`, `ml`, `laml`
- `select_true` / `select_false`
- `surface_<name>`: `snapshot`, `output`, `smoothcon`, `trace`, `kcheck`, `derivatives`, `regression`, `backend`
- `status_<name>`: `stable`, `known_gap`, `failing_or_warning`, `regression`

## Common Subset Runs
```bash
pytest tests/smooths -v
pytest tests/optimization -v
pytest tests/families -v
pytest -m "smooth_ps"
pytest -m "smooth_t2 and method_reml"
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
