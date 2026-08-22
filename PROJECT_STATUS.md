# GAM subsystem status

- Updated: 2026-08-18
- Upstream specification: vendored `mgcv` 1.9-4 R/C sources
- Stage: experimental strict-parity implementation

This is the maintained status page for `nampy/gam/`. The exact supported and
unsupported boundaries live in [GAM_IMPLEMENTED.md](GAM_IMPLEMENTED.md) and
[GAM_NOT_IMPLEMENTED.md](GAM_NOT_IMPLEMENTED.md). Future work belongs in
[backlog.md](backlog.md); resolved investigations should be captured by a
targeted regression test or a retained script under `debug/`, not accumulated
in additional status ledgers.

## Current state

The subsystem implements the full seven-stage pipeline used throughout the
repository:

1. formula parsing and canonical specifications;
2. runtime smooths and low-level bases;
3. constructed terms and constraint maps;
4. predictor and model compilation;
5. `gam.side`-style identifiability handling;
6. fitting and smoothing-parameter selection; and
7. prediction, inference, diagnostics, and parity serialization.

Within the declared surface, the implementation is intended to mirror `mgcv`
control flow, operand ordering, penalty/block assembly, factorization choices,
constraints, and edge cases. Unsupported branches raise explicitly rather than
falling back to an approximation.

The stable package-level GAM exports remain:

- `GAM`
- `fit_model_core`
- `solve_fit`
- `FitCoreSolution`

## Verification state

The maintained targeted suites do not currently declare an expected GAM
failure. The most recent recorded local audit covered the seven-stage gate,
the combination matrix, result ownership, general-family fitting, prediction,
post-processing, and diagnostics. Representative recorded commands include:

```text
pytest tests/parity/test_gam_seven_stage_pipeline_contracts.py -v

pytest tests/parity/test_gam_results_api_stage_owner_contracts.py -v

pytest tests/optimization/test_mgcv_postprocessing_final_fit_parity.py \
       -k 'gaulss_select_true_cr or select_true_postprocessing_at_mgcv_endpoint' -v

pytest tests/families/test_mgcv_gamlss_gaulss.py \
       tests/families/test_mgcv_gamlss_gammals.py \
       tests/families/test_gam_general_family_owner_contracts.py \
       tests/diagnostics/test_gam_diagnostics_owner_contracts.py \
       tests/parity/test_mgcv_anova_residual_df_stage_regression.py -v
```

These are historical results, not a claim that the current working tree was
revalidated after every documentation edit. No current full-suite result is
claimed. An older full-suite run used Python 3.10, outside the declared
Python 3.11-3.12 range, and predated several recorded fixes; it is not retained
as current release evidence.

Before release, the configured Python 3.11/3.12 and Linux/macOS/Windows jobs
must provide clean evidence using the vendored R package. Follow the
smallest-sufficient-slice policy locally; use a broader run only when its scope
is explicitly justified.

## Declared numerical invariants

These cases are passing behavioral contracts, not permission to weaken parity:

- Repeated or numerically zero eigenspaces may have non-unique raw basis
  orientation. Tests compare `mgcv`-relevant spaces, spectra, penalized
  operators, or fitted behavior when a coordinate representation is not
  uniquely defined.
- The `fs` null-space smoothing-parameter directions are exchangeable under an
  upstream row-permutation instability. Only that identified block is compared
  through the documented permutation invariant.
- The optimized `gaulss(select=True)` high-penalty coordinate is checked as a
  flat-tail endpoint invariant. Conditional behavior remains strict, and
  unconditional covariance/EDF2 are compared strictly at the shared `mgcv`
  endpoint.
- `optim` uses SciPy L-BFGS-B in place of R `stats::optim`; flat-boundary trace
  lengths may vary by a small documented amount while the common trace,
  boundary classification, and fitted behavior remain constrained.

New differences must first be localized to the corresponding vendored `mgcv`
routine. Do not add platform-specific LAPACK selection, eigenvector sign
forcing, heuristic canonicalization, or approximate derivative fallbacks.

## Current priorities

The active work is GAM-only:

1. obtain hosted cross-platform parity evidence;
2. close the supported combination gaps listed in [backlog.md](backlog.md);
3. port selected guarded surfaces only by following their upstream routines;
4. keep the implemented and unsupported inventories synchronized with code and
   owner tests.

Neural-model status and release work are intentionally outside this page.
