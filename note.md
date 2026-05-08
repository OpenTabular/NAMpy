# GAM First-Release Note

Last consolidated: 2026-05-08.

This is the single live note for the `nampy.gam` first release. It replaces the
old split notes in `todo.md`, `xfails.md`, `note2.md`, `note_parity.md`,
`MGCV_PARITY_REMAINING.md`, `failing_tests.md`, and `notes/`.

## Core Principle

`nampy.gam` is trying to mirror vendored upstream `mgcv`, not build a loosely
inspired GAM API. For parity-sensitive behavior, use the vendored `mgcv/R` and
`mgcv/src` sources as the behavioral specification.

Do not ship heuristic parity fallbacks. If strict parity is not implemented for
a surface, keep it unsupported, absent from the release support matrix, or
covered by an explicit unsupported-surface test.

## Current Status

- The previously tracked failing-test list has no remaining live failures from
  that list.
- Raw smooth-constructor known-gap registries are currently empty.
- Ordinary prediction / inference / diagnostics parity gap dictionaries are
  currently empty.
- General-family broad `t2` notes are stale because `t2` has been removed from
  the current `nampy.gam` scope.
- `gp` and `mrf` are also out of first-release scope. Stale public GAM test and
  taxonomy references have been removed; low-level non-GAM spline helpers are
  not part of this release surface.
- Packaging metadata now requires Python `>=3.10`, uses setuptools package
  discovery for all `nampy` subpackages, and the README has a scoped
  first-release `nampy.gam` support matrix.
- Historical failure dumps under the old `notes/` directory mixed real issues,
  stale removed surfaces, and raw investigation output. They are not a live
  backlog.

Do not claim the whole GAM subsystem is complete. The right first release is a
small, honest, mgcv-core subset.

## First-Release Support Matrix

### Public API

Treat the intended public GAM surface as:

- `GAM`
- `fit_model_core`
- `solve_fit`
- `FitCoreSolution`

Anything else should either be documented as internal/experimental or excluded
from first-release docs.

### Families

Safe first-release target:

- `gaussian`
- `poisson`
- `binomial`
- `gamma`
- `negbin` with fixed theta, if the targeted parity slice is green

Narrow / experimental:

- `gaulss`
- `gammals`

These general families should only be documented for the cases already locked by
targeted parity tests.

Out of first-release scope:

- `gevlss`
- `shashlss`
- `ziplss`
- broad GAMLSS/general-family behavior beyond the tested `gaulss` / `gammals`
  subset

### Smooths

Safe first-release target:

- univariate `s(...)` smooths using common bases such as `cr`, `cs`, `cc`, `ps`,
  `tp`, and `ts`
- common numeric `te(...)` and `ti(...)` tensor cases

Conditional / narrow:

- `re`, `fs`, and `sz` only for the exact surfaces covered by green parity tests

Out of first-release scope:

- `t2`
- `gp`
- `mrf`
- tensor `gp` marginals
- advanced tensor argument combinations that are not real upstream `ti` surfaces,
  such as old fake `ti_full` / `ti_ord_mc` cartesian cases

### Smoothing Selection

Safe first-release target:

- fixed smoothing parameters
- ordinary-family automatic smoothing selection only for tested methods and
  optimizers
- REML as the default recommended automatic smoothing path

Out of first-release scope unless explicitly tested green:

- NCV / QNCV
- broad EFS support
- automatic smoothing-selection combinations where the exact upstream derivative
  path is not implemented
- unsupported optimizer/family/method combinations

### Prediction

Safe first-release target:

- `type="link"`
- `type="response"`
- `type="terms"`
- `type="lpmatrix"`
- standard errors for supported surfaces

Narrow / guarded:

- unconditional standard errors only for tested supported surfaces
- general-family prediction only for tested `gaulss` / `gammals` cases

Out of first-release scope:

- multi-predictor general-family `iterms`
- prediction parameterizations wider than the fitted coefficient space
- grouped-term SEs spanning multiple non-parametric blocks unless explicitly
  tested
- transformed smooth `by=` expressions

### Diagnostics

Diagnostics can ship only with careful wording.

Parity-oriented surfaces with targeted tests may be documented. These include
the specific tested slices for residuals, `anova`, `k_check`, `sp_vcov`,
`gam_vcomp`, and related helpers.

Do not claim full `mgcv` parity for:

- `summary.gam`
- `plot.gam`
- `gam.check`
- direct BIC parity

Those should be documented as basic/local/experimental unless strict parity is
implemented and tested.

## Live Release Blockers

### P0. Freeze the Supported Test Matrix

Do not use broad cartesian products that create fake upstream surfaces. Keep
only real mgcv surfaces and explicit unsupported-surface tests.

Remove local-only tests that merely prove NumPy behavior, such as generic
`isfinite` covariance checks, unless they assert a user-visible mgcv parity
contract.

### P0. Decide Known Gaps Before Tagging

Current meaningful known-gap areas to either fix or document out of scope:

- `gaulss_select_true_cr` final-fit/post-processing covariance behavior
- gamma joint-scale optimizer trace rows if exact trace parity is promised
- exact `summary.gam`, `gam.check`, and BIC parity
- advanced tensor/factor/random-effect surfaces currently tracked as remaining
  snapshot gaps

Do not keep stale xfail registries for removed surfaces.

### P1. Add Public API Smoke Tests

Add or verify narrow smoke tests for:

- `GAM.fit`
- `GAM.predict`
- `GAM.save_model` / load path
- `bic`
- `predict_feature_vals`
- `gam_check`

If a method is not mgcv-parity, the test should lock the local contract and the
docs should not overclaim.

## High-Risk Areas To Keep Narrow

These are not all first-release blockers, but they should not be silently
broadened.

- prior/sample weights in scoring and final-fit objects
- non-default links
- factor level ordering and unused-level semantics
- factor `by` smooths beyond simple tested cases
- `fs` / `sz` / `re` prediction and unconditional covariance surfaces
- transformed smooth covariates and transformed `by=`
- tensor `d`, `m`, and marginal grouping semantics
- optimizer lifecycle trace parity beyond targeted slices
- Gaussian dynamic ML exact derivative backend
- ML/REML backend restrictions involving null-space-coupled penalty layouts
- general-family nonlinear `Sl` block layouts
- `sp_vcov(edge_correct=True)` beyond tested cases
- `gam_vcomp(rescale=True)` beyond tested cases

## Deferred Work

Keep these out of the first release unless the project explicitly expands scope:

- `t2` fit/predict parameterization and final covariance parity
- Gaussian process smooths (`gp`)
- Markov random field smooths (`mrf`)
- NCV / QNCV parity
- full GAMLSS families beyond the narrow current `gaulss` / `gammals` subset
- exact mgcv `summary.gam`
- exact mgcv `plot.gam`
- exact mgcv `gam.check`
- broad optimizer lifecycle parity for every method/family/smooth combination
- exhaustive tensor/factor/random-effect combinations

## Parity Test Policy

Use the smallest meaningful test slice.

Preferred order:

1. exact test function
2. exact test file
3. narrow `-k` expression within one file
4. broader neighboring tests only when the change crosses subsystem boundaries

Do not run the full test suite by default.

Representation policy:

- raw basis / constructor parity can use invariant comparisons when mgcv's
  representation is mathematically non-unique
- optimization state, branch-driving objects, final fits, predictions,
  smoothing parameters, covariance, EDF, and diagnostics need exact behavior
  parity
- do not hide behavior-sensitive differences behind invariant-only checks

## Suggested First-Release Validation Slices

Run focused slices before considering a release tag:

```bash
pytest tests/parity/test_mgcv_output_parity.py -q
pytest tests/parity/test_mgcv_prediction_inference_diagnostics_parity.py -q
pytest tests/optimization/test_mgcv_postprocessing_final_fit_parity.py -q
pytest tests/optimization/test_mgcv_outer_optimization_parity.py -q
pytest tests/smooths/test_mgcv_raw_constructor_parity.py tests/smooths/test_mgcv_smoothcon_parity.py -q
```

Add focused slices for any touched subsystem. A final broader run is reasonable
only as a release gate after the scoped slices are green.

## Upstream Anchors

Use these vendored upstream files first when debugging parity-sensitive behavior:

- `mgcv/R/mgcv.r`
  - `interpret.gam()`
  - `estimate.gam()`
  - `gam.outer()`
  - `predict.gam()`
  - `gam.vcomp()`
- `mgcv/R/gam.fit3.r`
  - `gam.fit3.post.proc()`
  - `Vb.corr()`
  - ordinary PIRLS / score derivative flow
- `mgcv/R/gam.fit4.r`
  - `gam.fit5()`
  - `gam.fit5.post.proc()`
  - general-family fit/post-fit flow
- `mgcv/R/fast-REML.r`
  - `Sl.setup()`
- `mgcv/R/smooth.r`
  - smooth constructors and unsupported constructor behavior
- `mgcv/src/gdi.c`
  - `gdiPK`
  - `ift1`
  - `ift2`
  - `pls_fit1`

## Maintenance Rules

- Keep this file short and live.
- Do not add raw failure dumps here.
- Remove resolved gaps in the same change that removes their xfails or
  unsupported-surface tests.
- Historical investigations belong in git history or purpose-built `debug/`
  scripts, not in a growing notes directory.
- If a new issue is important enough to keep, add it here with owner, upstream
  anchor, and the smallest validation slice.
