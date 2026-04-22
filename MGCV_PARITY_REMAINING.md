# Remaining `mgcv` Parity Inventory

This document is a live, repo-backed inventory of:

- remaining behaviors in `nampy/gam/` that are still untested for strict `mgcv` parity
- remaining implementation TODOs before the GAM subpackage can claim broader `mgcv` parity

Scope used for this audit:

- upstream specification: vendored `mgcv/R/`
- current implementation: `nampy/gam/`
- current evidence: `tests/`
- explicitly excluded: `notes/`

## Upstream anchors used in this audit

- `mgcv/R/mgcv.r`
  - `interpret.gam()`
  - `interpret.gam0()`
  - `estimate.gam()`
  - `gam.outer()`
  - `gam.vcomp()`
- `mgcv/R/gam.fit3.r`
  - `Vb.corr()`
  - `gam.fit3.post.proc()`
  - outer-Newton / BFGS score-derivative flow
- `mgcv/R/gam.fit4.r`
  - `gam.fit5()`
  - `gam.fit5.post.proc()`
  - `efsud()`
  - `efsudr()`
- `mgcv/R/fast-REML.r`
  - `Sl.setup()`
- `mgcv/R/gamlss.r`
  - general-family score / NCV / post-fit flow
- `mgcv/R/smooth.r`
  - `smooth.construct.gp.smooth.spec()`
  - `smooth.construct.fs.smooth.spec()`
  - `smooth.construct.sz.smooth.spec()`
  - random-effect `id` rejection

## Current state at a glance

- Ordinary prediction / inference / diagnostics parity is mostly green.
  - `tests/parity/test_mgcv_prediction_inference_diagnostics_parity.py` currently keeps
    `PREDICTION_GAP_REASONS`, `UNCONDITIONAL_GAP_REASONS`, `ITERMS_GAP_REASONS`,
    `ANOVA_GAP_REASONS`, `RESIDUAL_GAP_REASONS`, and `KCHECK_GAP_REASONS` empty.
- Raw-constructor parity is also mostly green.
  - `tests/smooths/test_mgcv_raw_constructor_parity.py` currently has no live known-gap
    entries.
- General-family parity is much better covered than before.
  - `tests/families/test_general_family_mgcv_parity.py` now covers newdata
    `link` / `response` / `terms` / `lpmatrix`, unconditional SE, `concurvity`,
    `sp_vcov(edge_correct=False)`, `one_se_rule()`, and `k_check()`.
  - `tests/optimization/test_mgcv_outer_optimization_parity.py` now covers
    public `smoothing_optimizer="optim"` parity and direct general-family EFS
    outer-trace parity for `gaulss`.
- `tests/optimization/test_mgcv_ncv_qncv_parity.py` now covers NCV / QNCV parity for all
  currently implemented GAMLSS families in `nampy/gam/families/gamlss/`.
- Ordinary PIRLS final-fit post-processing now carries `mgcv`-style unconditional
  covariance / `edf2` through the fitted object graph, including edge-correct
  derivative-state carry-through for outer Newton endpoints.
- The remaining live parity pressure is concentrated in a small number of post-fit and
  explicitly unsupported surfaces, plus a tail of fail-closed branches that still lack
  dedicated tests.

## 1. Live parity blockers already visible in tests

### 1.1 General-family `t2(...)` post-fit and newdata surfaces are still xfailed

Evidence in `tests/`:

- `tests/families/test_general_family_mgcv_parity.py`
  - `_GENERAL_KNOWN_GAP_TAGS = ("t2_",)`
  - strict xfails in:
    - `test_general_family_newdata_prediction_surfaces_match_mgcv()`
    - `test_general_family_newdata_unconditional_standard_errors_match_mgcv()`
- `tests/optimization/test_mgcv_postprocessing_final_fit_parity.py`
  - `_GENERAL_POSTPROC_KNOWN_GAP_TAGS = ("t2_",)`
  - strict xfail in:
    - `test_gam_fit5_postprocessing_final_fit_matches_mgcv()`

Affected case ids:

- `gaulss_t2_full_false`
- `gaulss_t2_full_true`
- `gammals_t2_full_false`
- `gammals_t2_full_true`
- `gevlss_t2_full_false`
- `gevlss_t2_full_true`
- `shashlss_t2_full_false`
- `shashlss_t2_full_true`
- `ziplss_t2_full_false`
- `ziplss_t2_full_true`

What remains to match upstream:

- exact `gam.fit5.post.proc()` covariance / `edf2` parity for general-family tensor-ANOVA terms
- exact newdata SE parity on those same fits
- exact downstream post-processing parity through the full `gam.fit5.post.proc()` path

Likely primary owners in `nampy/gam/`:

- `smooths/tensor/t2.py`
- `fit/solvers/general_family_solver.py`
- `fit/solvers/general_newton_solver.py`
- `fit/postprocess/unconditional_covariance.py`

Upstream anchors:

- `mgcv/R/gam.fit4.r::gam.fit5.post.proc()`
- `mgcv/R/gam.fit3.r::Vb.corr()`
- `mgcv/R/fast-REML.r::Sl.setup()`
- `mgcv/R/gamlss.r`

### 1.2 One upstream-supported surface is still intentionally unsupported in Python

This is already visible in tests and remains a real parity TODO.

Tensor `gp` marginals with `xt["max.knots"]`:

- Python guard: `nampy/gam/smooths/tensor/marginals.py`
- Upstream support: `mgcv/R/smooth.r::smooth.construct.gp.smooth.spec()`
- Evidence: `tests/regressions/test_gam_mgcv_patch_regressions.py::test_tensor_gp_max_knots_xt_is_explicitly_unsupported`
- Remaining work:
  - port `max.knots` handling through tensor marginal construction, not just the univariate `gp` path

## 2. Remaining implementation TODOs not yet pinned by direct parity tests

These are real parity TODOs visible from `nampy/gam/` against the vendored upstream, but
they do not currently have a dedicated direct test of their explicit unsupported contract.

### 2.1 Gaussian dynamic ML exact derivatives are still narrower than the upstream optimizer matrix

- Python implementation: `nampy/gam/smoothing_selection/criteria/gaussian_dyn.py`
- Current behavior:
  - `_gaussian_dynamic_reml_derivative_terms()` raises unless `method in {"REML", "LAML"}`
  - the outer driver now insists on exact first- and second-derivative support for strict ML/REML/LAML Newton parity
- Upstream anchors:
  - `mgcv/R/gam.fit3.r`
  - `mgcv/R/mgcv.r::estimate.gam()`
- Remaining work:
  - either port the exact Gaussian ML dynamic derivative/Hessian path
  - or prove that current backend selection already mirrors every upstream ML case that should be reachable

### 2.2 Automatic ML/REML remains narrower for some penalty layouts

- Python guard: `nampy/gam/fit/capabilities.py::raise_ml_reml_backend_error()`
- Current behavior:
  - explicit rejection when the current backend cannot handle
    null-space penalties coupling disconnected primary penalty components
- Upstream anchors:
  - `mgcv/R/mgcv.r::estimate.gam()`
  - `mgcv/R/gam.fit3.r`
  - `mgcv/R/fast-REML.r`
- Remaining work:
  - close the remaining structural backend gap or keep the fail-closed contract explicit with dedicated tests

## 3. Remaining untested behaviors

This section is the current test-gap inventory: parity-relevant or explicitly guarded
behaviors in `nampy/gam/` that do not appear to have a dedicated test yet.

## 3.1 Formula parsing and spec-building

Primary upstream anchors:

- `mgcv/R/mgcv.r::interpret.gam()`
- `mgcv/R/mgcv.r::interpret.gam0()`
- `mgcv/R/mgcv.r` offset / model-frame handling

Untested branches in `nampy/gam/`:

- `formula/parse.py`
  - multi-argument `offset(...)` rejection in `_parse_offset_call()`
  - `**kwargs` smooth-spec rejection in `_parse_smooth_call()`
  - smooth-interaction rejection beyond exact parsing in `parse_gam_formula()`
- `formula/extract.py`
  - rejection of multiple `offset(...)` terms on the same predictor
- `specs/build.py`
  - transformed smooth `by=` expressions are parsed but still rejected downstream
  - factor `by` expansion is implemented for `s(...)` only, not `te(...)`, `ti(...)`, or `t2(...)`
  - unsupported smooth kwargs that survive parsing are rejected at build time
  - `bs="fs"` factor-by fallback `xt` payload restrictions are untested
  - missing factor `by` variable at fit time is untested
  - missing factor `by` variable at prediction rebuild time is untested

Why this matters:

- these are all user-visible formula-surface differences relative to `mgcv`
- several are currently intentional fail-closed contracts, but they should still be pinned explicitly if they remain unsupported

Already covered and therefore not part of this gap bucket:

- transformed offsets
- mixed `list(...)` kwargs parsing
- non-identifier `xt` keys such as `"max.knots"`
- vector-valued `fx=` for `te(...)` / `ti(...)`
- multi-predictor distinct offsets for general families

## 3.2 General-family prediction / diagnostics / post-fit surfaces

Primary upstream anchors:

- `mgcv/R/mgcv.r` prediction helpers
- `mgcv/R/mgcv.r::gam.vcomp()`
- `mgcv/R/gamlss.r`
- `mgcv/R/gam.fit3.r::Vb.corr()`

Untested branches in `nampy/gam/`:

- `predict/general.py`
  - positive-path `predict(type="iterms")` for single-predictor general families is not directly parity-tested
  - current tests only cover the explicit multi-predictor rejection
- `predict/predictions.py`
  - grouped-term SE rejection when a non-parametric prediction group spans multiple blocks
- `diagnostics/residuals.py`
  - `_general_family_residual_fallback()` error path when fitted values do not expose a usable primary predictor column
- `smoothing_selection/postfit.py`
  - `sp_vcov(edge_correct=True)` parity is untested
  - `gam_vcomp(rescale=True)` parity is only covered for Gaussian fits; non-Gaussian ordinary and general-family `rescale=True` remain untested

Why this matters:

- these are user-facing diagnostics and prediction surfaces
- the default parity path is mostly covered, but a few meaningful branches still have no direct lock

## 3.3 General-family `Sl` setup and nonlinear block guards

Primary upstream anchors:

- `mgcv/R/fast-REML.r::Sl.setup()`
- `mgcv/R/gam.fit4.r::gam.fit5()`
- `mgcv/R/gamlss.r`

What is already covered:

- `tests/families/test_gam_general_family_owner_contracts.py` now covers:
  - non-contiguous coefficient blocks
  - non-contiguous penalty blocks
  - fallback penalty contiguity
  - non-reparameterized single-penalty blocks
  - non-reparameterized multi-penalty blocks
  - general-family `terms` raw-basis rejection
  - multi-predictor `iterms` rejection

Still untested in `nampy/gam/`:

- `fit/solvers/general_family_solver.py`
  - reparameterized nonlinear `Sl` blocks remain explicitly unsupported but untested
  - malformed nonlinear-block metadata requirements (`updateS`, `AS`, `AdS`, `ldS`, `St`) are untested

Why this matters:

- this is exactly the setup layer that still feeds the remaining `t2(...)` general-family gaps
- the covered owner-contract surface is much better than before, but the remaining unsupported nonlinear block shapes still need dedicated tests

## 3.4 Smoothing-selection and optimizer fail-closed contracts

Primary upstream anchors:

- `mgcv/R/mgcv.r::estimate.gam()`
- `mgcv/R/mgcv.r::gam.outer()`
- `mgcv/R/gam.fit3.r`
- `mgcv/R/gam.fit4.r`
- `mgcv/R/gamlss.r`

Untested branches in `nampy/gam/`:

- `smoothing_selection/criteria/dispatch.py`
  - exact-derivative rejection when no upstream-mirrored ML/REML/LAML derivative path exists
  - exact-Hessian rejection when no upstream-mirrored Hessian path exists
  - Gaussian dynamic exact-derivative / exact-Hessian rejection when the dynamic path returns `valid=False`
- `smoothing_selection/optimize/driver.py`
  - unsupported automatic smoothing-selection family/method combinations
  - strict outer-Newton rejection when exact first and second derivatives are unavailable
  - strict BFGS rejection when an exact gradient path is unavailable
  - strict EFS rejection outside `REML` / `LAML`
- `fit/capabilities.py`
  - explicit backend rejection for ML/REML structural layouts that the current backend still cannot mirror exactly
- `smoothing_selection/optimize/efs_mgcv.py`
  - explicit general-family EFS rejection
- `smoothing_selection/criteria/ncv.py`
  - unsupported `family_class` rejection
  - unsupported extended-family rejection for non-`negbin`
  - unsupported general-family rejection outside the implemented NCV/QNCV set
  - fixed-score rejection of joint `negbin` NCV/QNCV with `estimate_theta=True`

Why this matters:

- most positive-path optimizer parity is now covered
- the remaining gaps here are mostly about locking the current fail-closed contract so unsupported surfaces do not silently broaden or regress

## 3.5 Smooth-constructor fail-closed surfaces that still lack explicit tests

Primary upstream anchor:

- `mgcv/R/smooth.r`

Untested parity-acceptable guards in `nampy/gam/`:

- `smooths/categorical/fs.py`
  - `bs='fs'` requires exactly one factor variable
  - `bs="fs"` rejects multiply penalized base smooths
- `smooths/categorical/fs.py`
  - `bs="sz"` rejects multiply penalized base smooths

Why these are lower priority:

- the upstream constructors in `mgcv/R/smooth.r` also reject these shapes
- this is test debt on explicit rejection contracts, not an implementation parity blocker

## 3.6 Sidecar helper surfaces with parity-relevant TODOs but weak direct coverage

These are not the main public blockers, but they are still visible in `nampy/gam/`.

- `fit/postprocess/gaussian_smoothness_postprocess.py`
  - module docstring still marks `P-REML`, `P-ML`, `NCV`, and non-Gaussian paths as not implemented for this helper surface
  - no direct tests currently pin those explicit unsupported contracts
- `fit/solvers/irls_core.py`
  - still carries a general-family-not-implemented note even though public general-family fits route through the dedicated backend
  - this is a lower-priority parity cleanup item unless the helper remains a supported entrypoint

## 4. Prioritized TODOs to reach broader `mgcv` parity

### P0: close the remaining live xfail-backed gaps

1. Finish exact general-family `t2(...)` final-fit and newdata-SE parity.

### P1: implement the remaining upstream-supported surfaces that are still blocked outright

2. Port tensor `gp` marginal `xt["max.knots"]` support through the tensor path.

### P2: lock the remaining fail-closed contracts with direct tests

3. Add dedicated tests for the remaining formula/build rejection branches.
4. Add dedicated tests for the remaining general-family nonlinear `Sl` unsupported branches.
5. Add dedicated tests for smoothing-selection / optimizer rejection contracts.
6. Add dedicated tests for the remaining prediction / diagnostics / post-fit branches:
   - general-family single-predictor `iterms`
   - grouped prediction-block SE rejection
   - residual fallback malformed-fitted-value rejection
   - `sp_vcov(edge_correct=True)`
   - `gam_vcomp(rescale=True)` beyond Gaussian

### P3: resolve the narrower backend surfaces that still trail the upstream matrix

7. Close or explicitly justify the Gaussian dynamic ML exact-derivative gap.
8. Close or explicitly justify the ML/REML backend restriction on null-space-coupled penalty layouts.

## 5. Suggested minimal future test slices

When each item above is touched, the smallest useful slices are:

- general-family `t2(...)`
  - `pytest tests/families/test_general_family_mgcv_parity.py -k 't2_' -v`
  - `pytest tests/optimization/test_mgcv_postprocessing_final_fit_parity.py -k 't2_' -v`
- tensor `gp` `max.knots`
  - `pytest tests/regressions/test_gam_mgcv_patch_regressions.py -k 'max_knots' -v`
  - `pytest tests/smooths/test_mgcv_raw_constructor_parity.py -k 'gp and max_knots' -v`
- formula/build contracts
  - `pytest tests/parity/test_mgcv_formula_parse_parity.py -k 'offset or by or interaction' -v`
- general-family `Sl` contracts
  - `pytest tests/families/test_gam_general_family_owner_contracts.py -k 'Sl or iterms or terms' -v`
- optimizer / smoothing-selection contracts
  - `pytest tests/optimization/test_gam_owner_routing_objective_contracts.py -v`
  - `pytest tests/optimization/test_mgcv_outer_optimization_parity.py -k 'optim or newton or bfgs or efs' -v`
- diagnostics / post-fit side branches
  - `pytest tests/diagnostics/test_gam_diagnostics_owner_contracts.py -v`
  - `pytest tests/optimization/test_mgcv_vcomp_parity.py -v`
