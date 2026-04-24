# mgcv Parity Backlog Note

This note is based only on:

- vendored upstream `mgcv` sources under `mgcv/R` and `mgcv/src`
- the current `tests/` tree
- the current `nampy/gam/` implementation

It does not rely on repository guidance notes.

## What Looks Mostly Closed Right Now

- Raw smooth-constructor parity is no longer carrying an active known-gap registry.
  - `tests/smooths/test_mgcv_raw_constructor_parity.py` currently has empty `_KNOWN_RAW_GAPS_*` sets.
- Prediction / inference / diagnostics parity is not currently carrying registered surface xfails.
  - `tests/parity/test_mgcv_prediction_inference_diagnostics_parity.py` has empty `PREDICTION_GAP_REASONS`, `UNCONDITIONAL_GAP_REASONS`, `ITERMS_GAP_REASONS`, `ANOVA_GAP_REASONS`, `RESIDUAL_GAP_REASONS`, and `KCHECK_GAP_REASONS`.
- The dedicated remaining-gap registry is mostly documenting green or upstream-aligned unsupported surfaces, not a large live backlog.
  - `tests/parity/test_mgcv_remaining_gap_xfails.py` currently covers formula/parser wins and explicit upstream-aligned rejections like random-effect `id=`.
- Signed-weight stacked-QR work is no longer the main open item.
  - Recent regressions in `tests/regressions/test_gam_mgcv_patch_regressions.py` now cover signed-weight stacked-QR solve/state/`gdiPK`/`ift1` behavior.

The remaining backlog is concentrated in post-processing, derivative coverage, NCV/QNCV, general-family penalty setup, and a handful of parser / prediction branches.

## Highest-Priority Remaining Implementation Work

### 1. Non-Gaussian final-fit objects still do not consistently carry mgcv-style unconditional covariance / `edf2`

Evidence:

- `tests/optimization/test_mgcv_postprocessing_final_fit_parity.py` explicitly `xfail`s when the mgcv snapshot has `Vc` but the NAMpy final fit still has `None`:
  - `"Real implementation gap: non-Gaussian PIRLS final-fit objects do not yet carry mgcv-style unconditional covariance/edf2 post-processing."`

Upstream target:

- `mgcv/R/gam.fit3.r::gam.fit3.post.proc`
- `mgcv/R/gam.fit4.r::gam.fit5.post.proc`

Current Python shape:

- `nampy/gam/fit/solvers/general_newton_solver.py::postprocess_general_newton_fit()` already computes `Vc` and `edf2`.
- The gap is therefore not just formula construction or PIRLS itself; it is the end-to-end plumbing of that post-processing into the final fitted object surfaces that parity tests inspect.

Why it matters:

- This affects `Vc`, `edf2`, downstream unconditional standard errors, and any diagnostics that depend on the final public covariance surface.

### 2. General-family `t2` final-fit parity is still explicitly unresolved

Evidence:

- `tests/families/test_general_family_mgcv_parity.py` marks all `case_id` containing `t2_` as a known general-family gap.
- `tests/optimization/test_mgcv_postprocessing_final_fit_parity.py` also `xfail`s `t2_` general-family post-processing cases.

Current test language:

- `"Known general-family t2 final-fit gap; ... parity coverage is kept visible without claiming parity."`
- `"Known general-family post-proc gap: advanced/select/by/tensor surfaces do not yet have exact mgcv final-fit parity."`

Upstream target:

- `mgcv/R/gam.fit4.r::gam.fit5`
- `mgcv/R/gam.fit4.r::gam.fit5.post.proc`
- `mgcv/R/mgcv.r::predict.gam` for the associated covariance-driven prediction surfaces

What this likely means in practice:

- Pre-optimization and setup are already much healthier than final-fit/post-proc here.
- The remaining work is concentrated on the final covariance / `edf2` / post-proc representation for tensor general-family fits, not on basic `t2` term construction.

### 3. Exact PIRLS derivative coverage is still incomplete outside the currently ported families

Evidence in `nampy/gam/smoothing_selection/criteria/pirls_deriv.py`:

- Generic `gdi2` current-sp support is only complete for:
  - Gamma profile-scale
  - negbin
  - theta-free general families
- The code still raises for:
  - general families with `n_theta != 0`
  - any other family not yet given a dedicated port
- Exact PIRLS ML/REML gradients and Hessians are still limited to:
  - fixed-scale families
  - Gamma via the profiled-scale branch

Upstream target:

- `mgcv/src/gdi.c::{gdiPK, ift1, ift2, pls_fit1, applyP, applyPt}`
- `mgcv/R/gam.fit3.r`
- `mgcv/R/gam.fit4.r`

Why it matters:

- `nampy/gam/smoothing_selection/criteria/dispatch.py` now intentionally refuses numerical fallbacks for ML/REML/LAML when an exact upstream-mirrored derivative path is missing.
- So every missing derivative port is now a real user-visible parity boundary, not a hidden approximation.

### 4. NCV / QNCV coverage is still incomplete, and the signed-weight path still needs direct parity testing

Confirmed implementation gaps in `nampy/gam/smoothing_selection/criteria/ncv.py`:

- joint negative-binomial NCV/QNCV with `estimate_theta=True` is not implemented
- general-family NCV/QNCV is only implemented for:
  - `gaulss`
  - `gammals`
  - `ziplss`
  - `gevlss`
  - `shashlss`
- extended-family NCV/QNCV is only implemented for `negbin`

Upstream target:

- `mgcv/src/ncv.c::{ncv, Rncv, ncvls, Rncvls}`
- `mgcv/R/gam.fit3.r`
- `mgcv/R/gam.fit4.r`
- `mgcv/R/gamlss.r::gamlss.ncv`
- `mgcv/R/fast-REML.r::Sl.ncv`

Additional audit item:

- The current NCV implementation still works through `current.R`, triangular solves, and Cholesky update/downdate paths such as `_solve_from_upper_chol()` and `_chol_up()`.
- It also uses `sqrt(abs(...))` weight handling inside the fold-update kernels.
- After the recent signed-weight stacked-QR parity work, this area now needs direct mgcv parity tests instead of assumption. There is no evidence in `tests/` yet that NCV/QNCV has been audited against the same signed-weight corner cases.

### 5. General-family `Sl.setup` parity is still incomplete for several block layouts

Evidence in:

- `nampy/gam/fit/solvers/general_family_solver.py`
- `nampy/gam/fit/solvers/general_newton_solver.py`

Still unsupported:

- nonlinear term blocks whose coefficient slices are not contiguous
- fallback penalty blocks whose coefficient slices are not contiguous
- reparameterized nonlinear general-family `Sl` blocks
- non-reparameterized single-penalty blocks
- non-reparameterized multi-penalty blocks

Upstream target:

- `mgcv/R/fast-REML.r::Sl.setup`
- `mgcv/R/gam.fit4.r::gam.fit5`

Why it matters:

- These are core parity boundaries for general-family smoothing-parameter optimization, not cosmetic differences.
- They are also exactly the sort of block-assembly details that mgcv is sensitive to.

### 6. Exact Gaussian ML/REML backend coverage is still narrower than mgcv’s reachable surface

Evidence:

- `nampy/gam/smoothing_selection/criteria/gaussian.py::criterion_ml_reml_exact()` still rejects models whose penalties couple disconnected support components through null-space penalties.
- The same function intentionally returns `inf` for `fs` / `sz` surfaces so that the code falls back to the dynamic path instead of trusting a known-inexact Laplace implementation.
- `nampy/gam/smoothing_selection/criteria/gaussian_dyn.py` only provides exact dynamic derivatives for `REML` / `LAML`, not `ML`.

Upstream target:

- `mgcv/R/gam.fit3.r`
- `mgcv/R/fast-REML.r`

Priority:

- Lower than the active post-proc and NCV gaps above, because the dynamic path already preserves audited behavior on some surfaces.
- Still real parity debt if the goal is to remove backend-dependent surface restrictions rather than merely match the currently tested slices.

### 7. Formula / spec builder parity is still missing several mgcv-shaped surfaces

Evidence in:

- `nampy/gam/formula/parse.py`
- `nampy/gam/formula/extract.py`
- `nampy/gam/specs/build.py`

Still unsupported or only partially supported:

- `offset(...)` with more than one expression
- multiple `offset(...)` terms per predictor beyond parsing
- interactions involving smooth specials beyond exact parsing
- transformed smooth `by=` expressions are parsed, but not built downstream
- some formula value-expression forms still raise before they can be translated into the equivalent builder state

Why this belongs on the parity backlog:

- These are parser / builder surfaces where mgcv can express richer model specifications than the current builder can materialize.
- Existing tests already show several formula/parser wins, so the remaining holes are now specific and localized rather than broad front-end failure.

### 8. General-family prediction support is still narrower than mgcv in a few explicit branches

Evidence in `nampy/gam/predict/general.py`:

- `type='terms'` is unsupported when the prediction parameterization is wider than the fitted coefficient space
- `type='iterms'` is unsupported for multi-predictor general-family models
- response standard errors still depend on each family providing a suitable `predict(...)` implementation; otherwise the code raises

Related diagnostic caution:

- `nampy/gam/diagnostics/residuals.py` still uses a conservative fallback for some general-family residual cases instead of an obviously complete mgcv-family-specific residual port.

Priority:

- Medium.
- There are no active registered xfails in the prediction/diagnostic parity registry right now, so these are explicit code boundaries that need either implementation or direct targeted tests proving the currently exercised cases are enough.

## What Still Needs Targeted Tests

The highest-value new tests are not broad suite expansions. They should be small, exact parity slices tied to the remaining implementation boundaries.

### 1. Non-Gaussian final-fit post-processing

Add strict final-fit parity slices that specifically check:

- final public `Vc`
- final public `edf2`
- unconditional standard errors sourced from the final fitted object
- covariance-space consistency between fit-space and coefficient-space views

Target families:

- binomial / Poisson PIRLS fits
- negbin
- general-family fits already using `postprocess_general_newton_fit()`

### 2. General-family `t2` final-fit surfaces

The current known gap is explicit, so the tests to add after implementation are also explicit:

- `gaulss_t2_*`
- `gammals_t2_*`
- `gevlss_t2_*`
- `shashlss_t2_*`
- `ziplss_t2_*`

And they should compare at least:

- coefficients if orientation-stable
- `Vp`
- `Vc`
- `edf1`
- `edf2`
- prediction SEs, including unconditional SEs

### 3. PIRLS derivative parity beyond the current Gamma / negbin / theta-free-general coverage

Add direct parity tests against mgcv for:

- exact first derivatives
- exact Hessians
- families with extra parameters
- current-sp `gdi2` branches that are still missing

This should be kept separate from fit-level tests so derivative mismatches are diagnosed at the correct layer.

### 4. NCV / QNCV parity tests, especially after the signed-weight stacked-QR work

Add dedicated tests for:

- joint negbin NCV/QNCV when `theta` is estimated
- supported general-family NCV/QNCV cases with exact score / gradient comparison
- signed-weight or indefinite-kernel cases that force the Cholesky update/downdate paths
- fallback behavior when downdates fail and the MINRES branch is used

This area now needs its own parity coverage instead of relying on the earlier stacked-QR regressions.

### 5. General-family `Sl.setup` block-layout coverage

There should be small tests that isolate:

- contiguous vs non-contiguous term coefficient ownership
- contiguous vs non-contiguous fallback penalty ownership
- nonlinear blocks
- reparameterized nonlinear blocks
- non-reparameterized single- and multi-penalty blocks

Right now these are mostly explicit `NotImplementedError` boundaries in code, not a well-mapped tested surface.

### 6. Formula / spec builder edge cases

If these surfaces are implemented, they need dedicated front-end tests for:

- transformed smooth `by=` expressions
- more than one `offset(...)` term per predictor
- `offset(...)` with richer expressions
- interactions involving smooth specials

These should stay as formula/build tests, not full optimizer tests.

### 7. General-family prediction branches

If prediction support is widened, add targeted tests for:

- `type='terms'` where the prediction basis is wider than the fitted basis
- `type='iterms'` for multi-predictor general families
- response SEs for families that currently lack a `predict(...)` SE implementation
- residual types that currently fall back conservatively

## Lower-Priority Audit Holes

These are not the clearest active failures, but they still deserve explicit confirmation.

- Confirm that the dynamic Gaussian backend is sufficient on all currently allowed `fs` / `sz` ML/REML surfaces, or finish the exact path.
- Confirm whether any currently untested general-family prediction or residual branches are genuinely mgcv-parity complete versus merely unexercised.
- Confirm whether any remaining `advanced/select/by/tensor` general-family post-proc surfaces besides `t2_` still need explicit tagging once `t2_` is fixed.

## Things That Should Not Be Treated As Remaining Parity Work

These are intentionally unsupported because upstream mgcv also rejects them.

- random-effect smooths with `id=`
  - upstream: `mgcv/R/smooth.r` stops with `"random effects don't work with ids."`
  - local evidence: `tests/parity/test_mgcv_remaining_gap_xfails.py`
- data-aware `.` shorthand in formula-list / multi-predictor models
  - `nampy/gam/specs/build.py` already documents that upstream `mgcv::gam(list(...), data=...)` rejects this too
- full-rank shrinkage bases inside `bs="fs"`
  - `tests/smooths/test_mgcv_raw_constructor_unsupported.py` explicitly treats this as upstream-aligned unsupported behavior
- multiply penalized base smooths inside `bs="fs"` or `bs="sz"`
  - upstream `mgcv/R/smooth.r::smooth.construct.fs.smooth.spec` and `smooth.construct.sz.smooth.spec` both stop on multiply penalized bases
  - current NAMpy restrictions here are parity-preserving, not backlog

## Practical Next Order

If the goal is to reduce the real mgcv parity backlog rather than expand code surface area indiscriminately, the current order should be:

1. Make non-Gaussian final-fit `Vc` / `edf2` public and parity-correct.
2. Close general-family `t2` final-fit post-proc parity.
3. Add direct NCV/QNCV parity coverage, especially on signed-weight and joint-theta paths.
4. Expand exact PIRLS derivative coverage where `dispatch.py` currently hard-stops.
5. Finish the missing general-family `Sl.setup` block shapes.
6. Only then widen formula / prediction branches that are still explicitly unsupported.
