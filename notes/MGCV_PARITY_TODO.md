# MGCV Parity TODO

Live backlog for reaching full `mgcv` parity in `nampy/gam/`.

Basis for this file:
- repo guidance in `AGENTS.md` and `CLAUDE.md`
- live code scan on 2026-04-14
- current parity trackers in `tests/test_mgcv_known_gaps.py` and `tests/test_mgcv_parity_failing_and_warnings.py`
- current implementation state in `nampy/gam/`

This file intentionally excludes stale items already resolved structurally:
- ~~`gamlss.py` split is already done as `nampy/gam/families/gamlss/`~~
- ~~`gam/design/` deletion plan is stale; current code already uses `nampy/gam/compiler/`~~

## Definition Of Done

Parity complete means all of these hold:
- no `mgcv` behavior in supported surfaces depends on local heuristics, local Rscript fallback, or approximate diagnostics
- supported formulas/smooths/fit paths mirror upstream control flow closely enough that vendored `mgcv` source is recognizable in Python
- `tests/test_mgcv_known_gaps.py` is empty or reduced to explicitly unsupported upstream surfaces
- `tests/test_mgcv_parity_failing_and_warnings.py` is empty
- core snapshot/output/trace/pc-id/general-family parity suites pass without special-case warnings or relaxed tolerances added for NAMpy-specific behavior

## P0: Remove Remaining Non-Parity Core Paths

### 1. Replace local negative-binomial REML endpoint delegation with real Python port

Current evidence:
- `nampy/gam/smoothing_selection/optimize/driver.py:46-135` shells out to `Rscript` and returns `mgcv` endpoint directly
- `nampy/gam/smoothing_selection/optimize/driver.py:1005-1023` hard-requires that external endpoint for `estimate_theta=True`
- tracked gaps remain in `tests/test_mgcv_known_gaps.py` for estimated-theta REML endpoint matching
- parity failures include `negbin_theta_estimation` in `tests/test_mgcv_parity_failing_and_warnings.py`

Upstream reference:
- vendored `mgcv/R/gam.fit4.r`
- functions around `estimate.theta`, EFS / joint outer updates, and `gam.fit5` REML outer loop

TODO:
- port `mgcv` negative-binomial joint outer loop into Python
- remove `_optimize_negbin_reml_with_mgcv`
- keep theta update state and smoothing update state in one canonical optimizer path
- preserve upstream theta parameterization and update order exactly
- stop using repo-local `parity/mgcv_negbin_reml_opt.R` as execution dependency

Acceptance slices:
- `pytest tests/test_mgcv_known_gaps.py -k "negbin_estimated_theta_reml" -v`
- `pytest tests/test_mgcv_parity_failing_and_warnings.py -k "negbin_theta_estimation" -v`
- `pytest tests/test_mgcv_additional_scenarios.py -k "negbin_theta_0p5 or negbin_theta_2p0" -v`

### 2. Finish exact PIRLS derivative parity on canonical `gam.reparam()` state

Current evidence:
- `nampy/gam/smoothing_selection/reparam.py:36-45` still keeps old `ReparamState` mixed-model scaffolding
- `nampy/gam/smoothing_selection/reparam.py:112-163` still exposes `sl_group_indices`, `sl_lambda_vector`, and rank-scaling helpers from old block view
- `nampy/gam/smoothing_selection/reparam.py:746-820` still builds static blockwise `X_fix/Z_rand/sl_blocks`
- `nampy/gam/smoothing_selection/criteria/pirls_deriv.py:1456-1533` still restricts exact gradients/Hessians to fixed-scale families plus Gamma, with structural gates
- `nampy/gam/smoothing_selection/criteria/pirls_deriv.py:1450-1453` says generic `gdi2` current-sp port incomplete

Upstream reference:
- vendored `mgcv/R/gam.fit4.r`
- `gdi1`, `gdi2`, `gdiPK`, `ift1`, `ift2`
- vendored `mgcv/R/mgcv.r`
- `gam.reparam`

TODO:
- make canonical state from `Y/Z/U1/UrS/rp/T/St/Sr/Eb/Mp` sole owner for exact ML/REML derivative code
- remove exact-path dependence on static `ReparamState`, `sl_blocks`, `sl_lambda_vector`, `sl_group_indices`
- complete generic `gdi2` path beyond Gamma and NegBin where upstream supports it
- preserve derivative operand ordering from upstream to avoid trace/Hessian drift
- only keep legacy mixed-model builder if some non-parity path still truly needs it

Acceptance slices:
- `pytest tests/test_mgcv_trace_parity.py -k "reml or ml" -v`
- `pytest tests/test_mgcv_score_gamma_parity.py -v`
- `pytest tests/test_mgcv_snapshot_parity.py -k "gaussian_reml_sig2 or poisson_reml or binomial_reml or negbin_reml or gamma_reml" -v`

### 3. Remove remaining outer-optimizer rescue logic not in upstream `mgcv`

Current evidence:
- `nampy/gam/smoothing_selection/optimize/driver.py:1043-1120` mixes Newton and `L-BFGS-B` fallback passes in both directions
- optimizer result flags such as `lbfgsb_fallback`, `outer_newton_fallback`, `indefinite_hessian_lbfgsb_fallback` are NAMpy-local
- failing/warning triage module exists specifically for optimizer issues: `tests/test_mgcv_parity_failing_and_warnings.py`

Upstream reference:
- vendored `mgcv/R/gam.fit3.r`
- vendored `mgcv/src/magic.c`

TODO:
- separate true upstream optimizer control flow from NAMpy rescue logic
- delete or fully quarantine non-upstream fallback branches from parity-sensitive ML/REML paths
- keep only exact upstream Newton / magic-style behavior for supported criteria
- re-check all optimizer endpoint diagnostics after removal

Acceptance slices:
- `pytest tests/test_mgcv_trace_parity.py -v`
- `pytest tests/test_mgcv_parity_failing_and_warnings.py -k "warning or optimizer or gamma or tensor" -v`

## P1: Fix Known Failing Supported Smooth Surfaces

### 4. Complete `t2` parity, especially `by=` and problematic tensor branches

Current evidence:
- `nampy/gam/smooths/tensor/t2.py:76-79` raises `NotImplementedError` for `by`
- failing parity includes `gaussian_t2_full_false`, `gaussian_t2_ts_cr_reml_matches_mgcv`, and optimized `t2` snapshot cases
- `tests/test_mgcv_known_gaps.py` tracks strict `t2` response parity

Upstream reference:
- vendored `mgcv/R/smooth.r`
- `smooth.construct.t2.smooth.spec`

TODO:
- implement `t2` `by=` semantics exactly as upstream
- audit `full`, `ord`, penalty ordering, null-space constraint handling, and coefficient map ordering
- verify `t2` basis assembly against `smoothCon(..., absorb.cons=TRUE, scale.penalty=TRUE)`
- compare with `te`/`ti` handling to ensure tensor-family consistency

Acceptance slices:
- `pytest tests/test_mgcv_snapshot_parity.py -k "t2" -v`
- `pytest tests/test_mgcv_parity_failing_and_warnings.py -k "gaussian_t2_full_false or gaussian_t2_ts_cr_reml_matches_mgcv" -v`
- `pytest tests/test_mgcv_known_gaps.py::test_strict_t2_fixed_sp_response_parity -v`

### 5. Finish `fs` and `sz` factor-smooth parity

Current evidence:
- `nampy/gam/smooths/categorical/factor_smooth.py:83-86` restricts `xt`
- `nampy/gam/smooths/categorical/factor_smooth.py:115-125` restricts multivariate and extra-`xt` base support
- `nampy/gam/smooths/categorical/factor_smooth.py:557-560` and `803-806` require singly penalized base smooths
- `nampy/gam/specs/build.py:401-440` still carries `fs` factor-by fallback rewrite
- failing parity includes `factor_smooth_sz`, `gaussian_fs_select_reml_matches_mgcv`, FS term SE parity, FS 4-level scenarios, and `k.check` FS parity

Upstream reference:
- vendored `mgcv/R/smooth.r`
- `smooth.construct.fs.smooth.spec`
- `smooth.construct.sz.smooth.spec`
- helper `XZKr`

TODO:
- remove fallback rewrite for `fs` without factor feature; build upstream object flow directly
- support multi-penalty base smooths where upstream allows them
- audit factor-level ordering, contrast transform, null-space penalties, selection penalties, and standard-error propagation
- ensure `fs` / `sz` work with output parity, EDF parity, and `k.check` parity, not only fitted values

Acceptance slices:
- `pytest tests/test_mgcv_snapshot_parity.py -k "gaussian_fs or gaussian_sz" -v`
- `pytest tests/test_mgcv_additional_scenarios.py -k "fs or sz" -v`
- `pytest tests/test_mgcv_output_parity.py -k "terms_all_smooth_types or standard_errors" -v`
- `pytest tests/test_mgcv_k_check_parity.py -v`

### 6. Complete general linked `id=` basis sharing

Current evidence:
- `nampy/gam/compiler/linked_basis.py:62-68` rejects mixed supported/unsupported linked groups
- current linked pooling only applies to 1D cubic `s()` terms via `_eligible_id_pool_term`
- `nampy/gam/smooths/categorical/random_effect.py:130-131` still rejects `id=` for random effects
- parity tests still cover linked `id=` scenarios in `tests/test_mgcv_pc_id_parity.py`

Upstream reference:
- vendored `mgcv/R/mgcv.r`
- `gam.setup`
- `smoothCon`

TODO:
- generalize linked basis setup beyond pooled 1D cubic `s()`
- support `id=` for random effects and other smooth classes where upstream shares setup
- preserve upstream harmonization rules for `k`, knots, and marginal basis setup
- remove any remaining special-case warning behavior not present in upstream

Acceptance slices:
- `pytest tests/test_mgcv_pc_id_parity.py -v`
- `pytest tests/test_mgcv_output_parity.py::test_output_parity_newdata_terms_linked_id -v`

### 7. Audit `pc=` and constrained smooth parity across all supported bases

Current evidence:
- `tests/test_mgcv_pc_id_parity.py` still labels some `pc=` comparisons approximate
- parity failure already tracked for `gp_numeric_by_pc_reml_matches_mgcv`
- `nampy/gam/smooths/smooth_base.py` still limits point constraints to 1D forms

Upstream reference:
- vendored `mgcv/R/smooth.r`
- per-basis `smooth.construct.*`

TODO:
- verify point-constraint application timing against upstream for `s`, `gp`, tensor, and factor-by cases
- expand beyond current 1D-only `pc` machinery where upstream supports more
- ensure `pc=` interacts correctly with `by`, `id`, and null-space constraints

Acceptance slices:
- `pytest tests/test_mgcv_pc_id_parity.py -k "pc" -v`

## P2: Complete Formula / Spec Front-End Parity

### 8. Finish `interpret.gam` / `gam.setup` formula support

Current evidence:
- `nampy/gam/formula/parse.py:79-92` rejects positional `list(...)` args and richer formula value expressions
- `nampy/gam/formula/parse.py:114-120` requires smooth covariates to be bare variable names
- `nampy/gam/formula/parse.py:218-219` rejects `.`
- `nampy/gam/formula/parse.py:241-243` and `270-272` reject term removal with `-`
- `nampy/gam/formula/parse.py:254-257` supports only one offset per predictor
- `nampy/gam/specs/build.py:333-335` rejects vector `fx`
- `nampy/gam/specs/build.py:597-600` limits factor-`by` expansion to `s(...)`
- `nampy/gam/specs/build.py:935-938` still assumes one active linear predictor in offset handling
- `nampy/gam/specs/build.py:972-975` supports numeric offsets only

Upstream reference:
- vendored `mgcv/R/mgcv.r`
- `interpret.gam0`
- `interpret.gam`
- `gam.setup`
- `gam.setup.list`

TODO:
- support full subtractive formula semantics for parametric and smooth terms
- support multi-formula/list syntax used by multi-predictor and general-family fits
- support richer `list(...)`, `c(...)`, and non-bare smooth covariate expressions where upstream does
- align offset handling with upstream multi-predictor semantics
- remove local factor-by expansion limitations that exist only because of current front-end staging

Acceptance slices:
- add narrow parser/setup parity tests first
- `pytest tests/test_mgcv_snapshot_parity.py -k "offset or formula" -v`
- `pytest tests/test_general_family_mgcv_parity.py -v`

## P3: Diagnostics / Postfit / Output Surfaces Still Not Exact

### 9. Port `k.check`, `gam.check`, and residual diagnostics exactly

Current evidence:
- `nampy/gam/diagnostics/k_check.py:86-92` explicitly says approximate
- `nampy/gam/diagnostics/k_check.py:166-177` splits report into `mgcv_comparable` and `nampy_specific`
- `nampy/gam/diagnostics/residuals.py` still has custom branching and currently tracked strict residual gaps in `tests/test_mgcv_known_gaps.py`

Upstream reference:
- vendored `mgcv/R/plots.r` and related diagnostics code
- `k.check`
- `gam.check`
- `residuals.gam`

TODO:
- replace nearest-neighbor and residual logic with direct upstream port where feasible
- port exact `k.check` EDF/reporting path from upstream instead of local EDF stabilization or clipping logic
- match exact residual definitions and scaling across Gaussian, Poisson, Binomial, and supported general families
- remove custom diagnostic packaging from parity-sensitive API or clearly split it from strict parity surface

Acceptance slices:
- `pytest tests/test_mgcv_known_gaps.py -k "residual" -v`
- `pytest tests/test_mgcv_k_check_parity.py -v`

### 10. Finish `sp_vcov`, `gam_vcomp`, and Gaussian post-fit score parity

Current evidence:
- `nampy/gam/smoothing_selection/postfit.py:35-46` only uses strict stored Hessian / `fit_criterion_hessian`; no direct upstream joint Gaussian post-fit Hessian port exists yet
- `nampy/gam/smoothing_selection/postfit.py:68-71` leaves `gam_vcomp(rescale=True)` unsupported rather than using local rescaling heuristics
- `nampy/gam/fit/postprocess/gaussian_smoothness_postprocess.py` is Gaussian-only and documents missing P-REML / NCV / non-Gaussian paths

Upstream reference:
- vendored `mgcv/R/mgcv.r`
- `sp.vcov`
- `gam.vcomp`
- Gaussian post-processing paths around `gam.fit3.post.proc`

TODO:
- port exact post-fit Hessian / covariance path from upstream `mgcv::sp.vcov` and Gaussian post-processing code instead of finite-difference or other local reconstruction
- retain and thread exact upstream penalty rescaling metadata needed for `gam_vcomp(rescale=True)`
- implement full `edge_correct` semantics only from upstream path
- extend or split postprocess paths so reported smoothness scores and derivatives match upstream for all supported families/methods

Acceptance slices:
- add focused parity tests for `sp_vcov` and `gam_vcomp`
- `pytest tests/test_mgcv_score_gamma_parity.py -v`

### 11. Finish prediction / covariance / term SE parity for remaining failing surfaces

Current evidence:
- FS term standard-error parity still fails in `tests/test_mgcv_parity_failing_and_warnings.py`
- output parity suite covers `terms`, `se`, `lpmatrix`, and ANOVA comparisons

Upstream reference:
- vendored `mgcv/R/mgcv.r`
- `predict.gam`
- `vcov.gam`
- `anova.gam`

TODO:
- audit coefficient covariance assembly for term SE extraction
- ensure term naming, centering, and factor-level expansion match `mgcv`
- verify `lpmatrix` column order remains stable after linked-basis / factor-smooth fixes

Acceptance slices:
- `pytest tests/test_mgcv_output_parity.py -v`

## P4: Lower-Level Smooth Constructor Completeness

### 12. Close remaining smooth constructor restrictions that are now explicit errors

Current evidence:
- `nampy/gam/smooths/categorical/random_effect.py:130-131` rejects `re` with `id=`
- `nampy/gam/smooths/tensor/t2.py:76-79` rejects `t2` `by=`
- `nampy/gam/specs/build.py:289-294` still rejects unsupported `s()` bases and specials outright
- `nampy/gam/specs/build.py:786-789` rejects unknown smooth kwargs rather than mirroring upstream support

Upstream reference:
- vendored `mgcv/R/smooth.r`
- per-basis `smooth.construct.*` routines

TODO:
- inventory supported upstream bases versus current basis registry
- for each exposed basis, either finish exact parity or remove public exposure
- avoid broadening support ahead of parity-critical workstreams above

Acceptance slices:
- one exact parity test per newly completed constructor

## Recommended Execution Order

1. Port NegBin REML/theta optimizer path into Python and delete Rscript dependency.
2. Finish canonical `gam.reparam` derivative ownership and remove exact-path static mixed-model scaffolding.
3. Strip non-upstream optimizer rescue logic from parity-sensitive ML/REML flows.
4. Fix `t2` parity.
5. Fix `fs` / `sz` parity and term-SE parity.
6. Generalize linked `id=` and `pc=` parity.
7. Complete formula/setup parity.
8. Port diagnostics/postfit parity surfaces.
9. Sweep remaining explicit constructor restrictions case by case.

## Tracking Tests To Burn Down

Primary red/triage files:
- `tests/test_mgcv_known_gaps.py`
- `tests/test_mgcv_parity_failing_and_warnings.py`

Primary parity gates after each workstream:
- `tests/test_mgcv_snapshot_parity.py`
- `tests/test_mgcv_output_parity.py`
- `tests/test_mgcv_trace_parity.py`
- `tests/test_mgcv_pc_id_parity.py`
- `tests/test_general_family_mgcv_parity.py`
- `tests/test_mgcv_additional_scenarios.py`

## Ground Rules While Executing This Backlog

- use vendored upstream `mgcv` sources as behavioral spec before changing parity-sensitive code
- prefer direct control-flow ports over algebraic rewrites
- keep penalty ordering, constraint absorption, and coefficient slice ordering unchanged unless upstream requires change
- validate with smallest targeted pytest slice first, then only broaden if that workstream crosses subsystem boundaries
