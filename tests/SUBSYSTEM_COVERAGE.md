# GAM Subsystem Coverage

This matrix is the owner-level parity map for `nampy/gam/`.

Use it to answer two questions quickly:
- which test should fail first when a given subsystem drifts from `mgcv`?
- where should a new parity regression be added without duplicating broader snapshots?

End-to-end parity suites remain the broad backstop. The files listed here are
the primary owner-level localization points. Unresolved planning is kept in
local development notes rather than duplicated here.

## Coverage Matrix

| Subsystem | Primary owner(s) | Primary tests | Notes |
| --- | --- | --- | --- |
| Formula/spec parsing | `nampy/gam/formula/`, `nampy/gam/specs/` | `tests/parity/test_mgcv_formula_parse_parity.py` | Direct formula parity vs `mgcv`. |
| Smooth constructors / raw basis owners | `nampy/gam/smooths/`, `nampy/gam/splines/` | `tests/smooths/test_mgcv_raw_constructor_parity.py`, `tests/smooths/test_mgcv_smoothcon_parity.py`, `tests/parity/test_gam_spec_build_owner_contracts.py`, `tests/parity/test_mgcv_cp_combinations_parity.py`, `tests/parity/test_mgcv_bs_combinations_parity.py`, `tests/parity/test_mgcv_ds_combinations_parity.py`, `tests/parity/test_mgcv_gp_combinations_parity.py`, `tests/parity/test_mgcv_mrf_combinations_parity.py`, `tests/parity/test_mgcv_sos_combinations_parity.py` | Basis, penalty, constructor, and smoothCon surfaces, including cyclic P-splines (`cp`), integrated-derivative B-splines (`bs`), multivariate Duchon splines (`ds`), five-family stationary/nonstationary Gaussian-process smooths (`gp`), graph/polygon/direct-penalty Markov random fields (`mrf`), and all seven spherical-spline kernels (`sos`) across prediction, selection, linked bases, tensor/factor-smooth combinations, and explicit malformed-upstream boundaries. |
| `pc=` / linked `id=` routing | smooth metadata + linked basis owners | `tests/smooths/test_mgcv_pc_id_parity.py` | Localizes shared-smoothing and point-constraint issues, including `te`/`ti` point constraints. |
| Design / pre-fit assembly | `nampy/gam/compiler/`, `nampy/gam/fit/penalized_system.py` | `tests/optimization/test_mgcv_gam_setup_assembly_parity.py`, `tests/optimization/test_mgcv_preoptimization_blocks_parity.py`, `tests/optimization/test_mgcv_preoptimization_reparam_parity.py` | Setup, blocks, and reparameterization parity, including one global shared-component block with overlapping linear-predictor indices. |
| Term wrapping / by-variable / offset routing | predictor wrapping + compiled term owners | `tests/optimization/test_gam_term_wrapping_owner_contracts.py` | Localizes wrapped predictor blocks, offset routing, and general-family block ownership before broader prediction parity. |
| Side conditions / identifiability | `nampy/gam/constraints/` | `tests/optimization/test_mgcv_gam_side_parity.py` | Nested side conditions and current case matrix. |
| Fit backend dispatch | `nampy/gam/fit/backends.py`, `nampy/gam/fit/design_setup.py` | `tests/optimization/test_gam_fit_backend_owner_contracts.py`, `tests/optimization/test_gam_owner_routing_objective_contracts.py` | Backend selection, solver dispatch, compilation-time capability guards, and fixed-smoothing wrapper forwarding. |
| Gaussian covariance / post-fit assembly | `nampy/gam/fit/covariance.py`, `nampy/gam/fit/solvers/gaussian_exact.py` | `tests/optimization/test_gam_covariance_owner_contracts.py`, `tests/optimization/test_mgcv_postprocessing_final_fit_parity.py`, `tests/parity/test_mgcv_under_tested_supported_combinations.py` | Helper contracts plus exact/stacked-QR parity backstops; Gaussian-inverse REML directly covers logLik/AIC/BIC, covariance selection, residuals, summary/ANOVA, and k-check. |
| ML/REML backend routing | `nampy/gam/fit/selection/criteria/ml_reml.py` | `tests/optimization/test_gam_owner_routing_objective_contracts.py`, `tests/optimization/test_mgcv_gaussian_backend_selection.py` | Exact vs dynamic vs PIRLS vs general-family selection. |
| Objective wrappers / optimizer wiring | `nampy/gam/fit/selection/optimize/objectives.py`, `.../driver.py` | `tests/optimization/test_gam_owner_routing_objective_contracts.py`, `tests/optimization/test_mgcv_parametric_only_parity.py`, `tests/optimization/test_mgcv_outer_optimization_parity.py`, `tests/optimization/test_mgcv_optimization_lifecycle_parity.py` | Owner contracts first, direct empty-smoothing-vector parity, mgcv trace parity, then lifecycle parity. |
| Postfit smoothing diagnostics | `nampy/gam/fit/selection/postfit.py` | `tests/optimization/test_gam_postfit_owner_contracts.py`, `tests/optimization/test_mgcv_vcomp_parity.py`, `tests/optimization/test_mgcv_sp_vcov_stage_parity.py` | Endpoint diagnostics, Hessian sourcing, smoothing covariance surfaces, and stage-local `sp.vcov` / unconditional-covariance checkpoints. |
| General-family fixed-smoothing / postprocess | `nampy/gam/fit/solvers/general_family/fixed_smoothing.py`, `.../newton.py` | `tests/families/test_gam_general_family_owner_contracts.py`, `tests/optimization/test_mgcv_fixed_inner_fit_parity.py`, `tests/optimization/test_mgcv_general_family_preoptimization_parity.py`, `tests/families/test_general_family_mgcv_parity.py`, `tests/parity/test_mgcv_snapshot_extended_matrix.py`, `tests/parity/test_mgcv_under_tested_supported_combinations.py` | Owner precedence plus mgcv `gam.fit5` parity; reparameterized and original-coordinate singleton/multi-penalty `Sl` blocks are covered through setup, derivatives, roots/totals, and full fits. A four-smooth/two-predictor gaulss case covers wrapped blocks, `sp.vcov`, inference, and diagnostics. Structured `re`, `fs`, and linked-`sz` cases include fixed and optimized `fs` behavior and an `fs` block in linear predictor two. |
| Diagnostics owners | `nampy/gam/diagnostics/residuals.py`, `concurvity.py`, `summary.py`, `plots.py` | `tests/diagnostics/test_gam_diagnostics_owner_contracts.py`, `tests/diagnostics/test_gam_plot_and_public_api_contracts.py`, `tests/parity/test_mgcv_secondary_diagnostics_parity.py` | Owner-level residual/summary/plot contracts plus direct secondary-diagnostics parity. |
| Prediction / inference / diagnostics | `nampy/gam/predict/`, `nampy/gam/inference/`, `nampy/gam/diagnostics/` | `tests/parity/test_mgcv_output_parity.py`, `tests/parity/test_mgcv_prediction_arguments_parity.py`, `tests/parity/test_mgcv_prediction_inference_diagnostics_parity.py`, `tests/parity/test_mgcv_general_family_lpmatrix_stage_parity.py`, `tests/parity/test_mgcv_general_family_prediction_stage_parity.py`, `tests/parity/test_mgcv_inference_stage_parity.py`, `tests/diagnostics/test_mgcv_general_family_secondary_diagnostics_parity.py` | Public-surface parity plus direct `block.size`, `newdata.guaranteed`, `na.action`, `unconditional`, `iterms.type`, and stage-local general-family checkpoints. |
| Parity snapshot / trace tooling | `nampy/gam/parity/` | `tests/parity/test_gam_parity_owner_contracts.py`, `tests/parity/test_gam_results_api_stage_owner_contracts.py`, `tests/optimization/test_mgcv_score_hist_trace_parity.py`, `tests/optimization/test_mgcv_outer_optimization_parity.py`, `tests/optimization/test_mgcv_optimization_lifecycle_parity.py`, `tests/optimization/test_mgcv_inner_trace_parity.py`, `tests/optimization/test_mgcv_joint_branch_trace_parity.py` | Localizes serialization, criterion-view logic, outer-object trace schemas, lifecycle branch parity, and inner/joint trace branches. |
| Explicit regressions | cross-cutting | `tests/regressions/test_gam_mgcv_patch_regressions.py` | Add only for past bugs or easy-to-break branch contracts. |

## Seven-Stage Pipeline Gate and Combination Coverage

`tests/parity/test_gam_seven_stage_pipeline_contracts.py` is the direct release
gate for the repository's seven-stage GAM pipeline. It deliberately
uses one fixed-SP Gaussian formula with a parametric term, cubic smooth, and
formula offset so every transition is checked without conflating the stage
contract with outer-optimizer behavior. Stage 7 also compares newdata response
prediction with a committed `mgcv` reference fixture.

The vertical gate is complemented by
`tests/parity/test_gam_pipeline_combination_matrix.py`. The combination matrix
closes the former seven-stage under-tested backlog.

| Pipeline stage | Combination coverage now owned |
| --- | --- |
| 1. Formula parsing and canonical specs | Formula lists/shared components, transformed covariates, supported numeric/factor interactions, factor-by smooths, formula and fit offsets, intercept policies, weights, knots, `min_sp`, `drop_intercept`, and tensor-`m` warning/fallback behavior; the combined interaction recipe is rebuilt on newdata and compared with a committed `mgcv` reference fixture. |
| 2. Runtime terms and low-level bases | Train/newdata pairing for every supported basis (`bs/cr/cs/cc/cp/ds/gp/mrf/ps/sos/tp/ts/re/fs/sz/te/ti`), including univariate, multivariate Duchon/GP, categorical MRF, spherical SOS, and structured boundary response/SE parity. Row-permutation contracts cover linked and identified-FS terms, PS/TP/TS bases, `te`/`ti`, SZ, and factor-by smooths. Non-unique constructor representations use column-space projectors and penalized response operators instead of arbitrary coefficient orientation. |
| 3. Constructed/wrapped terms | Numeric-by plus linked `id=`, factor-by plus linked `id=`, tensor-by, and mixed fixed/free/select penalty ownership, numerical coefficient-map composition, and fitted response/SE parity. |
| 4. Predictor/model compilation | Multi-predictor layouts with unequal intercept and offset policies, overlapping shared-component coefficient indices, and three-term linked, fixed/free, select, and rank-deficient assembly; supported `gaulss` and `gammals` layouts are also fitted and compared with `mgcv`. |
| 5. Side conditions and identifiability | Repeated, nested, reverse-formula-order, tensor/main-effect, three-way, both identified near-rank regimes, zero-width, no-intercept, ordered factor-by, linked, general-family, two-predictor, SZ, and exempt random/factor smooth cases. Deletion rank and behavior replace raw pivot-column identity where QR/eigen choices are non-unique. |
| 6. Fitting and smoothing selection | The combination matrix covers Gaussian, binomial, Poisson, Gamma, negative-binomial, `gaulss`, and `gammals` fits across GCV/ML/REML and supported optimizers; weights, offsets, select penalties, boundary SPs, zero-weight rejection, and near-separated binomial behavior are exercised. Direct parity also covers Gaussian and Poisson REML formulas with an empty smoothing-parameter vector. |
| 7. Prediction, inference, and diagnostics | Term filters plus direct conditional/unconditional SE; exact row blocking, default NA pass/restoration plus omit/exclude/fail, guaranteed complete-newdata constructor skipping, and `iterms.type=2`; unconditional term inference for random effects, tensors, linked terms, FS, SZ, factor-by, and `ti`; shared-component link/response/SE/lpmatrix prediction and per-target decomposition; structured newdata and unseen-level rejection; and diagnostics/snapshot round trips. Gaussian-inverse REML directly covers post-fit likelihood criteria, covariance modes, residuals, summary/ANOVA, and k-check. Persistence covers shared-component layouts, random effects, `te`, `ti`, linked terms, FS, SZ, factor-by, `gaulss`, `gammals`, and binomial/Poisson/Gamma P-IRLS. Three-model ANOVA covers Gaussian, binomial, Poisson, Gamma, `gaulss`, and `gammals`. |

This is complete coverage of the declared supported and explicitly unsupported
surface, not an assertion that every Cartesian product of valid options is run.
When a new public feature is added, register its owner in the collection contract
and add the smallest behavior/parity case that crosses its affected stages.

## Current qualification

There is no live expected GAM failure declared by the targeted owner/parity
slices. This does not assert exhaustive Cartesian coverage or a current clean
full-suite run. The supported boundary is defined by public behavior, explicit
unsupported-input errors, and the committed owner/parity tests; unresolved
planning stays in local development notes.

## Sweep Rules

- Prefer one owner-level test per branch contract before adding more snapshot cases.
- Reuse existing `tests/mgcv_parity_utils.py` harnesses before creating new R helpers.
- If a failure is broad and hard to localize, add an owner test here first, then expand scenarios only if needed.
- Keep this file updated when a new parity-sensitive owner gets its first direct test file.
