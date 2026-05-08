# GAM Subsystem Coverage

This matrix is the owner-level parity map for `nampy/gam/`.

Use it to answer two questions quickly:
- which test should fail first when a given subsystem drifts from `mgcv`?
- where should a new parity regression be added without duplicating broader snapshots?

End-to-end parity suites remain the broad backstop. The files listed here are the
primary owner-level localization points, and the stage-local backlog below keeps
the remaining parity planning in this single document.

## Coverage Matrix

| Subsystem | Primary owner(s) | Primary tests | Notes |
| --- | --- | --- | --- |
| Formula/spec parsing | `nampy/gam/formula/`, `nampy/gam/specs/` | `tests/parity/test_mgcv_formula_parse_parity.py` | Direct formula parity vs `mgcv`. |
| Smooth constructors / raw basis owners | `nampy/gam/smooths/`, `nampy/splines/` | `tests/smooths/test_mgcv_raw_constructor_parity.py`, `tests/smooths/test_mgcv_smoothcon_parity.py` | Basis, penalty, constructor, and smoothCon surfaces. |
| `pc=` / linked `id=` routing | smooth metadata + linked basis owners | `tests/smooths/test_mgcv_pc_id_parity.py` | Localizes shared-smoothing and point-constraint issues. |
| Design / pre-fit assembly | `nampy/gam/compiler/`, `nampy/gam/fit/penalized_system.py` | `tests/optimization/test_mgcv_gam_setup_assembly_parity.py`, `tests/optimization/test_mgcv_preoptimization_blocks_parity.py`, `tests/optimization/test_mgcv_preoptimization_reparam_parity.py` | Setup, blocks, and reparameterization parity. |
| Term wrapping / by-variable / offset routing | predictor wrapping + compiled term owners | `tests/optimization/test_gam_term_wrapping_owner_contracts.py` | Localizes wrapped predictor blocks, offset routing, and general-family block ownership before broader prediction parity. |
| Side conditions / identifiability | `nampy/gam/constraints/` | `tests/optimization/test_mgcv_gam_side_parity.py` | Nested side conditions and current case matrix. |
| Fit backend dispatch | `nampy/gam/fit/backends.py`, `nampy/gam/fit/solve_ops.py` | `tests/optimization/test_gam_fit_backend_owner_contracts.py`, `tests/optimization/test_gam_owner_routing_objective_contracts.py` | Backend selection, solver dispatch, and fixed-smoothing wrapper forwarding. |
| Gaussian covariance / post-fit assembly | `nampy/gam/fit/covariance.py`, `nampy/gam/fit/solvers/gaussian_exact.py` | `tests/optimization/test_gam_covariance_owner_contracts.py`, `tests/optimization/test_mgcv_postprocessing_final_fit_parity.py` | Helper contracts plus exact/stacked-QR parity backstops. |
| ML/REML backend routing | `nampy/gam/smoothing_selection/criteria/ml_reml.py` | `tests/optimization/test_gam_owner_routing_objective_contracts.py`, `tests/optimization/test_mgcv_gaussian_backend_selection.py` | Exact vs dynamic vs PIRLS vs general-family selection. |
| Objective wrappers / optimizer wiring | `nampy/gam/smoothing_selection/optimize/objectives.py`, `.../driver.py` | `tests/optimization/test_gam_owner_routing_objective_contracts.py`, `tests/optimization/test_mgcv_outer_optimization_parity.py`, `tests/optimization/test_mgcv_optimization_lifecycle_parity.py` | Owner contracts first, mgcv trace parity second, lifecycle parity third. |
| Postfit smoothing diagnostics | `nampy/gam/smoothing_selection/postfit.py` | `tests/optimization/test_gam_postfit_owner_contracts.py`, `tests/optimization/test_mgcv_vcomp_parity.py`, `tests/optimization/test_mgcv_sp_vcov_stage_parity.py` | Endpoint diagnostics, Hessian sourcing, smoothing covariance surfaces, and stage-local `sp.vcov` / unconditional-covariance checkpoints. |
| General-family fixed-smoothing / postprocess | `nampy/gam/fit/solvers/general_family_solver.py`, `.../general_newton_solver.py` | `tests/families/test_gam_general_family_owner_contracts.py`, `tests/optimization/test_mgcv_fixed_inner_fit_parity.py`, `tests/families/test_general_family_mgcv_parity.py` | Owner precedence tests plus mgcv fit5 parity. |
| Diagnostics owners | `nampy/gam/diagnostics/residuals.py`, `concurvity.py`, `summary.py`, `plots.py` | `tests/diagnostics/test_gam_diagnostics_owner_contracts.py`, `tests/diagnostics/test_gam_plot_and_public_api_contracts.py`, `tests/parity/test_mgcv_secondary_diagnostics_parity.py` | Owner-level residual/summary/plot contracts plus direct secondary-diagnostics parity. |
| Prediction / inference / diagnostics | `nampy/gam/predict/`, `nampy/gam/inference/`, `nampy/gam/diagnostics/` | `tests/parity/test_mgcv_output_parity.py`, `tests/parity/test_mgcv_prediction_inference_diagnostics_parity.py`, `tests/parity/test_mgcv_general_family_lpmatrix_stage_parity.py`, `tests/parity/test_mgcv_general_family_prediction_stage_parity.py`, `tests/parity/test_mgcv_inference_stage_parity.py`, `tests/diagnostics/test_mgcv_general_family_secondary_diagnostics_parity.py` | Public-surface parity plus direct stage-local general-family checkpoints. |
| Parity snapshot / trace tooling | `nampy/gam/parity/` | `tests/parity/test_gam_parity_owner_contracts.py`, `tests/parity/test_gam_results_api_stage_owner_contracts.py`, `tests/optimization/test_mgcv_score_hist_trace_parity.py`, `tests/optimization/test_mgcv_outer_optimization_parity.py`, `tests/optimization/test_mgcv_optimization_lifecycle_parity.py`, `tests/optimization/test_mgcv_inner_trace_parity.py`, `tests/optimization/test_mgcv_joint_branch_trace_parity.py` | Localizes serialization, criterion-view logic, outer-object trace schemas, lifecycle branch parity, and inner/joint trace branches. |
| Explicit regressions | cross-cutting | `tests/regressions/test_gam_mgcv_patch_regressions.py` | Add only for past bugs or easy-to-break branch contracts. |

## Stage-Local Backlog

The following stage-local files exist now. Because this pass intentionally
skipped `pytest`, treat newly added slices as localized checkpoints but not
stage-status promotions.

| Stage | Direct file | Remaining or still-unvalidated backlog |
| --- | --- | --- |
| Stage 6-7 tensor marginal / tensor prediction | `tests/smooths/test_mgcv_te_stage_parity.py`, `tests/smooths/test_mgcv_ti_stage_parity.py` | mixed-basis term-level checkpoints are local now; strict raw-stage `ti_2d_cs_cs`, `ti_2d_cs_ps`, and `ti_2d_ps_cs` remain explicit known gaps |
| Stage 8 term wrapping | `tests/optimization/test_gam_term_wrapping_owner_contracts.py` | ordinary wrapped-block parity remains localized here; multi-smooth general-family wrapped-block parity is no longer a supported surface |
| Stage 14 optimizer trace | `tests/optimization/test_mgcv_joint_branch_trace_parity.py` | `gamma_joint_scale_trace` and `negbin_joint_theta_trace_labels` remain the explicit joint-branch known gaps |
| Stage 15 post-fit covariance / `sp.vcov` | `tests/optimization/test_mgcv_sp_vcov_stage_parity.py` | ordinary public-parameterization `sp.vcov`, one-standard-error, and `gam.vcomp(rescale=False)` slices are direct; multi-smooth general-family `sp.vcov` parity is no longer a supported surface |
| Stage 16 general-family `lpmatrix` | `tests/parity/test_mgcv_general_family_lpmatrix_stage_parity.py` | factor-level and NA-newdata behavior are now local |
| Stage 17 public prediction | `tests/parity/test_mgcv_general_family_prediction_stage_parity.py` | broader family/method and linked-`id=` coverage is now local |
| Stage 18 inference | `tests/parity/test_mgcv_inference_stage_parity.py` | ordinary model-comparison and general-family single-model slices are local; multi-smooth general-family inference parity is no longer a supported surface |
| Stage 19 diagnostics / summary | `tests/diagnostics/test_mgcv_general_family_secondary_diagnostics_parity.py` | summary scalars and extra supported residual branches are now local; multi-smooth general-family diagnostics parity is no longer a supported surface |
| Stage 20 results API / parity tooling | `tests/parity/test_gam_results_api_stage_owner_contracts.py` | fit-result and optimizer trace schema ownership is now local, pending targeted pytest validation |
| Stage 21 unsupported / guarded branches | `tests/optimization/test_gam_unsupported_branch_guards.py` | formula-list, multi-smooth general-family, fs-shrinkage, and wider-than-fit `terms` guards are now localized; keep adding new public-surface guards here instead of downstream parity files |

## Sweep Rules

- Prefer one owner-level test per branch contract before adding more snapshot cases.
- Reuse existing `tests/mgcv_parity_utils.py` harnesses before creating new R helpers.
- If a failure is broad and hard to localize, add an owner test here first, then expand scenarios only if needed.
- Keep this file updated when a new parity-sensitive owner gets its first direct test file.
