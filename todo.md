# Live TODO

This file tracks only current, actionable backlog.

Do not keep resolved work or historical notes here. Active strict parity xfails
live in `xfails.md`; explicit unsupported surfaces should be covered by tests
rather than described here forever.

## P0 Active Parity Failures

- General-family `t2` post-fit / final-fit parity still has `10` live xfails.
Source of truth: `xfails.md`
Coverage: `tests/optimization/test_mgcv_postprocessing_final_fit_parity.py`
Cases: `gaulss|gammals|gevlss|shashlss|ziplss` × `t2_full_false|t2_full_true`
Likely owners:
`nampy/gam/smooths/tensor/t2.py`
general-family final-fit / post-processing
Acceptance:
`pytest tests/optimization/test_mgcv_postprocessing_final_fit_parity.py -k "t2_full" -v`

## P1 Implementation Gaps

- Tensor `gp` marginals with `xt["max.knots"]` remain explicitly unsupported.
Current guard:
`tests/regressions/test_gam_mgcv_patch_regressions.py::test_tensor_gp_max_knots_xt_is_explicitly_unsupported`
Related parity surface:
`tests/smooths/test_mgcv_raw_constructor_parity.py`
- General-family IRLS/PIRLS backend is still unsupported; the live path is the
dedicated general-family solver instead.
Owner:
`nampy/gam/fit/solvers/irls_core.py`
- NCV/QNCV coverage is still restricted.
Current status:
general families limited to `gaulss`, `gammals`, `gevlss`, `shashlss`,
`ziplss`; extended families limited to `negbin`.
Owner:
`nampy/gam/smoothing_selection/criteria/ncv.py`
- EFS smoothing optimization remains restricted to `REML`/`LAML` and does not
support general families.
Owner:
`nampy/gam/smoothing_selection/optimize/efs_mgcv.py`
- Exact dynamic Gaussian derivative support remains `REML`/`LAML` only.
Owner:
`nampy/gam/smoothing_selection/criteria/gaussian_dyn.py`
- Outer smoothing driver still has explicit unsupported method/optimizer  
branches where strict mgcv-parity derivative paths are missing.  
Owner:  
`nampy/gam/smoothing_selection/optimize/driver.py`  
Existing targeted coverage:  
`tests/optimization/test_mgcv_outer_optimization_parity.py`  
`tests/optimization/test_gam_owner_routing_objective_contracts.py`
