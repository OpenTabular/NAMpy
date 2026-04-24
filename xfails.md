# Xfail Registry

This file tracks only active, unresolved `mgcv` parity xfails.

Resolved items should be removed from this file in the same change that removes
their xfail. Historical notes belong in git history, not in the live registry.

Current tracked xfails: `10`

Breakdown:

- `10` in [tests/optimization/test_mgcv_postprocessing_final_fit_parity.py](/home/ad32/projects/package/NAMpy/tests/optimization/test_mgcv_postprocessing_final_fit_parity.py)

## Rules

- Do not keep resolved behavior in this registry.
- Do not loosen assertions to retire an xfail.
- If `mgcv` rejects a surface, remove it from parity-xfail tracking and cover it with a strict unsupported-surface test instead.
- Prioritize behavior-affecting gaps over raw-only representation drift.
- For parity-sensitive work, mirror vendored `mgcv` control flow directly rather than replacing it with a fresh derivation.

## Priority Order

1. Close the `10` general-family post-fit/final-fit xfails.

## 1. General-Family Post-Fit / Final-Fit Xfails (`10`)

These cases are currently tracked through the general-family known-gap tags in
[tests/optimization/test_mgcv_postprocessing_final_fit_parity.py](/home/ad32/projects/package/NAMpy/tests/optimization/test_mgcv_postprocessing_final_fit_parity.py):
`t2_`.

### P1. General-family `t2` final-fit parity (`10`)

Cases:

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

Current gap:

- tensor-anova general-family post-fit parity still diverges

Primary owners:

- [nampy/gam/smooths/tensor/t2.py](/home/ad32/projects/package/NAMpy/nampy/gam/smooths/tensor/t2.py)
- general-family final-fit/post-processing

Solve focus:

1. Re-check general-family preoptimization parity on `t2`.
2. Then repair final-fit covariance and `edf2` drift.

Acceptance:

- `pytest tests/optimization/test_mgcv_postprocessing_final_fit_parity.py -k "t2_full" -v`

## 4. Maintenance Policy

- When a case turns green, remove it from the test xfail source and from this file in the same change.
- Do not add resolved writeups here; keep this file short and live.
- If a bucket splits into behavior-affecting versus representation-only work, only the behavior-affecting part belongs here.
