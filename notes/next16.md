Exact Now

- Strict-path code mostly fail-closed, not heuristic. Finite-difference / local rescue fallbacks removed for parity-sensitive outer optimization in nampy/gam/fit/solvers/general_fit5.py:225, nampy/gam/smoothing_selection/
criteria/dispatch.py:147, nampy/gam/smoothing_selection/optimize/driver.py:269.
- Constructor-level smoothCon parity is exact for big chunk of surface: te, ti, fs, sz, mrf, re, cc, ps, plus t2 marginal nat.param / block order. Evidence in tests/test_mgcv_smoothcon_parity.py:91.
- `t2` prefit object-level exactness was confirmed for requested cases (`ps, ps`, `tp, cr, full=True`, `tp, cr, full=True, ord=1`, `select=True`, `cc, cc`, `ts, cr`) in smoothCon-based checks. See tests/test_mgcv_smoothcon_parity.py.
- End-to-end Gaussian fixed-sp parity is exact for audited cases: plain cr, re, te, ti, t2, cc, ps, and several random-effect / coefficient-space checks. Evidence in tests/_mgcv_snapshot_parity_shared.py:730.
- pc= exact surfaces exist for cr, cs, factor-by cr, multivariate tp, gp, ts at fixed sp. Evidence in tests/test_mgcv_pc_id_parity.py:133.
- Linked id= exact surface exists only for pooled 1D cubic s() terms at fixed sp, including incompatible-k first-term convention and three-term sharing. Evidence in tests/test_mgcv_pc_id_parity.py:550.
- Gaussian cyclic cubic factor-by and linked-id REML surfaces now have targeted mgcv parity coverage: factor-by matches to machine precision and linked shared-basis REML stays tight at the optimizer endpoint. Automatic Gaussian ML for these shapes remains explicitly unsupported until exact ML derivatives/Hessians exist.
- Gaussian prediction/output exact surfaces exist for audited cases: newdata link, SE, lpmatrix, fixed-sp offset predictions. Evidence in tests/test_mgcv_output_parity.py:181.
- Gaussian output exactness for factor-smooth `fs` terms is now complete at the prefit/fit level in target output parity checks (including `with_se`). Evidence in tests/test_mgcv_output_parity.py:327.
- Outer Newton score history is exact for audited Gaussian REML case, tight to 1e-6 for audited binomial/poisson cases. Evidence in tests/test_mgcv_newton_exact_parity.py:82.
- GP and MRF constructor parity at the object level before optimization is fully aligned (including low-rank `mrf`, `k < n_areas`), with exact basis/penalty checks passing. Evidence in tests/test_mgcv_smoothcon_parity.py and tests/_mgcv_snapshot_parity_shared.py.
- `gaussian_t2_full_false` is no longer an REML-level triage case in requested parity; it passes targeted `gaussian_t2_full_false` checks (`tests/test_mgcv_parity_failing_and_warnings.py`).

Implemented But Not Strict

- Whole Gaussian parity surface is broad and serious: cr, cc, ps, gp, tp, ts, te, ti, t2, re, fs, sz, mrf, weights, offsets, concurvity, anova, trace, k_check. Evidence in tests/_mgcv_snapshot_parity_shared.py:729. But not
all exact.
- `sz` remains mixed on full parity after prefit alignment: factor_smooth_sz still fails requested coefficient parity/REML checks (tests/test_mgcv_output_parity.py, test_mgcv_parity_failing_and_warnings.py).
- ps penalty-scaling gap on REML / pc= looks resolved in tracked tests: ps pc surfaces now pass with tight tolerances, including factor-by and numeric-by REML cases (tests/test_mgcv_pc_id_parity.py:382, 398, 492).
- k_check is not strict whole-surface parity. Only k_prime and edf are checked strictly; k_index and p_value are validity-only because RNG/subsample path differs. fs k_check still triage. Evidence in tests/  
test_mgcv_k_check_parity.py:1. [DONE]
- Non-Gaussian single-predictor families are implemented and parity-tested: binomial logit/probit/cloglog, poisson, gamma log/identity/inverse, negbin. Strong parity, not blanket exact. Main matrix in tests/
_mgcv_snapshot_parity_shared.py:1657.
- General-family / GAMLSS path is real gam.fit5 port, not stub: gaulss, gammals, ziplss, gevlss, shashlss. Endpoints, outer derivatives, vcov, prediction, residual/anova surfaces have parity coverage, but tolerances are
looser. Evidence in nampy/gam/fit/solvers/general_fit5.py:1 and tests/test_general_family_mgcv_parity.py:87.
- estimate_theta=True for negbin is partial. Several audited cases now match mgcv, but not full joint surface. Two-smooth theta=0.5 REML still explicitly unsupported in tests/test_mgcv_known_gaps.py:181.

Not Implemented / Explicit Unsupported

- Formula parser is subset only. No ., no term removal via -, no negative offset, no multiple offset(...) per predictor, no **kwargs smooth specs, limited value-expression grammar in nampy/gam/formula/parse.py:81.
- Only smooth specials s, te, ti, t2 and bases cr, cs, cc, ps, tp, ts, gp, mrf, re, fs, sz are wired in nampy/gam/specs/build.py:290.
- Linked id= support is narrow: pooled 1D cubic s() only. Mixed/non-cubic linked groups raise. Random effects with id= raise. See nampy/gam/compiler/linked_basis.py:63 and nampy/gam/smooths/categorical/random_effect.py:128.
- fs / sz still require current narrow subset: exact one factor variable, singly penalized base smooth, some xt combos rejected. See nampy/gam/smooths/categorical/factor_smooth.py:530.
- Automatic Gaussian ML for cyclic cubic shared-basis / factor-by paths is still unsupported because the strict outer optimizer requires exact ML derivatives/Hessians; audited fixed-sp and REML slices now pass targeted parity tests.
- Formula mode still requires pandas.DataFrame; non-numeric offsets unsupported; multiple predictor-specific offsets unsupported; missing non-numeric fit data unsupported. See nampy/gam/data.py and nampy/gam/specs/
build.py:935.
- gam_vcomp(rescale=True) unsupported in nampy/gam/smoothing_selection/postfit.py:75.
- Some general-family surfaces explicitly unsupported. Example: predictive SE need family support in nampy/gam/predict/general.py:172, and shashlss deviance residual / k_check raise in nampy/gam/diagnostics/residuals.py:79
plus tests/test_general_family_mgcv_parity.py:571.
Upstream refs used: mgcv/R/mgcv.r (interpret.gam, predict.gam, concurvity, anova.gam), mgcv/R/smooth.r (smooth.construct.*, nat.param, t2.model.matrix, XZKr), mgcv/R/gam.fit3.r (gam.reparam, gam.fit3, newton), mgcv/R/
gam.fit4.r (gam.fit5, gam.fit5.post.proc), mgcv/R/gamlss.r (gaulss, gammals, ziplss, gevlss, shashlss), mgcv/R/fast-REML.r (ldetS), mgcv/R/plots.r (k.check), mgcv/R/efam.r (nb).
No pytest run in this audit. Report based on source scan plus existing parity suite.
