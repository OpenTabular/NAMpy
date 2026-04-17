High: Formula/front-end parity still subset, not full mgcv. Parser only
    accepts bare variable names inside smooths, one offset(...), no .
    expansion, no subtracting terms/smooths in RHS. See nampy/gam/formula/
    parse.py:114, nampy/gam/formula/parse.py:218, nampy/gam/formula/
    parse.py:241. Builder supports s() bases cr/cs/cc/ps/tp/ts/gp/mrf/re/fs/sz
    plus te/ti/t2, nothing beyond that. See nampy/gam/specs/build.py:222,
    nampy/gam/specs/build.py:283.

- Medium: Fit-layer separation weak. nampy/gam/fit/model_ops.py:35 mixes
backend capability checks, smoothing-param coercion, design compilation,
penalty assembly, solver wrappers, criterion wrappers, fit-result building.
One file sits across compiler, selection, engine, results. Hard for parity
work because small mgcv-port change touches god-module boundary.
- Medium: Term state duplicated in two representations. nampy/gam/compiler/
construct.py:25 ConstructedSmooth and nampy/gam/compiler/structures.py:48
CompiledTerm both own basis/predict callback/coefficient maps/feature
metadata/validation. Two sources of truth for constraint and prediction
transforms. Drift risk high for parity-sensitive bugs.
- Medium: General-family prediction path redundant. nampy/gam/predict/
general.py:41, nampy/gam/predict/general.py:66, nampy/gam/predict/
general.py:89 rebuild same predictor blocks three times for link, response,
lpmatrix, SE paths. Easy for one path to diverge from another.
- Medium: Remaining parity gaps still visible in tests. Analytic outer
gradient/Hessian for general-family ML/REML still unsupported: tests/
test_mgcv_gamlss_core.py:308. shashlss residual/k-check surfaces still
unsupported: tests/test_general_family_mgcv_parity.py:571. fs/sz dynamic
Gaussian REML still loose: tests/test_mgcv_additional_scenarios.py:265.
fs() k.check feature extraction still incomplete: nampy/gam/diagnostics/
k_check.py:16, tests/test_mgcv_k_check_parity.py:437.
  Implemented
- Core pipeline solid: formula parse/build -> predictor specs -> runtime term
instantiate -> compile -> identifiability -> fit -> predict.
- Smooth coverage good for current subset: univariate cr/cs/cc/ps/tp/ts/gp,
categorical re/mrf/fs/sz, tensor te/ti/t2. Registry proof: nampy/gam/
smooths/registry.py:1, smooth classes in smooths/.
- mgcv-port intent real in key backend pieces: smoothCon-like constructor
path in nampy/gam/compiler/construct.py:150, predictor-side identifiability
in nampy/gam/constraints/identifiability.py:82, gam.fit5/post-proc mirror
in nampy/gam/fit/solvers/gam_fit5.py:1, general-family backend in nampy/
gam/fit/solvers/general_fit5.py:1.
- Family coverage broad: gaussian, binomial logit/probit/cloglog, poisson,
gamma variants, negbin, gaulss/gammals/gevlss/shashlss/ziplss. See nampy/
gam/families/registry.py:1.
- Diagnostics/parity harness substantial: residuals, concurvity, k-check,
parity snapshots/traces/output tests. See nampy/gam/diagnostics/
residuals.py:36, nampy/gam/diagnostics/concurvity.py:71, nampy/gam/
  What Remains
- Bigger mgcv formula language.
- More basis families / edge-case options beyond current builder whitelist.
- Better decomposition of fit layer and term-representation layer.
- One canonical prediction-matrix path for all prediction modes.
- Close remaining parity gaps in fs/sz REML, fs/sz k.check, general-family
outer derivatives, and shashlss diagnostics



negbin REML theta-estimation path effectively dead code. In nampy/gam/

     smoothing_selection/optimize/driver.py:43, guard at nampy/gam/

     smoothing_selection/optimize/driver.py:61 compares x0.size against count

     of fixed params (~free_mask), not free params. For normal all-free models,

     helper returns None, then caller hard-raises at nampy/gam/

     smoothing_selection/optimize/driver.py:1026. This is live, not

     theoretical: pytest tests/test_mgcv_known_[gaps.py](http://gaps.py) -k

     negbin_estimated_theta_reml_endpoint_gap_tracked -v fails exactly here,

     while tests/test_mgcv_known_gaps.py:113 still expects partial parity.

  2. Same negbin path reconstructs smoothing params incorrectly if guard ever

     gets fixed. At nampy/gam/smoothing_selection/optimize/driver.py:112 it

     does np.exp(model.smoothing_params) even though model.smoothing_params

     already lives on natural scale, then writes optimized values into

     ~free_mask at nampy/gam/smoothing_selection/optimize/driver.py:113 and

     again at nampy/gam/smoothing_selection/optimize/driver.py:117. That is

     wrong mask direction and wrong scale. Caller then trusts this corrupted

     vector at nampy/gam/smoothing_selection/optimize/driver.py:1035. Severity

     high because first bug currently hides second.

  3. ti(..., mc=...) parity branch has no active regression coverage.

     Production branch lives at nampy/gam/smooths/tensor/ti.py:115 through

     nampy/gam/smooths/tensor/ti.py:136, but only dedicated parity case is

     explicitly skipped at tests/test_mgcv_parity_failing_and_warnings.py:72

     and tests/test_mgcv_parity_failing_and_warnings.py:113. For mgcv-parity

     code, skipped coverage on identifiability semantics is maintainability

     risk.



Findings

  1. One file doing too much in smoothing optimizer. nampy/gam/

     smoothing_selection/optimize/driver.py:1 mixes capability routing,

     initialization strategy, Gaussian special cases, negbin special cases,

     optimizer implementations, trace shaping, and result postprocessing. Hard

     to reason, hard to test in slices, easy to break parity by touching

     unrelated branch. Good architecture for parity code: keep orchestration

     thin, move family/backend-specific outer loops behind small strategy

     functions.

  2. Parity state spread across ad-hoc model attributes. Examples in nampy/gam/

     smoothing_selection/optimize/driver.py:1026, nampy/gam/fit/solvers/

     general_fit5.py:99, nampy/gam/smoothing_selection/postfit.py:297. Fields

     like *pirls*disable_theta_efs_, *general*fit5_outer_derivative_info,

     *gaussian*reml_sigma2_opt_, *optim*trace mutate across modules. Works, but

     maintainability weak: hidden coupling, no single schema, easy stale-state

     bug. Better: one typed optimizer-state/result object owned by smoothing-

     selection layer.

  3. Naming around scales/masks inconsistent enough to invite bugs. Same negbin

     helper uses free_mask, then writes through ~free_mask, and mixes log-

     scale/natural-scale storage in same block nampy/gam/smoothing_selection/

     optimize/driver.py:60 to nampy/gam/smoothing_selection/optimize/

     driver.py:118. Clean code issue, not only parity bug. In parity-sensitive

     numerics, names must encode scale exactly: log_sp_free, sp_full,

     fixed_mask, free_idx.

  4. Tensor / factor-smooth code carries important mgcv semantics, but comments

     only partially explain upstream mapping. Example nampy/gam/smooths/tensor/

     t2.py:114 and nampy/gam/smooths/categorical/factor_smooth.py:732. Logic

     may be right, but future maintainer must reverse-engineer why ordering/

     constraint placement/scaling chosen. For parity code, short “mirrors mgcv

     function X, stage Y” comments pay off.

  5. Test organization good overall, but signal split across “snapshot parity”,

     “known gaps”, and “failing/warnings” means live regressions can hide in

     quarantine buckets. Example negbin tracked gap now hard-fails unsupported,

     but still lives in tests/test_mgcv_known_gaps.py:113 rather than failing

     main parity surface. Maintainability risk: quarantine files become

     graveyard.

  What Clean Already

  - Clear subsystem split: specs/compiler/smooths/fit/predict mostly sensible.

  - Many unsupported surfaces raise explicitly, not heuristic fallback. Good

    for parity.

  - Public GAM API small. Good boundary.

  Best Cleanup Targets

  1. Split nampy/gam/smoothing_selection/optimize/driver.py:1 by backend/

     family:

     gaussian_[outer.py](http://outer.py), pirls_[outer.py](http://outer.py), negbin_[outer.py](http://outer.py), shared [result.py](http://result.py).

  2. Replace scattered model mutation with typed optimizer result/state

     dataclass.

  3. Standardize variable names for scale and masks repo-wide:

     log_sp, sp, free_mask, fixed_mask, free_idx, full_sp.

  4. Add upstream mgcv function refs at parity-critical blocks:

     newton(), gam.fit3, smooth.construct.*, XZKr, t2.model.matrix.

  5. Promote quarantined parity cases periodically:

     either unsupported with explicit skip rationale, or active regression in

     main suite.