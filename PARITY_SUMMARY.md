# NAMpy GAM — mgcv Parity Summary

  
**Status:** All listed parity tests currently pass. This document records what is
implemented, what is tested, at what tolerance, and what remains untested or unresolved.

---

## How to read this document


| Symbol       | Meaning                                                 |
| ------------ | ------------------------------------------------------- |
| **EXACT**    | Machine-precision parity (atol ≤ 1e-10, rtol ≤ 1e-10)   |
| **TIGHT**    | Numerical parity at atol ≤ 1e-6 / rtol ≤ 1e-6           |
| **LOOSE**    | Parity confirmed but tolerance is relaxed (atol > 1e-4) |
| **PARTIAL**  | Code exists and some tests pass; edge cases not covered |
| **UNTESTED** | Implementation present but no mgcv parity test exists   |
| **ABSENT**   | Feature not implemented in NAMpy                        |


---

## 1. Smooth Basis Types

### 1.1 Univariate smooths


| Smooth                        | mgcv equivalent | Basis parity | Penalty parity | Fixed-sp fit | REML fit  | Notes                                               |
| ----------------------------- | --------------- | ------------ | -------------- | ------------ | --------- | --------------------------------------------------- |
| `SplineTerm1D` bs=`cr`        | `s(x, bs="cr")` | **EXACT**    | **EXACT**      | **EXACT**    | **EXACT** | Core reference basis; most thoroughly tested        |
| `SplineTerm1D` bs=`cs`        | `s(x, bs="cs")` | **EXACT**    | **EXACT**      | **EXACT**    | **TIGHT** | cs shrinkage now EXACT via scipy DSYEVR; pc= now machine-precision |
| `SplineTerm1D` bs=`cc`        | `s(x, bs="cc")` | **EXACT**    | **EXACT**      | **EXACT**    | **EXACT** | Cyclic cubic; tests in `TestCyclicCubicSmooth`      |
| `PSplineTerm1D` bs=`ps`       | `s(x, bs="ps")` | **EXACT**    | **EXACT**      | **EXACT**    | **EXACT** | m=[2,2] default; m=[1,1] and m=[3,3] now also EXACT |
| `ThinPlateSplineTerm` bs=`tp` | `s(x, bs="tp")` | **TIGHT**    | **TIGHT**      | **EXACT**    | **EXACT** | Default basis for `s()`                             |
| `ThinPlateSplineTerm` bs=`ts` | `s(x, bs="ts")` | **TIGHT**    | **TIGHT**      | **EXACT**    | **EXACT** | Shrinkage variant of tp                             |
| `GPSmoothTerm` bs=`gp`        | `s(x, bs="gp")` | **EXACT**    | **EXACT**      | **TIGHT**    | **EXACT** | Single and two-smooth models tested                 |


### 1.2 Tensor product smooths


| Smooth                                       | mgcv equivalent | Basis parity | Penalty parity | Fixed-sp fit | REML fit  | Notes                                                                 |
| -------------------------------------------- | --------------- | ------------ | -------------- | ------------ | --------- | --------------------------------------------------------------------- |
| `TensorProductSplineTerm` bs=`te`            | `te(x,y,...)`   | **EXACT**    | **EXACT**      | **EXACT**    | **EXACT** | cr, ps, cc, ts, tp, gp marginals tested                               |
| `InteractionTensorProductSplineTerm` bs=`ti` | `ti(x,y,...)`   | **EXACT**    | **EXACT**      | **EXACT**    | **EXACT** | ANOVA decomposition correct; cc, ts, gp, tp marginals added           |
| `TensorANOVASplineTerm` bs=`t2`              | `t2(x,y,...)`   | **TIGHT**    | **TIGHT**      | **TIGHT**    | **TIGHT** | Natparam reparam passes at 1e-10 for cr×cr; cc and ts marginals added |


**Tensor marginal bases tested:**


| Marginal bs | te    | ti    | t2    |
| ----------- | ----- | ----- | ----- |
| `cr`        | EXACT | EXACT | TIGHT |
| `ps`        | EXACT | EXACT | TIGHT |
| `cc`        | EXACT | EXACT | TIGHT |
| `tp`        | EXACT | EXACT | TIGHT |
| `ts`        | EXACT | EXACT | TIGHT |
| `gp`        | TIGHT | TIGHT | TIGHT |


### 1.3 Categorical and spatial smooths


| Smooth                            | mgcv equivalent      | Basis parity | Penalty parity | Fixed-sp fit | REML fit  | Notes                                                               |
| --------------------------------- | -------------------- | ------------ | -------------- | ------------ | --------- | ------------------------------------------------------------------- |
| `RandomEffectTerm` bs=`re`        | `s(f, bs="re")`      | **EXACT**    | **EXACT**      | **EXACT**    | **EXACT** | Near-singular sp; stacked-QR solver tested                          |
| `FSmoothInteractionTerm` bs=`fs`  | `s(f,x,bs="fs")`     | **EXACT**    | **EXACT**      | —            | **EXACT** | cr and ps base smooths tested; 4-level REML parity now covered      |
| `SZSmoothInteractionTerm` bs=`sz` | `s(f1,f2,x,bs="sz")` | **EXACT**    | **EXACT**      | —            | **EXACT** | Shared-id penalty tested; 3x3-factor REML parity now covered        |
| `MarkovRandomFieldTerm` bs=`mrf`  | `s(r,bs="mrf")`      | **EXACT**    | **EXACT**      | —            | **EXACT** | Neighborhood adjacency; low-rank truncation (k < n_areas) now EXACT |


---

## 2. Smooth Modifiers

### 2.1 Point constraints (`pc=`)


| Basis              | Fixed-sp                  | REML      | Notes                                                                           |
| ------------------ | ------------------------- | --------- | ------------------------------------------------------------------------------- |
| `cr`               | **EXACT**                 | **EXACT** | Canonical reference; zero-at-point verified                                     |
| `cs`               | **EXACT**                 | **TIGHT** | Fixed-sp now machine-precision via DSYEVR fix (2026-04-09)                      |
| `cc`               | —                         | **EXACT** |                                                                                 |
| `ps`               | **EXACT** (zero-at-point) | **EXACT** |                                                                                 |
| `tp`               | **EXACT**                 | **EXACT** | Multivariate (2-d) tested                                                       |
| `ts`               | **EXACT**                 | **EXACT** | Multivariate tested                                                             |
| `gp`               | **EXACT**                 | **EXACT** | Multivariate tested                                                             |
| `cr` + by= factor  | **EXACT**                 | —         |                                                                                 |
| `cs` + by= factor  | **EXACT**                 | —         |                                                                                 |
| `ps` + by= factor  | **EXACT** (zero-at-point) | **EXACT** |                                                                                 |
| `tp` + numeric by= | —                         | **EXACT** |                                                                                 |
| `ts` + numeric by= | —                         | **EXACT** |                                                                                 |
| `gp` + numeric by= | —                         | **EXACT** |                                                                                 |
| `ps` + numeric by= | —                         | **EXACT** |                                                                                 |
| pc= + id= linked   | —                         | —         | mgcv does not support this combination; NAMpy asserts internal consistency only |


### 2.2 By-variables


| By type                   | Bases tested       | Status                                                                            |
| ------------------------- | ------------------ | --------------------------------------------------------------------------------- |
| Numeric by=               | cr, tp, ts, gp, ps | **EXACT** fixed-sp and REML for all listed                                        |
| Factor by=                | cr, cs, ps         | **EXACT** with and without pc=                                                    |
| Factor by= with link pred | cr                 | **TIGHT** (atol~5e-10 on link predictions) — tracked in `test_mgcv_known_gaps.py` |
| by= + select=True         | cr                 | **EXACT**                                                                         |


### 2.3 Linked basis (`id=`)


| Scenario                                | Status    | Notes                                                     |
| --------------------------------------- | --------- | --------------------------------------------------------- |
| Compatible k (k1 == k2) — fixed-sp      | **EXACT** | Shared basis verified                                     |
| Compatible k — REML                     | **EXACT** | Smoothing param count verified                            |
| Incompatible k (k1 ≠ k2) — fixed-sp     | **EXACT** | First k wins, harmonization tested                        |
| Incompatible k — REML                   | **EXACT** |                                                           |
| Reversed order (k2 < k1) — first k wins | **EXACT** |                                                           |
| ≥3 terms sharing same id=               | **EXACT** | Three-term fixed-sp parity and shared-sp count now tested |


### 2.4 `select=True` (shrinkage to zero)


| Family          | Method | Status                                         | Notes                                                                                                                               |
| --------------- | ------ | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Gaussian        | REML   | **EXACT**                                      | Null-space selection penalty tested                                                                                                 |
| Binomial        | REML   | **LOOSE** (predictions match; sp not compared) | sp values can diverge                                                                                                               |
| Poisson         | REML   | **LOOSE**                                      | Same as binomial                                                                                                                    |
| Gaussian (re)   | REML   | **EXACT**                                      | Includes near-singular intercept + `bs="re"` EDF attribution parity in snapshot output                                              |
| Gaussian (fs)   | REML   | **TIGHT**                                      | Base fs REML, fs term SE parity, and `select=True` ridge-stabilized endpoint metadata are covered for cr and ps marginals           |
| Gaussian (sz)   | REML   | **LOOSE**                                      |                                                                                                                                     |
| Gaussian (mrf)  | REML   | **TIGHT**                                      |                                                                                                                                     |
| Tensor (te, ti) | REML   | **TIGHT**                                      | `test_gaussian_te_select_reml_matches_mgcv`                                                                                         |
| Gamma           | REML   | **TIGHT**                                      | `select=True` REML now parity-tested on covered `cr` surface; predictions/criterion are exact-to-tight and `sp` agrees within ~1e-6 |
| Negbin          | REML   | **LOOSE**                                      | Fixed-theta `select=True` REML now parity-tested; predictions/criterion/theta match, but `log(sp)` can diverge on flat ridge        |


---

## 3. Families and Response Distributions


| Family              | Link     | Fixed-sp   | REML       | ML        | GCV       | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------- | -------- | ---------- | ---------- | --------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gaussian            | identity | **EXACT**  | **EXACT**  | **EXACT** | **EXACT** | Most thoroughly tested; scale (σ²) and RSS verified                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Binomial            | logit    | **EXACT**  | **EXACT**  | **LOOSE** | —         | ML: sp_log_atol=2.0 (optimizer path diverges)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Poisson             | log      | **EXACT**  | **EXACT**  | **LOOSE** | —         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Gamma               | log      | **EXACT**  | **EXACT**  | **LOOSE** | **LOOSE** | GCV sp_log_atol=0.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Negative binomial   | log      | **EXACT**  | **EXACT**  | —         | —         | Fixed-theta parity is exact; estimated-theta REML is now **PARTIAL** via `test_negbin_estimated_theta_reml_matches_mgcv`, `test_negbin_estimated_theta_reml_endpoint_gap_tracked`, and the two-smooth coverage in `test_mgcv_known_gaps.py` (response/theta/criterion parity is covered on the exercised surface; single-smooth `log(sp)` is within about 0.16 of mgcv after switching to a joint outer `(log sp, log theta)` solve seeded by the earlier EFS path, the two-smooth `theta=2.0` case is now within about 0.05, and the two-smooth `theta=0.5` case remains within about 0.35 on a broader low-theta ridge) |
| Tweedie             | —        | **ABSENT** | **ABSENT** | —         | —         | Not implemented                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Quasi/quasi-Poisson | —        | **ABSENT** | **ABSENT** | —         | —         | Not implemented                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |


**Prior weights (sample_weight=):**


| Family   | Fixed-sp  | REML      | Notes                                                                                           |
| -------- | --------- | --------- | ----------------------------------------------------------------------------------------------- |
| Gaussian | **EXACT** | **EXACT** | Full REML end-to-end with weights now tested in `test_weighted_reml_end_to_end_sp_matches_mgcv` |
| Poisson  | **EXACT** | —         | Fixed-sp at mgcv's sp value tested                                                              |
| Binomial | **EXACT** | —         | Fixed-sp at mgcv's sp value tested                                                              |


---

## 4. Smoothing Parameter Selection Methods


| Method        | NAMpy name     | Gaussian   | GLM               | Notes                                                                                                   |
| ------------- | -------------- | ---------- | ----------------- | ------------------------------------------------------------------------------------------------------- |
| REML          | `"REML"`       | **EXACT**  | **TIGHT**         | Primary method; most tests use this, including Gaussian accepted-step trace parity vs mgcv `score.hist` |
| ML            | `"ML"`         | **EXACT**  | **LOOSE**         | Criterion values differ by additive constant; predictions match                                         |
| GCV.Cp        | `"gcv"`        | **EXACT**  | **LOOSE** (Gamma) | Implemented; sp_log_atol=1e-5 (Gaussian), 0.1 (Gamma)                                                   |
| Fixed sp      | `sp=` per-term | **EXACT**  | **EXACT**         |                                                                                                         |
| P-REML / P-ML | —              | **ABSENT** | **ABSENT**        | mgcv variants for Poisson/negbin not implemented                                                        |
| fREML         | —              | **ABSENT** | **ABSENT**        | Faster REML approximation not implemented                                                               |


---

## 5. Fitting Backends and Solvers


| Component                     | mgcv analogue          | Status    | Tests                                                                                                                                      |
| ----------------------------- | ---------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Penalized LS (Gaussian exact) | `C_pls_fit1`           | **EXACT** | `test_gam_fit_nonnegative_penalized_qr_state.py`                                                                                           |
| Stacked-QR solver             | `qr.lm`-based          | **EXACT** | RE near-singular, QR state beta and log-det tested                                                                                         |
| Penalized IRLS                | `C_gdi1` / outer PIRLS | **EXACT** | Gradient and Hessian match for Poisson, Binomial, Gamma                                                                                    |
| REML criterion (Gaussian)     | `gam.fit3` Laplace     | **EXACT** | `test_gam_gaussian_reml_algebra.py`                                                                                                        |
| REML criterion (GLM)          | `gam.fit5` Laplace     | **TIGHT** | Gradient/Hessian at optimum tested in `TestMgcvTraceParity`                                                                                |
| Outer optimizer               | Newton + line search   | **TIGHT** | Trace parity; rollback and step-halving tested                                                                                             |
| Covariance matrix             | `Vp` / `Vc`            | **TIGHT** | Bayesian and frequentist covariances in snapshot                                                                                           |
| EDF computation               | `edf`, `edf1`          | **EXACT** | Per-term EDF in all snapshot parity tests; `bs="re"` term EDF attribution now aligned with `summary.gam` in intercept + RE REML edge cases |


---

## 6. Prediction and Inference


| Feature                                       | mgcv analogue                   | Status    | Notes                                                                                                                                                                                          |
| --------------------------------------------- | ------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type="response"`                             | `predict(..., type="response")` | **EXACT** | Tested across all families                                                                                                                                                                     |
| `type="link"`                                 | `predict(..., type="link")`     | **EXACT** |                                                                                                                                                                                                |
| `type="lpmatrix"`                             | `predict(..., type="lpmatrix")` | **EXACT** | lpmatrix tested for Gaussian                                                                                                                                                                   |
| `type="terms"`                                | `predict(..., type="terms")`    | **TIGHT** | Standalone mgcv suite covers cr/cs/cc/ps/tp/ts/gp/te/ti/t2/re/fs/sz/mrf; direct `t2` parity included; fs term decomposition and REML term SE are covered                                       |
| Standard errors (SE)                          | `se.fit=TRUE`                   | **TIGHT** | Link/response/newdata SE remain exact; `type="terms", se.fit=TRUE` is now covered for cr/cs/cc/ps/tp/ts/gp/te/ti/re/fs/sz/mrf, while `t2` remains a tight covariance/reparameterization case   |
| Prediction on new data                        | `predict(model, newdata)`       | **EXACT** |                                                                                                                                                                                                |
| Offset in prediction                          | `offset=`                       | **EXACT** | Tested with formula offset                                                                                                                                                                     |
| `anova()` model comparison                    | `anova.gam()`                   | **EXACT** | Chi-sq and F-test for nested models                                                                                                                                                            |
| Residuals (response/working/Pearson/deviance) | `residuals.gam()`               | **EXACT** | All residual types now pass at atol=1e-10 for Poisson and Binomial                                                                                                                             |
| Concurvity                                    | `concurvity()`                  | **EXACT** | `full=TRUE` and pairwise `full=FALSE` both snapshot-tested for Gaussian and Poisson models                                                                                                     |
| Basis dimension check                         | `k.check()`                     | **TIGHT** | `k_prime` EXACT; `edf` TIGHT (5e-6 Gaussian, 5e-3 Gamma), including mixed numeric + `bs="re"` Gaussian REML coverage; k_index validity only (RNG-dependent); see `test_mgcv_k_check_parity.py` |


---

## 7. Design and Constraint Pipeline


| Stage                          | Component                            | Status    | Notes                                                                     |
| ------------------------------ | ------------------------------------ | --------- | ------------------------------------------------------------------------- |
| Formula parsing                | `gam/formula/parser.py`              | **EXACT** | Python formula syntax tested                                              |
| TermSpec compilation           | `gam/formula/compiler.py`            | **EXACT** | Tested via `TestCompilePredictorSpecsFromFormula`                         |
| RuntimeTerm factory            | `gam/runtime/factory.py`             | **EXACT** | All basis types instantiated in tests                                     |
| Term construction (Stage 3)    | `gam/design/constructors.py`         | **EXACT** | Coefficient map tested in `test_gam_design_constraint_maps.py`            |
| Sum-to-zero identifiability    | `gam/constraints/identifiability.py` | **EXACT** | Column deletion, exempt terms, penalty transform tested                   |
| Explicit constraint absorption | `gam/constraints/absorption.py`      | **EXACT** | `apply_linear_constraint`, `full_term_sum_to_zero_constraint` unit tested |
| Penalty normalization          | `gam/penalties/subsystem.py`         | **EXACT** | Rank, null-space dim, normalization unit tested                           |
| Null-space selection penalty   | `build_null_space_selection_spec`    | **EXACT** | select=True penalty construction tested                                   |
| Tensor product penalties       | `gam/basis/tensor.py`                | **EXACT** | Kronecker sum structure, penalty scaling tested                           |
| t2 natural parameter reparam   | `t2_marginal_reparameterization`     | **EXACT** | cr×cr case passes at atol=1e-10; mixed marginals (tp, gp) remain TIGHT    |
| Linked basis (id=)             | `gam/design/linked_basis.py`         | **EXACT** | k harmonization, shared sp count                                          |
| Predictor compilation          | `gam/design/compiler.py`             | **EXACT** | Multi-predictor assembly structural independence now tested               |


---

## 8. Optimizer Trace and Diagnostics


| Feature                                         | Status      | Notes                                                                                                                                                    |
| ----------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trace serialization (save/load)                 | **EXACT**   | Schema roundtrip tested                                                                                                                                  |
| Optimizer endpoint metadata                     | **PARTIAL** | Endpoint metadata is covered; fs `select=True` ridge stabilization is now surfaced explicitly, while `sz + select=True` remains a looser flat-ridge case |
| Non-Gaussian gradient at optimum                | **EXACT**   | Poisson/Binomial matches mgcv                                                                                                                            |
| Non-Gaussian Hessian at optimum                 | **EXACT**   |                                                                                                                                                          |
| Gaussian optimizer trace                        | **TIGHT**   | Accepted-step trajectory and final trace row now compared against mgcv `outer.info$score.hist` / endpoint                                                |
| Log-sp seed matrix                              | **EXACT**   | `test_endpoint_log_sp_seed_matrix`                                                                                                                       |
| PIRLS exact derivatives (gradient)              | **EXACT**   | Finite-difference cross-checked; Gamma included                                                                                                          |
| PIRLS exact derivatives (Hessian)               | **EXACT**   | Including K1/K2 Laplace decomposition blocks                                                                                                             |
| Gamma Newton branch (working vs Fisher weights) | **EXACT**   | Regression test in `test_gam_mgcv_patch_regressions.py`                                                                                                  |
| Step-halving exhaustion behavior                | **EXACT**   | Returns failure without accepting bad step                                                                                                               |
| Optimizer rollback state                        | **EXACT**   | Stable metadata after rollback                                                                                                                           |


---

## 9. What Is Not Tested (Implementation Exists)

There are currently no major parity items in this category that are both implemented and completely untested in the main mgcv parity surface.

Previously untested items that are now covered: tensor marginals (cc/tp/ts/gp), MRF low-rank truncation, ps m=[1,1]/m=[3,3], weighted Gaussian REML end-to-end, Gaussian accepted-step trace parity, Gaussian final trace-row endpoint parity, factor-by `select=True`, linked `id=` with 3 terms, FS/SZ larger-factor cases, distributional-regression multi-predictor compilation, concurvity (full and pairwise), k.check, and near-singular Gaussian `bs="re"` EDF attribution parity.

---

## 10. What Is Not Implemented (Absent from NAMpy)

These are mgcv features with **no NAMpy implementation**:


| Feature                              | mgcv                                           | Priority |
| ------------------------------------ | ---------------------------------------------- | -------- |
| Tweedie family                       | `tw()`, `Tweedie()`                            | Low      |
| Quasi/quasi-likelihood families      | `quasi()`, `quasipoisson()`, `quasibinomial()` | Low      |
| Soap-film smooths                    | `bs="so"`                                      | Low      |
| Adaptive smooths                     | `bs="ad"`                                      | Low      |
| Cyclic P-splines                     | `bs="cp"`                                      | Low      |
| Scaled TP variant                    | `bs="sos"`                                     | Low      |
| fREML (faster REML)                  | `method="fREML"`                               | Medium   |
| P-REML / P-ML                        | `method="P-REML"`, `"P-ML"`                    | Low      |
| `xt=` advanced tensor options        | multi-knot groups in tensor marginals          | Low      |
| Linear inequality constraints        | `L`, `lsp0` in mgcv                            | Low      |
| Periodic boundary constraints        | (beyond cc cyclic)                             | Low      |
| Custom link functions                | `make.link()`                                  | Low      |
| Pre-specified error scale (`scale=`) | `gam(..., scale=)`                             | Medium   |
| `predict(type="iterms")`             | per-term with interaction effects              | Low      |


---

## 11. Known Tolerance Gaps (Passing Tests with Relaxed Tolerance)

These tests pass but at tolerances looser than machine precision. They are
actively tracked so that future work can tighten them.


| Test                                                              | Concept                                 | Tolerance        | Gap reason                                                                                                                                                             |
| ----------------------------------------------------------------- | --------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_strict_factor_by_link_parity`                               | factor by= link                         | atol=5e-10       | Gaussian exact REML outer endpoint now matches mgcv to near-machine precision; residual gap is endpoint selection, not prediction matrix construction                  |
| `test_binomial_ml_matches_mgcv`                                   | Binomial ML sp                          | sp_log_atol=2.0  | ML outer optimizer converges to a different local basin                                                                                                                |
| `test_gaussian_fs_select_reml_matches_mgcv`                       | fs + select=True                        | sp_log_atol=2.0  | Flat ridge; predictions remain exact-to-tight, criterion stays aligned                                                                                                 |
| `test_gaussian_sz_select_reml_matches_mgcv`                       | sz + select=True                        | sp_log_atol=4.1  | Same flat-ridge issue                                                                                                                                                  |
| `test_negbin_select_reml_matches_mgcv`                            | negbin + select=True                    | `check_sp=False` | Fixed-theta negbin `select=True` lands on flat ridge; predictions/criterion/theta match while `log(sp)` can differ by ~2.2                                             |
| `test_negbin_estimated_theta_reml_endpoint_gap_tracked`           | negbin estimated theta REML `log(sp)`   | sp_log_atol=0.2  | mgcv default REML uses `outer,newton`; NAMpy now uses a joint outer `(log sp, log theta)` objective with theta fixed inside PIRLS and seeded from the earlier EFS path |
| `test_negbin_estimated_theta_reml_two_smooth_theta2_gap_tracked`  | negbin two-smooth `log(sp)` (theta=2.0) | sp_log_atol=0.05 | Joint outer `(log sp, log theta)` path now lands essentially on the mgcv endpoint                                                                                      |
| `test_negbin_estimated_theta_reml_two_smooth_theta05_gap_tracked` | negbin two-smooth `log(sp)` (theta=0.5) | sp_log_atol=0.35 | Joint outer path improves theta/criterion strongly, but one smoother still drifts on the broader low-theta ridge                                                       |


---

## 12. Test File Map


| File                                                 | Coverage area                                                  | Test count (approx) |
| ---------------------------------------------------- | -------------------------------------------------------------- | ------------------- |
| `_mgcv_snapshot_parity_shared.py`                    | Shared fixture classes for all snapshot tests                  | ~140 tests          |
| `test_mgcv_snapshot_parity.py`                       | Entry point delegating to shared classes                       | ~12                 |
| `test_mgcv_smoothcon_parity.py`                      | Basis/penalty parity; cc/ps/gp model fits; ps m=[1,1],[3,3]    | ~53                 |
| `test_mgcv_output_parity.py`                         | Predictions on new data, SE, anova, lpmatrix                   | ~8                  |
| `test_mgcv_pc_id_parity.py`                          | Point constraints + linked basis; all basis combos             | ~50                 |
| `test_mgcv_trace_parity.py`                          | Optimizer trace, gradient/Hessian at optimum                   | ~10                 |
| `test_mgcv_known_gaps.py`                            | Tolerance-tracked strict parity assertions                     | 7                   |
| `test_mgcv_additional_scenarios.py`                  | select=True GLMs, weighted GLMs, tensor cc/ts/tp/gp            | ~20                 |
| `test_mgcv_gaussian_weighted_and_re_regressions.py`  | RE + weighting regressions; weighted REML end-to-end           | ~5                  |
| `test_gam_unit_coverage.py`                          | Unit tests for constraint, penalty, tensor, formula subsystems | ~40 classes         |
| `test_gam_runtime_term_contract.py`                  | Runtime term interface contracts                               | ~6                  |
| `test_gam_design_constraint_maps.py`                 | Coefficient map shape/type tests                               | ~2                  |
| `test_gam_gaussian_reml_algebra.py`                  | REML algebra: Laplace, scale, saturation                       | ~7                  |
| `test_gam_gaussian_smoothness_postprocess_parity.py` | Post-processing scale refinement                               | ~1 class            |
| `test_gam_tensor_pirls_reml.py`                      | Tensor REML derivatives for te/ti/t2                           | ~2                  |
| `test_gam_fit_penalized_irls_solver.py`              | IRLS solver correctness                                        | ~3                  |
| `test_gam_fit_nonnegative_penalized_qr_state.py`     | QR state: beta, log-det                                        | ~3                  |
| `test_pirls_exact_derivatives.py`                    | PIRLS gradient/Hessian finite-difference cross-check           | ~8                  |
| `test_gam_smoothing_selection_derivatives.py`        | Penalty derivative matrices                                    | ~2                  |
| `test_gam_mgcv_patch_regressions.py`                 | Regression tests: Newton branch, step-halving, rollback        | ~4                  |
| `test_parity_matrix_consistency.py`                  | Self-consistency of this document / parity registry            | ~3                  |


---

## 13. Remaining Priority Gaps

Ordered by impact on claiming full mgcv parity. All items 1–5 from the original list are **DONE** as of 2026-04-08.

1. **cs + pc= tolerance** — **RESOLVED** (2026-04-09). Root cause was `np.linalg.eigh` (DSYEVD) finding a different null-space basis than R's `eigen(symmetric=TRUE)` (DSYEVR). Switching to `scipy.linalg.eigh` (which defaults to DSYEVR) closes the gap to machine precision. cs fixed-sp and cs+pc= now EXACT; cs REML predictions now TIGHT (~1e-6).
2. **factor by= link predictions** — now down to about `5e-10`, but not yet machine exact.
3. **GLM ML optimizer basin** — sp values diverge; predictions still match. Structural gap (different local minima), unlikely fixable without matching mgcv's exact step sequence.
4. **negbin estimated-theta REML sp** — now down to about `0.0002` to `0.34` on the tracked surface after switching from the old joint-EFS outer path to a joint outer `(log sp, log theta)` solve with theta fixed inside PIRLS; single-smooth is about `0.16`, two-smooth `theta=2.0` is about `0.05`, and the remaining broader miss is the low-theta two-smooth surface.

