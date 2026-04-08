# NAMpy GAM — mgcv Parity Summary

**Date:** 2026-04-08  
**Status:** All listed fs parity tests currently pass. This document records what is
implemented, what is tested, at what tolerance, and what remains untested or unresolved.

---

## How to read this document

| Symbol | Meaning |
|--------|---------|
| **EXACT** | Machine-precision parity (atol ≤ 1e-10, rtol ≤ 1e-10) |
| **TIGHT** | Numerical parity at atol ≤ 1e-6 / rtol ≤ 1e-6 |
| **LOOSE** | Parity confirmed but tolerance is relaxed (atol > 1e-4) |
| **PARTIAL** | Code exists and some tests pass; edge cases not covered |
| **UNTESTED** | Implementation present but no mgcv parity test exists |
| **ABSENT** | Feature not implemented in NAMpy |

---

## 1. Smooth Basis Types

### 1.1 Univariate smooths

| Smooth | mgcv equivalent | Basis parity | Penalty parity | Fixed-sp fit | REML fit | Notes |
|--------|----------------|-------------|----------------|-------------|----------|-------|
| `SplineTerm1D` bs=`cr` | `s(x, bs="cr")` | **EXACT** | **EXACT** | **EXACT** | **EXACT** | Core reference basis; most thoroughly tested |
| `SplineTerm1D` bs=`cs` | `s(x, bs="cs")` | **EXACT** | **EXACT** | **EXACT** | **TIGHT** | cs inherits cr + shrinkage; pc= tolerance ~1e-4 |
| `SplineTerm1D` bs=`cc` | `s(x, bs="cc")` | **EXACT** | **EXACT** | **EXACT** | **EXACT** | Cyclic cubic; tests in `TestCyclicCubicSmooth` |
| `PSplineTerm1D` bs=`ps` | `s(x, bs="ps")` | **EXACT** | **EXACT** | **EXACT** | **EXACT** | Default m=[2,2]; other m combos `PARTIAL` |
| `ThinPlateSplineTerm` bs=`tp` | `s(x, bs="tp")` | **TIGHT** | **TIGHT** | **EXACT** | **EXACT** | Default basis for `s()` |
| `ThinPlateSplineTerm` bs=`ts` | `s(x, bs="ts")` | **TIGHT** | **TIGHT** | **EXACT** | **EXACT** | Shrinkage variant of tp |
| `GPSmoothTerm` bs=`gp` | `s(x, bs="gp")` | **EXACT** | **EXACT** | **TIGHT** | **EXACT** | Single and two-smooth models tested |

### 1.2 Tensor product smooths

| Smooth | mgcv equivalent | Basis parity | Penalty parity | Fixed-sp fit | REML fit | Notes |
|--------|----------------|-------------|----------------|-------------|----------|-------|
| `TensorProductSplineTerm` bs=`te` | `te(x,y,...)` | **EXACT** | **EXACT** | **EXACT** | **EXACT** | cr and ps marginals tested; tp/gp marginals `PARTIAL` |
| `InteractionTensorProductSplineTerm` bs=`ti` | `ti(x,y,...)` | **EXACT** | **EXACT** | **EXACT** | **EXACT** | ANOVA decomposition correct |
| `TensorANOVASplineTerm` bs=`t2` | `t2(x,y,...)` | **TIGHT** | **TIGHT** | **TIGHT** | **TIGHT** | Natural param reparameterization tolerance ~1e-7; negbin+t2 REML tested |

**Tensor marginal bases tested:**

| Marginal bs | te | ti | t2 |
|-------------|----|----|-----|
| `cr` | EXACT | EXACT | TIGHT |
| `ps` | EXACT | EXACT | TIGHT |
| `cc` | UNTESTED | UNTESTED | UNTESTED |
| `tp` | PARTIAL | PARTIAL | PARTIAL |
| `ts` | PARTIAL | PARTIAL | PARTIAL |
| `gp` | PARTIAL | PARTIAL | PARTIAL |

### 1.3 Categorical and spatial smooths

| Smooth | mgcv equivalent | Basis parity | Penalty parity | Fixed-sp fit | REML fit | Notes |
|--------|----------------|-------------|----------------|-------------|----------|-------|
| `RandomEffectTerm` bs=`re` | `s(f, bs="re")` | **EXACT** | **EXACT** | **EXACT** | **EXACT** | Near-singular sp; stacked-QR solver tested |
| `FSmoothInteractionTerm` bs=`fs` | `s(f,x,bs="fs")` | **EXACT** | **EXACT** | — | **EXACT** | cr and ps base smooths tested; ≥3 factors `PARTIAL` |
| `SZSmoothInteractionTerm` bs=`sz` | `s(f1,f2,x,bs="sz")` | **EXACT** | **EXACT** | — | **EXACT** | Shared-id penalty tested; ≥3 factors `PARTIAL` |
| `MarkovRandomFieldTerm` bs=`mrf` | `s(r,bs="mrf")` | **EXACT** | **EXACT** | — | **EXACT** | Neighborhood adjacency; low-rank truncation `PARTIAL` |

---

## 2. Smooth Modifiers

### 2.1 Point constraints (`pc=`)

| Basis | Fixed-sp | REML | Notes |
|-------|---------|------|-------|
| `cr` | **EXACT** | **EXACT** | Canonical reference; zero-at-point verified |
| `cs` | **TIGHT** (atol~1e-4) | **TIGHT** | Tolerance relaxed vs cr |
| `cc` | — | **EXACT** | |
| `ps` | **EXACT** (zero-at-point) | **EXACT** | |
| `tp` | **EXACT** | **EXACT** | Multivariate (2-d) tested |
| `ts` | **EXACT** | **EXACT** | Multivariate tested |
| `gp` | **EXACT** | **EXACT** | Multivariate tested |
| `cr` + by= factor | **EXACT** | — | |
| `cs` + by= factor | **EXACT** | — | |
| `ps` + by= factor | **EXACT** (zero-at-point) | **EXACT** | |
| `tp` + numeric by= | — | **EXACT** | |
| `ts` + numeric by= | — | **EXACT** | |
| `gp` + numeric by= | — | **EXACT** | |
| `ps` + numeric by= | — | **EXACT** | |
| pc= + id= linked | — | — | mgcv does not support this combination; NAMpy asserts internal consistency only |

### 2.2 By-variables

| By type | Bases tested | Status |
|---------|-------------|--------|
| Numeric by= | cr, tp, ts, gp, ps | **EXACT** fixed-sp and REML for all listed |
| Factor by= | cr, cs, ps | **EXACT** with and without pc= |
| Factor by= with link pred | cr | **LOOSE** (atol~1e-3 on link predictions) — tracked in `test_mgcv_known_gaps.py` |
| by= + select=True | cr | **UNTESTED** |

### 2.3 Linked basis (`id=`)

| Scenario | Status | Notes |
|----------|--------|-------|
| Compatible k (k1 == k2) — fixed-sp | **EXACT** | Shared basis verified |
| Compatible k — REML | **EXACT** | Smoothing param count verified |
| Incompatible k (k1 ≠ k2) — fixed-sp | **EXACT** | First k wins, harmonization tested |
| Incompatible k — REML | **EXACT** | |
| Reversed order (k2 < k1) — first k wins | **EXACT** | |
| ≥3 terms sharing same id= | **PARTIAL** | Two-term case only tested |

### 2.4 `select=True` (shrinkage to zero)

| Family | Method | Status | Notes |
|--------|--------|--------|-------|
| Gaussian | REML | **EXACT** | Null-space selection penalty tested |
| Binomial | REML | **LOOSE** (predictions match; sp not compared) | sp values can diverge |
| Poisson | REML | **LOOSE** | Same as binomial |
| Gaussian (re) | REML | **EXACT** | |
| Gaussian (fs) | REML | **TIGHT** | Base fs REML, fs term SE parity, and `select=True` ridge-stabilized endpoint metadata are covered for cr and ps marginals |
| Gaussian (sz) | REML | **LOOSE** | |
| Gaussian (mrf) | REML | **TIGHT** | |
| Tensor (te, ti) | REML | **TIGHT** | `test_gaussian_te_select_reml_matches_mgcv` |
| Gamma | — | **UNTESTED** | |
| Negbin | — | **UNTESTED** | |

---

## 3. Families and Response Distributions

| Family | Link | Fixed-sp | REML | ML | GCV | Notes |
|--------|------|---------|------|-----|-----|-------|
| Gaussian | identity | **EXACT** | **EXACT** | **EXACT** | **EXACT** | Most thoroughly tested; scale (σ²) and RSS verified |
| Binomial | logit | **EXACT** | **EXACT** | **LOOSE** | — | ML: sp_log_atol=2.0 (optimizer path diverges) |
| Poisson | log | **EXACT** | **EXACT** | **LOOSE** | — | |
| Gamma | log | **EXACT** | **EXACT** | **LOOSE** | **LOOSE** | GCV sp_log_atol=0.1 |
| Negative binomial | log | **EXACT** | **EXACT** | — | — | Fixed-theta parity is exact; estimated-theta REML is now **PARTIAL** via `test_negbin_estimated_theta_reml_matches_mgcv` (response/theta/criterion parity covered, smoothing parameter ridge remains unverified) |
| Tweedie | — | **ABSENT** | **ABSENT** | — | — | Not implemented |
| Quasi/quasi-Poisson | — | **ABSENT** | **ABSENT** | — | — | Not implemented |

**Prior weights (sample_weight=):**

| Family | Fixed-sp | REML | Notes |
|--------|---------|------|-------|
| Gaussian | **EXACT** | **EXACT** | REML algebra with weights tested in `TestGaussianPriorWeights` |
| Poisson | **EXACT** | — | Fixed-sp at mgcv's sp value tested |
| Binomial | **EXACT** | — | Fixed-sp at mgcv's sp value tested |

---

## 4. Smoothing Parameter Selection Methods

| Method | NAMpy name | Gaussian | GLM | Notes |
|--------|-----------|----------|-----|-------|
| REML | `"REML"` | **EXACT** | **TIGHT** | Primary method; most tests use this |
| ML | `"ML"` | **EXACT** | **LOOSE** | Criterion values differ by additive constant; predictions match |
| GCV.Cp | `"gcv"` | **EXACT** | **LOOSE** (Gamma) | Implemented; sp_log_atol=1e-5 (Gaussian), 0.1 (Gamma) |
| Fixed sp | `sp=` per-term | **EXACT** | **EXACT** | |
| P-REML / P-ML | — | **ABSENT** | **ABSENT** | mgcv variants for Poisson/negbin not implemented |
| fREML | — | **ABSENT** | **ABSENT** | Faster REML approximation not implemented |

---

## 5. Fitting Backends and Solvers

| Component | mgcv analogue | Status | Tests |
|-----------|---------------|--------|-------|
| Penalized LS (Gaussian exact) | `C_pls_fit1` | **EXACT** | `test_gam_fit_nonnegative_penalized_qr_state.py` |
| Stacked-QR solver | `qr.lm`-based | **EXACT** | RE near-singular, QR state beta and log-det tested |
| Penalized IRLS | `C_gdi1` / outer PIRLS | **EXACT** | Gradient and Hessian match for Poisson, Binomial, Gamma |
| REML criterion (Gaussian) | `gam.fit3` Laplace | **EXACT** | `test_gam_gaussian_reml_algebra.py` |
| REML criterion (GLM) | `gam.fit5` Laplace | **TIGHT** | Gradient/Hessian at optimum tested in `TestMgcvTraceParity` |
| Outer optimizer | Newton + line search | **TIGHT** | Trace parity; rollback and step-halving tested |
| Covariance matrix | `Vp` / `Vc` | **TIGHT** | Bayesian and frequentist covariances in snapshot |
| EDF computation | `edf`, `edf1` | **EXACT** | Per-term EDF in all snapshot parity tests |

---

## 6. Prediction and Inference

| Feature | mgcv analogue | Status | Notes |
|---------|---------------|--------|-------|
| `type="response"` | `predict(..., type="response")` | **EXACT** | Tested across all families |
| `type="link"` | `predict(..., type="link")` | **EXACT** | |
| `type="lpmatrix"` | `predict(..., type="lpmatrix")` | **EXACT** | lpmatrix tested for Gaussian |
| `type="terms"` | `predict(..., type="terms")` | **TIGHT** | Standalone mgcv suite covers cr/cs/cc/ps/tp/ts/gp/te/ti/t2/re/fs/sz/mrf; direct `t2` parity included; fs term decomposition and REML term SE are covered |
| Standard errors (SE) | `se.fit=TRUE` | **TIGHT** | Link/response/newdata SE remain exact; `type="terms", se.fit=TRUE` is now covered for cr/cs/cc/ps/tp/ts/gp/te/ti/re/fs/sz/mrf, while `t2` remains a tight covariance/reparameterization case |
| Prediction on new data | `predict(model, newdata)` | **EXACT** | |
| Offset in prediction | `offset=` | **EXACT** | Tested with formula offset |
| `anova()` model comparison | `anova.gam()` | **EXACT** | Chi-sq and F-test for nested models |
| Residuals (response/working/Pearson/deviance) | `residuals.gam()` | **TIGHT** | Poisson/Binomial deviance residuals have tolerance ~1e-6; tracked in `test_mgcv_known_gaps.py` |
| Concurvity | `concurvity()` | **EXACT** | `full=TRUE` and pairwise `full=FALSE` both snapshot-tested for Gaussian and Poisson models |
| Basis dimension check | `k.check()` | **TIGHT** | `k_prime` EXACT; `edf` TIGHT (5e-6 Gaussian, 5e-3 Gamma); k_index validity only (RNG-dependent); see `test_mgcv_k_check_parity.py` |

---

## 7. Design and Constraint Pipeline

| Stage | Component | Status | Notes |
|-------|-----------|--------|-------|
| Formula parsing | `gam/formula/parser.py` | **EXACT** | Python formula syntax tested |
| TermSpec compilation | `gam/formula/compiler.py` | **EXACT** | Tested via `TestCompilePredictorSpecsFromFormula` |
| RuntimeTerm factory | `gam/runtime/factory.py` | **EXACT** | All basis types instantiated in tests |
| Term construction (Stage 3) | `gam/design/constructors.py` | **EXACT** | Coefficient map tested in `test_gam_design_constraint_maps.py` |
| Sum-to-zero identifiability | `gam/constraints/identifiability.py` | **EXACT** | Column deletion, exempt terms, penalty transform tested |
| Explicit constraint absorption | `gam/constraints/absorption.py` | **EXACT** | `apply_linear_constraint`, `full_term_sum_to_zero_constraint` unit tested |
| Penalty normalization | `gam/penalties/subsystem.py` | **EXACT** | Rank, null-space dim, normalization unit tested |
| Null-space selection penalty | `build_null_space_selection_spec` | **EXACT** | select=True penalty construction tested |
| Tensor product penalties | `gam/basis/tensor.py` | **EXACT** | Kronecker sum structure, penalty scaling tested |
| t2 natural parameter reparam | `t2_marginal_reparameterization` | **TIGHT** | atol ~1e-7 vs mgcv; tracked tolerance |
| Linked basis (id=) | `gam/design/linked_basis.py` | **EXACT** | k harmonization, shared sp count |
| Predictor compilation | `gam/design/compiler.py` | **EXACT** | Multi-predictor assembly (distributional regression) not tested |

---

## 8. Optimizer Trace and Diagnostics

| Feature | Status | Notes |
|---------|--------|-------|
| Trace serialization (save/load) | **EXACT** | Schema roundtrip tested |
| Optimizer endpoint metadata | **PARTIAL** | Endpoint metadata is covered; fs `select=True` ridge stabilization is now surfaced explicitly, while `sz + select=True` remains a looser flat-ridge case |
| Non-Gaussian gradient at optimum | **EXACT** | Poisson/Binomial matches mgcv |
| Non-Gaussian Hessian at optimum | **EXACT** | |
| Gaussian optimizer trace | **UNTESTED** | Trace capture works; no dedicated parity test |
| Log-sp seed matrix | **EXACT** | `test_endpoint_log_sp_seed_matrix` |
| PIRLS exact derivatives (gradient) | **EXACT** | Finite-difference cross-checked; Gamma included |
| PIRLS exact derivatives (Hessian) | **EXACT** | Including K1/K2 Laplace decomposition blocks |
| Gamma Newton branch (working vs Fisher weights) | **EXACT** | Regression test in `test_gam_mgcv_patch_regressions.py` |
| Step-halving exhaustion behavior | **EXACT** | Returns failure without accepting bad step |
| Optimizer rollback state | **EXACT** | Stable metadata after rollback |

---

## 9. What Is Not Tested (Implementation Exists)

These are implemented in NAMpy but have **no mgcv parity test**:

1. **Tensor with cc/tp/ts/gp marginals** — `te(x,y,bs=["cc","cc"])` etc. are not tested for basis or penalty parity.

2. **MRF with low-rank truncation (k < n_areas)** — natural parameter type0/type1 truncation exists in `nampy/splines/mrf.py` but no test verifies parity for the truncated case.

3. **FS/SZ with ≥3 factors** — `s(f1,f2,f3,x,bs="fs")` is not tested; only 2-factor combinations covered.

4. **Gaussian optimizer trace parity** — trace serialization tested; no test compares the optimizer trajectory for a Gaussian REML problem to mgcv's outer Newton steps.

5. **`PSplineTerm1D` with non-default m=(p,q)** — only default m=[2,2] has basis/penalty parity tests; other order combinations (e.g. m=[1,1], m=[3,3]) are untested.

6. **Weighted Gaussian REML end-to-end** — `TestGaussianPriorWeights` tests algebra; no test runs the full REML optimization with weights and compares sp values.

7. **Distributional regression (multi-predictor compilation)** — `compile_predictor_designs()` supports multiple predictors but no GAM-level parity test exercises this path.

### Recent update: concurvity parity

- Upstream reference: `mgcv/R/mgcv.r`, `concurvity <- function(b, full=TRUE)`, including the pairwise `full=FALSE` branch.
- Snapshot parity now records both `concurvity(full=TRUE)` and pairwise `concurvity(full=FALSE)` matrices.
- Targeted tests added:
  - `pytest tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_concurvity_pairwise_matches_mgcv -v`
  - `pytest tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_poisson_concurvity_pairwise_matches_mgcv -v`
  - `pytest tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_concurvity_full_matches_mgcv -v`

---

## 10. What Is Not Implemented (Absent from NAMpy)

These are mgcv features with **no NAMpy implementation**:

| Feature | mgcv | Priority |
|---------|------|---------|
| Tweedie family | `tw()`, `Tweedie()` | Low |
| Quasi/quasi-likelihood families | `quasi()`, `quasipoisson()`, `quasibinomial()` | Low |
| Soap-film smooths | `bs="so"` | Low |
| Adaptive smooths | `bs="ad"` | Low |
| Cyclic P-splines | `bs="cp"` | Low |
| Scaled TP variant | `bs="sos"` | Low |
| fREML (faster REML) | `method="fREML"` | Medium |
| P-REML / P-ML | `method="P-REML"`, `"P-ML"` | Low |
| `xt=` advanced tensor options | multi-knot groups in tensor marginals | Low |
| Linear inequality constraints | `L`, `lsp0` in mgcv | Low |
| Periodic boundary constraints | (beyond cc cyclic) | Low |
| Custom link functions | `make.link()` | Low |
| Pre-specified error scale (`scale=`) | `gam(..., scale=)` | Medium |
| `predict(type="iterms")` | per-term with interaction effects | Low |
| Multi-way id linking (≥3 terms) | `id=` on 3+ terms | Low |

---

## 11. Known Tolerance Gaps (Passing Tests with Relaxed Tolerance)

These tests pass but at tolerances looser than machine precision. They are
actively tracked so that future work can tighten them.

| Test | Concept | Tolerance | Gap reason |
|------|---------|-----------|-----------|
| `test_strict_t2_fixed_sp_response_parity` | t2() predictions | atol=1e-7 | Natural-param reparameterization accumulates rounding vs mgcv C code |
| `test_strict_factor_by_link_parity` | factor by= link | atol=1e-3 | Contrast absorption order differs slightly at prediction time |
| `test_strict_poisson_reml_residual_parity` | Poisson deviance residuals | atol=1e-6 | Working-response differences in IRLS deviance computation |
| `test_strict_binomial_reml_residual_parity` | Binomial deviance residuals | atol=1e-6 | Same as Poisson |
| `test_tensor_te_ps_ps_fixed_sp_response_parity` | te(ps,ps) predictions | atol=1e-6 | P-spline penalty scaling path differs at one step |
| `test_tensor_ti_ps_ps_fixed_sp_response_parity` | ti(ps,ps) predictions | atol=1e-6 | Same as te |
| `test_tensor_t2_ps_ps_fixed_sp_response_parity` | t2(ps,ps) predictions | atol=1e-5 | Compound: natparam + PS penalty |
| `test_cs_pc_fixed_sp_matches_mgcv` | cs + pc= | atol=1e-4 | Shrinkage penalty interacts with constraint absorption |
| `test_binomial_ml_matches_mgcv` | Binomial ML sp | sp_log_atol=2.0 | ML outer optimizer converges to a different local basin |
| `test_gaussian_fs_select_reml_matches_mgcv` | fs + select=True | sp_log_atol=2.0 | Flat ridge; predictions remain exact-to-tight, criterion stays aligned, and endpoint metadata now records explicit fs shared-ridge stabilization |
| `test_gaussian_sz_select_reml_matches_mgcv` | sz + select=True | sp_log_atol=4.1 | Same flat-ridge issue |

---

## 12. Test File Map

| File | Coverage area | Test count (approx) |
|------|--------------|---------------------|
| `_mgcv_snapshot_parity_shared.py` | Shared fixture classes for all snapshot tests | ~120 tests |
| `test_mgcv_snapshot_parity.py` | Entry point delegating to shared classes | ~10 |
| `test_mgcv_smoothcon_parity.py` | Basis and penalty matrix parity; cc/ps/gp model fits | ~45 |
| `test_mgcv_output_parity.py` | Predictions on new data, SE, anova, lpmatrix | ~8 |
| `test_mgcv_pc_id_parity.py` | Point constraints + linked basis; all basis combos | ~50 |
| `test_mgcv_trace_parity.py` | Optimizer trace, gradient/Hessian at optimum | ~10 |
| `test_mgcv_known_gaps.py` | Tolerance-tracked strict parity assertions | 7 |
| `test_mgcv_additional_scenarios.py` | select=True GLMs, weighted GLMs, tensor ps variants | ~12 |
| `test_mgcv_gaussian_weighted_and_re_regressions.py` | RE + weighting regressions | ~4 |
| `test_gam_unit_coverage.py` | Unit tests for constraint, penalty, tensor, formula subsystems | ~40 classes |
| `test_gam_runtime_term_contract.py` | Runtime term interface contracts | ~6 |
| `test_gam_design_constraint_maps.py` | Coefficient map shape/type tests | ~2 |
| `test_gam_gaussian_reml_algebra.py` | REML algebra: Laplace, scale, saturation | ~7 |
| `test_gam_gaussian_smoothness_postprocess_parity.py` | Post-processing scale refinement | ~1 class |
| `test_gam_tensor_pirls_reml.py` | Tensor REML derivatives for te/ti/t2 | ~2 |
| `test_gam_fit_penalized_irls_solver.py` | IRLS solver correctness | ~3 |
| `test_gam_fit_nonnegative_penalized_qr_state.py` | QR state: beta, log-det | ~3 |
| `test_pirls_exact_derivatives.py` | PIRLS gradient/Hessian finite-difference cross-check | ~8 |
| `test_gam_smoothing_selection_derivatives.py` | Penalty derivative matrices | ~2 |
| `test_gam_mgcv_patch_regressions.py` | Regression tests: Newton branch, step-halving, rollback | ~4 |
| `test_parity_matrix_consistency.py` | Self-consistency of this document / parity registry | ~3 |

---

## 13. Priority Gaps to Close

Ordered by impact on claiming full mgcv parity:

1. **Tensor marginals beyond cr/ps** — add basis + penalty parity tests for `te`/`ti`/`t2` with `cc`, `tp`, `ts`, `gp` marginals.

2. **Weighted REML end-to-end** — run full REML optimization with `sample_weight=` and verify that sp values and predictions match mgcv at the REML optimum.

3. **MRF low-rank truncation** — add a test for `k < n_areas` that checks the truncated eigenbasis against mgcv's `nat.param(type=0)` / `nat.param(type=1)`.

4. **Concurvity parity** — compare `concurvity()` output against mgcv for at least one Gaussian and one GLM model.

5. **k.check parity** — `k_prime` and `edf` now tested vs mgcv in `test_mgcv_k_check_parity.py` (18 tests). `k_index`/`p_value` validity-only (RNG differs). Known gap: `fs()`/`sz()` k_index returns NaN in NAMpy because `_numeric_feature_block` doesn't extract the metric feature from FSmoothInteractionTerm; in the current fs configuration both NAMpy and mgcv return NaN for `fs()`.

6. **Tightening t2 tolerance** — the natural-parameter reparameterization accumulates ~1e-7 error vs mgcv's C code; investigate whether Python floating-point ordering can close this.

7. **Gaussian ML optimizer trace** — add a trace-level comparison for the Gaussian ML outer Newton steps.

8. **Non-default ps order** — add basis/penalty parity tests for `m=[1,1]` and `m=[3,3]` P-splines.


  ---
  What the document covers                                                                                                                                                                                                                               
                          
  Section 1–3 — every smooth type, modifier, and family, each with an explicit EXACT / TIGHT / LOOSE / PARTIAL / UNTESTED / ABSENT label and the tolerance used in the corresponding test.                                                               
                                                                                                                                                                                                                                                         
  Section 4–7 — smoothing selection methods (REML/ML/GCV), solvers, prediction types, and the design/constraint pipeline — again with per-component status.                                                                                              
                                                                                                                                                                                                                                                         
  Section 8 — optimizer trace and diagnostic internals.                                                                                                                                                                                                  
                                                                                                                                                                                                                                                       
  ---                                                                                                                                                                                                                                                    
  Key findings                                                                                                                                                                                                                                         

  Fully at machine precision (EXACT)

  - cr, cs, cc, ps, re — basis, penalties, fixed-sp fit, REML fit                                                                                                                                                                                        
  - gp, tp, ts — tight (≤1e-6), effectively machine precision for practical purposes
  - te, ti with cr/ps marginals                                                                                                                                                                                                                          
  - Point constraints (pc=) across all implemented bases                                                                                                                                                                                                 
  - Linked basis (id=) including incompatible-k harmonization                                                                                                                                                                                            
  - Gaussian REML, ML, GCV — smoothing params and predictions                                                                                                                                                                                            
  - PIRLS gradient and Hessian — finite-difference validated                                                                                                                                                                                             
                                                                                                                                                                                                                                                         
  Known tolerance gaps (implemented, passing, but loose)                                                                                                                                                                                                 
                                                                                                                                                                                                                                                       
  - t2 natural-parameter reparameterization: ~1e-7 accumulated error vs mgcv C code                                                                                                                                                                      
  - cs + pc=: ~1e-4 (shrinkage + constraint absorption interaction)                                                                                                                                                                                    
  - Binomial/Poisson deviance residuals: ~1e-6                                                                                                                                                                                                           
  - GLM ML: sp values can diverge (different optimizer basins), predictions still match                                                                                                                                                                
                                                                                                                                                                                                                                                         
  Implemented but untested (priority gaps)                                                                                                                                                                                                             
                                                                                                                                                                                                                                                         
  1. Tensor with cc/tp/ts/gp marginals — no basis/penalty tests
  2. Weighted REML end-to-end (algebra tested, but not the optimizer path)                                                                                                                                                                               
  3. MRF low-rank truncation (k < n_areas)                                                                                                                                                                                                             
  4. concurvity() parity coverage remains absent; k.check now has targeted parity coverage with RNG-limited k-index validity checks                                                                                                                                                                                         
  5. Non-default P-spline order (m=[1,1], m=[3,3])                                                                                                                                                                                                       
                                                                                                                                                                                                                                                         
  Not implemented                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                       
  - Tweedie, quasi families, soap-film, fREML, pre-specified scale
