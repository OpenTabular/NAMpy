# GAM subsystem — implemented surface

Snapshot date: 2026-08-14. This documents what `nampy/gam/` implements as a
strict-parity port of R's `mgcv` (the vendored sources under `mgcv/` are the
specification). Everything listed here is expected to match `mgcv` to the
tolerances used by the parity suite; the only excluded class of differences is
BLAS/LAPACK eigenvector/basis orientation inside mathematically indeterminate
subspaces. Whatever is *not* listed here is documented in
[GAM_NOT_IMPLEMENTED.md](GAM_NOT_IMPLEMENTED.md).

## Formula surface

- `s()`, `te()`, `ti()` smooth specials.
- Arguments: `k` (mgcv rounding warning; omitted `k` on `tp`/`ts` defers to the
  constructor rule `M + c(8, 27, 100)[min(d, 3)]` exactly like
  `smooth.construct.tp.smooth.spec`), `bs`, `by` (numeric, factor, ordered
  factor with dropped first level, transformed numeric expressions
  materialized into hidden columns), `id=` linkage with pooled covariates
  (multi-element `id` warns and uses the first), `pc=` on univariate
  `cc/cr/cs/ps/tp/ts`, scalar `fx` everywhere plus vector `fx` on `te`/`ti`
  (wrong length warns and resets), `sp`, `m` (`ps`/`tp`/`ts`; silently ignored
  elsewhere as upstream does), `xt` (`tp`/`ts`/`re`/`fs`/`sz`), `d=` tensor
  marginal dimensions with mgcv's coercion of multivariate `cr/cs/ps/cp`
  marginals to `tp`, `mc=` for `ti`, per-term and model-level `select`.
- `offset()` with mgcv's remember-for-prediction semantics. Multiple
  `offset()` terms in one formula keep only the first offset, matching
  `interpret.gam0`'s single-slot assignment, including R's
  "number of items to replace…" warning (verified against mgcv 1.9-4 in
  `debug/multi_offset_probe.R`).
- Formula lists for multi-predictor (gamlss) models with `lpi` numeric labels;
  `.` shorthand for single formulas; transformed responses; interactions
  `:`/`*`/`/`/`**` and intercept control `+1/-1/+0`; factor parametric terms
  with treatment contrasts; `I()` and `abs/sqrt/exp/log/sin/cos/tan`
  transforms; R literals `c()`, `list()`, `matrix()`, `diag()`, `TRUE/FALSE`.
- `knots=`, prior weights (`sample_weight`), fit-time `offset=` (deliberately
  not remembered for prediction, as in mgcv), `min_sp`, `drop_intercept`.

## Smooth bases

- Univariate: `cr`, `cs`, `cc`, `ps`, `tp` (full Lanczos/TPRS port including
  `xt=list(max.knots=, seed=)`), `ts`.
- Categorical: `re` (numeric/factor/n-way interactions, R column ordering),
  `fs` (base `cr/cc/ps/tp`; single factor; singly penalized base) and `sz`
  (base `cr/cs/cc/ps/tp/ts`; multiple factors via Kronecker sum-to-zero
  contrasts). Full-rank shrinkage bases under `fs` are rejected exactly as
  upstream rejects them.
- Tensors: `te` and `ti` over `cr/cs/cc/ps/tp/ts` marginals with mgcv's
  construction order (raw penalty → `np=TRUE` reparameterization →
  eigen-normalization → Kronecker → outer penalty rescale) and multi-feature
  `tp`/`ts` marginals.

## Constraints and identifiability

Sum-to-zero constraints, QR constraint absorption byte-matching base R's
`qr`/`qr.qty`, localized absorption, `pc=` point constraints, the `gam.side`
analogue with `augment.smX` penalty-aware dependence testing, exemption
policies for random-effect/factor smooths, zero-width term retention, and
per-term `select=TRUE` null-space penalties.

## Families

- `gaussian` (identity, log, inverse), `binomial` (logit, probit, cloglog,
  cauchit, log), `poisson` (log, identity, sqrt), `gamma` (log, identity,
  inverse).
- Negative binomial with mgcv's split: `{"name": "nb"}` /
  `{"name": "negbin", "estimate_theta": True}` estimates theta (EFS update
  inside PIRLS per `gam.fit4`, joint outer `(log theta, log sp)` optimization
  for ML/REML), `{"name": "negbin", "theta": ...}` is fixed-theta.
  Offsets, prior weights, and array (non-formula) construction are supported
  on the estimated-theta joint path.
- General families: `gaulss`, `gammals` with the `gamlss.etamu` / `gamlss.gH`
  / `trind.generator` ports.

## Smoothing criteria

- GCV and UBRE/Cp/AIC with exact `gdi1` derivatives (`mgcv/src/gdi.c` port:
  `multSk`, `applyP/applyPt`, `ift1`, `get_bSb`, operation order preserved).
- ML and REML on four backends: exact Gaussian (closed form), dynamic
  Gaussian, PIRLS-Laplace (exact `gam.fit3`-mirrored values, gradients, and
  Hessians for binomial/poisson/negbin, plus Gamma and Gaussian through the
  profiled-scale branch bordered exactly as `gam.fit3.r:628-637`), and the
  general-family `gam.fit5` route.
- LAML for general families, folded into the REML branch exactly as upstream.
- Joint outer optimization of extra parameters: Gaussian scale (identity and
  noncanonical links; `log(scale)` appended as the trailing coordinate exactly
  as `mgcv.r:2025-2037`), Gamma scale, and negbin theta.
- `initial.spg` port including the general-family `pen.reg` branch; the
  design-balance heuristic fallback is deliberately removed (strict error).
- `"gcv.cp"` is normalized as upstream `GCV.Cp` (GCV for unknown scale,
  UBRE/AIC for known scale, REML for extended families).

## Optimizers

- `outer_newton` — strict mirror of `gam.fit3.r::newton()` including
  edge-correction (`hess1`/`db.drho1`) support.
- `bfgs` — strict mirror of `mgcv::bfgs()` (including the inverted-Hessian
  approximation used for post-fit covariance).
- `efs` — mirror of `efsudr`/`efsud` (forces REML as upstream does).
- `optim` — SciPy L-BFGS-B standing in for R `stats::optim`; same call
  structure, documented as partial parity (see NOT_IMPLEMENTED).
- `lbfgsb` — NAMpy-only extra; auto-promoted to `outer_newton` for
  ML/REML/LAML when exact Hessians exist.

## Solvers

- `gam.fit3` PIRLS with the exact pre-refresh `eta`/`mu` state boundary around
  `gdiPK`, signed/negative-weight stacked QR (`pls_fit1` mirror with mgcv's
  `rank.tol = eps*100`, pivoted-QR column dropping, and zero-fill restore),
  and `gam.fit4` extended-family hooks (theta EFS).
- Gaussian exact/stacked-QR backend (the `magic`/`pls_fit1`-style linear
  algebra path).
- `gam.fit5` general-family Newton with reparameterized single- and
  multi-penalty `Sl` blocks, `ldetS`, `Sl.repara`, and `gam.fit5.post.proc`
  (including the deriv=0 behavior for efs/optim: no smoothing-uncertainty
  correction, `Vc = Vb`).

## summary.gam (added 2026-08-15)

`GAM.summary(*, dispersion=None, freq=False, re_test=True)` is a port of
`mgcv::summary.gam` (`mgcv/R/mgcv.r:3858-4068`): it prints the
`print.summary.gam` layout and returns a structured `GAMSummary` object.
Per-coefficient `p.table` (t/z branching on estimated scale, `dispersion=`
rescaling, `freq=` covariance switch), the parametric-term and smooth
significance tables (the existing `testStat`/`reTest` port), adjusted
r-squared (suppressed for general families), deviance explained via a full
`null.deviance` port (GLM closed form `gam.fit3.r:838-842`, offset-corrected
intercept-only refit `mgcv.r:2072-2075`, negbin `find.null.dev`
`efam.r:98-117`, gaulss/gammals postproc hooks `gamlss.r:910/2737`),
`-REML`/`-ML` score line, and the `Scale est. / n` trailer. Verified to
machine precision against live `summary.gam` for gaussian (plain + offset),
poisson, gamma joint-scale, negbin estimated-theta, and gaulss
(`tests/parity/test_mgcv_summary_parity.py`). Coefficient display names are
synthesized from term labels and intentionally differ from R's contrast
naming.

## Post-fit and inference

- Vp (Bayesian), Ve (frequentist), and the unconditional Vc with the full
  `Vb.corr` machinery, edge-correction second pass, and the joint-scale extra
  `L` row; mgcv's optimizer-dependent availability is mirrored exactly
  (GLM/extended + efs/optim → `Vc`/`edf2`/`V.sp` absent per `gam.fit3.r:1053`;
  general families always carry `Vc`, degenerating to `Vb`).
- `sp_vcov` (with mgcv's edge-correct regularization asymmetry), `gam_vcomp`,
  one-SE rule, optimizer endpoint diagnostics.
- edf/edf1/edf2 (public `GAM.edf1()`; `edf2`, `cov_unconditional`, and
  `cov_unconditional_space` are part of the `GAMFitResult` schema);
  `logLik`/`AIC`/`BIC` mirroring `logLik.gam`, parity-tested for poisson,
  negbin (fixed + estimated theta), and gaulss alongside gaussian.
- Rank-deficiency gauge: exactly aliased designs reproduce mgcv's
  representative (zero coefficient AND zero `Vp` row at the dropped canonical
  coordinate, `mgcv/src/gdi.c:2253-2292`), with the PIRLS inner solve at
  upstream's `rank.tol = eps*100` (`gam.fit3.r:131`).
- Unknown constructor arguments raise `TypeError` (23-key allowlist), so
  unported mgcv arguments (`paraPen=`, `absorb.cons=`, ...) fail loudly.
- `anova()` single-model (full `testStat` port with unpivoted QR, fractional
  rank, Davies/`liu2`/`psum.chisq` chi-square mixtures, and the `reTest`
  branch for full-rank-penalty terms) and multi-model comparison with mgcv's
  guards.
- Residuals: deviance, pearson, scaled.pearson, working, response.
- `concurvity` (full and pairwise, reproducing mgcv's parametric-block
  indexing quirk), `k_check` (with mgcv's deterministic subsample plan and
  tensor rescaling), `gam_check` (split into mgcv-comparable and
  NAMpy-specific sections; no plots).

## Prediction

- Types: `link`, `response`, `terms`, `iterms` (with mgcv's `cmX`
  mean-uncertainty broadcast), `lpmatrix`; standard errors for all of them
  (delta method on `response`).
- `terms=` / `exclude=` filtering for ordinary families with mgcv's
  "non-existent terms requested" warning behavior; `cov=` selection
  (bayes/freq/explicit matrix).
- Formula-driven newdata handling with hidden-column re-materialization,
  per-predictor offsets for general families, and offset defaults mirroring
  `predict.gam` semantics.

## Parity infrastructure

Snapshot/trace serialization comparing coefficients, covariances, criterion
values, optimizer traces (Newton/BFGS/EFS rows, joint scale/theta splits),
inner PIRLS traces, predictions, SEs, residuals, `k.check`, `anova` tables,
`sp.vcov`, `gam.vcomp`, and concurvity against live R `mgcv` runs with cached,
content-hashed references (`tests/mgcv_r_cache/`). The optimization lifecycle
registry pins 31 strict optimizer branch cases (including weighted negbin
estimated-theta and Gaussian log/inverse joint-scale branches).
