TL;DR — NAMpy status (2026-08-17, branch `mgcv`, implementation snapshot
through `e750f6d`)

The GAM subsystem is a strict-parity port of mgcv 1.9-4 whose supported
surface is documented in GAM_IMPLEMENTED.md / GAM_NOT_IMPLEMENTED.md and
whose verification ledger lives in PROJECT_STATUS.md. The independent audit
in REVIEW.md has been reconciled: each finding is marked fixed, verified
benign, or tracked open (see its status table).

What is solid (verified against live R at strict tolerances)

- Formula/constructor surface: s/te/ti over cr/cs/cc/ps/tp/ts/re/fs/sz with
  by/id/pc/fx/sp/m/xt/d/mc/knots/offsets, formula lists, mgcv-exact defaults
  (multivariate tp k defers to the constructor rule) and loud guards for
  everything unsupported (t2, pc outside univariate bases, unknown GAM
  kwargs, GACV, ...).
- Fitting: gam.fit3 PIRLS with the exact gdi1 port, gaussian stacked-QR,
  gam.fit5 general-family Newton; outer newton/bfgs/efs strict mirrors;
  joint scale/theta optimization for Gaussian (incl. log/inverse links),
  Gamma, and estimated-theta negbin (offset/weights/array inputs included);
  31 strict lifecycle branch cases.
- Rank deficiency now follows upstream's gauge: aliased designs give zero
  coefficient AND zero Vp row at mgcv's dropped canonical coordinate;
  PIRLS rank tolerance is upstream's eps*100.
- Post-fit: Vp/Ve/Vc with mgcv's optimizer-dependent availability, sp_vcov,
  gam_vcomp, anova/testStat with Davies mixtures, residuals, concurvity,
  k_check, prediction (link/response/terms/iterms/lpmatrix + SEs +
  terms=/exclude=), logLik/AIC/BIC (parity-tested for poisson, negbin,
  gaulss), and a full summary.gam port (GAMSummary + print layout +
  null.deviance for all family classes) at machine-precision parity.
- The only intended GAM xfail is now `gaulss_select_true_cr`. The former
  gammals endpoint/prediction xfails were a real `Sl.setup` triangle-convention
  bug and are fixed; gammals select=True initialization, optimized fit,
  predictions, SEs, and fit5 post-processing all pass strictly. The remaining
  gaulss start is upstream-sign-indeterminate: `estimate.gam` reparameterizes
  `G$X` with arbitrary-sign `DSYEVR` eigenvectors while passing an
  unreparameterized `G$Eb` to `initial.spg`. Matching one R build would require
  the platform/sign forcing that this project deliberately excludes.
- Neural first validation is complete: all five focused files pass (148 tests
  across architecture, task/multi-output, sklearn, SplineNAM, and public
  estimator fit/predict contracts).

Resolved after the audit

1. fs null-space ordering was proved exchangeable inside an mgcv-internal
   repeated eigenspace and is now compared by the declared invariant.
2. cs shrinkage now directly ports `(S+t(S))/2` and upstream's two ordered
   eigenvalue replacements. Ordinary cs output passes. Residual transformed
   and Gaussian-GCV differences are confined to R/SciPy orientation of the
   repeated zero eigenspace; no platform hook or heuristic was added.
3. `gam.side` no longer deletes aliased parametric columns, and the
   single-smooth EDF fallback heuristic was removed.
4. Near-singular Gaussian REML now follows `gdiPK`/`gdi1` QR-based deviance
   Hessian construction and `getXtMX` accumulation; the exact random-effect
   regression passes.
5. The fs prediction contribution heuristic, condition/model-selected Gaussian
   backend, dormant null-space/lstsq controls, and silent derivative/post-fit
   exception fallbacks were removed. Supported paths now follow the cited
   `predict.gam`, `gam.fit3`, `pls_fit1`, `gdiPK`, and `gdi1` state directly.
6. PIRLS now iterates in `gam.fit3`'s exact current-SP `T`/`St`/`Sr`/`Eb`
   coordinates. The Poisson-identity forced-Fisher and tolerance overrides were
   removed; noncanonical links use full Newton with only the upstream local
   indefinite-system Fisher retry.
7. Stacked QR no longer contains raw `ctypes`, LAPACK work-buffer/`JPVT`, or
   BLAS-accumulation plumbing. It uses SciPy's supported pivoted-QR interface
   while retaining the upstream `pls_fit1`/`gdiPK` behavioral algorithm.
8. The production package passes an AST guard against direct native numerical
   bindings and explicit solver-driver selection. Focused portability slices
   are configured on Linux, macOS, and Windows.
9. `gammals(select=True)` now matches ordinary mgcv directly. The multi-penalty
   `Sl.setup` path uses upstream's lower-triangle symmetric-eigen convention;
   final smoothing parameters match and optimized prediction differences are
   at most `3.9e-9`. No gammals xfail remains.

Remaining decisions and release work

1. Obtain the configured hosted Linux/macOS/Windows portability results. The
   guard, CI job, and retained evidence are committed; only the local Linux
   slices have been executed so far.
2. The manual built-wheel install/import smoke passes: the wheel installed into
   a temporary venv, imported `nampy` and mandatory `pretab 0.0.3`, instantiated
   `LinRegRegressor`, and exposed exactly the three-symbol GAM API from the
   installed artifact. Automating the same artifact check in CI remains optional
   release hardening.
3. Add multi-output fitting coverage beyond LinReg only if every public neural
   regressor is intended to guarantee it. The user-facing GAM support matrix,
   estimator parameter/clone contract, current multi-output evidence boundary,
   and SplineNAM preprocessing requirements are now documented in `README.md`.

The public-export conflict is resolved: `nampy.gam` exposes only
`fit_model_core`, `solve_fit`, and `FitCoreSolution`. The non-public `GAM`
implementation remains available to internal parity tests at
`nampy.gam.model.api`.

Deliberately out of scope: optim exact R L-BFGS-B behavior (negbin ML+optim
guarded), general-family term filters and formula-list dot shorthand, plot.gam
port, absent bases/families (documented), and BLAS/LAPACK orientation inside
indeterminate eigenspaces. These are not correctness defects in the declared
surface and must not acquire heuristic fallbacks.
