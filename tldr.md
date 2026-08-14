TL;DR — nampy/gam status (2026-08-15, branch mgcv @ e8c9b21)

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
- The only intended xfails are the two select=True general-family optimized
  endpoints, both proven to be mgcv-internal initial.spg orientation
  indeterminacy (mirrored basis reproduces NAMpy's endpoint exactly), with
  strict fixed-endpoint coverage.

Known open defects (todo.md P1, in priority order)

1. fs null-space penalty ordering: gaussian_reml_newton_fs_xt_ps lifecycle
   fails (last two log-sp swapped); the fs+select optimized endpoint is also
   off. Registry still marks the case stable — fix or mark first.
2. cs shrinkage parity: three cs cases in test_mgcv_output_parity.py fail at
   ~8.6e-5 against fresh live-R references (previously masked by a stale
   snapshot cache). Plain-cs cause bisected to the prior-session
   symmetrize_lower_triangle=True change in
   nampy/splines/univariate/cr.py::add_full_rank_shrinkage; transformed_cs
   has a second unlocalized cause in the same constructor changes.
3. Side-condition scope: aliased parametric columns are deleted pre-fit
   (upstream gam.side is smooths-only), shifting Mp/sp for such designs.
4. Two policy heuristics to remove/justify: _fallback_single_smooth_edf
   (fit/state.py:296) and the fs prediction shift
   (predict/predictions.py::_fs_term_penalty_adjustment).
5. Two stale unit tests failing in test_optimize_driver_mgcv_parity.py
   (mock/contract drift); gammals select=True optimized-prediction surface
   needs the mirrored-basis verification before tagging; public-exports
   doc/code mismatch needs a decision.

Deliberately out of scope: optim exact R L-BFGS-B behavior (negbin ML+optim
guarded), plot.gam port, absent bases/families (documented), BLAS/LAPACK
orientation inside indeterminate eigenspaces.
