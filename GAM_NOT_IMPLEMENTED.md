# GAM subsystem — not implemented / known deviations

Snapshot date: 2026-08-21. Companion to [GAM_IMPLEMENTED.md](GAM_IMPLEMENTED.md).
Policy: every unsupported surface must fail loudly (explicit
`NotImplementedError`/`ValueError`), never silently approximate. Items marked
**guarded** raise an explicit error today; items marked **absent** fail
through a generic guard (unknown basis/function/method).

## Smooth constructors

| Surface | Status |
| --- | --- |
| `t2()` tensor smooths | guarded: `t2(...) tensor product smooths are not supported; use te(...) or ti(...).` |
| `pc=` on `te`/`ti` (accepted by mgcv) and on `re`/`fs`/`sz` | guarded: `pc= is not supported for ...` |
| `bs=` `cp`, `bs`, `ds`, `gp`, `ad`, `mrf`, `sos`, soap (`so`/`sf`/`sw`), `sc`/`scad` | absent (generic `Unsupported s() basis` guard). The orphaned `gp` primitives were removed from `nampy/splines/`. |
| Tensor marginals outside `cr/cs/cc/ps/tp/ts` | guarded |
| `bs="fs"` with full-rank shrinkage base (`cs`/`ts`) | guarded (upstream mgcv rejects it too) |
| Random-effect smooths with `id=` | guarded with mgcv's own message (`random effects don't work with ids.`) |
| Matrix covariates / `by`-matrices | implemented through the shared linear-functional contract for ordinary `ps`/`cr`/`cs`/`cc` and SCAM's eight univariate `*By` shape bases; absent for other smooth classes |
| User-supplied constraint matrix `C` / `absorb.cons=FALSE` | absent (constructor kwargs are not plumbed) |
| `paraPen=` parametric-term penalties | absent |

## Shape-constrained (`scam`) boundaries

| Surface | Status |
| --- | --- |
| Automatic ML/REML/LAML selection | guarded; SCAM's GCV/UBRE criteria are implemented |
| Outer SP optimizers other than `bfgs_gcv.ubre` | guarded; no substitution with NAMpy's generic `optim`, EFS, or Newton paths |
| Coefficient optimizer other than SCAM Newton | absent |
| More than one linear predictor | fixed smoothing implemented through the generic transformed general-family kernel; automatic smoothing guarded pending transformed Laplace derivatives |
| Bivariate `by` terms | guarded |
| `select=True` null-space penalties on constrained terms | guarded; upstream SCAM does not expose this mgcv surface |
| AR(1) on ordinary GAMs | fixed smoothing and GCV implemented for Gaussian identity models through the shared observation transform |
| AR(1) ML/REML/LAML or non-Gaussian/non-identity models | guarded; correlated-likelihood determinant/derivative terms are not yet implemented |
| Estimated AR(1) correlation | absent; `ar1_rho` is supplied, as in upstream SCAM |
| Derivatives for matrix-valued linear functionals | guarded; scalar SCAM derivatives and scalar ordinary P-spline new-data derivatives are implemented |
| SCAM `plot.scam`, `vis.scam`, `scam.check`, and `qq.scam` specializations | absent; shared GAM plotting/checking remains available where its term contract applies |

## Formula constructs

| Surface | Status |
| --- | --- |
| `cbind(succ, fail)` binomial response | absent (expression-function whitelist) |
| `factor()`, `poly()`, `ordered()`, `cut()`, `interaction()` in formulas | absent (guarded per function name) |
| Ordered parametric factors (R ordered contrasts) | guarded |
| `%in%`, R's `^` operator (write `**`) | absent (parse error) |
| Data-aware `.` shorthand for formula lists | guarded (upstream rejects this too) |
| Shared linear-predictor components (`1 + 2 ~ ...`) | guarded (2026-08-18): mgcv shares ONE coefficient block across the labelled predictors; the former NAMpy expansion cloned terms with independent coefficients — a different model. Parsing keeps `interpret.gam` parity; building raises until coefficient sharing is ported |
| Per-term `select=` inside `s()/te()/ti()` | removed (2026-08-18): mgcv's `s()` has no `select` argument (smooth.r:614); use model-level `select` |
| Constructor `tensor_terms=`/`main_effects=` (non-formula tensor specs) | removed (2026-08-18): tensor terms are formula-only (`te()`/`ti()`); the array path builds one main-effect smooth per column |
| Constructor `side_condition_tol=` | removed (2026-08-18): upstream `gam.side` has no user-facing tolerance either (mgcv.r:1266) |
| Multiple `offset()` terms summing | intentionally NOT summed — upstream `interpret.gam0` keeps only the first offset (verified: `debug/multi_offset_probe.R`); NAMpy mirrors that including the R warning |

## Families absent entirely

`quasi`, `quasipoisson`, `quasibinomial`, `inverse.gaussian`;
extended families `scat`, `ziP`, `cnorm`,
`clog`, `cpois`, `bcg`; general families `multinom`, `ziplss`, `gevlss`,
`twlss`, `gumbls`, `shash`; `cox.ph`/`cox.pht`, `mvn`, `gfam`.
The implemented extended-family surface includes `nb`/`negbin`, `betar`,
`ocat`, and `tw`; `gaulss` and `gammals` are the implemented general families.
See [GAM_IMPLEMENTED.md](GAM_IMPLEMENTED.md) for their exact fitting routes.

## Criteria / optimizers

| Surface | Status |
| --- | --- |
| GACV (`GACV.Cp`) | absent; `gacv.cp` is rejected (no silent aliasing to GCV) |
| P-ML / P-REML as selectable criteria | absent (string literals appear only in the ML/REML-like score classification used by the upstream-mirrored optimizer convergence rules) |
| NCV / QNCV (`nei=`) | absent; the `nei=` parameter was removed |
| `nlm` optimizer | guarded |
| `magic` / performance iteration as a distinct optimizer identity | absent — Gaussian+GCV routes through outer optimization, so the reported optimizer name never equals `"magic"` |
| `scale=` argument (known-scale Gaussian/Gamma UBRE/Cp workflow) | absent — UBRE/AIC is blocked whenever `known_scale is None` |
| Parametric-only formulas with `optimize_smoothing=True` | guarded — the current smoothing-selection driver requires at least one smooth parameter; fixed fitting remains available |
| `optim` exact parity | partial — SciPy L-BFGS-B stands in for R `stats::optim`; the negbin estimated-theta **ML + optim** combination is guarded until the exact R L-BFGS-B flat-boundary behavior is ported |
| Tweedie (`tw`) joint ML/REML outer optimization | implemented for `outer_newton`, `bfgs`, and `optim` in mgcv's `[theta, log(sp), log(scale)]` order; `min_sp` and `efs` remain guarded |
| LAML for GLM/extended families | absent (flag off); only general families accept it, folded into REML as upstream |

## General-family (`gam.fit5`) scope

| Surface | Status |
| --- | --- |
| Multi-predictor `terms=`/`exclude=` prediction filters | guarded until coefficient-block selection mirrors `predict.gam` |
| `type="iterms"` for multi-predictor models | downgraded to `terms` with mgcv's warning (mirrors upstream) |
| Non-reparameterized single-/multi-penalty `Sl` blocks | guarded |
| Nonlinear (`linear=FALSE`) `Sl` blocks | **extension-only by decision (2026-08-14)**: the machinery mirrors upstream's adaptive-smooth `Sl` structure and is exercised by `tests/families/test_mgcv_gamlss_nonlinear_sl.py`, but no in-tree smooth constructor emits the `general_family_nonlinear_sl` metadata key, so no real fit reaches it; reparameterized nonlinear blocks are guarded |
| Multi-smooth general-family wrapped-block parity, `sp.vcov`, inference, diagnostics | declared out of the supported surface (see `tests/SUBSYSTEM_COVERAGE.md`) |

## Prediction arguments not ported

`block.size`, `newdata.guaranteed`, `na.action` (NAMpy rejects NaN instead of
`na.pass`), `unconditional=` as a predict kwarg (reachable via
`cov=model.vcov(unconditional=True)`), `iterms.type`. `se.fit` is spelled
`return_se`.

## Public-surface shape differences (not mgcv ports)

- `plot()` is a `plot.gam` port as of 2026-08-18 (see `GAM_IMPLEMENTED.md`):
  the data phase (grids, fits, CIs incl. `seWithMean`, partial residuals,
  too-far exclusion, re/fs/sz methods) is parity-tested against `plot.gam`'s
  returned data; rendering is matplotlib, so pure R graphics state (character
  expansion, contour-legend layout, device asking) is not mirrored. `vis.gam`
  and `deriv=TRUE` plots are still absent. (`summary()` IS a `summary.gam`
  port as of 2026-08-15; only the coefficient display names differ from R's
  contrast naming, e.g. `fac[b]` vs `facb`.)
- `gam_check()` returns data (mgcv-comparable vs NAMpy-specific split), no
  plots.
- No `gam()` functional entry point (class-based `GAM` only); no `bam`,
  `gamm`, `jagam`.

## Known numeric deviations (documented, evidence-backed)

- **`cs` repeated-zero eigenspace orientation** — the raw cubic-shrinkage
  penalty can differ slightly because base R and SciPy may choose different
  bases inside the repeated zero eigenspace before the two unequal shrinkage
  eigenvalues are assigned. Constructor and fitted behavior are compared with
  the documented invariant/tolerance; production code must not select a LAPACK
  driver solely to force one platform's raw representation.

- **Flat Poisson smoothing endpoints and `optim` traces** — a single-smooth
  UBRE optimum can differ from `mgcv` at approximately `1e-5` in log-SP on a
  flat objective. L-BFGS-B may also take a small number of extra trailing line
  searches after the smoothing coordinate has reached an effective infinite
  boundary. Tests constrain the common trace, endpoint class, score, and fitted
  behavior instead of requiring a platform-specific evaluation count.

- **Binomial AIC with non-unit prior weights** — the current binomial AIC
  kernel does not reproduce R `binomial()$aic`'s convention of interpreting
  non-unit prior weights as trial counts. Unit-weight AIC is within the
  supported parity surface; the weighted convention remains backlog work.

- **fs null-space penalty assignment order** — upstream assigns one sp per
  `nat.param(type=1)` null column (`mgcv/R/smooth.r:2067-2075`), whose order
  is a descending sort of numerically-zero eigenvalues: mgcv itself flips it
  under a row permutation of the same data
  (`debug/fs_null_order_stability_probe.py`), while the null directions are
  identical between NAMpy and mgcv. The lifecycle harness declares the block
  exchangeable and canonicalizes both sides by final log-sp before an
  otherwise-strict comparison (trace, endpoint, Vp/Ve, EDF, scale strict);
  Vc/edf2/AIC inherit the branch (NAMpy equals mgcv's row-permuted branch to
  7 digits) and are excluded for the affected case with the evidence in the
  registry comment.
  General-family fixed-`fs` parity therefore uses a common value for these
  exchangeable null penalties; optimized `gaulss`+`fs` behavior passes without
  comparing their raw order.

- **`gaulss(select=True)` optimized endpoint invariant** — passing, not an
  xfail. After correcting the multi-penalty `Sl.setup` triangle convention, the
  retained probe gives second log-sp `11.81049973` versus mgcv `11.91107097`.
  Both are above 10 on the same saturated high-penalty tail: the ML criteria
  differ by `3.98e-6`, both gradients are below `4.2e-5`, and the identified
  log-SP plus conditional covariance, EDF, trace, scale, AIC, predictions, and
  SEs pass. Unconditional covariance and EDF2 pass strictly when evaluated at
  mgcv's exact endpoint. Upstream `estimate.gam` transforms `G$X` with
  arbitrary-sign symmetric-eigen vectors but passes unreparameterized `G$Eb`
  into `initial.spg`, making the raw tail coordinate depend on a legal `DSYEVR`
  sign choice. This must not be hidden by a heuristic, sign canonicalizer, or
  platform-specific solver hook.

  The former `gammals_select_true_cr` endpoint, prediction, and EDF2 xfails are
  fixed. They were caused by `_sl_multi_penalty_block` using the upper triangle
  where `mgcv/R/fast-REML.r::Sl.setup` uses the lower-triangle symmetric-eigen
  convention. Gammals initialization, final smoothing parameters, predictions,
  SEs, and fit5 post-processing now pass ordinary mgcv strictly.
