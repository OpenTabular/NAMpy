# GAM subsystem — not implemented / known deviations

Snapshot date: 2026-08-14. Companion to [GAM_IMPLEMENTED.md](GAM_IMPLEMENTED.md).
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
| Matrix covariates / `by`-matrices (linear functional terms, signal regression, summation convention) | absent — the data path is DataFrame-column only |
| User-supplied constraint matrix `C` / `absorb.cons=FALSE` | absent (constructor kwargs are not plumbed) |
| `paraPen=` parametric-term penalties | absent |

## Formula constructs

| Surface | Status |
| --- | --- |
| `cbind(succ, fail)` binomial response | absent (expression-function whitelist) |
| `factor()`, `poly()`, `ordered()`, `cut()`, `interaction()` in formulas | absent (guarded per function name) |
| Ordered parametric factors (R ordered contrasts) | guarded |
| `%in%`, R's `^` operator (write `**`) | absent (parse error) |
| Data-aware `.` shorthand for formula lists | guarded (upstream rejects this too) |
| Multiple `offset()` terms summing | intentionally NOT summed — upstream `interpret.gam0` keeps only the first offset (verified: `debug/multi_offset_probe.R`); NAMpy mirrors that including the R warning |

## Families absent entirely

`quasi`, `quasipoisson`, `quasibinomial`, `inverse.gaussian`;
extended families `tw`/`Tweedie`, `betar`, `scat`, `ocat`, `ziP`, `cnorm`,
`clog`, `cpois`, `bcg`; general families `multinom`, `ziplss`, `gevlss`,
`twlss`, `gumbls`, `shash`; `cox.ph`/`cox.pht`, `mvn`, `gfam`.
Only `gaulss` and `gammals` of mgcv's general families and `nb`/`negbin` of
its extended families are ported.

## Criteria / optimizers

| Surface | Status |
| --- | --- |
| GACV (`GACV.Cp`) | absent; `gacv.cp` is rejected (no silent aliasing to GCV) |
| P-ML / P-REML as selectable criteria | absent (string literals appear only in score-scale heuristics) |
| NCV / QNCV (`nei=`) | absent; the `nei=` parameter was removed |
| `nlm` optimizer | guarded |
| `magic` / performance iteration as a distinct optimizer identity | absent — Gaussian+GCV routes through outer optimization, so the reported optimizer name never equals `"magic"` |
| `scale=` argument (known-scale Gaussian/Gamma UBRE/Cp workflow) | absent — UBRE/AIC is blocked whenever `known_scale is None` |
| `optim` exact parity | partial — SciPy L-BFGS-B stands in for R `stats::optim`; the negbin estimated-theta **ML + optim** combination is guarded until the exact R L-BFGS-B flat-boundary behavior is ported |
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

- `plot()` is bespoke matplotlib, not `plot.gam`: no confidence bands,
  `seWithMean`, partial residuals, rug, `scheme`/`pages`/`select`, no
  `vis.gam`. (`summary()` IS a `summary.gam` port as of 2026-08-15 — see
  `GAM_IMPLEMENTED.md`; only the coefficient display names differ from R's
  contrast naming, e.g. `fac[b]` vs `facb`.)
- `gam_check()` returns data (mgcv-comparable vs NAMpy-specific split), no
  plots.
- No `gam()` functional entry point (class-based `GAM` only); no `bam`,
  `gamm`, `jagam`.

## Known numeric deviations (documented, evidence-backed)

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

- **`select=True` general-family optimized endpoints**
  (`gaulss_select_true_cr`, `gammals_select_true_cr`) — the only live xfails.
  Both verified 2026-08-14 (`debug/gaulss_select_initial_spg_probe.py`,
  `debug/gammals_select_edf2_probe.py`): the divergence originates in
  `mgcv::initial.spg()` and the select-penalty endpoint is
  orientation-indeterminate inside mgcv itself — refitting mgcv on the
  mirrored basis (`x -> -x`) reproduces NAMpy's endpoint exactly in both
  cases (gaulss: log sp 11.79338762 vs 11.91107097 with scores agreeing to
  5e-6; gammals: sp 1387.5727 vs 1385.9021 with edf1/edf2 sums reproducing
  NAMpy's 3.690821 to 2.3e-7, criterion difference 7e-8, and NAMpy's endpoint
  the better-converged of the two). Only the endpoint-sensitive
  edf/edf1/edf2 scalars exceed tolerance; post-processing (Vc and edf2
  assembly) is tested strictly at a shared endpoint for both cases and
  matches at 5e-6.
