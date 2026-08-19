# GAM subsystem backlog

Updated: 2026-08-18. This file contains only unresolved or deliberately
deferred work for `nampy/gam/`. Completed investigations belong in targeted
regression tests, retained `debug/` probes when a test is impractical, or the
implemented/unsupported inventories.

## Priority 0: verification

1. Run the configured parity jobs on Python 3.11 and 3.12 across Linux, macOS,
   and Windows, using the vendored `mgcv` package rather than untracked caches.
2. Confirm the declared repeated-eigenspace and flat-boundary invariants on all
   hosted numerical stacks. Do not solve platform differences with native
   LAPACK bindings, driver selection, or sign forcing.
3. Keep `tests/regressions/test_gam_test_suite_contracts.py` and
   `tests/SUBSYSTEM_COVERAGE.md` synchronized whenever a supported leaf or
   explicit guard is added.

## Priority 1: close supported-surface gaps

These are coverage gaps within behavior that is already implemented. They are
not requests to expand the public surface.

1. Decide whether the undocumented `spline_1d` and `tensor` registry aliases are
   intentional compatibility surfaces. Remove them if stale; otherwise add
   small alias-equivalence contracts.

Use the smallest owner-level or parity test for each item. Do not create broad
snapshot duplication solely to increase scenario counts.

## Priority 2: guarded behavior worth porting

Each item requires an upstream-source port and a targeted parity test before
the guard can be removed.

1. Parametric-only formulas with `optimize_smoothing=True`. There are no smooth
   parameters to optimize today, so the path raises explicitly; supporting it
   would allow `mgcv`-style model-comparison chains starting from a purely
   parametric model.
2. Shared linear-predictor components in formula lists (`1 + 2 ~ ...`). `mgcv`
   shares one coefficient block; cloning independent blocks is not acceptable.
3. The non-unit-prior-weight convention in `binomial()$aic`, where upstream
   treats weights as trial counts.
4. Exact supported behavior for currently guarded `optim` combinations,
   especially estimated-theta negative-binomial ML at a flat L-BFGS-B boundary.
5. General-family `terms=`/`exclude=` filtering, including correct
   coefficient-block selection for multi-predictor models.
6. Prediction arguments with meaningful upstream semantics: `block.size`,
   `newdata.guaranteed`, `na.action`, `unconditional`, and `iterms.type`.
## Priority 3: optional surface expansion

These are product decisions, not assumed commitments:

- `t2()` and additional smooth constructors;
- additional extended/general families;
- matrix covariates, linear-functional terms, and `paraPen`;
- known-scale `scale=` workflows, GACV, P-ML/P-REML, and NCV/QNCV;
- `vis.gam` and derivative plots;
- high-level `gam`/`bam`/`gamm`/`jagam` entry points.

The complete guard and absence list is maintained in
[GAM_NOT_IMPLEMENTED.md](GAM_NOT_IMPLEMENTED.md). Do not partially expose one
of these surfaces through a heuristic fallback.

## API decisions

- Resolved: `GAM` is a stable package export (`from nampy.gam import GAM`),
  wrapped by the `nampy.models.GAMRegressor`/`GAMClassifier` adapters, and
  persistence is a classmethod `load_model` matching the shared
  `nampy.api.PersistableModel` contract.

## Cross-backend follow-ups

- GAMLSS sklearn adapter (gaulss/gammals): multi-column eta prediction and a
  score contract mirroring the neural LSS negative-mean-NLL; do in one
  dedicated change.
- `sample_weight` support in the neural training stack (dataset/datamodule/
  TaskModule loss), then in the `GAMResidual*` composers.
- `GAMResidualRegressor`: gamma family (needs a gamma NLL loss for the
  neural stage plus a dispersion story).
- Hybrid LSS composition (per-parameter offsets) — out of scope for the
  current hybrid backends.
- `GAMNetClassifier`: multiclass targets (`CompiledGAMTermsModule` already
  generalizes over `num_classes`); a lam grid-search helper (currently
  `gam_source=` REML lift is the documented path).
