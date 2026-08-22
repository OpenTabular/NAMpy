# Changelog

All notable changes are recorded here following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Normal mgcv, SCAM, NBM-SPAM, and SIAN parity tests now consume committed,
  versioned static fixtures. R and local upstream source clones are required
  only for explicit fixture refreshes, and normal CI no longer installs them.
- The entire `upstreams/` directory is local-only and ignored. Its tracked
  clone catalogue moved to `scripts/reference_generation/upstreams.json`.

## [0.2.0] - 2026-08-22

### Added

- `nampy.contracts`: backend-neutral `FeatureSchema`, `AdditivePrediction`, and
  `EnsembleAdditivePrediction` records shared by the GAM and neural backends.
- `GAMRegressor` / `GAMClassifier`: scikit-learn-style adapters around the
  mgcv-parity `GAM` (zero added numerics; automatic REML by default;
  label-encoded binary classification; `predict_components`,
  `standard_errors`, `lpmatrix`, and versioned pickle persistence).
- `score()` and `__sklearn_tags__` on all estimators without sklearn mixin
  classes: R² for regressors, accuracy for classifiers, negative mean NLL
  for LSS; `is_classifier`/`is_regressor` and `cross_val_score` now work.
- `predict_components()` on neural estimators, with fitted feature schemas
  recorded as `schema_`.
- `nampy.plotting`: backend-neutral term-plot renderer extracted from the
  `plot.gam` port; neural estimators gain `plot_terms()`.
- Per-sample link-scale offsets through the neural training stack
  (`fit(..., offset=)`), plus stratified automatic classification splits.
- Public classifier, regressor, and distributional-regression exports for the
  supported neural architectures, with matching base-model and config exports.
- Multi-output fitting coverage for every public neural regressor.
- `save_model()` and `load_model()` persistence for neural estimators, including
  fitted preprocessing and model state.
- PEP 561 `py.typed` marker and installed-wheel API smoke coverage.
- Paper-aligned GPNAM random Fourier features, automatic per-feature
  bandwidths, conjugate-gradient ridge fitting, and selected or all-pairs
  GP-NA2M interactions. Fixed-basis diagnostics, reproducible initialization,
  explicit interactions, and batched inference are shared neural contracts.
- IGANN regression and binary classification with the released linear
  initialization, feature-wise ELM boosting, validation truncation, optional
  ABESS-backed IGANN-Sparse selection, additive components, basis metadata,
  and native training history. Multiclass and IGANNLSS reuse the fixed ELM
  architecture through the generic objective engine. ``NeuralEnsemble`` now
  supports aligned bootstrap resampling for generic bagged additive estimators.
- SIAN regression, classification, and distributional regression with
  Archipelago interaction discovery over logical source-feature groups,
  arbitrary-order heredity search, explicit-interaction bypass, the released
  block-masked architecture and optional maximal-update residual network.
  Generic interaction-selection contracts, active-parameter diagnostics,
  lossless block/independent term conversion, and higher-order interaction
  plots are shared across neural architectures.
- NBM, SPAM, and NBM-SPAM estimator families, including dense and sparse NBM
  execution, Conv1D/einsum featurization, n-ary terms, low-rank polynomial
  effects, local term importance, and hybrid neural-basis/polynomial models.
- A declarative neural architecture registry that generates every supported
  regressor, classifier, and LSS estimator family from one capability record.
- A generic `NeuralEnsemble` for independently cloned neural regressors and
  classifiers, including bootstrap fitting and between-member component
  uncertainty.
- SCAM-compatible shape-constrained GAMs: all 24 univariate and 17 bivariate
  SCOP-spline classes, numeric-by and matrix-valued linear-functional terms,
  local/positive/endpoint constraints, dual optimization and prediction
  coefficient spaces, constrained Newton fitting, exact GCV/UBRE gradients
  and BFGS smoothing selection, transformed covariance and inference,
  derivatives, quantile residuals, and Gaussian-identity AR(1) sections with
  standardized residuals.

### Changed

- **Breaking:** LSS response families and distribution-specific keyword
  arguments are estimator constructor parameters (`family` and
  `distributional_kwargs`). LSS estimators now expose the conventional
  `fit(X, y)` contract required by sklearn model selection.
- Public package and estimator exports are resolved lazily. Backend dependency
  profiles are available as `nampy[gam]`, `nampy[neural]`, and `nampy[all]`;
  importing the GAM backend no longer initializes Torch or Lightning.
- **Breaking (naming sweep, no aliases):**
  `SklearnBaseRegressor`/`SklearnBaseClassifier`/`SklearnBaseLSS` →
  `NeuralRegressor`/`NeuralClassifier`/`NeuralLSS`
  (`nampy/models/{regressor,classifier,lss}.py`); `TaskModel` →
  `TaskModule` (`nampy/neural/task.py`); estimator
  `QNAM` → `QNAMLSS` and torch module `QNAMBase` → `QNAM`;
  `GAM.predict_feature_vals` → `GAM.predict_terms`; the raw-dict
  `predict_feature_vals` is removed from all estimators —
  `predict_components` (typed `AdditivePrediction`) is the one public
  term-contribution surface, now also implemented for LSS
  (multi-column, additive on the raw parameter scale).
- **Breaking:** the torch backend moved under `nampy/neural/` —
  `nampy.basemodels` → `nampy.neural.architectures`, `nampy.data_utils` →
  `nampy.neural.data`, `nampy.arch_utils` →
  `nampy.neural.architectures.components`, `nampy.configs` →
  `nampy.neural.configs`,
  `nampy.utils.distributions`/`distributional_metrics` →
  `nampy.neural.distributions`. No compatibility shims; public estimator
  class names are unchanged.
- **Breaking (correctness):** the PreTab preprocessor is now fit on training
  rows only — validation data no longer leaks into fitted statistics, so
  fitted neural models change numerically.
- **Breaking:** classifiers label-encode targets: `classes_` is populated and
  `predict` returns original labels instead of raw class indices.
- `ModelCheckpoint` now honours the user's `monitor`/`mode`, and each fit
  writes checkpoints into its own `checkpoint_path/<Class>-<id>/` directory
  instead of overwriting `best_model.ckpt`.
- Unfitted `predict` raises `sklearn.exceptions.NotFittedError` (a
  `ValueError` subclass); inference restores the module's training flag.
- The triplicated wrapper fit/predict/plot pipelines are consolidated in
  `nampy.models._base` and `nampy.models._plotting`; the forward-output key and
  penalty grammar has a single owner in `nampy.neural.contracts`.
- Neural preprocessing now targets pristine PreTab's public block contract.
  Generic preprocessing stays in PreTab; NAMpy owns only model-specific block
  interpretation. Unsupported experimental PreTab surfaces are no longer
  required.
- Supported Python versions are now explicitly Python 3.11 and 3.12.
- Package version metadata has one owner: `nampy.__version__` via
  `pyproject.toml` dynamic metadata.
- CI performs parity comparisons against committed `mgcv` reference results, enforces Ruff and
  mypy, validates wheels, and tests supported Python and operating-system
  matrices.
- Releases use validated artifacts and PyPI trusted publishing.
- API documentation is generated from live exports and built with warnings as
  errors; generated stubs are no longer committed.
- Contributor and release commands follow the repository's targeted-test policy.

### Fixed

- Neural optimizer settings from estimator constructors and `set_params()` now
  control training. Optional fit-time overrides use the same `lr`,
  `lr_patience`, `lr_factor`, and `weight_decay` names.
- QNAM now owns `DefaultQNAMConfig` and is exposed only through its supported
  distributional estimator, `QNAMLSS`.
- Public API documentation and examples now match the implemented model/task
  surface.
- Ruff violations and the existing `nampy` mypy error set were resolved.

### Removed

- Legacy `setup.py` metadata and committed autosummary output.
- Cache-only parity behavior from the hosted test matrix.

## [0.1.0] - 2024-01-07

### Added

- Initial public neural additive-model release with regression,
  classification, distributional regression, preprocessing, and interpretable
  feature contributions.

[Unreleased]: https://github.com/OpenTabular/NAMpy/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/OpenTabular/NAMpy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/OpenTabular/NAMpy/releases/tag/v0.1.0
