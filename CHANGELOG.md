# Changelog

All notable changes are recorded here following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `nampy.api`: backend-neutral contracts shared by all backends —
  `FeatureSchema`, `AdditivePrediction`, `Capabilities`, `PersistableModel`.
- `GAMRegressor` / `GAMClassifier`: scikit-learn-style adapters around the
  mgcv-parity `GAM` (zero added numerics; automatic REML by default;
  label-encoded binary classification; `predict_components`,
  `standard_errors`, `lpmatrix`, `capabilities`, pickle persistence).
- `score()` and `__sklearn_tags__` on all estimators without sklearn mixin
  classes: R² for regressors, accuracy for classifiers, negative mean NLL
  for LSS; `is_classifier`/`is_regressor` and `cross_val_score` now work.
- `predict_components()` and `capabilities()` on the neural estimators;
  fitted feature schemas recorded as `schema_`.
- `nampy.plotting`: backend-neutral term-plot renderer extracted from the
  `plot.gam` port; neural estimators gain `plot_terms()`.
- Per-sample link-scale offsets through the neural training stack
  (`fit(..., offset=)`), plus stratified automatic classification splits.
- `nampy.hybrid` (experimental, explicitly non-mgcv): `GAMPlusNeural`
  (frozen GAM baseline + offset-trained neural correction composed on the
  link scale) and `HybridJointRegressor`/`CompiledGAMTerms` (compiled
  mgcv-parity bases and penalties trained jointly with a neural net in
  Torch under fixed smoothing parameters).
- Public classifier, regressor, and distributional-regression exports for the
  supported neural architectures, with matching base-model and config exports.
- Multi-output fitting coverage for every public neural regressor.
- `save_model()` and `load_model()` persistence for neural estimators, including
  fitted preprocessing and model state.
- PEP 561 `py.typed` marker and installed-wheel API smoke coverage.

### Changed

- **Breaking:** the torch backend moved under `nampy/neural/` —
  `nampy.basemodels` → `nampy.neural.modules` (TaskModel:
  `nampy.neural.training`), `nampy.data_utils` → `nampy.neural.data`,
  `nampy.arch_utils` → `nampy.neural.layers` (shared) / `nampy.neural.modules`
  (single-architecture), `nampy.configs` → `nampy.neural.configs`,
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
- The triplicated wrapper fit/predict/plot pipelines are consolidated into
  `nampy.neural.training.engine` and `nampy.models._plotting`; the
  forward-output penalty grammar has a single owner
  (`nampy.neural.training.output_contract`).
- Supported Python versions are now explicitly Python 3.11 and 3.12.
- Package version metadata has one owner: `nampy.__version__` via
  `pyproject.toml` dynamic metadata.
- CI performs live comparisons against the vendored `mgcv`, enforces Ruff and
  mypy, validates wheels, and tests supported Python and operating-system
  matrices.
- Releases use validated artifacts and PyPI trusted publishing.
- API documentation is generated from live exports and built with warnings as
  errors; generated stubs are no longer committed.
- Contributor and release commands follow the repository's targeted-test policy.

### Fixed

- `QNAMBase` now owns `DefaultQNAMConfig` rather than the unrelated NAM config.
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

[Unreleased]: https://github.com/OpenTabular/NAMpy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OpenTabular/NAMpy/releases/tag/v0.1.0
