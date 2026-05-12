Changelog
=========

All notable changes to NAMpy are documented here.

For the complete changelog, see
`CHANGELOG.md <https://github.com/Ananyapam7/NAMpy/blob/main/CHANGELOG.md>`_
on GitHub.

Latest Release
--------------

Unreleased
~~~~~~~~~~

**Changed:**

* Breaking: model ``forward()``, estimator ``_predict()``, and
  ``predict_feature_vals()`` now return nested dictionaries with
  ``prediction``, ``terms``, ``intercept``, ``regularization``, and ``extras``.
* Per-term contributions moved from top-level result keys into ``terms``.
* Training penalties moved from top-level ``*_penalty`` keys into
  ``regularization``.

Version 0.1.0 (2024-01-07)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Added:**

* Initial public release
* 10+ model architectures (NAM, GPNAM, NBM, NATT, NAMformer, etc.)
* Support for regression, classification, and distributional regression
* Scikit-learn compatible API
* Flexible preprocessing options
* Multiple distribution families for LSS models
* Comprehensive test suite
* Full documentation

**Features:**

* Interpretable model architectures
* PyTorch Lightning backend
* Custom model implementation support
* Feature-level interpretability for additive models

Version History
---------------

* **0.1.0** - First stable public release
* **0.0.1** - Initial development version

For More Details
----------------

See the `full changelog <https://github.com/Ananyapam7/NAMpy/blob/main/CHANGELOG.md>`_
on GitHub for complete version history and migration guides.
