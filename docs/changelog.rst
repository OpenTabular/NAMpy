Changelog
=========

All notable changes to NAMpy are documented here.

For the complete changelog, see
`CHANGELOG.md <https://github.com/OpenTabular/NAMpy/blob/main/CHANGELOG.md>`_
on GitHub.

Latest Release
--------------

Version 0.2.0 (2026-08-22)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Added:**

* Strict ``mgcv``-aligned GAM backend with sklearn-style adapters
* Shape-constrained GAMs and expanded ordinary/general-family support
* Registry-generated NBM, SPAM, NBM-SPAM, SIAN, IGANN, and ensemble estimators
* Shared additive explanations, plots, importance tables, and persistence
* Distributional objectives and architecture-specific native training routes

**Changed:**

* Neural architectures now live under ``nampy.neural.architectures``
* Neural preprocessing targets pristine PreTab's public block contract
* LSS family configuration is constructor-owned and sklearn-cloneable
* Public imports are lazy and backend dependencies are split into extras

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

* **0.2.0** - Statistical GAM backend and expanded neural model platform
* **0.1.0** - First stable public release
* **0.0.1** - Initial development version

For More Details
----------------

See the `full changelog <https://github.com/OpenTabular/NAMpy/blob/main/CHANGELOG.md>`_
on GitHub for complete version history and migration guides.
