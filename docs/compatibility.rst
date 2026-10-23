mgcv and scam Compatibility
===========================

NAMpy contains a Python-native statistical GAM engine. ``mgcv`` is its
methodological and compatibility reference; ``scam`` is the reference for
shape-constrained smooths. Fitting and prediction run in Python and do not
require an R runtime.

Compatibility here means that a named workflow is implemented with the same
statistical construction and is checked against versioned upstream quantities
where a comparison exists. It does not mean that every function or numerical
iteration in either R package is reproduced.

Scope at a glance
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Status
     - Meaning
   * - Implemented
     - Available through a documented public NAMpy interface.
   * - Compared
     - Representative outputs are tested against versioned ``mgcv`` 1.9-4 or
       ``scam`` 1.2-22 results with quantity-specific tolerances.
   * - Intentionally different
     - The statistical objective is retained, but Python libraries or public
       conventions produce a documented difference.
   * - Unsupported
     - The combination is rejected explicitly instead of being approximated
       with another fitting route.

Implemented analytical workflow
-------------------------------

The public statistical engine covers:

* formulas, offsets, parametric effects, smooths, tensor products,
  factor-specific terms, random effects, MRFs, and linked penalties;
* thin-plate, cubic, cyclic, P-spline, B-spline, Duchon,
  Gaussian-process, adaptive, spherical, and shape-constrained bases;
* fixed smoothing and supported GCV, UBRE/Cp, ML, REML, P-ML, P-REML,
  LAML, NCV, and QNCV selection routes;
* ordinary, discrete, ordered-categorical, and supported multi-parameter
  response families;
* response, link, term, interaction, and linear-predictor-matrix prediction,
  including supported conditional and unconditional uncertainty;
* summaries, ANOVA, residuals, derivatives, concurvity, basis checks,
  variance components, likelihood criteria, and term plots.

The :ref:`statistical-gam-tools` guide maps these capabilities to public
symbols and runnable calls.

What is compared against upstream
---------------------------------

The validation suite contains focused comparisons rather than one aggregate
claim. Depending on the scenario, it compares design and penalty matrices,
coefficients, smoothing parameters, fitted link and response values,
deviance, effective degrees of freedom, covariance matrices, standard errors,
residuals, derivatives, diagnostics, and predictions.

Representative end-to-end cases include Gaussian, binomial, Poisson, and
Gamma responses; univariate and tensor smooths; factor-by and factor-smooth
terms; random effects; offsets; observation weights; and term selection.
Shape-constrained comparisons include monotone, curvature, positive,
numeric-by, local, bivariate, derivative, uncertainty, residual, and AR(1)
behavior.

The reference provenance and regeneration process are documented in
:doc:`development/reference_sources`. The repository keeps the executable
comparison cases under ``tests/parity/``, ``tests/smooths/``,
``tests/optimization/``, ``tests/diagnostics/``, and ``tests/scam/``.

Representative tolerances
-------------------------

Tolerances are attached to the quantity and scenario being compared; there is
no single package-wide threshold. The core REML matrix uses the following
defaults unless a named case documents an override:

.. list-table::
   :header-rows: 1
   :widths: 34 28 38

   * - Quantity
     - Representative tolerance
     - Reason for comparison
   * - Coefficients
     - ``1e-6 * (1 + abs(beta))``
     - Directly comparable when the basis representation is unique.
   * - Total effective degrees of freedom
     - Absolute difference below ``1e-4``
     - Compares fitted model complexity.
   * - REML criterion
     - Absolute difference below ``1e-4`` by default
     - Case-specific overrides cover different numerical scales.
   * - Link predictions for non-unique bases
     - ``1e-4 * (1 + abs(eta))``
     - Tests invariant prediction-space behavior instead of arbitrary basis
       representatives.
   * - Standard errors
     - Usually ``1e-6 * (1 + abs(se))``
     - Scenario-specific tolerances are recorded beside each fixture test.

SCAM and specialized family cases use their own documented tolerances, which
range from near machine precision for identical fixed-smoothing constructions
to wider prediction-space tolerances for independently optimized fits. The
test case is the source of truth; the documentation does not collapse those
different comparisons into one accuracy claim.

Reproduce the fixture comparisons
---------------------------------

The committed fixtures allow the main comparisons to run without an R
installation:

.. code-block:: bash

   pytest -q tests/parity/test_mgcv_snapshot_core_matrix.py
   pytest -q tests/scam/test_scam_fixed_smoothing_fit_parity.py

To regenerate upstream outputs with the recorded R package versions, follow
:doc:`development/reference_sources`. Review both the fixture provenance and
the per-case assertion before interpreting a passing comparison.

Intentional differences
-----------------------

* SciPy supplies the numerical machinery for R-style ``optim`` and ``nlm``
  routes. Objectives, analytic derivatives, bounds, and mapped tolerances are
  retained, but stopping codes and iteration-by-iteration traces are not
  expected to be identical to R.
* Python and scikit-learn conventions determine estimator construction,
  tabular inputs, persistence, and result objects.
* ``nthreads`` and ``ncv_threads`` are retained as metadata; they do not create
  parallel native kernels in the current implementation.

Guarded boundaries
------------------

The following combinations remain unsupported and fail explicitly:

* automatic ML, REML, or LAML selection for transformed SCAM models;
* automatic smoothing for transformed models with several linear predictors;
* ordinary-GAM ``nlm.fd`` and ``magic`` outside its Gaussian
  GCV/UBRE/Cp performance-iteration route;
* AR(1) likelihood selection outside supported Gaussian identity models with
  unknown scale;
* NCV/QNCV for general and extended families, and the optional post-fit
  jackknife pass when another criterion is fitted with ``nei=``.

The detailed optimizer matrix is maintained in
:doc:`development/mgcv_optimizer_boundaries`.

Choosing an interface
---------------------

Both interfaces accept formulas, but they serve different workflows:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Interface
     - Starting default
     - Choose it when
   * - :class:`nampy.gam.GAM`
     - Fixed smoothing unless automatic selection is requested.
     - You need direct control of formula compilation, fitting routes, or
       statistical internals.
   * - :class:`nampy.models.GAMRegressor`
     - Automatic REML smoothing selection.
     - You want estimator conventions such as ``score``, cloning, and a
       matrix-or-formula ``fit`` workflow.
   * - :class:`nampy.models.GAMClassifier`
     - Automatic REML smoothing selection for a binary binomial model.
     - You want class labels, probabilities, and estimator conventions.

For a first complete analysis, continue with :doc:`tutorials/first_gam`.
