GAM Extension API
=================

These namespaces expose the building blocks used to extend or validate the
statistical backend. Most analyses should start with :class:`nampy.gam.GAM`.

Controls and fitting
--------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Public symbol
     - Purpose
   * - ``nampy.gam.GAMControl``, ``gam_control``
     - Validated iteration, tolerance, threading, and optimizer controls.
   * - ``nampy.gam.fit.fit_model_core``
     - Compile and fit a configured statistical GAM.
   * - ``available_fit_backends``, ``resolve_fit_backend``
     - Inspect and choose a solver compatible with the compiled model.
   * - ``solve_fit``, ``solve_gaussian_fit``, ``solve_pirls_fit``
     - Run the general dispatcher or a backend-specific solve.
   * - ``solve_gaussian_given_smoothing``, ``solve_pirls_given_smoothing``
     - Solve coefficients at fixed smoothing parameters.
   * - ``FitState``, ``FitResult``, ``FitCoreSolution``
     - Structured numerical state returned by the fitting stages.
   * - ``build_bayes_and_freq_covariances``, ``select_covariance_matrix``
     - Construct or select coefficient covariance matrices.
   * - ``coerce_offset_array``, ``resolve_prediction_offset``
     - Validate fit-time and prediction-time offsets.
   * - ``irls_core``, ``fit_irls_from_model``
     - Low-level penalized iteratively reweighted least-squares routines.
   * - ``pls_fit1_nonneg_w``, ``solve_gaussian_penalized_ls_stacked_qr``
     - Stacked-QR penalized least-squares routines.

Declarative specifications
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Public symbol
     - Purpose
   * - ``TermSpec``, ``LinearPredictorSpec``, ``PenaltyGroupSpec``
     - Core declarative model specifications.
   * - ``SmoothSpec``, ``BaseSmoothSpec``
     - Shared smooth-term specifications.
   * - ``CubicRegressionSmoothSpec``, ``CubicShrinkageSmoothSpec``,
       ``CyclicCubicRegressionSmoothSpec``
     - Cubic regression spline specifications.
   * - ``PSplineSmoothSpec``, ``DerivativeBSplineSmoothSpec``
     - P-spline and derivative-penalized B-spline specifications.
   * - ``ThinPlateSmoothSpec``, ``ThinPlateShrinkageSmoothSpec``
     - Thin-plate spline specifications.
   * - ``DuchonSplineSmoothSpec``, ``GaussianProcessSmoothSpec``,
       ``SphericalSplineSmoothSpec``, ``AdaptiveSmoothSpec``
     - Specialized metric smooth specifications.
   * - ``TensorProductSmoothSpec``, ``TensorInteractionSmoothSpec``,
       ``AlternativeTensorProductSmoothSpec``
     - Tensor-product specifications.
   * - ``RandomEffectSmoothSpec``, ``MarkovRandomFieldSmoothSpec``,
       ``FactorSmoothInteractionSpec``, ``SumToZeroFactorSmoothSpec``
     - Structured smooth specifications.
   * - ``ShapeConstrainedSmoothSpec``
     - Shape-constrained smooth specification.
   * - ``build_smooth_spec``, ``smooth_spec_from_basis_options``,
       ``replace_smooth_spec``
     - Construct and update smooth specifications.

Compiled model structures
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Public symbol
     - Purpose
   * - ``CompiledPenalty``
     - A penalty matrix and its compiled metadata.
   * - ``CompiledTerm``
     - A compiled parametric or smooth term.
   * - ``CompiledPredictor``
     - Terms and coefficient indices for one linear predictor.
   * - ``CompiledModel``
     - Complete compiled design used by the fitting backend.

Prediction helpers
------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Public symbol
     - Purpose
   * - ``nampy.gam.predict.build_lpmatrix``
     - Construct the linear-predictor matrix for new data.
   * - ``predict_values``
     - Evaluate response, link, term, interaction, or matrix predictions.
   * - ``prediction_guaranteed_skip_contract``
     - Validate optimized prediction when excluded terms allow omitted columns.

Coefficient, constraint, and observation transforms
---------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Public symbol
     - Purpose
   * - ``CoefficientTransform``, ``IdentityCoefficientTransform``
     - Base and identity coefficient-space transforms.
   * - ``CoordinatewiseCoefficientTransform``, ``BlockCoefficientTransform``
     - Coordinate and block transforms for constrained coefficients.
   * - ``PositiveMap``, ``CovarianceScale``
     - Positive mappings and covariance transport metadata.
   * - ``compose_coefficient_transforms``
     - Combine independent coefficient transforms.
   * - ``apply_linear_constraint``, ``absorb_explicit_constraints``
     - Apply or absorb linear identifiability constraints.
   * - ``apply_global_side_conditions``
     - Apply side conditions across compiled terms.
   * - ``ObservationTransform``, ``IdentityObservationTransform``,
       ``AR1ObservationTransform``
     - Observation-space transforms, including known AR(1) whitening.
   * - ``make_observation_transform``, ``ar1_log_determinant_correction``
     - Construct an observation transform and its determinant correction.

Penalties
---------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Public symbol
     - Purpose
   * - ``PenaltySpec``
     - Declarative penalty definition.
   * - ``make_penalty_spec``, ``normalize_penalty_spec``
     - Build and validate penalty specifications.
   * - ``build_null_space_selection_spec``, ``null_space_penalty_from_penalty``
     - Construct null-space penalties used for term selection.
   * - ``tensor_product_penalties``, ``lifted_tensor_penalty``
     - Construct penalties for tensor-product terms.
   * - ``rescale_tensor_penalties_for_fit``
     - Apply the fitting-scale normalization used by tensor penalties.

Numerical and comparison utilities
----------------------------------

``nampy.gam.linalg`` exposes the pivoted-Cholesky, eigenvalue, rank,
null-space, shrinkage, matrix-norm, and subspace utilities used by the fitting
stack. ``nampy.gam.parity`` exposes snapshot, optimizer-trace, comparison, and
matrix-summary helpers for reference comparisons. These modules are intended
for backend development and numerical validation rather than ordinary model
analysis.


Advanced public utilities
-------------------------

These symbols support custom fitting, testing, and extensions. Most analyses
should use ``GAM`` or the estimator classes above.

.. list-table::
   :header-rows: 1
   :widths: 31 44 25

   * - Symbol
     - Purpose
     - Import
   * - ``GAMControl`` / ``gam_control``
     - Typed and validated fitting controls.
     - ``from nampy.gam import gam_control``
   * - ``fit_model_core``
     - Compile and fit a configured GAM object.
     - ``from nampy.gam import fit_model_core``
   * - ``solve_fit``
     - Dispatch a fixed-smoothing solve to the available backend.
     - ``from nampy.gam import solve_fit``
   * - ``FitCoreSolution``
     - Structured result returned by the low-level fitting stage.
     - ``from nampy.gam import FitCoreSolution``
   * - ``normalize_nei``
     - Validate neighborhood indices used by neighborhood cross-validation.
     - ``from nampy.gam import normalize_nei``


How constraints are fitted
--------------------------

NAMpy follows the ``scam`` 1.2-22 definitions for shape-constrained
P-splines. The runtime term constructs the basis, penalty, prediction matrix,
and derivative matrices. A coefficient transform maps unconstrained optimizer
coordinates to coefficients that satisfy the requested shape and transports
covariance into that prediction space.

These transforms are term-local. A formula can therefore combine constrained
and ordinary smooths, and the resulting model continues to use the statistical
engine's prediction, inference, residual, derivative, and result interfaces.
Observation transforms, including supported Gaussian AR(1) whitening, are
applied consistently to the response, offset, and design matrix.


Fitting compatibility
---------------------

.. include:: ../_includes/optimizer_support.inc

Numerical internals
-------------------

.. toctree::
   :maxdepth: 1

   Coefficient covariance correction <../statistical_reference/internals/covariance_correction>
