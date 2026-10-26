Compatibility
=============

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
* ordinary, quasi, discrete, ordered-categorical, and supported multi-parameter
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

The repository keeps the executable
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

To run or regenerate focused comparisons, follow :doc:`contributing`.
Review fixture provenance and the
per-case assertions before interpreting agreement.

Basis-dependent intermediate values need a shared representation. For example,
``gammals`` initialization with ``select=True`` can produce different starting
smoothing parameters after a valid eigenvector sign change, including in R.
Its initializer tests therefore compare exact values on a captured common
basis and separately check the design invariant. End-to-end fit and inference
tests remain separate checks of the native implementation.

Numerical differences in identified quantities still require investigation.
At a large smoothing parameter, the negative-binomial criterion evaluates its
penalty through roots, matching ``gam.fit4`` and ``get_bSb``. A dedicated R
fixture checks that penalty and criterion, alongside the complete optimizer
trace; a different stopping iteration is not excused as basis ambiguity.

Three-way nested smooth comparisons use independently sampled covariates and
check fitted values and standard errors at absolute tolerance ``2e-8``.
An older fixture accidentally made the third covariate proportional to the
first. That rank-deficient case retains deletion-rank checks, but portable
fitted-value parity for it has not been established.

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

* shape inequalities requested through mgcv's newer ``sc``/``scad`` bases;
  these need its separate ``scasm`` fitter. Their unconstrained constructor
  branches are available. SCAM's existing constrained bases remain separate;
* ``so`` and multi-penalty ``sf`` tensor/factor-smooth margins, full-rank ``sw``
  inside ``fs``, and soap factor nesting with matrix coordinates. Single-penalty
  ``sf``/``sw`` tensor margins, ordinary-input ``fs`` with ``sf``, and ``sz``
  with ``sf``/``sw`` are supported; see :ref:`soap-film-smooths`;
* known-boundary ``so``/``sf`` matrix fits, which fail during upstream offset
  aggregation, and weighted 2D adaptive matrix fits, whose upstream prediction
  drops the ``by`` variable. Free-boundary soap matrix fits, 1D adaptive matrix
  fits, and unit-weight 2D adaptive matrix fits are supported;
* matrix ``fs`` terms with the default ``tp`` base, because natural-parameter
  null-space orientation parity remains unresolved. Explicit ``cr`` and ``ps``
  bases support matrix functionals with a scalar group factor;
* linked matrix-valued tensor ``id`` groups and matrix pooling combinations
  without an implemented setup path; basis-specific guards remain authoritative;
* matrix ``t2`` with any ``bs`` margin: upstream type-3 natural parameterization
  normalizes null columns individually, so numerical null-space rotations can
  change the fitted metric. This is a NAMpy portable-parity limitation;
  ``te``/``ti`` with ``bs`` and scalar-input ``t2`` remain supported;
* all ``cs`` matrix fits, including distinct locations: unequal
  penalties on the nominal null space can amplify platform-dependent
  eigenvector orientation into different fixed and REML fits. Scalar ``cs``
  remains supported. This is an unresolved
  NAMpy parity restriction, rather than an upstream constructor rejection;
* automatic ML, REML, or LAML selection for transformed SCAM models;
* automatic smoothing for transformed models with several linear predictors;
* ordinary-GAM ``nlm.fd`` and ``magic`` outside its Gaussian
  GCV/UBRE/Cp performance-iteration route;
* inverse-Gaussian and quasi EFS selection on the joint smoothing/scale route;
* AR(1) likelihood selection outside supported Gaussian identity models with
  unknown scale;
* NCV/QNCV for general and extended families, and the optional post-fit
  jackknife pass when another criterion is fitted with ``nei=``.

Representative executed routes are shown in the
:doc:`optimizer report <development/optimizer_matrix>`.

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
     - Automatic REML smoothing selection for a binary binomial or quasibinomial model.
     - You want class labels, probabilities, and estimator conventions.

For a first complete analysis, continue with :doc:`tutorials/first_gam`.

Additional workflow coverage
----------------------------

The remaining extended families ``cpois`` and ``bcg`` and general families
``gumbls``, ``ziplss``, ``multinom``, ``gevlss``, ``shash`` and ``twlss`` are
implemented. Static mgcv references cover analytic likelihood derivatives,
initialization, fixed and selected smoothing, fitted values, predictions,
covariance, EDF, residuals and likelihood criteria. Separate public tests cover
natural parameters, additive components and persistence. These references
establish the tested routes, not every family/smooth/optimizer composition.

``cpois`` uses discrete intervals ``lo < Y <= hi`` and supports identity, log
and square-root links. ``bcg`` supports fixed and estimated Box--Cox and
Gaussian scale parameters, including the source vector EFS update. Upstream's
positive two-parameter fixed BCG constructor stores an invalid three-entry
working state and fails during fitting. NAMpy rejects that state explicitly;
the documented ``putTheta`` workaround supplies the source two-entry state.
Estimated BCG EFS requires at least one free smoothing parameter.

General-family predictions retain mgcv's family-specific response conventions;
``GAMLSS.predict`` returns natural distribution parameters instead. Multinomial
models have ``K`` predictors and ``K+1`` probability columns, including baseline
class zero. Natural-parameter standard errors use the full coefficient
covariance and transformation Jacobian. This public convention differs from
source prediction routines that clip cross-covariances or reuse a predictor
variance. Native prediction routines retain those source behaviors.
``twlss`` uses source EFS selection because its likelihood supplies only first
and second derivatives. Source likelihood weight conventions, offset guards,
support boundaries and constructor quirks are documented in
:doc:`api/gam_families`; no numerical derivative substitute is used for fitting.
``cox.ph``/``cox.pht``, ``mvn`` and ``gfam`` remain outside this task's scope.

Grouped-binomial ``cbind(success, failure)`` and two-column array responses
retain trial counts for initialization, likelihood and prediction semantics.
Inverse Gaussian supports ``1/mu^2``, inverse, log and identity links. Static
mgcv references exercise numerical family functions and public fit outputs.

Quasipoisson, quasibinomial and general quasi families support nine named
links; general quasi supports five variance functions and numeric power links.
Static references cover family derivatives and boundary behavior, fixed and
optimized fits, weights, offsets, grouped fractional quasibinomial responses,
predictions and covariance. ML/REML selection uses extended quasi-likelihood;
public log likelihood, AIC and BIC are ``NaN``. Distribution-dependent
randomized quantile residuals and custom R variance/link closures are
unavailable. See :doc:`user_guide/statistical_gams` for constructors and domains.

Fitted ``factor``, ``ordered``, ``interaction``, ``cut`` and ``poly``
transformations support merged factor labels, automatic cut breaks/labels, and
multivariate polynomials. Polynomial recurrence state persists for prediction;
scalar cut breaks and implicit factor-label mappings follow R reevaluation
semantics. Unsupported R options and unseen levels are rejected. These
functions do not imply general R evaluation.

MRF polygon-map effects are compared with ``plot.mrf.smooth``; the display uses
Matplotlib. Derivative plots reuse the supported derivative API and pointwise
Bayesian uncertainty. GP/SOS and other bases without derivative providers,
and new-data SCAM derivatives, remain unsupported.

Conditional prediction surfaces and SOS hemispheres use the upstream grid,
conditioning, rotation and masking rules. ``GAM.vis`` supports link/response
predictions, factors, selected linear predictors and pointwise SE surfaces;
``GAM.plot`` supports SOS hemisphere schemes 0/1 and generic schemes 2+.
Static fixtures compare data from ``vis.gam`` and ``plot.sos.smooth``; graphics
use Matplotlib rather than reproducing R device output.

Matrix-functional validation
----------------------------

Committed references compare numeric, categorical, adaptive and soap matrix
functionals, including numeric/factor random-effect interactions and full or
reduced-rank MRFs. They check fixed and REML fits, observation and quadrature
weights, prediction covariance kernels, EDF, smoothing parameters, aligned
penalties and persistence. Repeated-location fixtures test mgcv's setup packing
and constraint ordering. The matrix ``cs`` restriction above records a real
fitted-value discrepancy; it is not hidden by broader tolerances.

``fs`` matrix terms with ``cr``/``ps`` bases and scalar group factors, and
``sz`` matrix terms with ``tp``/``cr``/``ps`` bases and one or more scalar
factor columns, have separate references. Tensor tests cover paired coordinates and summation of
pointwise products, including numeric/factor random-effect margins with one
precision penalty. Multiple random-effect precision matrices remain rejected
as tensor margins. These checks establish the documented combinations,
not arbitrary nesting or every family/basis combination. See
:ref:`matrix-functional-smooths` and the runnable
``examples/example_matrix_functionals.py`` example.

Diagnostics and penalty controls
--------------------------------

``GAM.qq`` and ``GAM.plot_check`` implement family QQ/envelope and four-panel
check data. Shape-constrained fits use SCAM-specific term uncertainty, partial
residuals and check behavior. Static mgcv/SCAM comparisons cover numerical
coordinates and envelopes, not pixel-identical graphics. NumPy and R seeds do
not imply identical random streams; parity fixtures share explicit permutations
or response draws. Existing ``gam_check`` numerical reports remain available.

``para_pen`` supports multiple matrices per original parametric term, supplied
ranks, fixed/estimated smoothing and shared one-hot ``L`` maps. Arbitrary ``L``
maps and multi-formula parametric penalties are unavailable. In aliased models,
individual smoothing parameters and coefficient covariance allocations need not
be uniquely identified; compare estimable fitted functions and their covariance.

Explicit homogeneous constraints and raw/absorbed ``smooth_con`` construction
are available for ``cr/cs/cc/ps/cp`` without ``by``, ``pc`` or linked IDs.
Other bases/compositions and cancelling constraint-column sums are rejected.
Absorption can be disabled for construction inspection, but never for fitted
GAMs. See :doc:`user_guide/statistical_gams` for matrix coordinates and ordering.
