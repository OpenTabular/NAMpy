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

NCV/QNCV selection inflation uses ``score_gamma`` (mgcv's model ``gamma``).
The ``gamma`` entry in ``nei`` belongs to the upstream post-fit neighbourhood
or jackknife calculation and does not control smoothing selection.

What is compared against upstream
---------------------------------

The validation suite contains focused comparisons rather than one aggregate
claim. Depending on the scenario, it compares design and penalty matrices,
coefficients, smoothing parameters, fitted link and response values,
deviance, effective degrees of freedom, covariance matrices, standard errors,
residuals, derivatives, diagnostics, and predictions.

Working-information and NCV edge-case fixture identities normalize floating
inputs to twelve significant digits. Committed aliases map these portable keys
to the original saved R outputs, avoiding missing fixtures when math libraries
produce last-bit differences in ``exp`` or ``log``. Test inputs, R results and
comparison tolerances retain their original precision; normal tests do not run R.

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

Fitted log likelihood, AIC and BIC use the stored training means, including
formula and API offsets. Ordered-categorical likelihood uses the latent
predictor; beta regression uses its family density hook. Focused comparisons
cover fourteen family/link combinations, beta response boundaries, positive
and zero prior weights where supported, mixed offsets and model persistence.
Gaussian zero-weight
likelihoods follow the finite convention described below; weighted ``ziP``
fitting retains its explicit unsupported guard. Ordinary-family source
dispersion profiling uses tight outer convergence to isolate reporting errors.

Poisson and grouped-binomial densities use the R count-density construction
instead of subtracting large log-gamma terms. Source comparisons cover counts
and trial sizes through ``1e15``, tail probabilities, exact binomial probability
boundaries, noninteger Poisson support and zero weights. Actual large-count
``gam`` fits also check likelihood, AIC and BIC. The shared Poisson density
uses R 4.6's relative integer tolerance of ``1e-9``; this applies to censored
Poisson calculations as well.

Binomial initialization retains the raw ``stats::binomInitialize`` formula
at large denominators instead of clipping starting probabilities at ``1e-9``.
Saved references cover all five links, ordinary and grouped responses, zero
and nonunit weights, and trial counts through ``1e15``, with quasibinomial
controls. Binomial deviance and Bernoulli likelihood helpers preserve supplied
probabilities, including exact zero/one and tiny positive probabilities; inverse
links retain their own source-defined clipping. Four rare-event grouped fits
compare predictions, EDF, likelihood criteria and persistence with actual mgcv
fits at trial counts of ``1e9`` and ``1e12``.

Gaussian AIC with supplied dispersion follows ``gam.fit3`` and ``am.fit``:
the family hook receives ``scale * sum(prior.weights)`` in place of residual
deviance. Gamma log densities use the stable R ``dgamma`` construction.
Fixed-dispersion comparisons span parametric and cubic-spline models, Gaussian
identity and Gamma log/inverse links, offsets, weights and dispersions down
to ``1e-15``. Density-only checks additionally cover extreme scales and support
boundaries.

Gamma initialization, deviance and Pearson scale retain valid positive responses
below ``1e-9``. Their working-weight derivatives also avoid absolute denominator
floors that change the result when response units change. Comparisons across
log, identity and inverse links span response multipliers from ``1e-12`` to
``1e12``, with formula/API/mixed offsets, zero weights, residuals, prediction
standard errors, likelihood criteria and finite-difference GCV derivatives.
Automatic GCV comparisons use explicit log-SP bounds wide enough to contain
the rescaled identity/inverse-link optima; configured search bounds remain
absolute. The reference fits transport smoothing parameters with the response
units so each comparison represents the same penalized function.

Family working-information derivative hooks distinguish Fisher information
when no response is supplied from observed information when targets are supplied.
Saved mgcv analytic derivatives and independent source finite differences cover
noncanonical binomial, Gaussian, Poisson and negative-binomial links, small means,
signed inverse-link predictors and several negative-binomial shape parameters.
Gamma log retains its explicit requirement for targets. At machine-clipped link
tails, comparisons use mgcv's analytic derivative conventions. These hooks are
separate from the fitting core's working-information calculations. Negative-
binomial initialization also preserves tiny positive responses, with saved
references for all three links; fractional-response deviance follows the source
family's convention.
Gaussian family density helpers preserve supplied variance down to ``1e-15``
and scale residuals before squaring to avoid unnecessary overflow. Additional
saved R references check huge residuals/variances, zero/negative variance and
saturated likelihoods with varying and zero weights.

Basis-dependent intermediate values need a shared representation. For example,
``gammals`` initialization with ``select=True`` can produce different starting
smoothing parameters after a valid eigenvector sign change, including in R.
Its initializer tests therefore compare exact values on a captured common
basis and separately check the design invariant. End-to-end fit and inference
tests remain separate checks of the native implementation.

For a non-unique basis, parity checks compare represented function spaces,
penalty operators in common coordinates, fitted predictions, EDF, criteria
and prediction covariance. If ``X_other = X @ T``, the corresponding penalty
is ``T.T @ S @ T``; changing design columns without transporting every penalty
does not test a coordinate change. Raw coefficient arrays, coefficient
covariances and individual smoothing-parameter labels are compared only after
the corresponding alignment. Equal penalty spectra alone do not establish
equal penalized models: the penalty must act on the same functions.

Some upstream constructors assign unequal penalties to numerically chosen
null directions, or normalize those directions individually before adding
new penalties. Repeating that construction after rotating its input can
change the model itself. Such a discrepancy must be demonstrated through
fitted functions or covariance, rather than inferred from differing columns.

Numerical differences in identified quantities still require investigation.
The higher-order smoothness-uncertainty correction ``Vb.corr`` differentiates
a Cholesky root and depends on coefficient coordinates. AR(1) comparisons
evaluate this upstream correction in common coordinates, checking the design
and penalty transport first. Conditional predictions, covariance, and selected
smoothing parameters retain direct fitted-model comparisons.

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

* Gamma saturated likelihood and scale derivatives retain the requested
  dispersion below ``1e-9`` and evaluate the small-dispersion terms through
  the Stirling expansion. The recorded upstream digamma/trigamma subtraction
  loses precision and can reverse the second derivative at ``1e-15``.
  These cases use independent 100-digit differentiation of the Gamma density;
  direct upstream captures are retained, and ordinary dispersions also compare
  directly with ``fix.family.ls``. The profile-scale solver also retains valid
  initial scales below ``1e-12`` rather than moving them away from the solution
  and triggering a Newton-step underflow.
* NCV keeps known offsets in held-out predictors and retains the signs of
  Newton weights when deleting neighbourhoods. The recorded mgcv ``Rncv``
  path omits these offsets and uses absolute weights for negative-weight
  downdates. Affected cases are checked with independent signed one-step
  equations using actual mgcv fits, designs, penalties and family derivatives;
  the ordinary cases also retain direct upstream score comparisons.
* Gaussian log likelihood, AIC and BIC omit zero-prior-weight observations
  from the log-precision term and remain finite for otherwise well-defined
  fits. The recorded mgcv 1.9-4 library can return ``NaN`` for these criteria
  because it includes zero weights inside a logarithm. Gaussian GCV/UBRE
  retains upstream positive-weight-row scoring and scale estimation.
* A Gaussian AR(1) fit without ``ar_start`` represents one series and uses
  ``n - 1`` transitions in likelihood normalization. The recorded ``bam``
  final AIC path counts a missing ``AR.start`` as zero starts; reference calls
  explicitly mark the first observation to represent the same one-series model.
* Beta-regression mixed mean/log-precision derivatives use polygamma
  recurrences to cancel boundary poles analytically. Near mean zero or one,
  this avoids severe cancellation also present in mgcv 1.9-4. These boundary
  derivatives are checked against independent 100-digit differentiation;
  ordinary-range derivatives retain direct mgcv parity checks.
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

Tweedie inner-fit dispersion retains a Pearson fallback. Scale-dependent
diagnostics can differ from mgcv's separately profiled outer dispersion,
especially when smoothing optimization is disabled. The fixed-inner residual
comparisons cover working, response, deviance and unscaled Pearson residuals.

The following combinations remain unsupported and fail explicitly:

* shape inequalities requested through mgcv's newer ``sc``/``scad`` bases;
  these need its separate ``scasm`` fitter. Their unconstrained constructor
  branches are available. SCAM's existing constrained bases remain separate;
* soap tensor/factor margins with multiple penalties: ``so`` with any free
  boundary and ``sf`` with multiple free loops. These are explicit upstream
  rejections. Fully known-boundary ``so`` tensors and ``sz`` are supported;
  single-penalty ``sf/sw`` tensor and accepted matrix-coordinate factor
  compositions are also available. Full-rank ``sw`` inside a real ``fs``
  interaction reaches an upstream zero-null indexing failure and remains
  rejected. Scalar factors with paired matrix soap coordinates work, as do
  explicit matrix-valued factors supplied as categorical rows with consistent
  levels and quadrature widths of at least two.
  See :ref:`soap-film-smooths` for the source boundary-offset convention;
* known-boundary ``so``/``sf`` matrix fits, which fail during upstream offset
  aggregation, and weighted 2D adaptive matrix fits, whose upstream prediction
  drops the ``by`` variable. Free-boundary soap matrix fits, 1D adaptive matrix
  fits, and unit-weight 2D adaptive matrix fits are supported;
* matrix ``fs`` terms with the default ``tp`` base, because natural-parameter
  null-space orientation parity remains unresolved when several null directions
  receive independent penalties. Univariate ``tp`` with ``m=1`` has one null
  direction and is supported. Explicit ``cr`` and ``ps`` bases also support
  matrix functionals with a scalar group factor;
* explicit matrix-factor ``fs`` bases with multiple independently penalized
  null directions, and multiple matrix factors in ``sz`` with empty joint
  training groups. These remain NAMpy penalty/rank parity limitations.
  Supported single-null ``fs`` and fully observed ``sz`` routes are documented
  under :ref:`matrix-functional-smooths`. Unused declared matrix-factor levels
  fail upstream during dimension-dropping level removal and are rejected;
* linked matrix-valued tensor ``id`` groups and matrix pooling combinations
  without an implemented setup path; basis-specific guards remain authoritative;
* matrix ``t2`` mixing range/null components with multidimensional ``bs`` null
  spaces: upstream type-3 natural parameterization
  normalizes null columns individually, so numerical null-space rotations can
  change the fitted metric. This is a NAMpy portable-parity limitation;
  range-only ``ord=[number_of_margins]`` and ``bs`` margins with zero or one
  null direction are supported. ``te``/``ti`` with ``bs`` and scalar-input
  ``t2`` remain supported;
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
and square-root links. Concentrated rare-event derivatives use conditional-count
cumulants and independent 100-digit checks, avoiding cancellation artifacts in
the upstream double-precision tail expressions. Ordinary-event source parity
checks retain their original tolerances.

``bcg`` supports fixed and estimated Box--Cox and
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

General-family unconditional covariance requires the outer derivative state
from smoothing selection. A genuinely fixed-SP fit does not acquire that
uncertainty just because its smoothing parameters equal a selected model's
endpoint. Tests that isolate selected-endpoint postprocessing explicitly
reconstruct the analytic outer Hessian at that endpoint.

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
ranks, fixed/estimated smoothing and arbitrary finite log-SP ``L`` maps,
including signed/fractional entries and rectangular maps. Analytic derivatives
and supported smoothing uncertainty are transported into the underlying
parameter coordinates. Combined smooth/parametric maps preserve mgcv's setup
ordering, including its ordering quirks. Nonidentity maps with automatic EFS
selection remain guarded because upstream omits the map in that optimizer;
unused free map coordinates and multi-formula parametric penalties are rejected.
Automatic initialization of rank-deficient free maps follows the source
rejection. Positive ``min_sp`` combined with nonidentity maps needs separate
full-penalty minimum-penalty assembly and remains unsupported, as do mapped
transformed-coefficient fits. General-family formula lists reject ``para_pen``
even when there is only one predictor.
In aliased models,
individual smoothing parameters and coefficient covariance allocations need not
be uniquely identified; compare estimable fitted functions and their covariance.

Explicit homogeneous constraints and raw/absorbed ``smooth_con`` construction
are available for ``cr/cs/cc/ps/cp/bs`` without ``by``, ``pc`` or linked IDs.
Other bases/compositions and cancelling constraint-column sums are rejected.
Absorption can be disabled for construction inspection, but never for fitted
GAMs. See :doc:`user_guide/statistical_gams` for matrix coordinates and ordering.

.. toctree::
   :hidden:
   :maxdepth: 1

   Published PreTab boundary <user_guide/pretab_compatibility>
