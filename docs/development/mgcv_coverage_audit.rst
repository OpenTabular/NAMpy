mgcv coverage audit
===================

This audit covers NAMpy's implemented statistical features, against mgcv 1.9.4
and the separate SCAM reference where appropriate. A registered feature having
one passing case is not evidence that every supported combination is covered.
The seven-stage registry matrix covers 65 smooth codes and 30 canonical family
names. Its exact cases and limitations are recorded in
``tests/parity/seven_stage_coverage.json``.

Measured execution coverage
---------------------------

On 2026-10-28, all 7,918 existing statistical tests passed. The backfill adds
548 crossing/diagnostic tests, including provenance gates. Combined branch
instrumentation of ``nampy.gam`` reports 34,399/38,536 executable lines
(89.3 percent) and 9,673/12,836 branches (75.4 percent): 4,137 lines and 3,163
branches were not executed. This measurement is on Python 3.12; it is not a
claim that neural, benchmark or documentation test execution was measured.

``docs/development/mgcv_coverage_snapshot.json`` records the environment,
per-file source hashes, coverage summaries and missed line numbers. These
numbers measure code execution, not the accuracy or breadth of numerical
assertions. Some missing code is a guard or compatibility path bypassed by the
normal dispatcher; it should be classified before adding numerical scenarios.

.. list-table:: Concrete code paths with low branch coverage
   :header-rows: 1
   :widths: 48 12 40

   * - Source area
     - Branches
     - What needs investigation
   * - ``parity/compare.py`` and ``parity/validator.py``
     - 0 percent
     - Public snapshot comparison is unexercised. The legacy comparator still
       compares coefficients, basis matrices and coefficient covariance entries.
       It needs an invariant-only migration and tests for equivalent
       reparameterizations, malformed inputs and real numerical disagreement.
   * - ``results/snapshots.py``
     - 25.7 percent
     - Snapshot loading/coercion, round trips, missing/optional fields and
       fallback reporting. The validator's coercion path was not reached.
   * - ``fit/selection/criteria/gaussian.py``
     - 30.4 percent
     - Exact/dynamic fallback variants and degenerate profiles. Many legacy
       dynamic branches are bypassed for normal ML/REML/LAML dispatch; this
       does not mean that Gaussian REML fitting itself is untested.
   * - ``fit/selection/optimize/trace.py``
     - 36.4 percent
     - Trace normalization for joint theta/scale coordinates, missing optimizer
       traces and alternative score-history layouts.
   * - ``fit/selection/optimize/shape_bfgs.py``
     - 43.3 percent
     - Line-search expansion/interpolation, step halving, curvature-condition
       failures, nonfinite objectives and stopping paths.
   * - ``fit/solvers/general_family/sl_transforms.py``
     - 47.6 percent
     - Inverse/one-sided/two-sided transformations, vector versus matrix
       inputs and diagonal versus dense penalty-block transforms.

What is already covered
-----------------------

* All registered smooths have separate formula, runtime, constructed-term,
  assembly, identifiability, fitting and public-output checks. There are also
  two-dimensional adaptive cases. Every regular mgcv smooth now has these
  stages for binomial, Poisson and Gamma responses as well as Gaussian.
* Ordinary links, inverse Gaussian links, quasi variance/link combinations,
  fixed and estimated nuisance parameters, grouped binomial responses and
  censoring have dedicated family suites. These should not be described as
  untested merely because the registry matrix uses one representative link.
* The optimization lifecycle registry currently has 31 stable cases. Dedicated
  suites cover fixed fitting, GCV, GACV, UBRE, ML/REML/LAML, Pearson variants,
  NCV/QNCV, Newton, BFGS, EFS, optim and NLM routes. This is route coverage,
  not the complete product of every family, penalty layout and optimizer.
* Separate suites cover matrix linear functionals, factor smooths, linked IDs,
  point constraints, selection penalties, parametric penalty maps, offsets,
  weights, known Gaussian AR(1) transforms, prediction filters, diagnostics,
  inference, persistence and the installed wheel.

Remaining gaps, in priority order
---------------------------------

.. list-table:: Coverage depth still needing work
   :header-rows: 1
   :widths: 18 38 44

   * - Area
     - Existing evidence
     - Missing guarantee
   * - Reference identity
     - The main fixture manifest records a local Newton-sign mutation. Some
       historical probes restore the CRAN behavior; the new shared probe now
       restores it in-process too.
     - Package version alone does not identify an oracle. Every capture needs
       auditable source identity and any process-local corrections. Historical
       references must retain their stated scope rather than silently being
       treated as pristine upstream results.
   * - Strict numerical acceptance
     - ``mgcv_parity_utils._assert_basic_mgcv_parity`` checks fitted responses,
       EDF, deviance and selection criteria for many integration cases.
     - Its default EDF/deviance/criterion tolerances include relative bounds
       of about 5--8 percent. These cases need independently justified tighter
       acceptance and the full observable-statistic checks used by the newer
       invariant suites. A loose test passing is not strict numerical parity.
   * - Selected smoothing and uncertainty
     - ``test_mgcv_optimization_lifecycle_parity.py`` and
       ``test_mgcv_structured_unconditional_inference_parity.py`` test selected
       routes and structured covariance kernels.
     - There is no registry-wide optimized counterpart of the fixed-smoothing
       matrix. EDF2 totals, likelihood degrees of freedom, unconditional kernels,
       and criteria need crossings with recent smooths, nuisance estimation,
       multiple penalties and mixed fixed/free smoothing parameters.
   * - Shape-constrained crossings
     - All 41 shape codes have Gaussian fixed-smoothing SCAM comparisons;
       dedicated shape optimizer and derivative suites exist.
     - The new non-Gaussian matrix covers regular mgcv smooths, not all shape
       codes. Binomial/count families, constraint activation, positive-transform
       choices and supported selection routes still need a corresponding
       explicit matrix. Transformed LSS automatic Laplace smoothing remains
       an explicitly unsupported route, not an expected numerical success.
   * - Distributional model compositions
     - Every general family has fitting/output tests; Gaussian and Gamma LSS
       also have tensor and multismooth workflows.
     - Every general family is not crossed with every supported tensor,
       adaptive, graph, GP, soap, factor or shared-predictor design. Conditional
       fits do not establish joint optimization or cross-predictor uncertainty.
   * - Smooth configuration space
     - Constructor and combination suites vary knots, dimensions, penalty
       orders, by variables, linked IDs, point constraints and matrix inputs.
     - These variations do not form an exhaustive fitted-invariant matrix.
       Higher dimensions, tensor margin/order combinations, graph components,
       boundary geometry and extrapolation need explicit coverage accounting.
   * - Inference beyond ordinary cases
     - Summary, ANOVA, variance components and covariance-option suites exist.
       Sandwich comparisons currently concentrate on Gaussian and Poisson.
     - Supported extended/general-family and structured-design crossings need
       more invariant Wald statistics, hypothesis tests, variance components,
       dispersion rescaling and sandwich prediction kernels. Raw covariance
       entries are not an acceptable substitute for function-space comparisons.
   * - Boundary combinations
     - Dedicated suites cover rare tails, censoring, near separation, zero
       weights, rank deficiency, missing prediction rows and unseen factors.
     - Each edge is not crossed with every applicable family/design/optimizer.
       Combinations of zero weights, aliases, extreme nuisance values and
       multiple predictors deserve more attention than more easy central cases.
   * - Random diagnostics
     - QQ diagnostics have shared-draw references. k-check tests verify EDF,
       basis dimensions and broadly bounded statistics/p-values.
     - Broad k-index tolerances and p-value range checks alone do not establish
       equality of the permutation calculation. A new shared-plan suite closes
       this gap for binomial/Poisson/Gamma crossed with cr, gp, t2, re, mrf, fs and
       sos, both with and without subsampling. Other diagnostic-family and
       structure combinations still need this stronger check.
   * - Portability and invariance
     - Row-permutation tests cover selected nonunique bases; frozen reference
       inputs prevent a basis change from changing the test dataset.
     - Every applicable feature still needs accounting for row order, factor
       level order, covariate units, equivalent constraints/parameterizations,
       persistence and differing BLAS/LAPACK implementations. Existing legacy
       coefficient-coordinate comparisons also need review against the
       invariant-only policy. The full suite runs on Linux/Python 3.11--3.12;
       macOS/Windows run selected slices, with one pinned dependency set.
       Minimum-versus-current dependency combinations are not a CI matrix.

Explicit limits are not missing tests
-------------------------------------

The registry matrix fixes Tweedie scale and ordered-categorical cutpoints to
isolate conditional fitting. Other suites exercise nuisance estimation, but
the conditional matrix cannot claim that guarantee. Three NAMpy joint families
(``student_lss``, ``beta_lss`` and ``nb_lss``) have no corresponding smoothly
varying two-parameter mgcv family; independent SciPy optimization and existing
gamlss.dist/Torch references provide their evidence.

NAMpy also does not expose the complete mgcv product. For example,
``cox.ph``, ``mvn``, general ``gamm`` mixed-model fitting, ``jagam``, and full
``bam(discrete=TRUE)`` fitting are not implemented by the statistical engine.
The known Gaussian AR(1) route reproduces a particular bam calculation; it does
not imply the rest of bam is supported. These are implementation gaps, not
features for which adding tests alone establishes support.

Acceptance policy
-----------------

For a supported route, compare fitted link and response functions, likelihood,
EDF/EDF1/EDF2 totals where defined, AIC/BIC, deviance, scale/nuisance parameters,
residuals, prediction standard errors and full prediction covariance kernels.
Undefined quantities must remain undefined. Test failures and explicit
unsupported-route errors are part of the contract.

Across reference implementations, never require matching basis columns,
eigenvectors, coefficient vectors or covariance entries in coefficient
coordinates. Internal ownership checks may reconstruct quantities within one
object's own coordinates. Compare function-space projectors and prediction or
contrast kernels when representations are nonunique.

Frozen reference captures are used in normal CI. Generation is explicit and
records the R probe, input-source and output hashes. The new non-Gaussian
crossing matrix has 72 independent captures and 504 stage checks, plus a
provenance check. It fills one finite coverage gap; it is not evidence that
all of the combinations above are complete.

The k-check backfill has 42 source captures. It supplies identical selected
rows and 37 permutations to Python and a local copy of the upstream R function
body. Only its ``sample()`` dependency is replaced; fitting, residuals,
neighbor selection, tensor rescaling, the observed k-index and the permutation
p-value are still computed independently. The tests require tight agreement
of the complete diagnostic table, including exact undefined entries for
factor smooths, rather than checking only that p-values lie in [0, 1].
