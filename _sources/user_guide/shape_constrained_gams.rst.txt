Shape-constrained GAMs
======================

NAMpy ports the shape-constrained P-spline machinery from ``scam`` 1.2-22
into the classical :class:`nampy.gam.GAM` pipeline. A constrained and an
ordinary smooth can be combined in one formula. The compiler represents the
constraint as a coefficient transform, then dispatches by model capability
rather than by a hard-coded list of SCAM terms.

Architecture
------------

Three contracts separate reusable model behavior from SCAM-specific basis
construction:

* runtime terms own basis construction, penalties, prediction matrices, and
  derivative matrices;
* coefficient transforms map unconstrained optimization coordinates to the
  coefficients used for prediction and transport covariance between those
  spaces;
* observation transforms apply row operations such as known AR(1) whitening
  consistently to responses, offsets, and design matrices.

Ordinary ``mgcv``-compatible terms use identity coefficient transforms. SCAM
terms declare the constrained coordinates and SCAM's released covariance
transport policy. Independent transform blocks compose across terms and
distributional linear predictors. This keeps the ordinary identity path
unchanged while allowing constrained terms to use the same compiler,
prediction, diagnostics, and result contracts.

Basic use
---------

Use a SCAM basis code in an ordinary ``s()`` term.  Fixed smoothing and
SCAM-compatible automatic GCV/UBRE selection are both available::

   from nampy.gam import GAM

   fixed = GAM(
       formula="y ~ s(x, bs='mpi', k=10)",
       family="gaussian",
       smoothing_method="fixed",
       smoothing_params=[0.5],
   ).fit(data=df)

   selected = GAM(
       formula="count ~ s(x, bs='mpi', k=10)",
       family="poisson",
       optimize_smoothing=True,
       smoothing_method="ubre",
       smoothing_optimizer="bfgs",
   ).fit(data=df)

``positive_transform="exp"`` is the SCAM default.  Set
``positive_transform="softplus"`` to use SCAM's ``not.exp=TRUE`` map;
``softplus_beta`` and ``softplus_threshold`` expose its two numerical
parameters.

Univariate bases
----------------

The supported univariate basis codes are:

* monotone: ``mpi`` and ``mpd``;
* convex/concave: ``cx`` and ``cv``;
* joint monotonicity and curvature: ``micx``, ``micv``, ``mdcx``, and
  ``mdcv``;
* positive: ``po``, ``ipo``, ``dpo``, and cyclic-positive ``cpop``;
* anchored monotone: ``miso`` (start at zero) and ``mifo`` (finish at zero);
* numeric-by variants without the ordinary centering constraint:
  ``mpiby``, ``mpdby``, ``micxby``, ``micvby``, ``mdcxby``, ``mdcvby``,
  ``cxby``, and ``cvby``;
* local constraints: ``lmpi`` and ``lipl``.  Supply the change point as
  ``xt=list(xc=...)`` in the formula.

For example::

   local = GAM(
       formula="y ~ s(x, bs='lmpi', k=12, xt=list(xc=0.4))",
       family="gaussian",
       smoothing_method="fixed",
       smoothing_params=[0.7],
   ).fit(data=df)

Bivariate bases
---------------

All 17 bivariate classes from the upstream release are available:
``tedmi``, ``tedmd``, ``temicx``, ``temicv``, ``tedecx``, ``tedecv``,
``tecxcx``, ``tecvcv``, ``tecxcv``, ``tescx``, ``tescv``, ``tesmi1``,
``tesmd1``, ``tesmi2``, ``tesmd2``, ``tismi``, and ``tismd``.  They are
written as two-covariate ``s()`` terms::

   surface = GAM(
       formula="y ~ s(x, z, bs='tedmi', k=c(6, 5), m=c(2, 2))",
       family="gaussian",
       smoothing_method="fixed",
       smoothing_params=[0.4, 0.6],
   ).fit(data=df)

Linear functionals
------------------

Matrix-valued covariates use a shared row-wise weighted basis-aggregation
contract. It is available for ordinary ``ps``, ``cr``, ``cs``, and ``cc``
smooths and for SCAM's ``*by`` bases.

For SCAM signal regression, store each row of evaluation locations and
weights as array-valued DataFrame cells, then use the location column as the
smooth covariate and the weight column as ``by``::

   model = GAM(
       formula="y ~ s(locations, by=weights, bs='mpdby', k=12)",
       family="gaussian",
       smoothing_method="fixed",
       smoothing_params=[0.5],
   ).fit(data=df)

Inference and diagnostics
-------------------------

Prediction uses the shape-valid coefficient space, while optimization state
remains available as ``fit_result().coef_optimization``.  The corresponding
Bayesian and frequentist covariance matrices are exposed in both spaces.
``predict(..., return_se=True)``, ``summary()``, and all ordinary residual
types use the transformed covariance, as ``scam`` does.

``derivative(smooth_number=1, deriv=1)`` returns first- or second-derivative
values and Bayesian standard errors for a univariate constrained smooth.  The
smooth number is one-based to match ``scam::derivative.scam``::

   derivative = model.derivative(smooth_number=1, deriv=1)
   derivative.derivative
   derivative.se

Derivative evaluation is term-owned. Ordinary P-splines also support exact
first and second derivatives at training or new data. SCAM derivatives retain
the upstream training-data semantics; unsupported new-data derivative
requests fail explicitly.

Continuous quantile residuals are available with
``residuals(type="rquantile")`` in addition to ``deviance``, ``pearson``,
``scaled.pearson``, ``working``, and ``response``.

Gaussian AR(1) errors
---------------------

For any supported Gaussian identity-link GAM, constrained or ordinary,
supply a known correlation and optional starts of independent series::

   model = GAM(
       formula="y ~ s(time, bs='mpi')",
       family="gaussian",
       ar1_rho=0.6,
       ar_start="new_series",
       smoothing_method="fixed",
       smoothing_params=[0.5],
   ).fit(data=df)

``ar_start`` may instead be a boolean array. Standardized response residuals
are returned by ``ar1_standardized_residuals()``. Fixed smoothing and GCV use
the shared observation transform. ML/REML/LAML are guarded until their
correlation-determinant contribution is implemented. Non-Gaussian and
non-identity likelihoods are rejected before fitting.

Current boundaries
------------------

For transformed single-predictor ordinary-family models, automatic smoothing
selection is the upstream ``bfgs_gcv.ubre`` route only; ML/REML/LAML and
alternative outer policies are not substituted. Fixed-smoothing transformed
terms compose across multiple distributional predictors, including LSS
families. Automatic smoothing for that combination remains guarded until the
higher-order transformed Laplace derivatives are implemented. Bivariate
``by`` terms, model-level ``select=True``, user-supplied constraint matrices,
and SCAM's ``efs``, ``optim``, ``nlm``, and coefficient-BFGS variants remain
unsupported and raise explicit errors where reachable.
