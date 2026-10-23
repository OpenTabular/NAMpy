Shape-constrained GAMs
======================

Use a shape constraint when the direction or curvature of an effect follows
from the problem rather than from the observed sample alone. For example, a
dose effect may be expected to increase but not decrease over the studied
range. A constrained smooth encodes that assumption during fitting; it is not
a visual adjustment applied afterward.

Compare an unconstrained and monotone fit
-----------------------------------------

The only formula change below is the basis code. ``cr`` fits an ordinary cubic
regression spline, while ``mpi`` fits a monotone-increasing P-spline.

.. code-block:: python

   from nampy.gam import GAM

   ordinary = GAM(
       formula='y ~ s(x, bs="cr", k=12)',
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data)

   increasing = GAM(
       formula='y ~ s(x, bs="mpi", k=12)',
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="gcv",
       smoothing_optimizer="bfgs",
   ).fit(data)

.. image:: ../_static/shape_constraint_comparison.png
   :alt: Unconstrained and monotone-increasing smooths fitted to noisy data
   :class: scientific-figure
   :align: center

The unconstrained curve can turn downward locally when the observations are
noisy. The monotone fit cannot: its first derivative remains nonnegative over
the fitted range. This is useful only when monotonicity is scientifically or
operationally defensible. Otherwise the constraint can conceal real structure.

Inspect the fitted constraint
-----------------------------

The derivative is an estimand, not merely a visual property of the plotted
line. For the first smooth, request its first derivative and Bayesian standard
error:

.. code-block:: python

   derivative = increasing.derivative(smooth_number=1, deriv=1)
   derivative.derivative
   derivative.se

Check the derivative over the fitted range and report the scientific reason
for imposing the constraint. A nonnegative derivative confirms what the
chosen basis enforces; it is not an independent test that monotonicity is true.

Fit a constrained smooth
------------------------

Use a ``scam`` basis code in an ordinary ``s()`` term. Fixed smoothing and
compatible automatic GCV/UBRE selection are both available:

.. code-block:: python

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
values and Bayesian standard errors for a univariate constrained smooth. The
smooth number is one-based.

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
the shared observation transform. For ordinary Gaussian identity models with unknown scale, ML/REML/LAML use
the determinant-corrected bam route. Transformed-coefficient SCAM models
continue to reject automatic ML/REML/LAML selection. Non-Gaussian and
non-identity observation-transform likelihoods are rejected before fitting.

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

Current boundaries
------------------

.. include:: ../_includes/optimizer_support.inc
