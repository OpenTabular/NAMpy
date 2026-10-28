How smoothness selection works
==============================

Smoothing selection chooses the strength of regularization within a specified
model. It does not select an arbitrary basis dimension, invent missing
interactions, or validate a response law. The :doc:`criterion and optimizer
pages <index>` explain each supported choice separately.

Two coupled optimization problems
---------------------------------

For a given set of penalty strengths, fit coefficients. Then evaluate how well
that fitted model balances fit and complexity, and update the free strengths:

.. math::

   \widehat\theta(\lambda)=\arg\min_\theta
   \{-\ell(\theta)+\tfrac12\theta^\mathsf TS_\lambda\theta\},
   \qquad
   \widehat\rho=\arg\min_\rho C\{\widehat\theta(e^\rho),e^\rho\},
   \quad \rho_j=\log\lambda_j.

The inner problem estimates effects; the outer problem selects smoothness.
Every outer candidate must have an adequate inner solve. The fitted
coefficients depend on rho, so outer derivatives must include that dependence.
A conditional derivative holding coefficients fixed is generally insufficient.
Positive log coordinates apply to free penalties; fixed zero penalties are
handled separately rather than taking log(0).

The criterion and optimizer are independent choices. REML is a statistical
criterion; Newton and BFGS are algorithms that can optimize it where the
derivatives and dispatch support that combination. Fixed smoothing omits the
outer loop entirely.

Predictive criteria
-------------------

For a Gaussian linear smoother, A maps observations to fitted values.
Leave-one-out residuals involve the diagonal leverages A_ii. GCV replaces each
leverage by tr(A)/n:

.. math::

   \operatorname{GCV}=\frac{D/n}{\{1-\gamma\mathrm{EDF}/n\}^2},
   \qquad\mathrm{EDF}=\operatorname{tr}(A).

D is RSS in the basic Gaussian case; supported PIRLS routes use their deviance
convention. The complexity correction discourages overfitting. GACV uses a
different local approximation involving Pearson residual information. UBRE
uses known dispersion to estimate risk with an additive EDF correction.
Their detailed pages define the actual formulas and scale assumptions.

Neighborhood CV specifies which observations are deleted together and which
rows are predicted. This can align validation with local dependence or
clustered prediction tasks. NAMpy's NCV uses approximate deletion updates;
QNCV also approximates the held-out loss quadratically. These are not exact
repeated full refits. An external held-out test remains useful.

Marginal likelihood and the random-effects view
-----------------------------------------------

Split the penalty into its zero and positive eigenvalue directions. The
unpenalized directions act as fixed effects. Penalized directions can be
interpreted as Gaussian random effects with variance inversely related to
lambda. A marginal likelihood integrates over those random effects, so its
score includes both fit and curvature/volume terms.

For Gaussian compatible designs this integration is exact. For general smooth
models a Laplace approximation evaluates local curvature around the penalized
mode. Schematically, with scale and fixed-effect terms suppressed,

.. math::

   \log L_M\approx\ell(\widehat\theta)
   -\tfrac12\widehat\theta^\mathsf TS_\lambda\widehat\theta
   +\tfrac12\log|S_\lambda|_+
   -\tfrac12\log|H|+c.

The positive determinant accounts for the singular penalty's range space;
H is penalized negative likelihood curvature in identified coordinates.
REML additionally accounts for estimation of unpenalized coefficients.
Consequently fixed-effect rank and null spaces are part of the selection
problem, not just implementation bookkeeping. See :doc:`ml`, :doc:`reml` and
:doc:`laml` for the distinctions and NAMpy's dispatch conventions.

Interpreting the selected model
-------------------------------

Large lambda suppresses penalized directions. An ordinary cubic penalty
therefore favors a line, while a shrinkage/full-rank penalty can suppress the
whole smooth. Small lambda allows the available basis directions to follow
the data; it cannot compensate for an inadequate k. Assess the basis dimension
and residual structure after selecting smoothness.

Finite objective values do not establish convergence. Inspect both coefficient
and outer stopping information. Flat criterion regions, near-zero variance
components and overlapping smooths can make smoothing parameters unstable
even when predictions are comparatively stable. EDF and fitted curves are
often more interpretable than the raw numerical lambda, whose scale depends
on penalty normalization.

Comparison rules
----------------

Compare criterion values only under compatible response, weighting and
normalization conventions. REML scores for different unpenalized fixed-effect
spaces are not ordinary likelihood comparisons; use an appropriate ML or
external predictive comparison. Quasi-family scores do not supply a normalized
likelihood. Standard coefficient covariance usually conditions on the selected
smoothing parameters; smoothing uncertainty corrections require support from
the fitted route. See :doc:`../inference` and the
:doc:`fitting support matrix </api/gam_advanced>`.

Further reading
---------------

* Craven and Wahba (1979), `Smoothing noisy data with spline functions
  <https://doi.org/10.1007/BF01404567>`_.
* Wood (2011), `Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models
  <https://doi.org/10.1111/j.1467-9868.2010.00749.x>`_.
* Wood, Pya and Säfken (2016), `Smoothing Parameter and Model Selection for
  General Smooth Models <https://doi.org/10.1080/01621459.2016.1180986>`_.
* Wood and Fasiolo (2017), `A generalized Fellner-Schall method for smoothing
  parameter optimization with application to Tweedie location, scale and shape
  models <https://doi.org/10.1111/biom.12666>`_.
