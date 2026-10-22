Additive Models in NAMpy
========================

Additive models describe a prediction as a sum of named effects. For a model
with main effects and selected interactions,

.. math::

   \eta(x) = \beta_0
   + \sum_j f_j(x_j)
   + \sum_{(j,k) \in \mathcal{I}} f_{jk}(x_j, x_k),

where :math:`\eta` is the linear predictor and :math:`\mathcal{I}` contains the
interactions included in the model. This structure makes it possible to inspect
how individual terms contribute without reducing the model to one global
importance score.

NAMpy supports two distinct ways to estimate these effects: statistical
generalized additive models and neural additive models. They use different
representations, optimizers, and uncertainty statements; NAMpy preserves those
differences while giving both model families direct access to predictions and
term effects.

Statistical generalized additive models
---------------------------------------

A statistical GAM represents an effect with basis functions,

.. math::

   f_j(x) = \sum_{r=1}^{K_j} b_{jr}(x)\theta_{jr},

and estimates the coefficients by balancing fit against smoothness penalties,

.. math::

   \underset{\theta}{\operatorname{minimize}}\;
   -\ell(\theta) + \frac{1}{2}\sum_m
   \lambda_m\theta^\mathsf{T}S_m\theta.

The smoothing parameters :math:`\lambda_m` determine how much flexibility each
penalized component retains. NAMpy can estimate them with GCV, UBRE, ML, REML,
or Laplace approximate marginal likelihood, depending on the family and model.

NAMpy contains a Python-native statistical GAM engine rather than an interface
that sends models to R. Its formula compiler, smooth constructors, penalized
solvers, prediction routines, inference, and diagnostics cover the main
analytical stages familiar from ``mgcv``:

* formulas with parametric terms, transformations, offsets, ``s()``, ``te()``,
  ``ti()``, and ``t2()`` terms;
* thin-plate, cubic, cyclic, P-spline, Duchon, Gaussian-process, adaptive,
  spherical, random-effect, MRF, factor-specific, tensor, and
  shape-constrained smooths;
* automatic smoothness selection, shrinkage and term selection, linked
  smoothing parameters, observation weights, and Gaussian AR(1) correction;
* Gaussian, binomial, Poisson, Gamma, negative-binomial, beta, Tweedie,
  ordered-categorical, and supported multi-parameter families;
* response, link, term, interaction, and linear-predictor-matrix predictions,
  including standard errors and unconditional covariance where supported;
* summaries, ANOVA, residuals, smooth derivatives, concurvity, basis-dimension
  checks, variance components, likelihood criteria, and GAM plots.

The :ref:`statistical-gam-tools` page is the complete, symbol-level inventory.
The :doc:`api/gam` reference provides exact signatures.

Neural additive models
----------------------

A neural additive model keeps the additive predictor but learns each effect
with a feature network,

.. math::

   f_j(x_j) = h_j(x_j; w_j).

The network parameters :math:`w_j` are fitted by gradient-based optimization.
Different architectures change the feature function, parameter sharing, or
interaction mechanism, but an effect interpreted as :math:`f_j(x_j)` remains a
function of the stated feature. NAMpy exposes the resulting term contributions,
interaction contributions, and feature importance through the fitted
estimator. See :doc:`models/index` for the theoretical description and code for
each architecture.

Distributional additive models
------------------------------

Mean regression is not enough when conditional spread, skewness, or tail
behavior changes with the predictors. Distributional models assign a separate
additive predictor to each distribution parameter,

.. math::

   \eta_k(x) = \beta_{0k} + \sum_j f_{jk}(x_j),
   \qquad \theta_k(x) = g_k^{-1}\!\left(\eta_k(x)\right).

NAMpy returns both natural distribution parameters :math:`\theta_k(x)` and
point summaries such as the conditional mean. Parameter-wise effects remain on
their predictor scales unless a plotting method explicitly applies the inverse
links. See :doc:`user_guide/gamlss` for the exact semantics.

Interpreting additive effects
-----------------------------

For a fitted model, the intercept and returned terms reconstruct the linear
predictor. This is an accounting identity, not a claim that every effect is
uniquely identified. Centering removes constant-shift ambiguity for main
effects, while interactions and correlated predictors can still admit several
similar decompositions. Effect plots should therefore be read together with
the model specification, reference data, and uncertainty or stability checks.
The :doc:`user_guide/interpretability` page defines these contracts precisely.

Continue with NAMpy
-------------------

* :doc:`quickstart` fits and inspects complete statistical and neural models.
* :doc:`user_guide/statistical_gams` documents the statistical GAM workflow.
* :doc:`models/index` explains every model architecture with code.
* :doc:`notebook_tutorials` contains executable analyses.
* :doc:`api/index` is the signature-level reference.
