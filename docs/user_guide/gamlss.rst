Distributional GAMs and GAMLSS
==============================

GAMLSS extends additive regression beyond the mean (Rigby and Stasinopoulos, 2005).

This guide covers ``GAMLSS`` formulas, parameter ordering, predictions, and
scoring. For fitted statistical examples, see :doc:`../notebooks/02_statistical_gams`
and :doc:`../notebooks/gamma_location_scale`. For a separate comparison of neural
distributional, quantile, and ensemble models, see
:doc:`../notebooks/04_distributional_models`.

The lower-level :class:`nampy.gam.GAM` also accepts supported multi-parameter
families and exposes their statistical parameterization directly.

.. _distributional-model-concepts:

Decide which aspects of the response to model
---------------------------------------------

A mean model assumes that the conditional expectation is the main quantity
that changes with predictors. Distributional regression is useful when
variability, skewness, tail behavior, or another parameter also changes.

For parameter :math:`\theta_k`, NAMpy fits an additive predictor
:math:`\eta_k` and applies the parameter's inverse link,

.. math::

   \eta_k(x)=\beta_{0k}+\sum_j f_{jk}(x_j)
   +\sum_{S\in\mathcal I_k}f_{Sk}(x_S)+o_k(x),\qquad
   \theta_k(x)=h_k^{-1}\{\eta_k(x)\}.

Together these parameters specify the response distribution,
:math:`Y\mid x\sim\mathcal D\{\theta_1(x),\ldots,\theta_K(x)\}`.
The term definitions and identifiability considerations follow
:ref:`additive-predictor-interpretation`, separately for each parameter.

Terms add on the :math:`\eta_k` scale. Natural parameters generally do not add
after a nonlinear inverse link. Conditional standard deviation describes
observation-level predictive spread. It is different from a standard error
for an estimated mean and from variation across repeated fits.

Quantile models instead estimate selected conditional quantiles using pinball
loss. These quantiles do not by themselves specify a complete probability
density; their spacing and empirical coverage describe different properties.

Public parameter contract
-------------------------

``GAMLSS.predict`` returns the family-defined point prediction.
``predict_parameters`` returns natural distribution parameters with columns
in ``parameter_names_`` order. ``predict_link`` returns additive predictors
with columns in ``predictor_names_`` order. See :doc:`model_workflow`.

.. list-table:: Built-in families
   :header-rows: 1

   * - Public aliases
     - ``parameter_names_``
     - Meaning
   * - ``normal``, ``gaussian``, ``gaulss``
     - ``("mu", "sigma")``
     - Normal conditional mean and standard deviation
   * - ``gamma``, ``gammals``
     - ``("mu", "sigma")``
     - Gamma conditional mean and coefficient of variation, so
       ``Var(Y | X) = mu**2 * sigma**2``

``predict_point`` returns the family's declared point summary, usually a conditional
mean or median. Use ``predict_distribution(...).mean()`` or ``.quantile(0.5)``
when a specific summary is required and supported.

``score`` returns weighted mean log likelihood (higher is better), while the
default ``evaluate`` result reports its negative. For censoring targets, both
use the supplied observation bounds rather than evaluating the target as an
exact response; see :doc:`distributional_observations`.

Formula and array modes
-----------------------

A named mapping is the preferred formula form. It prevents accidental changes
to parameter order and requires secondary formulas to be one-sided:

.. code-block:: python

   from nampy.models import GAMLSS

   model = GAMLSS(
       family="normal",
       formula={
           "mu": "y ~ s(age, bs='cr', k=10) + group",
           "sigma": "~ s(age, bs='cr', k=8)",
       },
       smoothing_method="reml",
   ).fit(data)

   parameters = model.predict_parameters(data)
   standard_errors = model.standard_errors(data)
   components = model.predict_components(data)
   components.validate_additive_reconstruction()


An ordered formula list remains supported for compatibility. Without a
formula, each distribution parameter receives its own independently compiled
copy of the configured main-effect smooths:

.. code-block:: python

   model = GAMLSS(family="gamma", k=10, basis="tp").fit(X, y)

Offsets are supplied on the link scale. Follow the shape documented for the
method you are calling; ``predict_distribution`` accepts an array with one row
per observation and one column per predictor.

For fitting through ``GAMLSS.fit``, supply an array with shape
``(n_rows, n_predictors)`` in the family's parameter order. A single column is
not broadcast across parameters. Contributions in ``predict_components`` are
zero-padded matrices, so every term has the same
``(n_samples, n_parameters)`` shape as the link prediction.

For neural LSS effect plots, centering references, and uncertainty envelopes,
see :doc:`interpretability`.

Statistical smoothness selection
--------------------------------

Supported multi-parameter fitting routes use the general smooth-model
framework described by Wood, Pya, and Säfken (2016), including Laplace
approximate marginal likelihood where available. See :doc:`../api/gam_advanced`
for compatible family and fitting combinations.

Query the conditional distribution
----------------------------------

Both statistical ``GAMLSS`` and neural ``*LSS`` estimators provide
``predict_distribution`` for normal, Gamma, Student-t, beta, negative-binomial,
lognormal and Weibull
families. Statistical ``betabinomial_lss`` and ``zaga_lss`` additionally support
trial counts and mixed zero/positive outcomes, respectively. It returns a shared
``nampy.distributions.DistributionPrediction`` containing a snapshot of the
conditional probability law at each prediction row:

.. code-block:: python

   distribution = model.predict_distribution(X_test)
   log_density = distribution.log_prob(y_test)  # (n,), unweighted
   probabilities = distribution.cdf(y_test)    # (n,)
   intervals = distribution.quantile([0.05, 0.95])  # (n, 2)
   draws = distribution.sample(n_draws=500, random_state=7)  # (500, n)
   means = distribution.mean()                # (n,)
   variances = distribution.variance()        # (n,)
   available = distribution.spec.operations

``cdf_left`` returns ``P(Y < y)``; ``cdf`` returns ``P(Y <= y)``. They agree
for the supplied continuous laws; a custom discrete or mixed law must handle
jumps explicitly. ``log_cdf``, ``survival`` and ``log_survival`` call the
probability implementation's corresponding operations rather than subtracting
or taking logarithms of rounded CDF values. The supplied laws use SciPy kernels;
extreme Gamma tails can still underflow. No arbitrary-precision tail calculation
or parameter derivatives of CDFs are implied.

Evaluation accepts a scalar response value or exactly one value per prediction
row. Quantiles use a separate probability axis: even one probability returns
``(n, 1)``. Prediction rows retain their order, including all-NaN missing rows
under statistical ``na_action='pass'``. Parameter vectors are copied and made
read-only. Samples condition on fitted parameters; they do not include
coefficient uncertainty, bootstrap variation or an ensemble predictive mixture.
Scoring weights remain the estimator's responsibility.

See :doc:`distributional_probabilities` for the adapter parameter table, moment
existence rules, numerical boundaries and custom probability-law contracts.


Read conditional density slices
-------------------------------

.. figure:: /_static/features/distribution_slices.png
   :alt: Three fitted lognormal density slices shift right, spread out and become more skewed as x increases; dotted lines mark medians.
   :width: 100%

   The fitted lognormal scale grows from about 0.28 at x=-0.8 to 0.68
   at x=0.8, while log-location also rises. The resulting density moves
   right, spreads out and becomes more skewed, so a location change alone
   would miss much of the response behavior. Dotted lines mark conditional
   medians, which differ from means in these asymmetric distributions.

Choose a focused task
---------------------

.. toctree::
   :maxdepth: 1

   distributional_families
   distributional_observations
   distributional_boxcox
   distributional_comparison
   distributional_diagnostics
   distributional_probabilities

References
----------

The methodological references include [Rigby2005]_ and [Wood2016]_, with
[Cole1992]_, [Rigby2004]_ and [Rigby2006]_ for BCCG/LMS, BCPE and BCT.
The corresponding neural LSS framework is [ThielmannLSS2024]_. See
:doc:`/references` for links to these papers and the additional response-family
references, including censored/truncated observation likelihoods.


Rigby and Stasinopoulos (2005), `Generalized additive models for location,
scale and shape <https://doi.org/10.1111/j.1467-9876.2005.00510.x>`_. Journal
of the Royal Statistical Society: Series C 54:507–554.

Wood, Pya, and Säfken (2016), `Smoothing Parameter and Model Selection for
General Smooth Models <https://doi.org/10.1080/01621459.2016.1180986>`_. Journal
of the American Statistical Association 111:1548–1563.
