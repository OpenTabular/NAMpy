Choose an API and response family
=================================

Start with an estimator from ``nampy`` or ``nampy.models``. Both paths expose the
same estimator classes. Advanced numerical controls and custom backend extensions
remain under ``nampy.gam`` and ``nampy.neural``.

.. list-table:: Public tasks
   :header-rows: 1

   * - Task
     - Import from
   * - Fit a model
     - ``nampy.models`` (also available directly from ``nampy``)
   * - Query a response probability law
     - ``nampy.distributions``
   * - Check residuals, prepare worms or centiles
     - ``nampy.diagnostics``
   * - Inspect contributions, effects and importance
     - ``nampy.inspection``
   * - Render prepared results
     - ``nampy.plotting``
   * - Calibrate future-observation intervals
     - ``nampy.calibration``
   * - Construct survival targets and evaluate curves
     - ``nampy.survival``

Understand model predictions
----------------------------

``predict`` follows the estimator's task: regression values, classification
labels, or natural distribution parameters for GAMLSS/LSS. For distributional
models, ``predict_point`` returns the documented family-specific point summary.
``predict_distribution`` returns a shared ``DistributionPrediction`` with explicit
probability operations. ``predict_components`` describes additive raw/link
predictors, intercepts and offsets, separately from natural parameters.

A response draw conditions on fitted parameters. It is different from coefficient
uncertainty, variation across refits and calibrated future-observation coverage.
Quantile-loss models return selected conditional quantiles; those values do not
supply a complete density or CDF.

Choose a family
---------------

GAMLSS accepts ordinary names such as ``normal``, ``gamma``, ``studentt``, ``beta``,
``negativebinom``, ``lognormal``, ``weibull``, ``bccg``, ``bct``, ``bcpe``, ``betabinomial``, ``zaga``
and ``beinf``. Neural LSS uses the same ordinary names where supported through
``ObjectiveConfig(family=...)``. Source-specific mgcv family names such as
``gaulss`` and ``gammals`` remain available for advanced statistical use.

Equal family names do not imply equal raw parameterizations, links, optimizer
routes or inference. For example, statistical Gamma location/spread and neural
Gamma shape/rate are converted explicitly into the shared probability law.
Student-t degrees of freedom are fixed configuration in the statistical adapter
and modeled in the neural family. Beta-binomial trials are row-specific
statistical context but fixed neural family configuration.

The table below is generated from the existing backend family registries and
shared probability-law metadata. It describes distributional training adapters
and available post-fit queries, not a guarantee of every smoothing or inference
combination. See :doc:`gamlss`, :doc:`training` and :doc:`survival` for restrictions.
Ordinary single-predictor GAM families have their separate source-aligned API.

.. include:: ../_generated/family_support.inc

BCT and BCPE statistical fits currently require fixed smoothing; their neural
objectives provide first-order gradients with a checked CPU support-normalizer
calculation. Observation fitting supports normal, lognormal and Weibull responses
with fixed truncation bounds and exact/left/right/interval-censored targets.
A missing probability adapter does not disable a family's existing fitting or
scoring API.

Extend the owning layer
-----------------------

Implement probability mathematics under ``nampy.distributions``. Statistical
likelihood derivatives and fitting declarations belong to ``nampy.gam.families``;
Torch transforms and differentiable likelihoods belong to
``nampy.neural.families``. Neural quantile loss belongs to
``nampy.neural.objectives``. Register constructor configuration and prediction
adaptation together in the corresponding backend registry.

Distribution contracts live in ``nampy.distributions.contracts`` and are exported
by ``nampy.distributions``. Generic feature-schema and additive-prediction
contracts remain in ``nampy.contracts``. Statistical AR(1) row transformations
live in ``nampy.gam.correlation``; censoring/truncation observations live in
``nampy.distributions.observations``.

The layout refactor removes old module paths directly. Version-6 fitted artifacts
reject earlier estimator headers before unpickling; there are no migration
readers or forwarding modules.
