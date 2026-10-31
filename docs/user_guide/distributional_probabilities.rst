Distribution parameters and probability contracts
=================================================

This reference complements :doc:`gamlss` with backend parameter mappings and
numerical restrictions.

The snapshot uses explicit probability-law parameter names, which can differ
from the estimator's existing prediction columns. The table below explains
selected mappings. The registry-generated family table in
:doc:`api_and_families` lists all declared adapters and their probability
operations; the :doc:`Box--Cox <distributional_boxcox>`,
:doc:`discrete/mixed <distributional_families>` and
:doc:`observation <distributional_observations>` guides cover their additional
parameterizations and conditions.

.. list-table:: Selected parameter mappings
   :header-rows: 1

   * - Backend family
     - Existing natural columns
     - Shared distribution parameters
   * - Statistical ``normal`` / ``gaulss``
     - ``mu``, ``sigma`` (mean and SD)
     - ``loc=mu``, ``scale=sigma``
   * - Statistical ``gamma`` / ``gammals``
     - ``mu``, ``sigma`` (mean and coefficient of variation)
     - ``shape=1/sigma**2``, ``rate=shape/mu``
   * - Neural ``normal``
     - ``mean``, ``scale``
     - ``loc=mean``, ``scale=scale``
   * - Neural ``gamma``
     - ``shape``, ``rate``
     - ``shape``, ``rate``
   * - Statistical ``student_lss``
     - ``mu``, ``sigma``; fixed ``df`` in the family
     - ``df``, ``loc=mu``, ``scale=sigma``
   * - Neural ``studentt``
     - ``df``, ``loc``, ``scale``
     - ``df``, ``loc``, ``scale``
   * - Statistical ``beta_lss``
     - ``mu``, ``phi`` (mean and precision)
     - ``alpha=mu*phi``, ``beta=(1-mu)*phi``
   * - Neural ``beta``
     - ``alpha``, ``beta``
     - ``alpha``, ``beta``
   * - Statistical ``nb_lss``
     - ``mu``, ``alpha`` (NB2 mean and dispersion)
     - ``mean=mu``, ``dispersion=alpha``
   * - Neural ``negativebinom``
     - ``mean``, ``dispersion``
     - ``mean``, ``dispersion``
   * - Statistical ``lognormal_lss`` / ``lognormal`` and neural ``lognormal``
     - ``loc``, ``scale`` (location and SD of log(Y))
     - ``loc``, ``scale``
   * - Statistical ``weibull_lss`` / ``weibull`` and neural ``weibull``
     - ``scale``, ``shape`` (WEI convention)
     - ``scale``, ``shape``
   * - Statistical ``betabinomial_lss``
     - ``mu``, ``kappa``; context ``trials``
     - ``mu``, ``kappa``; context ``trials``
   * - Statistical ``zaga_lss``
     - Positive-component ``mu``, ``sigma``; atom probability ``pi0``
     - ``shape=1/sigma**2``, ``rate=shape/mu``, ``pi0``

Statistical adapters use the fitted family's existing natural-parameter
conversion, including configured links and bounds. Neural adapters use the
fitted family's transformed parameters once, then detach to NumPy for inference.
Neural loss computation and gradients remain in Torch. Existing ``predict``,
``score``, statistical fitting and mgcv response conventions retain their
original behavior.

Student-t probability operations require positive degrees of freedom, but moments
need stronger conditions. ``mean()`` returns NaN for df <= 1;
``variance()`` returns NaN for df <= 1 and infinity for 1 < df <= 2.
Neural ``predict_point`` still returns the location in these regimes, while the
statistical ``student_lss`` family retains its existing fixed df > 2 requirement.

Probability queries evaluate the supplied response without training-time target
clipping. In particular, the existing neural beta loss clips targets near zero
and one; its training loss need not equal ``-log_prob`` at those targets.
The neural negative-binomial loss also clamps its parameter conversion. An
adapter request that activates those clamps raises ``NotImplementedError``
instead of silently returning a different law. Other negative-binomial
probability queries use the NB2 law, with variance
``mean + dispersion * mean**2``. Numerically unrepresentable size/probability
regimes are rejected; a Poisson approximation is not substituted. This differs
from small-dispersion Poisson branches in some R GAMLSS NBI probability functions.

Probability adaptation requires an explicit backend declaration. Ordinary mgcv
``scat``, ``betar`` and ``nb`` do not acquire a shared adapter merely by matching
a distributional family's name. Families and subclasses without a declared
adapter raise ``NotImplementedError`` from ``predict_distribution``; their
existing fitting, prediction and scoring remain available. Vector events and
inference bands for centile estimates are not added by this interface. Censoring and truncation use the separate
:doc:`distributional_observations` contract. A quantile loss does not define a normalized probability law.

To add a law, implement the structural ``nampy.distributions.DistributionLaw``
protocol and construct ``DistributionPrediction(law, parameters)``. Its
``DistributionSpec`` declares named parameters, a parameterization version,
response kind, event shape and available operations. The law validates its
parameter domains and evaluates the declared operations. Rowwise fixed inputs,
such as trial counts, can be declared in ``spec.context_names`` and supplied
through ``DistributionPrediction(..., context=...)``. The object copies and
slices that context with the same row mask as the parameters. No inheritance from
a concrete estimator, statistical family or Torch module is required.
Unavailable operations raise explicit errors even when a method exists on the
shared prediction object. Declaring probability support does not enable any
fitting or smoothing route.

Writing a statistical extension family
--------------------------------------

The internal ``_DistributionalFamily`` adapter derives predictor count from
the declared predictor names and links. It assembles likelihood derivatives,
weights, offsets and initialization for any positive predictor count, reusing
the existing joint solver. Natural-parameter bounds are declared per named
column; no particular parameter position is implicitly positive.

The default natural transforms support independent identity, log and logit
links. Coupled transforms or a different number of natural parameters require
family-owned transforms and Jacobians. Full Jacobians retain covariance between
predictor blocks; treating coupled parameters as independent is not supported.
When natural and predictor counts differ, public additive components additionally
need a family-owned padded link representation and predictor-index mapping;
padding is not inferred. A custom family also needs an explicit probability
adapter before ``predict_distribution`` is available.
Initialization must return one target per predictor. New families supply their
response validation, normalized likelihood, initialization, response moments
and parameter meanings; they do not supply a new solver.

The default extension is fixed-smoothing only. Declare the available likelihood
derivative order and opt into automatic fitting capabilities separately, after
validation; automatic smoothing requires fourth-order derivatives here.
Three- and four-predictor normalized test families exercise derivative packing,
fitting and initialization, and a coupled three-predictor transform exercises
four natural outputs. These checks do not establish support for every new law
or formula composition. Existing source-specific mgcv general families retain
their current implementations and numerical paths.

For methodological citations and paper links, see :doc:`/references`.
