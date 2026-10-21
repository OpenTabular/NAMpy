GAM Families
============

Family objects define links, variance functions, likelihoods, and response
transformations for statistical GAMs.

Family classes
--------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Public symbol
     - Family and link
   * - ``GaussianIdentityFamily``, ``GaussianLogFamily``,
       ``GaussianInverseFamily``
     - Gaussian responses with identity, log, or inverse links.
   * - ``BinomialLogitFamily``, ``BinomialProbitFamily``,
       ``BinomialCloglogFamily``, ``BinomialCauchitFamily``,
       ``BinomialLogFamily``
     - Binomial responses with the corresponding link.
   * - ``PoissonLogFamily``, ``PoissonIdentityFamily``,
       ``PoissonSqrtFamily``
     - Poisson responses with log, identity, or square-root links.
   * - ``GammaInverseFamily``, ``GammaLogFamily``,
       ``GammaIdentityFamily``
     - Gamma responses with inverse, log, or identity links.
   * - ``NegativeBinomialLogFamily``
     - Negative-binomial response with fixed or estimated shape.
   * - ``BetaRegressionFamily``
     - Beta regression for responses in the unit interval.
   * - ``OrderedCategoricalFamily``
     - Ordered categorical response with fitted cut points.
   * - ``TweedieTwFamily``
     - Tweedie response with an estimated power parameter.
   * - ``GaulssFamily``, ``gaulss``
     - Gaussian location-scale family and constructor.
   * - ``GammalsFamily``, ``gammals``
     - Gamma location-scale family and constructor.

Base classes and constructors
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Public symbol
     - Purpose
   * - ``BaseFamily``
     - Shared family contract.
   * - ``GLMFamily``
     - Single-predictor exponential-family contract.
   * - ``ExtendedFamily``
     - Single-predictor family with additional fitted parameters.
   * - ``GeneralFamily``
     - Multi-predictor likelihood contract.
   * - ``GamlssFamily``
     - Base contract for distributional statistical GAMs.
   * - ``make_gam_family``
     - Resolve a family name, tuple, dictionary, or family instance.
   * - ``clone_gam_family``
     - Copy a family so fitted state is not shared between models.

See :doc:`../user_guide/statistical_gams` for accepted family names and short
examples.

