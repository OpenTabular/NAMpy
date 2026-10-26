.. _statistical-gam-tools:

Statistical GAMs in Python
==========================

NAMpy provides a Python implementation of the statistical GAM workflow and
uses ``mgcv`` terminology where the corresponding behavior is available. This
page collects the formula syntax, smooth constructors, fitting controls,
prediction methods, inference, and diagnostics in one place. The
:doc:`../api/gam` reference gives full call signatures.

Coverage map
------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Specify

      Formulas, transformed covariates, offsets, parametric interactions,
      smooth main effects, tensor products, factor smooths, linked penalties,
      and multiple linear predictors.

   .. grid-item-card:: Estimate

      Penalized Gaussian, PIRLS, transformed-coefficient, and general-family
      solvers with fixed or automatically selected smoothing parameters.

   .. grid-item-card:: Predict

      Response and link values, term and interaction contributions, prediction
      matrices, conditional or unconditional uncertainty, offsets, and
      missing-data controls.

   .. grid-item-card:: Diagnose

      Summaries, ANOVA, residuals, smooth derivatives, concurvity, basis checks,
      likelihood criteria, effective degrees of freedom, and term plots.

.. _smoothness-and-basis-dimension:

Choose basis capacity and smoothness
------------------------------------

The basis dimension ``k`` gives a smooth enough capacity to represent a range
of curves. The smoothing penalty determines how much of that capacity the fit
uses. Increasing ``k`` therefore does not instruct the model to fit a more
irregular curve; it raises the ceiling on representational complexity.

For a statistical GAM, each smooth has a basis representation,

.. math::

   f_j(x)=\sum_{r=1}^{K_j}b_{jr}(x)\theta_{jr}.

The coefficients balance fit against roughness by minimizing

.. math::

   -\ell(\theta)+\frac{1}{2}\sum_m\lambda_m\theta^\mathsf T S_m\theta.

The matrices :math:`S_m` encode penalties, and the smoothing parameters
:math:`\lambda_m` control their strength. Effective degrees of freedom
summarize the flexibility left after penalization and need not be integers.

For statistical GAMs, fixed smoothing parameters are useful when a penalty is
known in advance. GCV, UBRE, ML, REML, and the supported related criteria
estimate smoothing from the data. After fitting, inspect effective degrees of
freedom, residual patterns, and ``k_check()``. A basis-dimension diagnostic is
a warning tool rather than a universal acceptance test.

The tutorial :doc:`../tutorials/first_gam` shows these choices in a complete fit. The :doc:`../api/gam` reference documents the available bases, criteria, and fitting controls.

Model interfaces
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Symbol
     - Use it for
     - Minimal use
   * - ``nampy.gam.GAM``
     - Formula-based GAMs and direct control of statistical fitting.
     - ``GAM(formula="y ~ s(x)").fit(data=df)``
   * - ``nampy.models.GAMRegressor``
     - Regression from a feature matrix with an estimator interface.
     - ``GAMRegressor(k=10).fit(X, y)``
   * - ``nampy.models.GAMClassifier``
     - Binary classification with class and probability predictions.
     - ``GAMClassifier(k=10).fit(X, y)``
   * - ``nampy.models.GAMLSS``
     - Multi-parameter distributional GAMs.
     - ``GAMLSS(family="normal", formula=formulas).fit(df)``

.. _gam-formula-language:

Formula language
----------------

For a modeling-question-first introduction with fitted plots, see
:ref:`formula-visual-reference`. The complete ``bs=`` code inventory is in
:ref:`gam-smooth-bases`.

.. list-table::
   :header-rows: 1
   :widths: 26 39 35

   * - Symbol
     - Meaning
     - Example
   * - ``x``
     - Linear numeric or factor term.
     - ``y ~ x + group``
   * - ``x1:x2``
     - Parametric interaction.
     - ``y ~ x1 + x2 + x1:x2``
   * - ``x1*x2``
     - Main effects and their parametric interaction.
     - ``y ~ x1*x2``
   * - ``s(...)``
     - One- or multi-variable smooth using the selected ``bs`` code.
     - ``y ~ s(x, bs="tp", k=12)``
   * - ``te(...)``
     - Full tensor-product smooth.
     - ``y ~ te(x, z, bs=c("cr", "cr"))``
   * - ``ti(...)``
     - Tensor interaction used with lower-order main effects.
     - ``y ~ s(x) + s(z) + ti(x, z)``
   * - ``t2(...)``
     - Alternative tensor product with an ANOVA-style penalty decomposition.
     - ``y ~ t2(x, z, bs=c("cr", "cr"))``
   * - ``offset(...)``
     - Known additive contribution to the linear predictor.
     - ``count ~ s(x) + offset(log_exposure)``
   * - ``0`` or ``-1``
     - Remove the intercept.
     - ``y ~ 0 + s(x)``
   * - ``I()``, ``log()``, ``sqrt()``, ``exp()``
     - Numeric transformations evaluated from data columns.
     - ``y ~ s(log(x)) + I(z**2)``

Fitted formula transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``factor(group)`` uses treatment coding; ``ordered(group)`` uses R-style
orthogonal polynomial contrasts. Both accept explicit ``levels`` and ``labels``;
repeated labels merge levels. Supply explicit levels when supplying labels to
make their mapping stable across new-data subsets. ``interaction(a, b)``
constructs a factor, with ``drop``, ``sep`` and ``lex_order`` options.

``poly(x, 3)`` produces three orthogonal polynomial columns;
``poly(x, 3, raw=True)`` produces ordinary powers. For multivariate polynomials,
use ``poly(cbind(x, z), degree=2)``; orthogonal recurrence coefficients are
retained for prediction and persistence. The alternative ``poly(x, z,
degree=2)`` follows R's positional-argument rules, which can fail on single-row
prediction; the ``cbind`` form avoids that ambiguity.

``cut(x, breaks=[0, 1, 2], labels=["low", "high"], include_lowest=True)``
uses explicit bins. ``cut(x, breaks=4)`` selects equal-width intervals and
creates R-style interval labels. Use ``labels=False`` for numeric bin codes.
As in R, a scalar break count recomputes the range on each evaluation batch;
new labels may then fail the fitted model's factor-level check. Use explicit
break points and labels for stable prediction bins.

Unseen factor values and values outside supplied cut breaks are rejected
explicitly; prediction rows with missing inputs follow ``na_action``. These
functions form a supported formula subset, not an arbitrary R evaluator.

Grouped binomial responses
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use two response columns with the low-level statistical GAM:

.. code-block:: python

   model = GAM(
       formula="cbind(success, failure) ~ s(x, bs='ps', k=8)",
       family="binomial",
       optimize_smoothing=True,
       smoothing_method="REML",
   ).fit(data=df)
   probability = model.predict(df, type="response")

The array API also accepts an ``(n, 2)`` response in success/failure order.
Trial counts multiply any observation weights, and zero-trial rows contribute
zero weight. Predictions remain success probabilities. ``GAMClassifier``
continues to require one binary class label per row; grouped counts use ``GAM``.

Smooth options
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 43 35

   * - Symbol
     - Purpose
     - Example
   * - ``k=``
     - Set the basis dimension.
     - ``s(x, k=20)``
   * - ``m=``
     - Set basis- or penalty-order parameters.
     - ``s(x, bs="ps", m=c(3, 2))``
   * - ``fx=True``
     - Treat the term as fixed rather than estimating smoothness.
     - ``s(x, fx=True)``
   * - ``sp=``
     - Supply a term-level smoothing parameter.
     - ``s(x, sp=0.5)``
   * - ``select=True``
     - Add null-space shrinkage for term selection.
     - ``GAM(formula="y ~ s(x)", select=True)``
   * - ``pc=``
     - Apply a point constraint when supported by the basis.
     - ``s(x, pc=0.0)``
   * - ``xt=``
     - Pass basis-specific information such as a graph or change point.
     - ``s(region, bs="mrf", xt=graph)``
   * - ``knots=``
     - Supply user-defined knot locations.
     - ``model.fit(data=df, knots={"x": knots})``

Shape constraints
-----------------

Shape constraints use the same ``s(..., bs=...)`` syntax as ordinary
smooths. See :doc:`shape_constrained_gams` for the modeling workflow and
:ref:`shape-constrained-basis-codes` for the exhaustive code inventory.

Matrix-valued linear functionals
--------------------------------

Store each observation's evaluation locations as an array-valued DataFrame
cell. ``s(X, Z, by=L, bs='tp')`` sums the smooth over paired locations using
quadrature weights from ``L``. All coordinate and weight rows must have the
same width. Omitting ``by`` uses unit weights; ``sample_weight`` separately
weights observations. New-data rows can use a different quadrature width.

Numeric and factor random effects and full/reduced-rank MRFs also support this
summation convention. Use lists of labels for categorical matrix cells.
For ``fs`` functionals with scalar group factors,
validated bases include ``cr``, ``ps``, one-null-direction ``tp`` and
single-penalty ``sf``. ``sz`` accepts ``tp``, ``cr``, ``ps``, single-penalty
``sf``, ``sw`` and fully known-boundary ``so`` with scalar factors.
Soap factor interactions discard marginal boundary offsets, following mgcv.
Factors may also vary across quadrature columns when each factor row is an
explicit ``pandas.Categorical`` with consistent declared levels and width of
at least two. For these matrix factors, ``fs`` supports ``sf``, univariate
``tp`` with ``m=1``, and ``ps`` with ``m=[2,1]``. ``sz`` supports the documented
numeric and soap bases, with every joint group observed when using several
matrix factors. See ``examples/example_matrix_factor.py`` and
:ref:`matrix-functional-smooths` for the input recipe and remaining guards.
Tensor functionals
sum products evaluated at each paired location.

Matrix ``cs`` terms are rejected even for distinct locations because portable
null-space shrinkage parity remains unresolved. Scalar ``cs`` is supported.

Run ``python examples/example_matrix_functionals.py`` from a source checkout
for complete surface and tensor fits with prediction standard errors.
See :ref:`matrix-functional-smooths` for basis-specific compositions, repeated
location packing, and explicit restrictions on shrinkage, soap offsets,
adaptive terms and linked tensor groups.

Families and links
------------------

.. list-table::
   :header-rows: 1
   :widths: 26 39 35

   * - Family
     - Available links or form
     - Example
   * - ``gaussian``
     - Identity, log, and inverse.
     - ``GAM(family="gaussian")``
   * - ``binomial``
     - Logit, probit, complementary log-log, cauchit, and log.
     - ``GAM(family="binomial")``
   * - ``poisson``
     - Log, identity, and square root.
     - ``GAM(family="poisson")``
   * - ``Gamma``
     - Inverse, log, and identity.
     - ``GAM(family="Gamma")``
   * - ``inverse_gaussian``
     - ``1/mu^2`` (default), inverse, log, and identity; positive responses.
     - ``GAM(family=("inverse_gaussian", "log"))``
   * - ``quasipoisson`` / ``quasibinomial``
     - Quasi variance ``mu`` / ``mu(1-mu)``; default log / logit links.
     - ``GAM(family="quasipoisson")``
   * - ``quasi``
     - Five variance functions; default constant variance and identity link.
     - ``GAM(family={"name": "quasi", "variance": "mu^2", "link": "log"})``
   * - ``nb`` / ``negbin``
     - Estimated or fixed negative-binomial shape.
     - ``GAM(family="nb")``
   * - ``ziP`` / ``zip``
     - Hurdle Poisson counts with linked intensity and presence probability.
     - ``GAM(family={"name": "ziP", "theta": [-0.5, -0.7], "b": 0.3})``
   * - ``scat``
     - Scaled Student t location; identity, log, or inverse link.
     - ``GAM(family={"name": "scat", "theta": [5, 0.3]})``
   * - ``cnorm``
     - Censored normal location; identity, log, or square-root link.
     - ``GAM(family={"name": "cnorm", "theta": 1})``
   * - ``clog``
     - Censored logistic location; identity, log, or square-root link.
     - ``GAM(family="clog")``
   * - ``cpois``
     - Censored Poisson mean; log, identity, or square-root link.
     - ``GAM(family="cpois")``
   * - ``bcg``
     - Censored Box-Cox Gaussian location; estimated lambda and SD.
     - ``GAM(family="bcg")``
   * - ``betar``
     - Beta regression.
     - ``GAM(family="betar")``
   * - ``tw``
     - Tweedie regression.
     - ``GAM(family="tw")``
   * - ``ocat``
     - Ordered categorical response.
     - ``GAM(family="ocat")``
   * - ``gaulss`` / ``normal``
     - Gaussian location and scale.
     - ``GAMLSS(family="normal", formula=formulas)``
   * - ``gammals`` / ``gamma``
     - Gamma mean and scale.
     - ``GAMLSS(family="gamma", formula=formulas)``
   * - ``gumbls``
     - Gumbel location and log scale.
     - ``GAMLSS(family="gumbls", formula=formulas)``
   * - ``ziplss``
     - Separate hurdle-Poisson intensity and presence predictors.
     - ``GAMLSS(family="ziplss", formula=formulas)``
   * - ``multinom``
     - ``K`` log-odds predictors for categories ``0,...,K``.
     - ``GAMLSS(family={"name": "multinom", "K": 2}, formula=formulas)``
   * - ``gevlss``
     - Generalized extreme value location, log scale, and shape.
     - ``GAMLSS(family="gevlss", formula=formulas)``
   * - ``shash``
     - Sinh-arcsinh location, scale, skewness, and tail shape.
     - ``GAMLSS(family="shash", formula=formulas)``
   * - ``twlss``
     - Tweedie mean, power, and dispersion; EFS smoothing selection.
     - ``GAMLSS(family="twlss", formula=formulas)``

Scaled Student t responses
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``scat`` for a continuous location model with Student t errors. Its
single linear predictor models location; degrees of freedom and scale are
constant distribution parameters that can be fixed or estimated.

.. code-block:: python

   fixed = GAM(
       formula="y ~ s(x, bs='cr', k=7)",
       family={"name": "scat", "theta": [5, 0.3]},
       smoothing_params=[0.4],
       optimize_smoothing=False,
   ).fit(data=data)

   estimated = GAM(
       formula="y ~ s(x, bs='cr', k=7)",
       family="scat",
       smoothing_method="REML",
       optimize_smoothing=True,
   ).fit(data=data)

``theta=[df, scale]`` fixes both parameters when both entries are positive.
Omit ``theta`` to estimate both; signed constructor starts and ``min_df``
follow the rules in :doc:`../api/gam_families`. A tuple such as
``("scat", {"theta": [5, 0.3], "link": "identity"})`` or a
``ScaledTFamily`` instance is also accepted. ``log`` and ``inverse`` links
apply to location, not to the distribution scale.

SCAT uses fixed smoothing or likelihood-based ``ML``/``REML`` selection;
``LAML`` selects the REML route. As in mgcv, a ``GCV.Cp`` request uses the
extended-family REML route. SCAT does not supply a separate GCV or UBRE
criterion. ``predict(..., return_se=True)`` gives uncertainty for the fitted
location, rather than a response prediction interval.

The executable ``examples/example_scat_gam.py`` demonstrates fixed distribution
parameters and fixed smoothing. Offsets, nonunit weights, multiple penalties,
and fixed or estimated ancillary parameters are covered by static mgcv
references. Fixed smoothing and REML also support zero observation weights.
``ML`` with zero weights raises an explicit error because the corresponding
mgcv 1.9-4 reference runs crash. Joint ancillary-parameter estimation with
``smoothing_optimizer="efs"`` is unsupported.

When a design drops dependent coefficients, fits requiring outer-parameter
uncertainty corrections raise an explicit error. The upstream full-Hessian
Cholesky branch is sensitive to rounding in this case. Fixing both ancillary
parameters and smoothing retains conditional fitting for such designs.

The external ``GAM(scale=...)`` dispersion is distinct from the Student t
scale in ``theta``. Its default is one. A positive override affects the
selection criterion and covariance; it can change estimated smoothing or
ancillary parameters. Only when both are fixed does changing this dispersion
leave the fitted location unchanged and multiply its coefficient covariance.

For uncertainty comparisons near a flat smoothing optimum, use the same
convergence controls in both implementations. The difficult zero-weight,
two-smooth reference cases use ``control={"epsilon": 1e-10,
"newton": {"conv_tol": 1e-10}}`` for unconditional covariance comparisons.
The original default-control fits remain tested. At a boundary, validation
compares physical degrees of freedom and scale, or the vanishing penalized
contribution as a smooth becomes linear, rather than an arbitrary finite
log-parameter representative. Working residuals can be sensitive near zero
observed curvature and are undefined on zero-weight rows; see
:doc:`../api/gam_families` for the residual comparison contract.

Hurdle Poisson counts
~~~~~~~~~~~~~~~~~~~~~

``ziP`` models nonnegative counts using one latent log-intensity predictor.
A positive count follows a zero-truncated Poisson distribution; its probability
is tied to the same predictor. This is the single-predictor mgcv family,
not a model with an independently specified presence formula.

.. code-block:: python

   count_model = GAM(
       formula="y ~ s(x, bs='cr', k=7)",
       family={"name": "ziP", "theta": [-0.5, -0.7], "b": 0.3},
       smoothing_params=[0.4],
       optimize_smoothing=False,
   ).fit(data=data)

   expected_count, mean_se = count_model.predict(type="response", return_se=True)
   latent_log_intensity = count_model.predict(type="link")

The two supplied ``theta`` values are working-scale parameters and fix both
ancillary parameters. Omit ``theta`` to estimate both; use
``smoothing_method="REML", optimize_smoothing=True`` to select smoothing too.
The ``b`` argument is a lower bound on the slope tying presence to intensity.
See :doc:`../api/gam_families` for the exact parameterization and current scope.

Only unit observation weights are currently accepted. Binary responses,
nonidentity links, positive ``min_sp``, and joint ancillary estimation with
``smoothing_optimizer="efs"`` are rejected. Response residuals compare counts
with predicted expected counts; stored fitted values are latent log intensities.
Designs that drop dependent coefficients are guarded when outer-parameter
uncertainty is required; fixed-theta, fixed-smoothing conditional fits remain
available. The example ``examples/example_zip_gam.py`` simulates hurdle counts and fits a
model with fixed ancillary parameters and smoothing.

Censored normal observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a paired response to retain a detection limit or an interval instead of
pretending its endpoint is an exactly observed outcome. This example fixes
the underlying normal standard deviation at one and fixes smoothing:

.. code-block:: python

   from nampy.gam.families import CensoredNormalFamily

   censored_model = GAM(
       formula="cbind(value, bound) ~ s(x, bs='cr', k=7)",
       family=CensoredNormalFamily(theta=1.0),
       smoothing_params=[0.7],
       optimize_smoothing=False,
   ).fit(data=data)

   location, location_se = censored_model.predict(
       data[["x"]], type="response", return_se=True
   )

For an exact observation set ``bound=value``. Set ``bound=-np.inf`` for a
left-censor threshold, ``bound=np.inf`` for a right-censor threshold, or a
different finite bound for an interval. Both finite endpoint orders are
accepted. For example, ``[2, np.inf]`` means the response exceeds two;
``[2, 3]`` means it lies between two and three. Predictions require only the
covariates and describe the latent normal location.

``examples/example_cnorm_gam.py`` creates all four observation types and
fits this fixed-SD, fixed-smoothing model with unit weights. Positive weights
have precision semantics in the upstream family, rather than frequency
semantics. Simulation and quantile diagnostics are unavailable. See
:doc:`../api/gam_families` for the source constructor and AIC quirks and the
current validation scope. Omitting ``theta`` estimates SD; use
``smoothing_method="REML", optimize_smoothing=True`` to estimate smoothing
too. ML and REML have direct endpoint and joint-derivative references.
Zero precision weights and positive ``min_sp`` are rejected; estimated SD
with ``smoothing_optimizer="efs"`` remains guarded for ``cnorm``.

Censored logistic observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``family={"name": "clog", "theta": 0.7}`` with the same
``cbind(value, bound)`` formula to fit a latent logistic location with fixed
scale. Omit ``theta`` to estimate scale. Positive observation weights have
precision semantics; each row's logistic scale is divided by the square
root of its weight. The logistic scale differs from its standard deviation.
Predictions need only the covariates; they return latent location rather
than a censoring probability. To estimate logistic scale and smoothing together:

.. code-block:: python

   censored_model = GAM(
       formula="cbind(value, bound) ~ s(x, bs='cr', k=7)",
       family="clog",
       smoothing_method="REML",
       smoothing_optimizer="efs",
       optimize_smoothing=True,
   ).fit(data=data)

EFS also supports log/square-root links and positive precision weights.
For estimated-scale ML with a log/square-root link use
``smoothing_optimizer="optim"``: the corresponding Newton/BFGS source
routes have captured crashes. Newton fits that fail selection raise without
replacing an earlier fitted state; successful weighted and nonidentity
fits are supported. Nonconverged BFGS/nlm fits issue warnings.

``examples/example_clog_gam.py`` simulates logistic responses and records
exact, interval, left- and right-censored observations. It uses
``GAMRegressor`` and checks the shared additive prediction contract.
See :doc:`../api/gam_families` for the source AIC convention, failed-source
state evidence, and shared extended-family restrictions. With all smoothing
parameters fixed, estimate scale using Newton, BFGS or optim; the equivalent
EFS source call errors.

Censored counts and Box-Cox responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``cpois`` for a Poisson mean observed exactly or through censor bounds:

.. code-block:: python

   count_model = GAM(
       formula="cbind(count, bound) ~ s(x, bs='cr', k=7)",
       family="cpois",
       smoothing_method="REML",
       optimize_smoothing=True,
   ).fit(data=data)

   mean, mean_se = count_model.predict(newdata, return_se=True)

Equal endpoints denote an exact count; a ``-np.inf`` bound denotes a left
threshold and ``np.inf`` a right threshold. A finite interval has discrete
semantics ``lo < Y <= hi``: ``[2, 4]`` includes three and four. The source
likelihood hooks ignore observation weights, so they do not act as frequency
weights. All three mean links, fixed/ML/REML smoothing and the tested Newton,
BFGS, optim, nlm and EFS routes have direct references. Source quantile and
simulation hooks are unavailable. ``examples/example_cpois_gam.py`` demonstrates
censored count fitting.

``bcg`` models Gaussian location after a Box-Cox transformation, with lambda
and log SD estimated by default. It accepts identity, log and square-root
location links. Use ``cbind(value, bound)``; here a nonpositive second-column
bound denotes left censoring. Predictions remain on the transformed Gaussian
location scale. Positive weights have precision semantics.

.. code-block:: python

   transformed_model = GAM(
       formula="cbind(value, bound) ~ s(x, bs='cr', k=6)",
       family="bcg",
       smoothing_method="REML",
       optimize_smoothing=True,
   ).fit(data=data)

   lambda_and_sd = transformed_model.family.getTheta(trans=True)

Fixed/estimated parameters, exact/mixed censoring and fixed/ML/REML smoothing
have source references. EFS supports the referenced two-parameter ancillary
updates and selected fits; estimating the parameters with EFS requires a free
smoothing parameter. The source's positive fixed constructor creates an
invalid three-entry state and is rejected. To specify a fixed working state,
create the family then call its own ``putTheta([lambda, log_sd])`` before
fitting, as shown in :doc:`../api/gam_families`. That page also documents
source likelihood/Jacobian conventions and guarded combinations.
``examples/example_bcg_gam.py`` uses the default estimated family.

Multiple distribution predictors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``GAMLSS`` to fit separate additive predictors for distribution
parameters. Supply one formula per predictor, with a response only in the
first formula. ``predict`` returns natural parameters in ``parameter_names_``
order; ``predict_link`` or ``predict(raw=True)`` returns the additive
predictors. Natural-parameter uncertainty comes from
``standard_errors(type="response")``. These parameter errors differ from
the response quantities and source SE conventions of low-level ``GAM``.

.. list-table:: Natural parameters and point predictions
   :header-rows: 1
   :widths: 18 45 37

   * - Family
     - ``GAMLSS.predict`` columns
     - ``predict_point``
   * - ``gumbls``
     - ``location, scale``
     - ``location + EulerGamma * scale``
   * - ``ziplss``
     - ``lambda, presence``
     - Expected hurdle count.
   * - ``multinom``
     - ``p0,...,pK`` for ``K+1`` categories.
     - First most-probable category.
   * - ``gevlss``
     - ``mu, sigma, xi``
     - Source conditional-mean expression.
   * - ``shash``
     - ``mu, sigma, epsilon, delta``
     - Conditional median.
   * - ``twlss``
     - ``mu, p, phi``
     - Mean ``mu``.

For example, fit a GEV location smooth with constant scale and shape:

.. code-block:: python

   from nampy import GAMLSS

   model = GAMLSS(
       family="gevlss",
       formula={"mu": "y ~ s(x, bs='cr', k=6)",
                "sigma": "~ 1", "xi": "~ 1"},
       smoothing_method="REML",
   ).fit(data)

   parameters = model.predict(newdata)
   parameter_se = model.standard_errors(newdata)
   components = model.predict_components(newdata)
   components.validate_additive_reconstruction()

Gumbel location uses identity, with identity or the bounded source log-scale
link. Hurdle-Poisson ``ziplss`` uses two identity links on latent intensity
and presence predictors, with independently specified formulas. GEV location
accepts identity/log, log scale uses identity, and shape accepts identity or a
shifted logit confining it to ``(-1, 0.5)``. Its source zero-shape cutoff and
residual expressions are retained, including an undefined point-mean ratio
at exactly zero shape.

For multinomial responses use integer categories ``0,...,K`` and one log-odds
formula for each nonreference category:

.. code-block:: python

   categories = GAMLSS(
       family={"name": "multinom", "K": 2},
       formula={"eta1": "category ~ s(x, bs='cr', k=6)",
                "eta2": "~ s(z, bs='cr', k=6)"},
   ).fit(data)

   probabilities = categories.predict(newdata)       # p0, p1, p2
   log_odds = categories.predict_link(newdata)       # eta1, eta2
   predicted_category = categories.predict_point(newdata)

Multinomial raw prediction has ``K`` columns. Its additive components include
an additional zero-logit baseline column, and natural-probability standard
errors use the full probability Jacobian and cross-predictor covariance.
``GAM`` probability SEs preserve the source covariance-clipping expression.
The source likelihood includes a class-zero constant; normalized public
density scoring uses actual probabilities. See :doc:`../api/gam_families`
for its fourth-order source return-value quirk.

Sinh-arcsinh ``shash`` uses identity/logeb/identity/identity links and a
positive lower scale bound. Its source fitting ridge on log tail shape is
omitted from normalized public density scoring. Nonzero offsets are rejected;
an exactly zero standardized residual retains the source infinite fitting
likelihood anomaly, while public density stays finite there.

Tweedie ``twlss`` accepts log/identity/sqrt mean links and identity links for
transformed power and log dispersion, with ``1 < a < b < 2`` power bounds.
Smoothing selection automatically uses EFS because the source exposes
likelihood derivatives only through second order. Fixed smoothing and EFS
fits are supported. Source captures for a smooth after a parametric-only
middle predictor specify fixed smoothing inside the individual ``s(...,
sp=...)`` terms because the equivalent external-SP source call is rejected.

The six native source likelihoods ignore prior weights; residual and
initialization weighting can still differ by family. Direct tests cover
their documented fixed/selected routes, all seven pipeline stages, additive
reconstruction and save/load. Examples are
``example_gumbls_gam.py``, ``example_ziplss_gam.py``,
``example_multinom_gam.py``, ``example_gevlss_gam.py``,
``example_shash_gamlss.py`` and ``example_twlss_gamlss.py`` in ``examples/``.
The API family reference gives exact links, residual availability, source
SE/likelihood conventions and the bounded validation scope for each family.

Quasi families
~~~~~~~~~~~~~~

Quasi families specify a mean/variance relationship and estimate dispersion.
All three accept ``log``, ``identity``, ``sqrt``, ``inverse``, ``1/mu^2``,
``logit``, ``probit``, ``cloglog`` and ``cauchit`` links. General ``quasi``
accepts ``constant``, ``mu``, ``mu(1-mu)``, ``mu^2`` and ``mu^3`` variance
functions. A valid response and fitted mean must satisfy the selected
variance and link domains; unusual combinations can fail to converge.

Names select defaults; tuples select a link or an options dictionary.
Family objects from ``nampy.gam.families`` are also accepted:

.. code-block:: python

   GAM(family="quasipoisson")
   GAM(family=("quasibinomial", "probit"))
   GAM(family={"name": "quasi", "variance": "mu^2", "link": "log"})
   GAM(family=("quasi", {"variance": "mu^3", "power": 0.5}))

``("quasi", "log")`` retains constant variance. The ``power`` option belongs
only to general ``quasi``: a positive nonunit power uses ``mu**power``, one
uses identity, and nonpositive values use log, following R's ``power()``.
Use it with the default link or ``link="power"``; combining it with another
named link is rejected. Custom R variance/link closures are not accepted.

Raw ``GAM`` accepts quasibinomial proportions and grouped
``cbind(success, failure)`` or two-column array responses, including fractional
counts and zero-weight rows. ``GAMClassifier(family="quasibinomial")`` provides
binary-label classification. Regression quasi families can be used with
``GAMRegressor``.

Fixed smoothing, GCV and the supported ML/REML and Pearson-scale selection
routes use quasi deviance and dispersion. ML/REML here use extended
quasi-likelihood for selection: there is no response probability distribution.
Consequently public log likelihood, AIC and BIC are ``NaN``; randomized
quantile residuals are unavailable. Deviance, Pearson, response and working
residuals, predictions, covariance and standard errors remain available.
See ``examples/example_quasi_gam.py`` for complete count, grouped-proportion
and general-quasi fits.

Smoothing and fitting
---------------------

.. list-table::
   :header-rows: 1
   :widths: 29 36 35

   * - Symbol
     - Purpose
     - Example
   * - ``smoothing_method="fixed"``
     - Fit with supplied smoothing parameters.
     - ``GAM(smoothing_params=[0.5])``
   * - ``"gcv"`` / ``"ubre"``
     - Select smoothness with prediction-error criteria.
     - ``GAM(optimize_smoothing=True, smoothing_method="gcv")``
   * - ``"ml"`` / ``"reml"``
     - Select smoothness by likelihood criteria.
     - ``GAM(optimize_smoothing=True, smoothing_method="reml")``
   * - ``"p-ml"`` / ``"p-reml"``
     - Pearson-scale ML or REML criteria for supported PIRLS models.
     - ``GAM(optimize_smoothing=True, smoothing_method="p-reml")``
   * - ``"ncv"`` / ``"qncv"``
     - Exact or quadratic neighborhood cross-validation using ``nei=``.
     - ``GAM(optimize_smoothing=True, smoothing_method="ncv", nei=nei)``
   * - ``"laml"``
     - Laplace approximate marginal likelihood for supported general families.
     - ``GAM(optimize_smoothing=True, smoothing_method="laml")``
   * - ``smoothing_optimizer=``
     - Choose ``outer_newton``, ``bfgs``, ``efs``, or a guarded ``optim`` route.
     - ``GAM(smoothing_optimizer="outer_newton")``
   * - ``gam_control(...)``
     - Validate iteration tolerances, step controls, threads, and optimizer
       settings.
     - ``GAM(control=gam_control(maxit=100))``
   * - ``sample_weight=``
     - Apply prior observation weights.
     - ``model.fit(data=df, sample_weight=w)``
   * - ``min_sp=``
     - Set lower bounds for smoothing parameters.
     - ``GAM(min_sp=[1e-6, 1e-6])``
   * - ``score_gamma=`` / ``scale=``
     - Adjust selection scores or supply a known dispersion scale.
     - ``GAM(score_gamma=1.4, scale=1.0)``
   * - ``coefficient_optimizer=``
     - Select Newton or BFGS coefficient optimization where supported.
     - ``GAM(coefficient_optimizer="bfgs")``
   * - ``start=``
     - Supply starting coefficients for compatible fitting routes.
     - ``GAM(start=beta0)``
   * - ``ar1_rho=`` / ``ar_start=``
     - Use known Gaussian AR(1) residual correlation.
     - ``GAM(ar1_rho=0.5, ar_start="new_series")``

Prediction and term inspection
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Symbol
     - Returned information
     - Typical call
   * - ``predict``
     - Response, link, terms, interaction terms, or linear-predictor matrix;
       optionally with standard errors.
     - ``model.predict(new, type="response", return_se=True)``
   * - ``terms=`` / ``exclude=``
     - Include or omit named terms without changing the fitted model.
     - ``model.predict(new, type="terms", exclude=["s(z)"])``
   * - ``unconditional=True``
     - Include smoothing-parameter uncertainty in prediction covariance when
       available.
     - ``model.predict(new, return_se=True, unconditional=True)``
   * - ``na_action=``
     - Pass, omit, exclude, or reject incomplete prediction rows.
     - ``model.predict(new, na_action="exclude")``
   * - ``block_size=``
     - Evaluate large prediction sets in bounded-memory blocks.
     - ``model.predict(new, block_size=20_000)``
   * - ``newdata_guaranteed=True``
     - Use the optimized prediction route when new data satisfy the documented
       completeness and factor-level contract.
     - ``model.predict(new, exclude=["s(z)"], newdata_guaranteed=True)``
   * - ``predict_terms``
     - Named smooth and parametric term contributions.
     - ``model.predict_terms(new)``
   * - ``lpmatrix``
     - Linear-predictor design matrix for new data.
     - ``model.lpmatrix(new)``
   * - ``fit_result``
     - Coefficients, covariance, smoothness, fitted state, and term summaries.
     - ``result = model.fit_result()``
   * - ``plot``
     - One- and two-dimensional fitted effects with uncertainty and residual
       overlays.
     - ``model.plot(residuals=True, pages=1)``
   * - ``save_model`` / ``load_model``
     - Persist and restore a fitted statistical GAM.
     - ``GAM.load_model("model.pkl")``

Inference and diagnostics
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 31 39 30

   * - Symbol
     - Returned information
     - Typical call
   * - ``summary``
     - Parametric and smooth-term inference with fit statistics.
     - ``model.summary()``
   * - ``anova`` / ``anova_gam``
     - Single-model term tests or comparisons between fitted models.
     - ``model.anova(other, test="Chisq")``
   * - ``residuals``
     - Deviance, Pearson, scaled Pearson, working, response, or randomized
       quantile residuals.
     - ``model.residuals(type="deviance")``
   * - ``derivative``
     - First or second derivative and Bayesian standard error for a supported
       smooth.
     - ``model.derivative(smooth_number=1, deriv=1)``
   * - ``concurvity``
     - Full or pairwise concurvity diagnostics.
     - ``model.concurvity(full=True)``
   * - ``k_check``
     - Basis-dimension diagnostic for smooth terms.
     - ``model.k_check(seed=1)``
   * - ``gam_check``
     - Residual checks and the basis-dimension diagnostic together.
     - ``model.gam_check(type="deviance")``
   * - ``vcov`` / ``sp_vcov``
     - Coefficient or smoothing-parameter covariance matrices.
     - ``model.vcov(unconditional=True)``
   * - ``loglik`` / ``aic`` / ``bic``
     - Log likelihood and information criteria.
     - ``model.aic()``
   * - ``gam_vcomp``
     - Variance-component representation of smoothing parameters.
     - ``model.gam_vcomp()``
   * - ``one_se_rule``
     - One-standard-error selection from recorded smoothness candidates.
     - ``model.one_se_rule()``
   * - ``edf1``
     - Per-coefficient upper-bound effective degrees of freedom.
     - ``model.edf1()``

For low-level fitting and extension interfaces, see :doc:`../api/gam_advanced`.

Full concurvity follows mgcv's QR calculation without dropping dependent
columns. With repeated covariates across predictors, the joint design can be
rank deficient and these diagnostics can vary across numerical libraries.
Pairwise comparisons remain identifiable when the individual compared blocks
have full column rank; a full-design estimate is not a portable reference in
this case.

For fitted illustrations of marginal and joint terms, see the
:ref:`formula visual reference <formula-visual-reference>`.

Derivative curves and region maps
---------------------------------

``model.plot_derivative(smooth_number=1, deriv=1, level=0.95)`` displays a
supported univariate smooth's slope and pointwise Bayesian band. Use
``deriv=2`` for curvature. It returns matplotlib figures and is also available
on the estimator adapters. By default it sorts the observed covariate values;
``X=`` supplies evaluation rows where the derivative API permits them. Bands
are conditional, not simultaneous, and derivatives are on the additive
predictor scale. Formula-transformed covariates retain their transformed axis.
The existing basis restrictions apply: ``ps`` and ``bs`` provide derivative
matrices; ``cr``, GP and SOS do not. SCAM derivatives remain training-data only.

``model.plot()`` renders an MRF region map when its smooth has
``xt={"polys": polygons}``. ``polygons`` maps region names to two-column
coordinate arrays; NaN rows separate disconnected loops or nested holes.
Map values follow the polygon names, including when their order differs from
factor-level order. Scheme 0 uses heat colors and other schemes use gray.
MRF smooths supplied only with a neighbor graph or penalty have no polygon
geometry and are omitted from plots. Matplotlib rendering is not pixel-identical
to R graphics; the plotted fitted effects are checked against ``mgcv``.

Conditional prediction surfaces
-------------------------------

``vis`` displays a full-model prediction surface while holding other predictors
fixed. It differs from a term plot, which displays an individual smooth's
contribution.

.. code-block:: python

   surface = model.vis(
       view=["x", "z"], cond={"group": "control"},
       type="response", plot_type="contour", n_grid=40, too_far=0.1,
   )
   figures = surface["figures"]
   predictions = surface["fit"]

The default view uses the first two varying predictors. Other numeric predictors
use mgcv's observed type-3 median; factors use their most frequent level.
``cond`` overrides those values. Factor views are supported; distance masking
with ``too_far`` requires numeric view variables. ``lp`` selects a one-based
linear predictor for multi-predictor fits.

Use ``plot_type="persp"`` for a perspective surface. A positive ``se`` adds
plus/minus that many pointwise standard errors; requesting contour plots with
positive ``se`` warns and draws perspective surfaces, following mgcv. ``color``
accepts ``heat``, ``topo``, ``cm``, ``terrain``, ``gray`` or ``bw``. The returned
``fit`` and ``se_fit`` arrays vary the first view fastest; reshape them to
``(n_grid, n_grid)`` for rows along the second view and columns along the first.
Use ``prepare_vis_gam_data`` from ``nampy.gam.diagnostics`` to obtain data without
rendering. Estimator adapters also expose ``vis``.

Spherical hemisphere plots
--------------------------

``model.plot(scheme=0, theta=30, phi=30)`` draws a rotated SOS hemisphere with
colored fitted effects, latitude/longitude lines, contours and visible
observations. Scheme 1 gives a contour-only hemisphere. Coordinates remain
latitude then longitude in degrees; ``theta`` and ``phi`` set the viewing
orientation. ``n2`` controls the disk resolution. Schemes 2 and above delegate
to the ordinary rectangular smooth plots with the scheme reduced by two, as
in mgcv. Hemisphere plots do not add standard-error bands, and SOS derivative
matrices remain unavailable.

These displays use Matplotlib; numerical projection and prediction data are
compared against mgcv, while graphics are not claimed to be pixel-identical.

QQ and four-panel model checks
------------------------------

``model.qq(rep=200, level=0.9, seed=7)`` plots observed residuals against
family-based reference quantiles, with a pointwise simulation envelope.
``rep=0`` averages randomized quantile calculations where the family supplies a
quantile function; otherwise it uses 50 simulations without an envelope.
``level>=1`` draws individual simulated curves. These are diagnostic envelopes,
not simultaneous confidence bands or a calibrated goodness-of-fit test.

``model.plot_check(type="deviance", seed=7)`` draws the QQ panel, residuals
against the linear predictor, a residual histogram using R's bin rules, and
response against fitted values. ``old_style=True`` uses normal QQ quantiles.
Both methods return prepared numerical data plus a ``figures`` list and are
also available on the sklearn-style GAM adapters. ``model.gam_check()`` retains
its numerical convergence/rank/basis-check report.

Use ``prepare_qq_gam_data`` and ``prepare_gam_check_data`` from
``nampy.gam.diagnostics`` for plotting-free data. Quantile and simulation support
covers the implemented ordinary, extended and location-scale families; families
without either capability use the upstream normal-QQ fallback. A NumPy ``seed``
is reproducible but does not reproduce R's RNG stream. Exact reference tests
supply ``quantile_permutations`` (zero-based, shape ``(s_rep, n)``) or
``response_draws`` (shape ``(rep, n)``) to both implementations.

Shape-constrained fits automatically use SCAM display semantics: one-dimensional
term plots default to two standard errors, two-dimensional plots to one;
partial residuals use SCAM's normalized square-root IRLS weights. Bivariate
shape plots center on their plotting grid; ``vis`` continues to show full-model
predictions. Shape uncertainty uses transformed coefficient covariance;
``se_with_mean`` does not alter shape-term uncertainty. ``plot_check`` additionally
accepts ``type="rquantile"`` for SCAM fits. The explicit preparation functions
``prepare_plot_scam_data``, ``prepare_vis_scam_data``, ``prepare_qq_scam_data`` and
``prepare_scam_check_data`` are available for custom displays. SCAM check panels
do not run a k-index test. Graphics use Matplotlib and do not promise identical
R pixels.

Parametric penalties and explicit smooth constraints
----------------------------------------------------

``para_pen`` is the Python spelling of mgcv's ``paraPen``. It maps original
parametric formula labels to one or more matrices, in that term's fitted column
order. For example, this shrinks the coefficient of ``x`` while fitting a smooth
of ``z``:

.. code-block:: python

   model = GAM(
       formula="y ~ x + s(z, bs='cr', k=7)",
       para_pen={"x": {"penalties": [np.eye(1)], "sp": [-1]}},
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data=df)

Each entry accepts ``penalties``, optional ``rank`` (one integer per matrix),
``sp`` (nonnegative for fixed, negative for estimated), and ``L``. Omitting
``sp`` marks its parameters for estimation; smoothing optimization must be
enabled to estimate them. Matrices are not rescaled. A factor term uses its
expanded contrast columns; a matrix-valued term uses its expanded columns.
Several penalties may act on the same term. ``L`` maps log underlying smoothing
parameters onto log penalty multipliers. It may be rectangular and contain
finite signed or fractional entries: with positive underlying parameters,
the full multipliers are ``np.exp(L @ np.log(sp))``. Fixed parameters are
removed from optimization and contribute through the fixed log-SP offset.
Zero rows specify a constant multiplier of one. Fixed zeros use mgcv's
effective-zero construction. See ``examples/example_parametric_penalty_map.py``
for a fit with two underlying parameters and three penalties.
Parametric smoothing parameters precede smooth parameters, in formula-term
order. As in mgcv, unmatched labels are ignored and the intercept is not a
parametric-penalty target. Multi-formula ``para_pen`` is rejected, including
general-family formula lists with one predictor.

When combining nonidentity parametric maps with smooth penalties, setup follows
mgcv's combined ``L`` row/column ordering exactly. In mgcv 1.9-4 this ordering
differs from simply prepending a parametric block to the smooth map; fixed and
selected fits follow the captured upstream setup state. Automatic EFS selection
with nonidentity maps is guarded: upstream's EFS branch omits ``L`` and its
offset, causing rectangular-map failures or inconsistent square-map states.
Newton and supported alternative outer optimizers use the mapped analytic
derivatives. Unused free map columns are rejected rather than estimated.
More generally, automatic initialization rejects rank-deficient free maps,
matching source initialization; explicit initial smoothing parameters may
bypass that initialization. Positive ``min_sp`` with nonidentity maps requires
a separate full-penalty ``H`` implementation and remains guarded. Mapped
penalties are unavailable for transformed-coefficient fits.
``gam_vcomp(rescale=False)`` reports underlying parameter names separately
from full-penalty standard deviations and uses the accepted REML variance for
Gaussian fits. Rescaling parametric variance components remains guarded where
the required scale metadata is unavailable.

``constraints={1: C}`` replaces the first smooth's default centering with
``C @ beta_raw = 0``. A unique smooth label may replace the one-based index.
Fitting supports ``cr``, ``cc``, ``ps``, ``cp`` and ``bs`` without ``by``,
point constraints, or linked IDs. ``C`` acts on the uncentered basis, must be
finite with independent rows, and must leave at least one coefficient.
A matrix with zero rows requests no local constraint; global model
identifiability conditions still apply. Columns with cancelling nonzero row
sums are rejected because upstream's localized absorption cannot represent them.
Other bases/compositions remain guarded rather than accepting a constraint in
a different raw parameterization.
Explicit ``cs`` constraints are guarded: mgcv assigns unequal shrinkage to
numerically arbitrary null eigenvectors, which changes fitted predictions
across LAPACK implementations. Use ``cr`` with ``select=True`` for a portable
penalized null space. Raw ``smooth_con`` inspection of ``cs`` remains available
with invariant penalty and constraint checks.

Derivative-penalized B-splines (``bs='bs'``) also support multiple derivative
penalties through ``m``. The same constraint transform applies to every
penalty, and ``select=True`` adds a null-space penalty after constraint
absorption when a null space remains. See ``examples/example_bs_constraints.py``
for a runnable fit with an explicit coefficient-sum constraint.
For an empty constraint matrix combined with multiple B-spline penalties and
``select=True``, default GCV performance iteration can stop at different
endpoints in NAMpy and mgcv. Fixed-SP reconstruction agrees; the targeted
automatic-selection comparison uses the same tighter ``mgcv_tol=1e-12`` in
both implementations. This is an optimizer stopping limitation, not a
constraint-operator difference.

Inspect the raw basis before specifying ``C``:

.. code-block:: python

   from nampy.gam.compiler.construct import smooth_con
   from nampy.gam.specs import TermSpec, build_smooth_spec

   term = TermSpec(
       kind="smooth", features=("x",), label="s(x)",
       smooth_spec=build_smooth_spec(special="s", bs="cr", k=7),
   )
   raw = smooth_con(term, df[["x"]].to_numpy(), ["x"], absorb_cons=False)
   C = raw.fit_constraint_operator
   constrained = smooth_con(term, df[["x"]].to_numpy(), ["x"], constraints=C)
   X_new = constrained.compiled_term.predict_matrix(new_df[["x"]].to_numpy())

The returned object contains ``compiled_term``, ``penalty_specs`` and
``fit_constraint_operator``. Absorption transports the basis, every penalty,
prediction and derivative coefficient maps together. ``absorb_cons=False`` is
a construction/inspection control; fitted GAMs always absorb constraints,
matching mgcv's high-level fitting route. It is not a switch to fit an
unconstrained model accidentally.
