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

See :doc:`../tutorials/first_gam` for a complete example and
:ref:`statistical-gam-tools` for the available bases and fitting controls.

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
     - ``s(x, select=True)``
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
   * - ``nb`` / ``negbin``
     - Estimated or fixed negative-binomial shape.
     - ``GAM(family="nb")``
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
   * - ``parity_snapshot``
     - Export coefficients, predictions, smoothness, and optional covariance
       summaries for numerical validation.
     - ``model.parity_snapshot(new)``

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

Advanced public utilities
-------------------------

These symbols support custom fitting, testing, and extensions. Most analyses
should use ``GAM`` or the estimator classes above.

.. list-table::
   :header-rows: 1
   :widths: 31 44 25

   * - Symbol
     - Purpose
     - Import
   * - ``GAMControl`` / ``gam_control``
     - Typed and validated fitting controls.
     - ``from nampy.gam import gam_control``
   * - ``fit_model_core``
     - Compile and fit a configured GAM object.
     - ``from nampy.gam import fit_model_core``
   * - ``solve_fit``
     - Dispatch a fixed-smoothing solve to the available backend.
     - ``from nampy.gam import solve_fit``
   * - ``FitCoreSolution``
     - Structured result returned by the low-level fitting stage.
     - ``from nampy.gam import FitCoreSolution``
   * - ``normalize_nei``
     - Validate neighborhood indices used by neighborhood cross-validation.
     - ``from nampy.gam import normalize_nei``

