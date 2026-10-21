Statistical GAM Tools
=====================

NAMpy provides a Python implementation of the statistical GAM workflow and
uses ``mgcv`` terminology where the corresponding behavior is available. This
page collects the reader-facing formula syntax, smooth constructors, fitting
controls, prediction methods, and diagnostic tools in one place. The
:doc:`../api/gam` reference gives full call signatures.

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

Formula syntax
--------------

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
   * - ``offset(...)``
     - Known additive contribution to the linear predictor.
     - ``count ~ s(x) + offset(log_exposure)``
   * - ``0`` or ``-1``
     - Remove the intercept.
     - ``y ~ 0 + s(x)``
   * - ``I()``, ``log()``, ``sqrt()``, ``exp()``
     - Numeric transformations evaluated from data columns.
     - ``y ~ s(log(x)) + I(z**2)``

Smooth bases
------------

The basis code is supplied through ``bs=`` inside ``s()``, ``te()``, or
``ti()``. Tensor products accept one code per margin.

.. list-table::
   :header-rows: 1
   :widths: 20 43 37

   * - Code
     - Basis
     - Example
   * - ``tp``
     - Thin-plate regression spline.
     - ``s(x, bs="tp")``
   * - ``ts``
     - Thin-plate spline with null-space shrinkage.
     - ``s(x, bs="ts")``
   * - ``cr``
     - Cubic regression spline.
     - ``s(x, bs="cr")``
   * - ``cs``
     - Cubic regression spline with shrinkage.
     - ``s(x, bs="cs")``
   * - ``cc``
     - Cyclic cubic regression spline.
     - ``s(day, bs="cc")``
   * - ``ps``
     - P-spline.
     - ``s(x, bs="ps", m=c(2, 2))``
   * - ``cp``
     - Cyclic P-spline.
     - ``s(day, bs="cp")``
   * - ``bs``
     - B-spline with derivative-based penalties.
     - ``s(x, bs="bs", m=c(3, 2))``
   * - ``ds``
     - Duchon spline for one or more covariates.
     - ``s(x, z, bs="ds")``
   * - ``gp``
     - Gaussian-process smooth.
     - ``s(x, z, bs="gp")``
   * - ``ad``
     - Adaptive smooth with locally varying smoothness.
     - ``s(x, bs="ad")``
   * - ``sos``
     - Spline on a sphere from latitude and longitude.
     - ``s(lat, lon, bs="sos")``

Structured smooths
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 43 37

   * - Code
     - Structure
     - Example
   * - ``re``
     - Random-effect smooth.
     - ``s(group, bs="re")``
   * - ``mrf``
     - Markov random field over neighboring regions.
     - ``s(region, bs="mrf", xt=graph)``
   * - ``fs``
     - Factor-specific smooth with a shared smoothing parameter.
     - ``s(x, group, bs="fs")``
   * - ``sz``
     - Sum-to-zero factor smooth.
     - ``s(x, group, bs="sz")``
   * - ``by=``
     - Numeric or factor modulation of a smooth.
     - ``s(x, by=group)``
   * - ``id=``
     - Share smoothing parameters across compatible terms.
     - ``s(x, id=1) + s(z, id=1)``

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

Shape-constrained bases
-----------------------

Shape constraints use the same ``s(..., bs=...)`` syntax as ordinary
smooths. See :doc:`shape_constrained_gams` for the coefficient transforms,
derivative behavior, linear functionals, and supported fitting routes.

.. list-table::
   :header-rows: 1
   :widths: 27 38 35

   * - Constraint
     - Codes
     - Example
   * - Monotone
     - ``mpi``, ``mpd``
     - ``s(x, bs="mpi")``
   * - Convex or concave
     - ``cx``, ``cv``
     - ``s(x, bs="cx")``
   * - Monotone with curvature
     - ``micx``, ``micv``, ``mdcx``, ``mdcv``
     - ``s(x, bs="micx")``
   * - Positive
     - ``po``, ``ipo``, ``dpo``, ``cpop``
     - ``s(x, bs="po")``
   * - Anchored monotone
     - ``miso``, ``mifo``
     - ``s(x, bs="miso")``
   * - Numeric-by
     - ``mpiby``, ``mpdby``, ``micxby``, ``micvby``, ``mdcxby``,
       ``mdcvby``, ``cxby``, ``cvby``
     - ``s(x, by=z, bs="mpiby")``
   * - Local
     - ``lmpi``, ``lipl``
     - ``s(x, bs="lmpi", xt=list(xc=0.4))``
   * - Bivariate
     - ``tedmi``, ``tedmd``, ``temicx``, ``temicv``, ``tedecx``,
       ``tedecv``, ``tecxcx``, ``tecvcv``, ``tecxcv``, ``tescx``,
       ``tescv``, ``tesmi1``, ``tesmd1``, ``tesmi2``, ``tesmd2``,
       ``tismi``, ``tismd``
     - ``s(x, z, bs="tedmi")``

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

