Quick start — Fitting a model
-----------------------------

Estimators provide ``fit``, ``predict``, and ``score`` methods and work with
scikit-learn cross-validation and hyperparameter search.

Generalized Additive Model (GAM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The estimator adapters default to automatic REML smoothness selection.
The simulated group has no effect on the response, so its fitted coefficient
is small and the two group prediction curves nearly overlap.

.. showcase-gam-start

.. code-block:: python

   import numpy as np
   import pandas as pd

   from nampy.models import GAMRegressor

   rng = np.random.default_rng(7)
   x = np.linspace(0.0, 1.0, 300)
   data = pd.DataFrame(
       {
           "x": x,
           "group": rng.choice(["a", "b"], size=x.size),
           "y": np.sin(2 * np.pi * x) + rng.normal(scale=0.15, size=x.size),
       }
   )

   model = GAMRegressor(
       formula="y ~ s(x, bs='cr', k=12) + group",
       smoothing_method="reml",
   ).fit(data)

   prediction = model.predict(data)
   summary = model.summary()
   components = model.predict_components(data)

.. showcase-gam-end

.. include:: /_generated/home_gam.inc

Effective degrees of freedom describe the fitted smooth's flexibility;
explained deviance summarizes the training fit. These summary entries do not
measure held-out prediction quality. In the component table, adding the
intercept, smooth contribution, and group contribution reconstructs the
prediction for that row.

.. image:: /_static/home_gam.png
   :alt: GAM smooth effect and fitted response by group
   :class: homepage-demo scientific-figure

Without a formula, the adapter creates one smooth main effect per input column:

.. showcase-gam-adapter-start

.. code-block:: python

   X_train = data[["x"]].iloc[::2]
   y_train = data["y"].iloc[::2]
   X_test = data[["x"]].iloc[1::2]

   model = GAMRegressor(k=10, basis="tp").fit(X_train, y_train)
   prediction = model.predict(X_test)
   standard_error = model.standard_errors(X_test)

.. showcase-gam-adapter-end

.. include:: /_generated/home_gam_adapter.inc

.. image:: /_static/home_gam_adapter.png
   :alt: Held-out GAM predictions with pointwise uncertainty
   :class: homepage-demo scientific-figure

This numeric single-column adapter example predicts alternating held-out
rows. The standard-error column and pointwise band concern the fitted mean,
conditional on the statistical model and smoothing convention; they do not
represent the scatter of future observations.

Distributional GAM (GAMLSS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit separate predictors for the conditional mean and standard deviation:

.. showcase-gamlss-start

.. code-block:: python

   from nampy.models import GAMLSS

   rng_dist = np.random.default_rng(11)
   data_dist = data.copy()
   noise_scale = 0.08 + 0.30 * np.sin(2 * np.pi * data_dist["x"]) ** 2
   data_dist["y"] = (
       np.sin(2 * np.pi * data_dist["x"])
       + 0.25 * (data_dist["group"] == "b")
       + rng_dist.normal(scale=noise_scale)
   )

   model = GAMLSS(
       family="normal",
       formula={
           "mu": "y ~ s(x, bs='cr', k=10) + group",
           "sigma": "~ s(x, bs='cr', k=8)",
       },
   ).fit(data_dist)

   parameters = model.predict(data_dist)       # columns: mu, sigma
   point_prediction = model.predict_point(data_dist)
   raw_predictors = model.predict(data_dist, raw=True)

.. showcase-gamlss-end

.. include:: /_generated/home_gamlss.inc

.. image:: /_static/home_gamlss.png
   :alt: GAMLSS fitted conditional mean and standard deviation
   :class: homepage-demo scientific-figure

The displayed natural columns are ``mu`` (conditional mean) and ``sigma``
(conditional standard deviation). ``raw=True`` returns the predictors before
the family transforms; its scale predictor is not the natural ``sigma``.
The scale formula has no group term, so its group curves coincide.

Neural Additive Model (NAM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This housing-like dataset is entirely synthetic. Each feature has a known
nonlinear contribution; the names do not identify a real housing study.

.. showcase-nam-start

.. code-block:: python

   import numpy as np
   import pandas as pd

   from nampy.models import NAMRegressor

   rng = np.random.default_rng(7)
   n = 1_800
   X = pd.DataFrame({
       "median_income": rng.uniform(1.0, 10.0, n),
       "house_age": rng.uniform(1.0, 52.0, n),
       "latitude": rng.uniform(32.0, 42.0, n),
       "rooms_per_household": rng.lognormal(1.5, 0.35, n),
   })
   y = (
       2.2 * np.tanh((X["median_income"] - 5.0) / 1.7)
       + 0.9 * np.sin((X["house_age"] - 5.0) / 8.0)
       - 0.55 * (X["latitude"] - 37.0) ** 2 / 6.0
       + 0.75 * np.log1p(X["rooms_per_household"])
       + rng.normal(0.0, 0.3, n)
   )

   model = NAMRegressor(
       layer_sizes=[48, 24],
       dropout=0.0,
       numerical_preprocessing="minmax",
   )
   model.fit(
       X,
       y,
       max_epochs=100,
       lr=0.005,
       batch_size=128,
       patience=12,
       random_state=7,
   )

   prediction = model.predict(X)
   components = model.predict_components(X, center=True, reference_X=X)
   importance = model.term_importance(X, center=True, reference_X=X)
   figures = model.plot_terms(X, center=True, reference_X=X, pages=1)
   truth_functions = {
       "median_income": lambda x: 2.2*np.tanh((x-5)/1.7),
       "house_age": lambda x: .9*np.sin((x-5)/8),
       "latitude": lambda x: -.55*(x-37)**2/6,
       "rooms_per_household": lambda x: .75*np.log1p(x),
   }
   for axis, feature in zip(figures[0].axes, X.columns):
       values = np.sort(X[feature].to_numpy())
       truth = truth_functions[feature]
       axis.lines[0].set_label("Fitted NAM")
       axis.plot(values, truth(values)-truth(values).mean(), "--", label="Known centered truth")
       axis.legend()

.. showcase-nam-end

.. image:: /_static/homepage_demo_light.png
   :alt: Centered NAM feature effects for the four input features
   :class: homepage-demo scientific-figure
   :align: center

Effects are centered against the training input rows ``X``: their averages
are transferred to the intercept. Red background shading shows feature
density, not an uncertainty band. Dashed curves show the known synthetic
effects centered against the same rows; discrepancies indicate incomplete
recovery, including in sparsely observed ranges.

.. include:: /_generated/home_nam.inc

.. image:: /_static/home_nam_importance.png
   :alt: NAM feature importance measured by mean absolute centered contribution
   :class: homepage-demo scientific-figure
   :align: center

Neural distributional regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neural LSS estimators also provide point-and-spread effect plots:

.. showcase-neural-lss-start

.. code-block:: python

   from nampy.models import NAMLSS

   rng_dist = np.random.default_rng(11)
   X_dist = pd.DataFrame({
       "median_income": rng_dist.uniform(1.0, 10.0, n),
       "house_age": rng_dist.uniform(1.0, 52.0, n),
   })
   location = (
       1.8 * np.tanh((X_dist["median_income"] - 5.0) / 1.7)
       + 0.7 * np.sin((X_dist["house_age"] - 5.0) / 8.0)
   )
   scale = 0.25 + 0.45 / (1.0 + np.exp(-(X_dist["median_income"] - 5.0)))
   y_dist = location + rng_dist.normal(0.0, scale, n)

   lss = NAMLSS(
       family="normal",
       layer_sizes=[48, 24],
       dropout=0.0,
       numerical_preprocessing="minmax",
   )
   lss.fit(
       X_dist,
       y_dist,
       max_epochs=100,
       batch_size=128,
       patience=12,
       random_state=11,
   )
   figures = lss.plot_distributional_effects(
       X_dist,
       reference_X=X_dist,
       feature_names=["median_income", "house_age"],
       display_names={
           "median_income": "Median income",
           "house_age": "House age",
       },
   )

.. showcase-neural-lss-end

.. image:: /_static/readme_distributional_effects.png
   :alt: Distributional feature effects with conditional spread and predictor density
   :class: homepage-demo scientific-figure
   :align: center

For this normal-family display, black curves show the centered point effect
plus and minus one conditional standard deviation at the reference setting
of the other terms. They describe response spread. They are not confidence
bounds for the learned effect. ``reference_X=X_dist`` defines empirical
centering; zero centered terms specify a synthetic reference configuration.
