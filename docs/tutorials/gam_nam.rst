Compare Statistical and Neural Feature Effects
==============================================

Do a statistical GAM and a neural additive model learn the same relationships
when the data and train/test split are held fixed? This analysis fits both
representations to a four-feature synthetic housing dataset, evaluates them on
the same held-out rows, and compares every learned feature effect.

Create data with four nonlinear effects
---------------------------------------

The response combines saturating, periodic, quadratic, and logarithmic
relationships. Because the data are synthetic, each predictor has a known role
without presenting the example as a real housing study.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split

   rng = np.random.default_rng(41)
   n = 1_200
   data = pd.DataFrame({
       "median_income": rng.uniform(1.0, 10.0, n),
       "house_age": rng.uniform(1.0, 52.0, n),
       "latitude": rng.uniform(32.0, 42.0, n),
       "rooms_per_household": rng.lognormal(1.5, 0.35, n),
   })
   data["value"] = (
       2.2 * np.tanh((data["median_income"] - 5.0) / 1.7)
       + 0.9 * np.sin((data["house_age"] - 5.0) / 8.0)
       - 0.55 * (data["latitude"] - 37.0) ** 2 / 6.0
       + 0.75 * np.log1p(data["rooms_per_household"])
       + rng.normal(0.0, 0.3, n)
   )

   train, test = train_test_split(data, test_size=0.2, random_state=41)
   features = [
       "median_income",
       "house_age",
       "latitude",
       "rooms_per_household",
   ]

Fit both representations
------------------------

The GAM uses a cubic regression smooth for every predictor and selects the
smoothness by REML. The NAM uses an independent neural network for every
source feature. Both models are additive, but their function classes,
regularization, and fitting algorithms remain different.

.. code-block:: python

   from nampy.models import GAMRegressor, NAMRegressor

   gam = GAMRegressor(
       formula=(
           "value ~ s(median_income, bs='cr', k=12)"
           " + s(house_age, bs='cr', k=12)"
           " + s(latitude, bs='cr', k=12)"
           " + s(rooms_per_household, bs='cr', k=12)"
       ),
       family="gaussian",
   ).fit(train)

   nam = NAMRegressor(
       layer_sizes=[48, 24],
       dropout=0.0,
       numerical_preprocessing="minmax",
   )
   nam.fit(
       train[features],
       train["value"],
       max_epochs=60,
       batch_size=128,
       patience=10,
       random_state=41,
   )

Evaluate the same held-out rows
-------------------------------

``score`` returns :math:`R^2` for both regressors, so the predictive comparison
does not require task-specific adapter code.

.. code-block:: python

   scores = {
       "GAM": gam.score(test, test["value"]),
       "NAM": nam.score(test[features], test["value"]),
   }

These scores answer whether each model predicts the held-out response. They do
not show whether the models learned the same decomposition, so inspect their
components next.

Check the shared additive contract
----------------------------------

The GAM returns spline terms and the NAM returns feature-network terms through
the same :class:`nampy.contracts.AdditivePrediction` type. The neural effects
are centered against the training rows so their arbitrary additive constants
are transferred to the intercept.

.. code-block:: python

   gam_parts = gam.predict_components(test)
   nam_parts = nam.predict_components(
       test[features],
       center=True,
       reference_X=train[features],
   )

   gam_parts.validate_additive_reconstruction()
   nam_parts.validate_additive_reconstruction()

For each model, its intercept, named terms, and optional offset reconstruct the
linear predictor. This is an accounting guarantee. It does not imply that the
individual effects are uniquely identified or that neural effects carry the
GAM's inferential interpretation.

Compare every feature
---------------------

The curves below are centered link-scale contributions evaluated across each
feature's training range. Every other feature is held at its training median;
because these models contain main effects only, the selected value of the
other predictors does not change the plotted term.

.. image:: ../_static/gam_nam_all_features.png
   :alt: Statistical GAM and NAM centered contributions for all four features
   :class: scientific-figure
   :align: center

Both models recover the dominant saturating income effect, the curved latitude
effect, the age pattern, and the diminishing return from additional rooms.
Where the curves differ, the disagreement is useful information: spline basis
dimension and REML regularization govern the GAM, while network capacity,
optimization, and early stopping govern the NAM.

Use the comparison responsibly
------------------------------

The shared workflow makes representation sensitivity visible without erasing
important distinctions. Use the GAM when statistical smoothness selection,
standard errors, hypothesis tests, and diagnostics matter. Use the NAM when
feature-specific neural functions are the desired representation. Compare
their centered effects as fitted relationships, not as interchangeable
uncertainty statements.

Continue with :doc:`first_gam` for a complete statistical diagnostic workflow
or :doc:`nam_nbm` to compare two neural representations.
