Compare GAM and NAM feature effects
===================================

A GAM and a NAM can make similar predictions while learning different feature effects. We will compare them on the same synthetic dataset, using both test accuracy and the known relationships that generated the response.

Create the data
---------------

The response combines four effects: a saturating income relationship, an age pattern, a curved latitude effect, and diminishing returns from additional rooms. The housing-like names make the example easy to read; these are simulated data, not measurements from a housing study.

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


Fit the two models
------------------

The GAM uses a cubic regression spline for each feature and selects smoothing strength with REML. The NAM uses a separate neural network for each feature. We give both models the same training data and evaluate their predictions on the same test rows.

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
       lr=5e-3,
       batch_size=128,
       patience=10,
       random_state=41,
   )


Compare test predictions
------------------------

Both regressors report R2 through score. This compares predictive accuracy, but it does not tell us which feature relationships each model has learned. We examine those relationships next.

.. code-block:: python

   scores = {
       "GAM": gam.score(test, test["value"]),
       "NAM": nam.score(test[features], test["value"]),
   }
   pd.DataFrame({"Model": list(scores), "Test R2": list(scores.values())}).style.hide(axis="index").format({"Test R2": "{:.3f}"})

.. include:: /_generated/gam_nam_scores.inc

Compare the feature effects
---------------------------

We center the learned effects and the true effects over the same training rows. This removes differences caused only by moving a constant between an effect and the intercept. Each panel varies one feature across its training range.

.. code-block:: python

   import matplotlib.pyplot as plt

   gam_parts = gam.predict_components(test)
   nam_parts = nam.predict_components(
       test[features],
       center=True,
       reference_X=train[features],
   )

   gam_parts.validate_additive_reconstruction()
   nam_parts.validate_additive_reconstruction(atol=1e-5)


   # Match each named term to its source feature.
   def feature_effect(parts, feature):
       key = next(name for name in parts.terms if feature in name)
       return np.asarray(parts.terms[key]).reshape(-1)

   gam_reference = gam.predict_components(train)
   figure, axes = plt.subplots(2, 2, figsize=(10, 7))
   for axis, feature in zip(axes.flat, features):
       lower, upper = np.quantile(train[feature], [0.01, 0.99])
       values = np.linspace(lower, upper, 240)
       grid = pd.DataFrame({name: np.full(values.size, train[name].median()) for name in features})
       grid[feature] = values
       gam_effect = feature_effect(gam.predict_components(grid), feature)
       gam_effect -= feature_effect(gam_reference, feature).mean()
       nam_grid = nam.predict_components(grid, center=True, reference_X=train[features])
       axis.plot(values, gam_effect, label="GAM")
       truth = {
           "median_income": lambda x: 2.2*np.tanh((x-5)/1.7),
           "house_age": lambda x: .9*np.sin((x-5)/8),
           "latitude": lambda x: -.55*(x-37)**2/6,
           "rooms_per_household": lambda x: .75*np.log1p(x),
       }[feature]
       axis.plot(values, truth(values)-truth(train[feature]).mean(), "k--", label="Centered truth")
       axis.plot(values, feature_effect(nam_grid, feature), label="NAM")
       axis.set(title=feature, xlabel="Feature value", ylabel="Centered contribution")
   axes[0, 0].legend()
   figure.tight_layout()


.. image:: /_static/gam_nam_all_features.png
   :alt: GAM, NAM, and known centered effects for four synthetic housing-like features
   :class: scientific-figure

The income effect saturates, the latitude effect is curved, and the rooms effect increases more slowly at larger values. Compare the fitted age curve with its known pattern as well. Differences from the dashed truth show where a model has flattened, exaggerated, or missed part of a relationship.

The GAM's curves depend on its spline basis and roughness penalty. The NAM's curves depend on its networks, regularization, and optimization. Similar test scores can therefore coexist with visible differences in effect recovery.

.. _compare-independent-networks-with-shared-bases:

The :doc:`neural architecture comparison <../notebooks/03_neural_models>` examines what changes when feature networks share learned basis functions.
