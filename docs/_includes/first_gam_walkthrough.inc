How does energy demand change with outdoor temperature and time of day? We will fit a GAM to synthetic demand data, inspect its effects, and check whether the chosen smooths capture the main patterns.

Create the data
---------------

Demand rises at very low and very high temperatures and follows a daily cycle. We add random noise so the observations vary around these relationships.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split

   rng = np.random.default_rng(21)
   n = 900

   temperature = rng.uniform(-5.0, 35.0, n)
   hour = rng.uniform(0.0, 24.0, n)
   demand = (
       45.0
       + 0.12 * (temperature - 18.0) ** 2
       + 5.0 * np.sin(2.0 * np.pi * (hour - 7.0) / 24.0)
       + rng.normal(0.0, 2.0, n)
   )

   data = pd.DataFrame(
       {"demand": demand, "temperature": temperature, "hour": hour}
   )
   train, test = train_test_split(data, test_size=0.2, random_state=21)


Choose the smooths
------------------

Temperature gets a cubic regression spline. Hour gets a cyclic cubic spline so that the fitted curve joins at the start and end of the day. The basis dimension k limits the available flexibility; REML selects the smoothing strength within that basis.

.. code-block:: python

   from nampy.models import GAMRegressor

   model = GAMRegressor(
       formula=(
           'demand ~ s(temperature, bs="cr", k=12) '
           '+ s(hour, bs="cc", k=12)'
       ),
       family="gaussian",
   )
   model.fit(train)


Evaluate predictions
--------------------

We predict demand for the held-out observations and report RMSE and R2. RMSE is measured in the same units as demand; R2 compares the squared prediction error with a constant prediction based on the test-set mean.

.. code-block:: python

   from sklearn.metrics import mean_squared_error, r2_score

   predicted = model.predict(test)
   rmse = mean_squared_error(test["demand"], predicted) ** 0.5
   r2 = r2_score(test["demand"], predicted)
   print(f"RMSE: {rmse:.2f}")
   print(f"R²:   {r2:.3f}")


.. code-block:: text

   RMSE: 1.95
   R²:   0.989


Interpret the effects
---------------------

.. code-block:: python

   plot_data = model.plot(
       pages=1,
       scheme=1,
       shade_col="#f3b4c2",
       scale=0,
       figsize=(10, 4),
   )


.. image:: /_static/first_gam_effects.png
   :alt: Estimated temperature and hour effects with pointwise uncertainty
   :class: scientific-figure

Demand is lowest at mild temperatures and rises toward both extremes. The hour curve shows the daily pattern and joins at the day boundary. Both curves are centered contributions; the shaded bands show pointwise uncertainty about the fitted effects.

Inspect residuals and basis dimensions
--------------------------------------

The residual plot shows the errors left after fitting the temperature and hour effects. There is no strong trend against fitted demand in this example. We also inspect the model summary and basis diagnostic to see whether either smooth needs more flexibility.

.. code-block:: python

   import matplotlib.pyplot as plt

   components = model.predict_components(test)
   components.validate_additive_reconstruction()

   fitted = model.predict(train)
   residual = model.gam_.residuals(type="deviance")

   plt.scatter(fitted, residual, s=15, alpha=0.5)
   plt.axhline(0.0, color="#b4234d")
   plt.xlabel("Fitted demand")
   plt.ylabel("Deviance residual")


.. image:: /_static/first_gam_residuals.png
   :alt: Deviance residuals against fitted energy demand
   :class: scientific-figure

.. code-block:: python

   summary = model.summary()
   basis_check = model.gam_.k_check(n_rep=20, seed=21)
   print(basis_check)


.. include:: /_generated/first_gam_summary.inc

.. code-block:: text

                   k_prime    edf  k_index  p_value
   label
   s(temperature)        11  9.977     1.028     0.75
   s(hour)               10  7.114     1.001     0.70


The diagnostic gives no indication that either basis is too small here. If it suggested remaining structure, we would increase the relevant k, refit, and check whether the curve and residuals changed.
