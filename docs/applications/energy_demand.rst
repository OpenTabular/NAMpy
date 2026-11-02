How does temperature move demand through the day?
=================================================

Electricity demand is high at cold and hot temperatures and follows a daily cycle. This page uses the synthetic demand series from :doc:`the demand tutorial </tutorials/first_gam>`. Temperature is drawn uniformly from −5 to 35, hour from 0 to 24, and the mean is

.. code-block:: text

   45 + 0.12 (temperature − 18)² + 5 sin(2 π (hour − 7) / 24)

with independent Gaussian noise of standard deviation 2. There are 900 rows, drawn with seed 21. The split is ``train_test_split(..., test_size=0.2, random_state=21)``, so 180 rows are held out.

What is the problem?
--------------------

The target is held-out demand. The scientific question is whether the fitted temperature curve has a minimum at mild weather, near the vertex at 18 in the generator, and whether the hour curve joins at midnight.

Why is a linear model insufficient?
-----------------------------------

Ordinary least squares in temperature and hour, fit on the same training rows, has test RMSE 15.46 and R² 0.286. The quadratic temperature effect and the daily cycle are both invisible to that plane: a plane in temperature and hour has one slope for each variable and cannot rise at both ends of the temperature range or repeat every 24 hours. The linear fit is a baseline for this page; the tutorial itself reports the GAM.

Why this model?
---------------

``GAMRegressor`` uses a cubic regression spline for temperature and a cyclic cubic spline for hour:

.. code-block:: python

   GAMRegressor(
       formula=(
           'demand ~ s(temperature, bs="cr", k=12) '
           '+ s(hour, bs="cc", k=12)'
       ),
       family="gaussian",
       knots={"hour": [0.0, 24.0]},
   ).fit(train)

``k=12`` is the basis dimension, the maximum flexibility available to each smooth. REML selects the smoothing strength inside that basis. The cyclic basis, together with knots fixed at 0 and 24, is the part that knows hour 0 and hour 24 are the same point on the clock. A cubic regression spline on hour would treat those endpoints as unrelated.

What the fit showed
-------------------

The GAM has test RMSE 1.95 and R² 0.989. RMSE is in the same units as demand. The noise added by the generator has standard deviation 2, so a test RMSE of 1.95 is close to the variation that no smooth of temperature and hour can remove. R² compares the squared prediction error with a constant prediction equal to the test-set mean.

``model.plot(pages=1, scheme=1, scale=0)`` draws one centered contribution per smooth. In the temperature panel the curve is lowest near mild weather, the vertex of ``(temperature − 18)²`` in the generator, and it rises toward both cold and hot temperatures. In the hour panel the cyclic basis, with knots fixed at 0 and 24, makes the contribution meet at the two ends of the day. The shaded band on each panel is the pointwise uncertainty of that centered effect under the fitted Gaussian model.

.. image:: /_static/first_gam_effects.png
   :alt: Estimated temperature and hour effects with pointwise uncertainty
   :class: scientific-figure
   :align: center

``predict_components(test)`` rebuilds the linear predictor as intercept plus the two smooths, and ``validate_additive_reconstruction`` checks that sum. Deviance residuals of the training fit, plotted against fitted demand, show no strong remaining trend in this example.

The basis check ``k_check(n_rep=20, seed=21)`` reports:

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Smooth
     - k'
     - EDF
     - k-index
     - p-value
   * - ``s(temperature)``
     - 11
     - 9.977
     - 1.028
     - 0.75
   * - ``s(hour)``
     - 10
     - 7.116
     - 1.001
     - 0.70

``k'`` is the basis dimension after the side condition that separates the smooth from the intercept. EDF near ``k'`` would say the penalty is using almost every available column. Here both EDFs sit below their ``k'``, and the k-index p-values do not suggest that either basis is too small for this sample.

.. image:: /_static/first_gam_residuals.png
   :alt: Deviance residuals against fitted energy demand
   :class: scientific-figure
   :align: center

Limits of the interpretation
----------------------------

The series is synthetic, so the RMSE gap measures recovery of a known mean. The generator has no weather-forecast error, holiday, or season. Hour is a daily cycle. A season-by-temperature interaction would be a different model. When one covariate's effect changes with another, the separate :doc:`tensor-product tutorial </notebooks/tensor_interactions>` is the relevant next fit, and its RMSE change belongs to that simulation. The uncertainty bands assume the fitted Gaussian model. If a later residual plot or k-index said a smooth should change, the next step is to raise that term's ``k``, refit, and compare the curve, as in the :doc:`diagnostic tutorial </notebooks/gam_diagnostics>`.

:notebook-download:`../tutorials/first_gam`
