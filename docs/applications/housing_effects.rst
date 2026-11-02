Which variables have nonlinear effects, and where do they saturate?
===================================================================

Similar test scores do not say that two models learned the same feature curves. This page uses the housing-style simulation in :doc:`Do a GAM and a NAM recover the same effects? </tutorials/gam_nam>`. The column names are income, house age, latitude, and rooms per household. They are simulated draws with housing-style names.

There are 1,200 rows, drawn with seed 41. Income is uniform on 1 to 10, house age uniform on 1 to 52, latitude uniform on 32 to 42, and rooms per household lognormal. The response is the sum of four known effects plus Gaussian noise of standard deviation 0.3:

.. code-block:: text

   2.2 tanh((income − 5) / 1.7)
   + 0.9 sin((age − 5) / 8)
   − 0.55 (latitude − 37)² / 6
   + 0.75 log(1 + rooms)

The split is ``train_test_split(..., test_size=0.2, random_state=41)``.

What is the problem?
--------------------

The target is the held-out response and, separately, the shape of each feature contribution. Saturation means the income contribution stops rising as income increases, which is the shape of the tanh. That shape is a property of the centered effect. The R² does not display it.

Why is a linear model insufficient?
-----------------------------------

Ordinary least squares in the four features, fit on the same training rows, has test R² 0.752. The linear fit can capture an average slope. It cannot flatten the income effect, follow the age oscillation, or bend with latitude. This linear score is a baseline for this page. The tutorial reports the GAM and the NAM.

Why this model?
---------------

The GAM is a Gaussian ``GAMRegressor`` with one cubic regression spline per feature:

.. code-block:: text

   value ~ s(median_income, bs='cr', k=12)
         + s(house_age, bs='cr', k=12)
         + s(latitude, bs='cr', k=12)
         + s(rooms_per_household, bs='cr', k=12)

REML selects the smoothing strength of each spline. The NAM is a ``NAMRegressor`` with ``DefaultNAMConfig(layer_sizes=[48, 24], dropout=0.0)`` and min-max preprocessing. Its training run uses 60 epochs, learning rate 0.005, batch size 128, patience 10, and seed 41. Each feature has its own network. Both models are main-effects models. Neither is asked to represent an interaction, and both see the same training rows.

What the fit showed
-------------------

``score`` on the test rows gives test R² 0.975 for the GAM and 0.971 for the NAM. Those two numbers say the held-out squared error is similar. They do not say the four curves match.

The figure evaluates each effect on a grid of 240 values between the 1st and 99th percentiles of that feature in the training data, holding the other three features at their training medians. ``predict_components`` supplies the named term, and both decompositions are checked with ``validate_additive_reconstruction``. Each curve is then centered by subtracting its mean on the training rows, and the known effect is centered the same way. The vertical axis is that centered contribution. The horizontal axis is the feature value.

.. image:: /_static/gam_nam_all_features.png
   :alt: GAM, NAM, and known centered effects for four synthetic housing-like features
   :class: scientific-figure
   :align: center

Upper left is median income. The known curve is ``2.2 tanh((income − 5) / 1.7)``, which flattens as income increases. Upper right is house age, the oscillation ``0.9 sin((age − 5) / 8)``. Lower left is latitude, the downward curve ``−0.55 (latitude − 37)² / 6``, centered near 37. Lower right is rooms per household, ``0.75 log(1 + rooms)``, which keeps rising but more slowly. In every panel the GAM and the NAM are the solid curves and the centered generator is dashed. A gap between a solid curve and the dashed curve is where that fit flattened, exaggerated, or missed part of the relationship. The GAM's shape depends on the basis and the penalty. The NAM's shape depends on the network, the regularization, and the optimization. Rooms per household were drawn lognormal with underlying normal parameters 1.5 and 0.35.

Limits of the interpretation
----------------------------

These are not sale prices, and latitude here is not a spatial smooth. A high R² for both models is not evidence that their income curves are interchangeable. Centering removes a constant that could have been stored in the intercept. It does not identify a causal effect. The four features are drawn independently, so a signal that would be shared by correlated housing measurements in a real extract is absent from this simulation. The NAM scores can also move if the epoch limit, learning rate, or hidden widths change. The GAM scores can move if ``k`` or the smoothing criterion changes.

:notebook-download:`../tutorials/gam_nam`
