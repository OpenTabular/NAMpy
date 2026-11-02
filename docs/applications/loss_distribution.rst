How does the distribution of positive losses change with covariates?
====================================================================

A positive loss can change its mean and its relative spread. A single residual variance, or a fixed Gamma dispersion, collapses those into one number. This page uses the simulated Gamma location-scale tutorial. There ``σ`` is a coefficient of variation: the conditional variance is ``μ²σ²``.

What is the problem?
--------------------

The target is the conditional distribution of a positive response. A tail probability depends on both fitted parameters, so a mean by itself leaves the spread unspecified.

The notebook draws 700 values of ``x`` uniformly on 0 to 1, with seed 108. The generating mean and coefficient of variation are

.. code-block:: text

   μ = exp(0.5 + 0.6 sin(2 π x))
   σ = 0.2 + 0.3 x

and ``y`` is Gamma with that mean and that coefficient of variation. The split is ``train_test_split(..., test_size=0.25, random_state=108)``. A grid of 180 values of ``x`` is used only to draw the curves. It is not part of the fit.

Why is a linear model insufficient?
-----------------------------------

A linear mean with one residual scale assumes constant variance on the response or, after a log transform, a different and still rigid variance law. In this simulation the mean is periodic on the log scale and the coefficient of variation increases linearly with ``x``. A mean-only Gamma model can follow the average and still misstate the spread. The fixed-dispersion Gamma tutorial states the restriction explicitly: ``Var(Y|x) = φ μ(x)²`` when ``φ`` does not depend on covariates. Here ``σ`` plays the role of ``sqrt(φ)`` and is allowed to depend on ``x``.

Why this model?
---------------

The location-scale fit is a ``GAMLSS`` with ``family="gamma"`` and a separate formula for each parameter:

.. code-block:: python

   GAMLSS(
       family="gamma",
       formula={
           "mu": "y ~ s(x, bs='cr', k=10)",
           "sigma": "~ s(x, bs='cr', k=8)",
       },
   ).fit(train)

The mean uses the log link, which matches the exponential in the generator. The coefficient of variation has its own cubic regression spline, with a smaller basis dimension than the mean. ``predict_parameters(grid)`` returns both parameters on a grid. ``σ`` is not the standard deviation. Multiplying ``μ`` by ``σ`` produces the conditional standard deviation plotted in the third panel.

What the fit showed
-------------------

The figure has three panels, each with the fitted curve and the curve used to generate the data: the conditional mean, the coefficient of variation, and their product. Test mean RMSE is 0.584, computed from the predicted mean and the held-out responses. The negative log likelihood is 0.843 and scores the fitted distribution together with its mean. A model can have a small RMSE and a poor likelihood when the mean is close and the spread is not.

.. image:: /_static/howto_gamma_parameters.png
   :alt: Same Gamma fit showing conditional mean, coefficient of variation, and their product giving response standard deviation
   :class: scientific-figure
   :align: center

Each panel uses the same 180-point grid of ``x``, drawn only for the figure. The dashed curve is the generator and the solid curve is ``predict_parameters``. The left panel is the conditional mean. The generator is ``exp(0.5 + 0.6 sin(2 π x))``, so the curve is periodic on the log scale and the log link is what lets the spline stay on that scale. The middle panel is ``σ``, the coefficient of variation, generated as ``0.2 + 0.3 x`` and given its own spline with ``k=8``. The right panel multiplies the two fitted parameters. That product is the conditional standard deviation, ``μ σ``, because a Gamma variance is ``μ² σ²``. A change in the right panel can come from the mean, from ``σ``, or from both. Reading only the middle panel misses the absolute spread.

The :doc:`Tweedie tutorial </notebooks/tweedie_response>` is the related example when losses can be exact zeros. The :doc:`distributional neural tutorial </notebooks/04_distributional_models>` compares a normal location-scale network with quantile forecasts on a different simulation.

Limits of the interpretation
----------------------------

The response is a simulated positive variable with a known mean and coefficient of variation. Agreement with the known curves is possible because those curves were used to simulate the sample. A small mean RMSE does not by itself validate the tails. ``σ`` must be read as a coefficient of variation. The standard deviation of a future loss is the product in the third panel, and a confidence band for the mean is a different object again.

:download:`Download notebook <../notebooks/gamma_location_scale.ipynb>`
