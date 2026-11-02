How should exposure and extra-Poisson variation enter expected counts?
======================================================================

A claim count depends on how long, or over how much exposure, the policy was observed. It also often varies more than a Poisson mean allows. Age and territory would enter the same formula as further smooths. The executed examples isolate the two pieces that change the model even before those covariates are added: a log-exposure offset, and a negative-binomial variance.

The rows are simulated. In the overdispersion notebook the mean is ``exposure * exp(0.7 + sin(2 π x))`` and the counts are negative binomial with ``θ = 2``.

What is the problem?
--------------------

The target is an expected count. Exposure stays inside the mean through an offset, which keeps the smooth on the rate scale while the prediction remains a count.

The overdispersion notebook draws 900 rows. ``x`` is uniform on 0 to 1 and exposure is uniform on 0.5 to 3. The log exposure is stored as its own column so the formula can name it. Counts are drawn from a negative binomial whose mean is that exposure times a periodic rate, and whose variance is ``μ + μ²/θ`` with ``θ = 2``. The split is ``train_test_split(..., test_size=0.25, random_state=106)``.

Why is a linear model insufficient?
-----------------------------------

The simulated rate is periodic in ``x``. A straight line in ``x`` cannot represent that oscillation, and a cubic regression spline with ``k=12`` can. A Poisson variance, ``Var(Y) = μ``, is also too small for the process that generated the counts. Under the negative binomial used here, the variance exceeds the mean by ``μ²/θ``. With ``θ = 2`` that extra term is half the squared mean, so the discrepancy grows quickly as the expected count grows.

Why this model?
---------------

Both fits use the same mean structure, the same training rows, and the same offset:

.. code-block:: python

   formula = "y ~ s(x, bs='cr', k=12) + offset(log_exposure)"

   GAM(formula=formula, family="poisson",
       optimize_smoothing=True, smoothing_method="reml").fit(train)

   GAM(formula=formula,
       family=NegativeBinomialLogFamily(theta=1, estimate_theta=True),
       optimize_smoothing=True, smoothing_method="reml").fit(train)

``theta=1`` is only the starting value. ``estimate_theta=True`` replaces it with the fitted dispersion. REML selects the smoothing parameter of ``s(x)``. The offset coefficient stays fixed at one, so the smooth describes a rate and multiplying by exposure returns a count.

The notebook then builds a grid of 180 values of ``x`` with ``log_exposure = 0``, which is exposure one, and plots both fitted means against the known rate ``exp(0.7 + sin(2 π x))``. Pearson residuals are computed from each family's own variance, so a residual of a given size means a different standardized error in the two panels.

.. image:: /_static/applications_overdispersion.png
   :alt: Fitted mean at exposure one for Poisson and negative binomial, with Pearson residuals against fitted count
   :class: scientific-figure
   :align: center

The left panel is the rate at exposure one. The dashed curve is the rate used to simulate the counts. The two solid curves are the Poisson and negative-binomial fitted means on the same grid. The middle panel is the Poisson Pearson residual against the Poisson fitted count, and the right panel is the negative-binomial Pearson residual against its own fitted count. The horizontal line is zero. Because each residual divides by that family's standard deviation, a Poisson panel that fans out as the fitted count grows is the extra ``μ²/θ`` term showing up as a variance the Poisson model does not have.

The companion :doc:`count tutorial </notebooks/02_statistical_gams>` adds a group shift and plots expected counts at three exposure levels. Doubling exposure doubles the fitted count when the other inputs are held fixed.

.. image:: /_static/formula_exposure_offset.png
   :alt: Fitted expected counts at three exposure levels on the response scale
   :class: scientific-figure
   :align: center

What the fit showed
-------------------

``getTheta(trans=True)`` on the negative-binomial fit returns 2.429, compared with the simulation value 2. On the shared test rows, the mean negative log probability uses ``poisson.logpmf`` for the Poisson fit and ``nbinom.logpmf`` for the negative-binomial fit, both evaluated at that fit's predicted mean. It falls from 2.662 under the Poisson fit to 2.207 under the negative binomial fit. Mean absolute errors stay similar. The probability score detects the variance misspecification that the mean error largely misses, because a wrong variance changes the probability of the observed count even when the predicted mean is close.

Limits of the interpretation
----------------------------

The generated table contains ``x`` and exposure. Age and territory are absent, so this fit does not estimate them. Adding ``s(age)`` or a regional smooth would use the same formula interface on a different table. The negative-binomial improvement is one generated sample with seed 106. A mean standard error is not a prediction interval for a future count. Larger ``θ`` approaches a Poisson variance, and smaller ``θ`` permits more extra-Poisson variation. The estimated 2.429 is close to the generating value on this sample. It is one draw from the estimator, not a confidence interval for ``θ``.

:download:`Download the overdispersion notebook <../notebooks/overdispersed_counts.ipynb>`

:download:`Download the exposure notebook <../notebooks/02_statistical_gams.ipynb>`
