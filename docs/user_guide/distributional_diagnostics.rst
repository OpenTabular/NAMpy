Assess distributional residuals and centiles
============================================

Use held-out response distributions to examine calibration and residual shape.
These diagnostics do not refit a model or supply coefficient uncertainty.

Shared quantile residuals and worm plots
----------------------------------------

Use the fitted distribution on observed rows to obtain normalized quantile
residuals. Continuous observations use ``Phi^-1(F(y))``. At a probability jump,
the operation draws a uniform value between ``F(y-)`` and ``F(y)`` before applying
the normal quantile function. It works for discrete and mixed laws through
their left-limit CDF operations, without assuming every jump lies on an integer.

.. code-block:: python

   from nampy.diagnostics import prepare_worm_plot
   from nampy.plotting import render_worm_plot

   distribution = model.predict_distribution(X_test)
   from nampy.diagnostics import quantile_residuals
   residuals = quantile_residuals(distribution, y_test, random_state=7)
   worm_data = prepare_worm_plot(residuals, x=X_test["age"], n_bins=4)
   figure = render_worm_plot(worm_data)

The ``QuantileResiduals`` result contains ``residuals``, ``pit``, ``cdf_lower``,
``cdf_upper``, ``randomized`` and ``clipped`` arrays. Default clipping is disabled:
zero/one probabilities, including rounded tails, can produce infinite residuals.
To request clipping explicitly, pass ``clip=1e-12`` and inspect the ``clipped``
flags. The original CDF bounds remain available. Repeated randomizations can use
successive calls with a single NumPy Generator. Alternatively, supply an explicit
``uniforms`` vector, especially when comparing languages whose seeds use
different random-number streams. Observation weights are not expanded or used
to weight these diagnostics. Responses must respect the law's observation
support; probability-threshold queries are more permissive than observations.

Worm plots detrend normal QQ points and optionally fit a cubic trend.
``groups=...`` accepts category labels; ``x=...`` forms numeric bins. Explicit
``breaks`` define left-closed bins with the last right-closed. Tied quantile
edges collapse. Missing group labels/nonfinite x are counted separately; empty
explicit bins remain visible. Each panel records original row indices, total,
missing and infinite-residual counts. Cubic fitting requires four finite rows.
QQ positions follow R's default ``ppoints`` convention.

Shading is a pointwise reference envelope for independent standard normal
observations. It is not simultaneous coverage or an adjustment for fitting the
model. Prefer held-out or cross-fitted diagnostics for calibration. These shared
functions leave ``gam_.residuals`` and mgcv QQ behavior unchanged; family-specific
residual types can still be unsupported there. ``plot_worm`` is a convenience
that prepares data and renders it in one call.

Conditional centiles and calibration
------------------------------------

Choose explicit prediction rows for a centile curve. With several covariates,
vary the desired covariate on a grid while fixing the others to a chosen reference
row. Averaging conditional quantiles does not produce marginal quantiles.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from nampy.diagnostics import prepare_centiles, centile_calibration
   from nampy.plotting import render_centiles

   grid = pd.concat([X_test.iloc[[0]]] * 100, ignore_index=True)
   grid["age"] = np.linspace(X_test["age"].min(), X_test["age"].max(), 100)
   grid_distribution = model.predict_distribution(grid)
   curves = prepare_centiles(grid_distribution, grid["age"], probs=[0.05, 0.5, 0.95])
   figure = render_centiles(curves, xlabel="Age")

   held_out = model.predict_distribution(X_test)
   calibration = centile_calibration(held_out, y_test, probs=[0.05, 0.5, 0.95])

``CentileCurves`` contains sorted x, probabilities, quantiles, original
``row_indices`` and the count of excluded nonfinite x. Missing prediction rows
remain gaps. Requested probabilities must be strictly increasing inside (0,1).
``plot_centiles`` combines preparation and rendering. These are conditional
outcome quantiles, with no confidence-band shading or parameter-uncertainty
calculation. Discrete curves can have steps and repeated quantile values.
They are rendered as step curves between supplied grid rows.

``CentileCalibration`` reports ``observed`` coverage, the mean attainable
``model_probability = F(q_p)`` and per-centile ``n_used``. For discrete laws,
``F(q_p)`` can exceed the nominal probability p even under a correctly specified
model. Missing/nonfinite observations or quantiles are excluded with counts;
there are no implicit weights or independence-based confidence intervals.

Read Gamma parameter meanings
-----------------------------

For Gamma responses, ``sigma`` is a coefficient of variation, so
:math:`\operatorname{SD}(Y\mid x)=\mu(x)\sigma(x)`. These three panels
use the same fitted model and prediction rows from the
:doc:`Gamma location–scale example <../notebooks/gamma_location_scale>`.
Absolute spread can change through either fitted parameter.

.. image:: /_static/howto_gamma_parameters.png
   :alt: Same Gamma fit showing conditional mean, coefficient of variation, and their product giving response standard deviation
   :class: scientific-figure

For methodological citations and paper links, see :doc:`/references`.
