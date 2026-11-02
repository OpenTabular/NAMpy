How do we report conditional quantiles instead of one forecast?
===============================================================

A point forecast is one summary of a conditional distribution. A quantile forecast says which level of the response the model is targeting, and an interval built from two quantiles has a coverage to check. Three estimators answer different questions. ``QuantileGAM`` estimates a quantile, ``CalibratedQuantileGAM`` calibrates uncertainty for that estimated function, and QNAM estimates several ordered quantiles with a neural additive model.

The examples are simulated. In the distributional tutorial the mean is ``sin(3x)`` and the standard deviation increases with ``x``. In the QNAM tutorial the conditional quantiles are known because the noise is standard normal.

What is the problem?
--------------------

The decision needs a lower and upper level of demand, or of another continuous response, rather than only the conditional mean. Coverage of those levels on held-out rows is part of the result.

The distributional tutorial draws 1,200 values of ``x`` uniformly on −1 to 1, with seed 11. The response is ``sin(3x)`` plus noise whose standard deviation is ``0.12 + 0.35 (x + 1) / 2``, so the noise is smallest at the left of the range and largest at the right. Twenty percent of the rows are held out with ``random_state=7``, which leaves 240 test rows. The remaining rows are split again, with ``test_size=0.25`` and ``random_state=8``, into the training and validation sets used by the neural fit.

Why is a linear model insufficient?
-----------------------------------

A linear mean with constant variance produces parallel quantile lines. Both tutorials use a scale that changes with ``x``, so the gap between a lower and an upper quantile widens as ``x`` increases. A single squared-error fit estimates the conditional mean. A residual standard deviation from that fit is one number for the whole sample, and it is not a quantile model.

Why this model?
---------------

``QuantileGAM`` minimizes pinball loss at a stated level. In the uncertainty tutorial, level 0.5 is the median. ``predict_interval`` on the calibrated estimator is a confidence statement about that median function. Split conformal prediction, applied to the already fitted quantile model, targets a future response. Those widths are not interchangeable.

The distributional comparison fits two neural models on the same split. The normal location-scale model builds an interval as ``μ ± 1.282 σ``, the central 80% of a normal distribution. The quantile model is a ``QNAMLSS`` with hidden widths 24 and 12, dropout 0, and quantile levels 0.1, 0.5, and 0.9. It minimizes pinball loss at those levels and constrains the predicted quantiles to stay ordered. Ordering prevents the 0.1 curve from crossing the 0.9 curve. It does not set the coverage.

The :doc:`QNAM calibration page </in_depth/quantile_calibration>` repeats the ordered-quantile fit on a simulation whose conditional distribution is standard normal noise around known feature curves, and checks coverage on 280 test observations. Age is one of the features, and coverage is also tabulated inside age bins with edges 18, 31, 44, 57, and 70.

What the fit showed
-------------------

On the 240 test rows, the normal location-scale interval covers 77.5% of responses with mean width 0.739. The quantile interval covers 91.25% with mean width 1.071. The nominal target is 80%. The normal interval is closer on this normal-data simulation. The wider quantile interval overcovers. Ordering stopped the quantiles from crossing and did not make the coverage exact.

The QuantileGAM tutorial uses a different sample, with a linear conditional median, to separate two intervals that share an ``x`` axis. Six hundred rows are split into 300 training, 150 calibration, and 150 test rows. ``level=0.5`` is the median of the response. ``predict_interval(..., level=0.95)`` on the calibrated estimator is 95% confidence for that median function.

.. image:: /_static/features/quantile_targets.png
   :alt: Pointwise confidence band for the conditional median beside a wider interval for a future response
   :class: scientific-figure
   :align: center

Both panels plot the same held-out responses and the same dashed true median, ``0.4 + 0.8 x``. The left band is the 95% pointwise interval for the median function. The right band is the 90% interval for a future response. The mean width of the median-function interval is about 0.30, and the future-response interval is about 2.08. Most held-out points fall outside the left band because that band targets the median, not the noisy response. The right band is built to contain a new response.

.. image:: /_static/features/conformal_intervals.png
   :alt: Uncalibrated response band beside conformal bounds, with held-out coverage and mean width
   :class: scientific-figure
   :align: center

The left panel compares a deliberately uncalibrated band of plus or minus 0.5 around the fitted median with the 90% conformal bounds. On the 150 test rows that band covers about 55% of responses, and the conformal interval covers 88%. The middle panel is that coverage, with a line at the 90% target. The right panel is the mean width: 1.0 for the uncalibrated band and about 2.08 for the conformal interval. The narrow band has no coverage claim. The wider conformal interval is the one whose target is a future response.

.. image:: /_static/detailed_quantile_median.png
   :alt: Predicted versus known conditional medians
   :class: scientific-figure
   :align: center

The QNAM median figure compares the fitted median with the known conditional median on the calibration simulation. Only 39.3% of the 280 test responses fall below that fitted median, where about half should. The curve can look smooth and still sit above too many of the responses.

.. image:: /_static/detailed_quantile_calibration.png
   :alt: Test quantile calibration and fitted versus known age contributions
   :class: scientific-figure
   :align: center

The calibration figure has two parts. One is the coverage check: 3.9% of test responses fall below the lower quantile, where about 10% should, and 97.5% fall below the upper quantile, where about 90% should. The interval therefore covers 93.6% rather than 80%. The other part is the fitted age contribution beside the known age curve. Age is tabulated again inside bins with edges 18, 31, 44, 57, and 70. Coverage of the interval and the shape of the age curve are separate checks.

Limits of the interpretation
----------------------------

One test coverage is one sample. Overcoverage from a wider interval can look conservative while the quantile function itself is biased, which is what the QNAM median shows: 39.3% of responses fall below a curve that was asked to be the median. Ensemble spread among several NAM fits, also plotted in the distributional tutorial, is disagreement between models and is not a prediction interval. A demand series with trend, season, and forecast-error in the covariates is not one of these simulations.

:download:`Download the distributional comparison <../notebooks/04_distributional_models.ipynb>`

:download:`Download the QuantileGAM notebook <../notebooks/quantile_uncertainty.ipynb>`

:notebook-download:`../in_depth/quantile_calibration`
