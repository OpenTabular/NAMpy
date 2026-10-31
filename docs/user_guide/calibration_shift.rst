Calibrate under covariate shift
===============================

Use supplied density ratios only when the covariate-shift assumptions are
appropriate. Start with :doc:`calibration` for the independent data roles.

Covariate-shift weighting
-------------------------

Supply ``density_ratio`` values representing
``dP_target(X)/dP_calibration(X)``, on a common arbitrary scale, while the
conditional response distribution remains unchanged. These are not case,
frequency, survey or optimizer loss weights. A density-ratio function must be
known or fitted separately; NAMpy does not estimate one during calibration.

.. code-block:: python

   weighted = calibrate_regressor(
       model, X_cal, y_cal, density_ratio=ratio_cal, level=.9,
       weight_source="known")
   interval = weighted.predict_interval(X_test, density_ratio=ratio_test)

The weighted empirical measure includes each new observation's weight as an
atom at positive infinity. Consequently corrections depend on the supplied
new-point ratios; there is no single global rank/correction. Explicit positive
finite test ratios are required. Calibration ratios can be zero. If calibration
mass is insufficient, including all-zero mass, the result is unbounded.
No mean-weight substitution, clipping or percentile interpolation occurs.
``ConformalCalibration.corrections(ratio_test)`` exposes the per-row corrections.

``weight_source="known"`` labels marginal coverage under known covariate shift.
``"estimated"`` labels approximate coverage with estimated ratios; accuracy and
coverage are not guaranteed merely because estimates were supplied. Ratio
estimation or method selection using calibration responses would invalidate the
stated split procedure. Array-level ``calibrate_conformal`` accepts these same
arguments. Positive rescaling leaves the mathematical quantile unchanged.

The method follows Tibshirani, Barber, Candès and Ramdas (2019),
`Conformal Prediction Under Covariate Shift
<https://www.stat.berkeley.edu/~ryantibs/papers/weightedcp.pdf>`_.

For methodological citations and paper links, see :doc:`/references`.
