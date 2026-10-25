Fit and assess conditional quantiles
====================================

A quantile model estimates selected points of the conditional response distribution. Here we fit the 10th, 50th, and 90th percentiles, then examine whether they recover the simulated quantiles and achieve the expected coverage on test observations.

Generate a response with known quantiles
----------------------------------------

The response has additive location and scale functions with normal noise. Its conditional quantiles are therefore mu(x) + Phi_inverse(tau) sigma(x). This gives us a reference for evaluating each fitted quantile, including the feature effects that produce it.

.. math::

   Y = \mu(x) + \sigma(x)\epsilon,\quad \epsilon\sim N(0,1),
   \qquad Q_\tau(Y\mid x)=\mu(x)+\Phi^{-1}(\tau)\sigma(x).

.. code-block:: python

   import tempfile
   import numpy as np
   import pandas as pd
   import torch
   from sklearn.model_selection import train_test_split
   from nampy.models import QNAMLSS


.. literalinclude:: ../../examples/example_qnam.py
   :language: python
   :start-after: def main():
   :end-before: # Fit QNAM
   :dedent: 4

Fit the quantiles
-----------------

QNAM minimizes pinball loss at the requested quantile levels. For residual u = y - q_tau(x), the loss is tau*u when u is positive and (tau-1)*u otherwise. The asymmetry makes the fitted curve target the chosen percentile.

The objective sums the losses across the requested levels and averages over observations. QNAM orders its full predictions so that the lower quantile cannot exceed the upper one.

.. literalinclude:: ../../examples/example_qnam.py
   :language: python
   :start-after: # Fit QNAM
   :end-before: # Predictions
   :dedent: 4

Assess coverage
---------------

About 10% of outcomes should fall below a well-calibrated 10th-percentile prediction, 50% below its median, and 90% below its 90th percentile. The interval between the lower and upper predictions should cover about 80% of outcomes. We estimate these quantities using the 280 test observations.

.. code-block:: python

   prediction = np.asarray(model.predict(X_test))
   for column, quantile in enumerate(quantiles):
       coverage = np.mean(y_test <= prediction[:, column])
       assert 0 <= coverage <= 1

   lower, median, upper = prediction.T
   interval_coverage = np.mean((y_test >= lower) & (y_test <= upper))
   mean_width = np.mean(upper - lower)


.. include:: /_generated/detailed_quantile_coverage.inc

.. include:: /_generated/detailed_quantile_interval.inc

The fitted quantiles are ordered, but their coverage is inaccurate in this example. Only 3.9% of test responses fall below the lower quantile and 97.5% fall below the upper quantile. Together, the interval covers 93.6% rather than the target 80%. The fitted median also sits too low overall: only 39.3% of responses fall below it.

This is a reason to revisit fitting choices using validation data. The test results diagnose the current model; they should not become the target for repeated tuning.

.. image:: /_static/detailed_quantile_calibration.png
   :alt: Test quantile calibration and fitted versus known age contributions
   :class: scientific-figure

The calibration panel compares the requested quantile levels with their observed test frequencies. The diagonal marks agreement. The age-effect panel compares the fitted quantile contributions with the known centered effects; the weak separation between the fitted curves shows incomplete recovery of how age affects different parts of the distribution.

Inspect the quantile effects
----------------------------

Each component array has one column per requested quantile. Use the same quantile labels when plotting those columns. We center the fitted and true effects over the same training rows so their vertical positions are comparable.

Ordering applies to complete quantile predictions. Individual centered feature contributions can cross because centering and the other terms also affect their sums.

.. image:: /_static/detailed_quantile_median.png
   :alt: Predicted versus known conditional medians
   :class: scientific-figure

The predicted median broadly follows the known conditional median, but agreement in this plot is not enough to establish calibration. The coverage table shows the remaining bias and the wider errors in the tails.

:download:`Download the complete example <../../examples/example_qnam.py>`
