Fit and assess conditional quantiles
====================================

A fitted quantile should have a corresponding proportion of observations below it. We will compare estimated quantiles with their known values and examine coverage across the predictor range.

Ordered quantiles avoid crossing, but ordering alone does not ensure accurate coverage.

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

   age_bin_edges = [18, 31, 44, 57, 70]
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

Age bins are fixed at 18, 31, 44, 57, and 70 before examining the test outcomes.
Within each bin, we compare the fraction below each fitted quantile with its
nominal level. Counts show how much test evidence supports each comparison.

.. code-block:: python

   age_bins = pd.cut(X_test.age, age_bin_edges, include_lowest=True)
   binned_coverage = pd.DataFrame({
       "Age bin": age_bins,
       **{f"Fraction below q={level:g}": np.asarray(y_test) <= prediction[:, column]
          for column, level in enumerate(quantiles)},
   }).groupby("Age bin", observed=False).agg(["mean", "count"])

.. include:: /_generated/detailed_quantile_binned_coverage.inc

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

Diagnose the fit using validation rows
--------------------------------------

Before another final evaluation, use the validation split to investigate the
failure. Here are the same coverage quantities computed without test outcomes:

.. code-block:: python

   validation_prediction = np.asarray(model.predict(X_val))
   validation_coverage = pd.DataFrame({
       "Quantile": quantiles,
       "Fraction below": [np.mean(y_val <= validation_prediction[:, j])
                          for j in range(len(quantiles))],
   })
   print(validation_coverage)

.. include:: /_generated/detailed_quantile_validation_coverage.inc

Validation shows the same tail pattern: about 4.6% of responses fall below
the lower quantile and 98.9% below the upper one. The median fraction is
45.0%, closer to its target than the test fraction but still below 50%.

Check validation pinball loss and coverage together: wider intervals can
increase coverage while making quantile estimates worse. Inspect the training
and validation objective histories for incomplete optimization; compare a
prespecified learning-rate range with a longer training budget using the same
validation rows and random seeds. If both losses remain high, examine target
scaling and capacity. If training improves while validation worsens, investigate
regularization and feature dependence. Check validation coverage within the
same fixed age bins to distinguish a global bias from local shape errors.

Choose any revised procedure using validation loss and these diagnostics,
then freeze it before evaluating fresh test data. The test results already
shown describe the original fit; this diagnostic recipe does not claim they
have improved. Ordered predictions prevent crossing but do not ensure calibration.

:download:`Download the complete example <../../examples/example_qnam.py>`
