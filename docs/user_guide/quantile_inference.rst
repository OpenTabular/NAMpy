Calibrated quantile-function inference
======================================

Choose the target before choosing an interval. ``QuantileGAM`` estimates one
conditional quantile with pinball or smooth ELF loss. The separate
``CalibratedQuantileGAM`` calibrates approximate uncertainty in that fitted
quantile function using qgam 2.0.1's ``calFast`` route. Future-response intervals
are supplied by :doc:`calibration`; they include response variation and answer
a different question.

Quantile level and confidence level
-----------------------------------

``level`` on the estimator is the conditional quantile level, such as ``.3``.
``level`` on ``predict_interval`` is the confidence level for the estimated
function, such as ``.95``. Neither changes the other's target. A 95% confidence
interval for a conditional 30th percentile is not an interval intended to
contain 95% of future responses.

.. literalinclude:: /../examples/example_calibrated_quantile.py
   :language: python

Compare function uncertainty with response variation
----------------------------------------------------

.. figure:: /_static/features/quantile_targets.png
   :alt: Narrow 95% confidence band for an estimated median beside a wider 90% conformal future-response interval.
   :width: 100%

   In the :doc:`quantile tutorial </notebooks/quantile_uncertainty>`, mean
   width is about 0.30 for the 95% pointwise confidence interval for the
   median function and 2.08 for the 90% future-response interval. They cover different targets:
   noisy future responses need not fall inside a confidence band for
   their conditional median. Future-response calibration uses a separate
   cohort.

Choose bandwidth and smoothing together
---------------------------------------

The four combinations of supplied/adaptive bandwidth and fixed/selected
penalties are deliberate fitting choices:

.. list-table:: Fitting controls
   :header-rows: 1

   * - Control
     - Supplied or fixed
     - Estimated or selected
   * - ELF bandwidth
     - ``bandwidth=0.1`` supplies a positive bandwidth in response units.
     - ``bandwidth=None`` estimates the Gaussian-residual SHASH bandwidth.
   * - Spline penalties
     - ``optimize_smoothing=False`` uses supplied ``smoothing_params``.
     - ``optimize_smoothing=True`` selects REML penalties at each calibrated
       scalar loss scale using the existing extended-family PIRLS engine.

Both modes calibrate one constant loss scale over the declared
``log_scale_bounds``. A boundary minimum raises so the caller can choose an
adequate range explicitly. Fixed-bandwidth, fixed-penalty fits retain their
coefficient-score route. Automatic smoothing uses the existing statistical
compiler, constraints, penalties and exact outer derivatives.

``fit_levels(X, y, levels=[.2, .5, .8])`` returns independent calibrated models.
When bandwidth is adaptive, it shares the initial Gaussian/SHASH preparation.
Quantiles may cross: independent fits do not impose joint or noncrossing
inference.

Read intervals and fit diagnostics
----------------------------------

``predict_interval`` returns ``QuantileFunctionInterval`` with estimate,
standard error and bounds. Its ``uncertainty_kind`` is
``calFast_calibrated_quantile_function`` in both fitting modes; coverage metadata
states the conditioning:

.. list-table:: Interval interpretation
   :header-rows: 1

   * - Smoothing mode
     - ``coverage``
     - Interpretation
   * - Fixed penalties
     - ``approximate_pointwise_fixed_smoothing``
     - Pointwise uncertainty for the quantile function under the supplied
       smoothing strengths.
   * - Selected penalties
     - ``approximate_pointwise_conditional_on_selected_smoothing``
     - Pointwise uncertainty conditional on selected smoothing strengths;
       it does not include their selection uncertainty.

``loss_scale_``, ``posterior_covariance_``, ``sandwich_covariance_``,
``calibration_loss_`` and ``calibration_profile_`` retain the completed fit.
The profile keeps scalar evaluation results rather than intermediate models.
``fit_diagnostics_`` reports coefficient fitting, smoothing selection and
loss-scale calibration separately. Missing diagnostics are ``None``, not a
zero residual or zero iterations: bounded loss-scale optimization, for example,
does not report a score derivative. Fixed-penalty fitting has no smoothing
selection stage. Requesting automatic smoothing when the design has no free
penalties also leaves this stage absent and retains fixed-smoothing interval
metadata. Coefficient optimality is the measured infinity norm of the
penalized score; function evaluations remain distinct from solver iterations.

The model's ``capabilities()`` declares calibrated coefficient covariance and
``uncertainty.quantile_function`` separately from the ordinary quantile-loss
route. Failed refits preserve successful fitted state, parameter updates
invalidate it, and trusted fitted-state save/load preserves predictions and
intervals.

Conditions and unsupported routes
---------------------------------

The implemented route uses independent unweighted observations, an identity
link and a constant loss scale. Covariate-dependent or heteroscedastic loss
scales, bootstrap calibration, weighted inference, simultaneous bands,
noncrossing multiple quantiles and classical likelihood/AIC inference remain
unsupported. The same restrictions apply when bandwidth or smoothing is
selected.

See :doc:`/statistical_reference/estimators/CalibratedQuantileGAM` for the
constructor and :doc:`/statistical_reference/quantile_calibration` for the
calibration calculations and independent validation evidence.

.. toctree::
   :maxdepth: 1

   /statistical_reference/quantile_calibration

References
----------

The quantile-loss foundation is [Koenker1978]_. The implemented qgam-style ELF,
bandwidth and calibrated loss-scale routes are based on [FasioloQuantile2021]_
and the software paper [FasioloQgam2021]_. See :doc:`/references` for full paper
links.
