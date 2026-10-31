Survival prediction, evaluation and uncertainty API
===================================================

See :doc:`/user_guide/survival` for the task workflow.

.. autoclass:: nampy.survival.SurvivalTarget
   :members: from_frame, take

.. autoclass:: nampy.survival.SurvivalPrediction
   :members: survival, log_survival, cumulative_hazard, cumulative_hazard_at, hazard, quantile, median, rmst, take

.. automodule:: nampy.survival.evaluation
   :members:

.. automodule:: nampy.diagnostics.survival
   :members:

.. automodule:: nampy.survival.uncertainty
   :members:

.. automodule:: nampy.survival.bootstrap
   :members:

Numerical integration contract
------------------------------

Weibull/lognormal RMST uses adaptive quadrature of the shared log-survival law,
with a logarithmic time change and a median break point. Absolute error control
is ``1e-10`` in time units and relative control is ``1e-9``; integration failures
raise explicitly.

Cox risk scaling
----------------

Partial likelihood and coefficient derivatives are invariant to a common
predictor shift within each stratum. Risk calculations normalize their
exponentials; fitted ``CoxGAM`` baselines and their uncertainty share a saved
stratum-specific predictor reference. Survival prediction uses predictors
relative to the same reference, so baseline scaling does not change fitted
curves or their standard errors. Raw ``predict_link`` values and explicit
new-row offsets retain their original predictor scale.
