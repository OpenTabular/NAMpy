Calibration
===========

Post-fit calibration stores scores from a calibration sample and applies them
to later predictions. The calibration rows stay separate from the rows used to
fit or tune the estimator. Coverage is a property of that exchangeability
assumption and of the selected score, not of the fitted model class.

:ref:`howto-user-guide-calibration` is the procedure for scalar regression
intervals. Grouped and covariate-shift routes are in
:ref:`howto-user-guide-calibration-groups` and
:ref:`howto-user-guide-calibration-shift`. Survival bounds are in
:ref:`howto-user-guide-calibration-survival`. The shared result contract is in
:ref:`howto-user-guide-calibration-contract`.

.. currentmodule:: nampy.calibration

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: noindex-class

   ConformalCalibration
   ConformalInterval
   ConformalPredictor
   GroupedConformalCalibration
   SurvivalConformalCalibration
   SurvivalConformalPredictor

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: noindex-function

   calibrate_conformal
   calibrate_regressor
   calibrate_grouped
   calibrate_survival
   calibrate_survival_estimator
