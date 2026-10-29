Calibrated quantile-function inference
======================================

``CalibratedQuantileGAM`` implements fixed and adaptive bandwidth/smoothing variants of the
``calFast`` route from qgam 2.0.1. It fits an ELF conditional quantile and
calibrates one scalar loss scale by comparing posterior and shrinkage-sandwich
variances over the training design. It uses the existing statistical compiler,
constraints and spline penalties. ``QuantileGAM`` retains its existing fitting
behavior.

.. literalinclude:: /../examples/example_calibrated_quantile.py
   :language: python

``level`` on the estimator is the conditional quantile level. ``level`` on
``predict_interval`` is the interval confidence level. The returned
``QuantileFunctionInterval`` contains the estimate, standard error and bounds,
with ``uncertainty_kind='calFast_calibrated_quantile_function'`` and
``coverage='approximate_pointwise_fixed_smoothing'``. These intervals target the
conditional quantile function. Future-response prediction intervals are supplied
separately by :doc:`calibration`.

``bandwidth`` is a supplied positive ELF bandwidth in response units (upstream
``co``); ``smoothing_params`` is fixed. Calibration optimizes log loss scale
inside the declared ``log_scale_bounds``. A boundary minimum raises so that the
caller must choose an adequate search range. The coefficient solver uses exact
score/Hessian calculations and certifies the final Newton score step;
``max_iter`` limits score evaluations per scale. Covariance construction uses
qgam's prescribed diagonal-scaled spectral inverse and relative eigenvalue
cutoff of 1e-6. It does not introduce a fallback into the mgcv fitting engine.

``loss_scale_``, ``posterior_covariance_``, ``sandwich_covariance_``,
``calibration_loss_`` and ``calibration_profile_`` describe the completed fit.
The profile retains scalar evaluation results rather than intermediate models.
Failed refits preserve the successful fitted state, parameter updates invalidate
it, and trusted fitted-state save/load preserves predictions and intervals.

Scope and validation
--------------------

The supported route uses independent unweighted observations, an identity link
and a constant loss scale. Covariate-dependent loss scale estimation,
heteroscedastic loss scales, bootstrap calibration, weighted inference,
simultaneous bands, noncrossing multiple quantiles, and classical likelihood/AIC
inference are unsupported. Adaptive bandwidth and REML selection are described below; heteroscedastic
qgam calibration remains outside this supported subset.

Independent R fixtures compare criterion profiles at four scales, posterior and
sandwich prediction variances, and the optimized calibrated endpoints. Both a
small basis and a larger basis exercising score-covariance shrinkage are covered.
A separate prespecified 100-replicate linear-median simulation checks nominal
90% pointwise coverage with an exact binomial test. It validates that regime,
not universal calibration. R is needed only to regenerate the fixtures, not
to fit models in Python.

.. autoclass:: nampy.gam.inference.quantile.QuantileFunctionInterval
   :no-members:

See :doc:`/statistical_reference/estimators/CalibratedQuantileGAM` for the
constructor. Mathematical reference: Fasiolo et al. (2021), *Fast calibrated
additive quantile regression*, JASA; source specification:
`qgam <https://github.com/mfasiolo/qgam>`_, version 2.0.1, ``.llkGrads``,
``.biasedCov``, ``.sandwichLoss`` and ``.invert_psdef_matrix``.

Adaptive bandwidth and smoothing
--------------------------------

Set ``bandwidth=None`` for qgam's Gaussian residual SHASH bandwidth estimate.
Set ``optimize_smoothing=True`` to select REML penalties at every scalar loss
scale, using the existing extended-family PIRLS engine and exact outer
derivatives. Fixed-bandwidth, fixed-penalty fits retain their established route.
``fit_levels(X, y, levels=[.2, .5, .8])`` returns independent calibrated models
and shares initial Gaussian/SHASH preparation when bandwidth is adaptive.
Quantiles may cross; no joint or noncrossing inference is claimed. Intervals
condition on the selected smoothing parameters. Weighted calibration,
covariate-dependent loss scales, bootstrap calibration and simultaneous
quantile-function bands remain unsupported.
