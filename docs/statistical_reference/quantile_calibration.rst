Calibrated quantile calculations and validation
===============================================

Start with :doc:`/user_guide/quantile_inference` to choose the fitting mode and
interpret the interval target. This reference describes the numerical owner
and the scope of the evidence independently of declared implementation support.

ELF fitting and loss-scale calibration
--------------------------------------

``CalibratedQuantileGAM`` fits an ELF conditional quantile and calibrates one
scalar loss scale by comparing posterior and shrinkage-sandwich variances over
the training design. It reuses the existing statistical compiler, constraints
and spline penalties. Ordinary ``QuantileGAM`` fitting behavior is unchanged.

``bandwidth`` is a supplied positive ELF bandwidth in response units (upstream
``co``), or ``None`` for the Gaussian-residual SHASH estimate. For fixed
penalties, the coefficient solver uses exact score/Hessian calculations and
certifies the final Newton score step; ``max_iter`` limits score evaluations
per loss scale. Automatic smoothing selects REML penalties at every scalar
loss scale using the extended-family PIRLS engine and exact outer derivatives.
There is no alternate-solver retry or change to the native fitting engine's
stopping rules.

Covariance construction uses qgam's prescribed diagonal-scaled spectral inverse
and relative eigenvalue cutoff of ``1e-6``. This source-specific rank rule is
part of the calibration calculation, not a general covariance-jitter policy.
Intervals with selected smoothing condition on that selection; calibration
alone does not propagate smoothing-selection uncertainty.

Numerical diagnostics
---------------------

Coefficient convergence, smoothing selection and bounded loss-scale calibration
are separate entries in ``fit_diagnostics_``. Automatic coefficient iteration
counts come from the underlying solver, and the coefficient score is evaluated
at the returned coefficients with the selected penalty and loss scale. A
solver's function-evaluation count is not reported as an iteration count.
Unavailable iteration counts or derivative residuals are ``None``. In
particular, bounded scalar calibration reports convergence and evaluations
without claiming a measured perfect derivative residual.

Independent validation evidence
-------------------------------

Independent R fixtures compare criterion profiles at four scales, posterior and
sandwich prediction variances, and optimized calibrated endpoints. Both a small
basis and a larger basis exercising score-covariance shrinkage are covered.
A separate prespecified 100-replicate linear-median simulation checks nominal
90% pointwise coverage with an exact binomial test. It validates that regime,
not universal calibration. R is needed only to regenerate the fixtures, not
to fit models in Python. Adaptive bandwidth and REML evidence is recorded in
:doc:`/development/neighbor_parity`.

.. autoclass:: nampy.gam.inference.quantile.QuantileFunctionInterval
   :no-members:

References
----------

Mathematical reference: Fasiolo et al. (2021), *Fast calibrated additive quantile
regression*, JASA [FasioloQuantile2021]_; software reference
[FasioloQgam2021]_. Source specification:
`qgam <https://github.com/mfasiolo/qgam>`_, version 2.0.1, ``.llkGrads``,
``.biasedCov``, ``.sandwichLoss`` and ``.invert_psdef_matrix``.
