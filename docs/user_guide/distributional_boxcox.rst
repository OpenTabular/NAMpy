Box--Cox location, scale and shape families
===========================================

BCCG, BCT and BCPE describe positive outcomes with linked location, spread, power
and, for BCT/BCPE, tail parameters. Their numerical support differs; select a
family before selecting a fitting route.

BCCG probabilities, moments and joint REML fitting
--------------------------------------------------

``nampy.distributions.BCCGLaw`` supplies Box-Cox Cole-Green probabilities through
the same backend-neutral container and diagnostics as the other scalar laws::

    from nampy.distributions import BCCGLaw, DistributionPrediction
    from nampy.diagnostics import prepare_centiles

    prediction = DistributionPrediction(
        BCCGLaw(), {"mu": [1., 2., 3.], "sigma": [.4, .5, .6], "nu": [-1., 0., 1.]}
    )
    centiles = prepare_centiles(prediction, x=[0., 1., 2.])
    log_density = prediction.log_prob([.8, 2., 3.5])
    from nampy.diagnostics import quantile_residuals
    residuals = quantile_residuals(prediction, [.8, 2., 3.5])

The parameters require finite ``mu > 0``, ``sigma > 0`` and finite real ``nu``.
For positive responses, define
``z = ((y / mu)**nu - 1) / (nu * sigma)``. For positive ``nu`` the normal
variable is conditioned above ``-1 / (nu * sigma)``; for negative ``nu`` it is
conditioned below that boundary. Density includes the response Jacobian
``(y / mu)**nu / (sigma * y)`` and divides by
``Phi(1 / (sigma * abs(nu)))``. At exactly zero power the law is lognormal with
log-location ``log(mu)`` and log-standard-deviation ``sigma``.

``mu`` is generally neither the exact median nor the response mean. The declared
operations are log density, CDF/left CDF, survival, their logarithms, quantiles,
sampling, mean and variance. Shared quantile residuals, worm-plot preparation
and centiles use those capabilities directly. ``prediction.mean()`` and
``prediction.variance()`` evaluate response moments without substituting ``mu``.
For negative ``nu``, a positive-order moment of order ``k`` exists only when
``nu < -k``. In particular the mean diverges for ``-1 <= nu < 0``; the second
moment diverges for ``-2 <= nu < 0``. The mean returns positive infinity where
divergent. Variance returns NaN when the mean is infinite and positive infinity
for ``-2 <= nu < -1``, where the mean is finite but the second moment diverges.
These mathematical results are distinct from missing parameter rows, which
retain the shared all-NaN row policy.

Finite moments use checked adaptive quadrature of the normalized law. Positive
powers use log response centered at the moment integrand's mode; negative
powers use algebraic endpoint integration of the positive truncated-normal
variable ``U = (Y / mu)**nu``. Its distribution is normal with location one
and standard deviation ``abs(sigma * nu)``, conditioned on positivity. An
analytic endpoint correction retains accuracy just below the moment-existence
boundary. Small variances use centered integrals instead of subtracting nearly
equal raw moments. Exactly zero power uses the existing lognormal moment
implementation. Response units are applied in log space. Mathematically finite
moments may overflow or underflow float64; unsupported integration ranges or
failed error checks raise ``NotImplementedError``. Moments are numerical
inference operations, not Torch training objectives or differentiable losses.
Finite negative-power variance with ``sigma < sqrt(machine epsilon)`` is
explicitly unsupported: the current centered integral relies on a mean whose
precision has not been validated at that scale. This restriction does not
change the mathematical divergence results above or positive-power variance.

The statistical ``bccg_lss`` family supplies the normalized
log likelihood, exact score and observed Hessian in ``(log(mu), log(sigma), nu)``
predictors. It reuses the generic extension adapter's weighted coefficient
assembly, including offsets and shared coefficient columns. The removable
zero-power singularity is handled analytically, including mixed derivatives;
the positive-support normalizer and response Jacobian are retained. Independent
high-precision differentiation and fixed-penalty coefficient differences validate
these first/second derivatives. Saved R first-derivative hooks agree at ordinary
nonzero powers; R's second-derivative hooks supply working curvature and are
not substituted for the observed joint Hessian. The recorded R power score is
undefined at zero; the analytic kernel has its finite continuous limit.

Statistical fitting is available as ``GAMLSS(family='bccg_lss')``.
For a DataFrame ``data`` containing positive ``y`` and covariate ``x``::

    from nampy import GAMLSS

    model = GAMLSS(
        family="bccg_lss",
        formula={"mu": "y ~ s(x, bs='cr', k=5)",
                 "sigma": "~ s(x, bs='cr', k=4)",
                 "nu": "~ s(x, bs='cr', k=4)"},
        optimize_smoothing=True,
        smoothing_optimizer="outer_newton",
    ).fit(data)
    prediction = model.predict_distribution(data)
    parameters = model.predict_parameters(data)  # mu, sigma, nu
    errors = model.standard_errors(data)  # conditional natural-parameter errors

Automatic joint REML is validated with ``outer_newton`` (default), ``bfgs`` and
``efs``, including all three predictors smoothed. To hold smoothing strengths
fixed, set ``optimize_smoothing=False`` and ``smoothing_params=[.3, .5, .7]``.
Other method labels retain the existing mgcv general-family coercion to REML;
this is not separate ML/GCV/UBRE/NCV smoothing-selection support.

The likelihood has analytic mixed derivatives through fourth order, including
the zero-power limit and the truncated-normal normalizer. Independent
high-precision differentiation checks the tensors. Outer REML gradients and
Hessians are checked against finite differences with three penalties, offsets,
nonunit and zero weights. A derivative-free scalar REML profile independently
checks the selected penalty and fitted predictions for all three optimizers.
No finite-difference derivatives are used by fitting. Generic covariance
transport includes the existing approximate smoothing-uncertainty correction
when requested; this does not provide simulation/refit centile confidence bands.

Initialization uses upstream ``mu=(y+mean(y))/2``, ``sigma=.1``, ``nu=.5``;
the generic adapter projects their linked targets and subtracts predictor offsets.
At least three distinct positive-weight responses are required. This necessary
guard does not guarantee estimability or a unique optimum for arbitrary formulas.
Positive/negative-power unpenalized fits and nonunit/zero weights match saved
R GAMLSS endpoints. Three-predictor fixed-penalty fits are checked against an
independent normalized-likelihood optimum and observed covariance. Smoothed
GAMLSS RS/backfitting parity is not claimed. With frequency weights, hold physical
knots fixed when comparing to duplicated rows: automatic quantile knots otherwise
describe a different penalized model.

Natural-parameter errors, additive components, missing-row alignment, shared
diagnostics and persistence use the existing generic adapters. ``predict_point``
returns the response mean, including infinity when the fitted power implies
divergence. A power near zero is not rounded to zero to make moments finite.
Response/Pearson residuals require their moments to exist. Tight stopping rules
may report nonconvergence at a float64 objective plateau; the existing solver
status is retained and such fits are ineligible for comparison ranking.

Derivatives above order four, alternate links, BCCG censoring/truncation
and RS/backfitting remain unsupported. ``optim`` lacks the family-defined null
deviance it needs for objective scaling; ``magic`` and ``nlm.fd`` are also
unavailable. Existing transformed-fit and general-family NCV guards remain in
force. Method labels do not unlock another smoothing criterion.
This law is distinct from the
existing mgcv censored ``bcg`` family, whose fitting behavior is unchanged.
BCT/BCPE fitting and existence-checked moments, are described in this guide; continuous truncation and observation fitting
are covered in :doc:`distributional_observations`.

Ordinary values are checked against saved ``gamlss.dist`` 6.1.1 ``d/p/qBCCG``
results. Recorded upstream out-of-support log-density and log-CDF quirks are
deliberately not reproduced: nonpositive responses have zero probability and
log density/CDF of negative infinity. Normal interval masses are evaluated in
log space, with controlled Gaussian quadrature near the truncation boundary;
boundary quantiles invert the same interval in log width. Independent
high-precision checks cover tails that upstream CDF subtraction rounds to zero.
Quantiles and samples retain float64 outcome limits: an interior probability
can produce zero by underflow or infinity by overflow. These values cannot
round-trip through the CDF. Unsupported numerical ranges raise explicit errors
instead of substituting another distribution.

Run ``examples/example_bccg_probabilities.py`` for probabilities, moments and
their mathematical existence boundaries.

BCT and BCPE fitting and probability laws
-----------------------------------------

``BCTLaw`` and ``BCPELaw`` use the shared ``DistributionPrediction`` interface.
The fitting adapters use the same natural parameters as these laws::

    import numpy as np
    from nampy.distributions import BCTLaw, BCPELaw, DistributionPrediction

    parameters = {"mu": np.array([1., 2., 3.]),
                  "sigma": np.array([.4, .5, .6]),
                  "nu": np.array([-1., 0., 1.]),
                  "tau": np.array([5., 8., 3.])}
    prediction = DistributionPrediction(BCTLaw(), parameters)
    centiles = prediction.quantile([.05, .5, .95])
    draws = prediction.sample(500, random_state=7)
    from nampy.diagnostics import quantile_residuals
    residuals = quantile_residuals(prediction, [.8, 2., 3.5])

Both laws require finite positive ``mu, sigma, tau`` and finite real ``nu``.
They share ``z = expm1(nu * log(y / mu)) / (nu * sigma)`` and its exact
zero-power limit. The normalized response density is
``f_Z(z) * (y / mu)**nu / (sigma * y * F_Z(a))``, where
``a = 1 / (sigma * abs(nu))``. Positive powers condition the symmetric latent
law above ``-a``; negative powers condition it below ``a``. At zero power there
is no truncation and ``log(Y / mu) = sigma * Z``.

.. list-table:: Latent parameter meanings
   :header-rows: 1

   * - Law
     - Symmetric latent distribution
     - Meaning of ``tau`` and ``sigma``
   * - BCT
     - Ordinary Student-t
     - ``tau`` is degrees of freedom; ``sigma`` scales the t variable and is
       not its standard deviation.
   * - BCPE
     - Variance-one power exponential
     - ``tau`` is generalized-normal shape; its scale is
       ``sqrt(Gamma(1/tau) / Gamma(3/tau))``. ``sigma`` scales that latent
       variable before truncation.

BCPE at ``tau=2`` is exactly BCCG. BCT approaches BCCG as its finite degrees
of freedom increase, while retaining its Student-t law. At ``nu=0`` the laws
are log-t and log-power-exponential, respectively; BCT does not become
lognormal simply because its Box-Cox power is zero. ``mu`` is generally neither
the response mean nor the exact median.

Declared operations are log density, CDF/left CDF, survival, logarithmic tails,
quantiles and sampling. These support existing shared quantile residuals,
worm plots, centiles, calibration and atom-mixture probability composition.
Mean and variance use the existence checks and numerical integration described
below. Tail parameters and powers can give divergent moments; no location
parameter is substituted for an expectation.

Ordinary density, CDF, survival and quantile values are checked against saved
``gamlss.dist`` 6.1.1 outputs. Separate discrepancy fixtures record:

* Nonpositive-response and log-tail quirks. The installed ``pBCPE`` source is
  captured because its boundary behavior differs from the vendored source.
  The shared laws return zero CDF and negative-infinite log density/CDF for
  nonpositive responses, with survival one.
* Upstream ``dBCT`` switches only its density to Gaussian when ``tau > 1e6``;
  its probability functions remain Student-t. ``BCTLaw`` keeps an internally
  consistent Student-t density, CDF, quantiles and samples for every supported
  finite ``tau``. This discrepancy is deliberately not copied.
* At large BCPE shape, the intermediate gamma probability or quantile can
  underflow although the final power-exponential value remains representable.
  Log-scale evaluation retains that value instead of creating a false median.

Probability calculations retain log latent magnitudes, including Student-t
values too large to represent directly. Stable special functions, an
incomplete-gamma continued fraction, and checked integration evaluate the same
law in extreme tails. Near a finite truncation boundary, interval probabilities
retain the explicit log width; boundary quantiles invert that interval.
Numerical failures raise ``NotImplementedError`` rather than substituting a
Gaussian law or silently clipping parameters. Extremely small BCPE shapes whose
standardized latent scale is outside float64 range are explicitly unsupported.
Outcome quantiles and draws can underflow to zero, overflow to infinity or
collapse to the same floating-point value; CDF round trips are not guaranteed
outside representable outcome resolution.

Use ``GAMLSS(family='bct', ...)`` or ``family='bcpe'`` with formula keys
``mu, sigma, nu, tau`` and fixed smoothing. Links are log/log/identity/log.
The joint fitting engine uses observed first and second likelihood derivatives,
including derivatives of the support normalizer. Checked adaptive quadrature
integrates analytic density derivatives; it never substitutes finite differences.
Automatic smoothing and alternate links remain unsupported. BCPE observed
derivatives at ``y=mu`` are explicitly guarded; BCT derivatives at ``nu=0``
require tail shape exceeding the derivative order.

Neural LSS accepts ``ObjectiveConfig(family='bct')`` or ``'bcpe'`` with the same
four predictor links. Its normalized objective provides first-order gradients;
the support normalizer uses vectorized checked CPU quadrature and returns
gradients to the input device. Higher autograd derivatives are unsupported.
This CPU calculation can limit accelerator throughput. Fitting does not assert
GAMLSS RS/backfitting parity or convergence for arbitrary data.

BCT/BCPE censoring/truncation fitting and untruncated-family aliases remain
unavailable. Run ``examples/example_boxcox_tail_probabilities.py`` for a standalone
probability and centile example.

Box--Cox moment existence
-------------------------

BCT/BCPE means and variances have analytic existence gates and checked
quadrature. A divergent mean is infinity and its variance is undefined (NaN);
a finite mean with a divergent second moment has infinite variance. BCT's
log-t case has no positive moments. BCPE's log-PE case has the usual
MGF boundary at tau=1. Finite integrals outside the certified numerical range
raise explicitly. The existence gates also apply to fitted predictions.

For methodological citations and paper links, see :doc:`/references`.
