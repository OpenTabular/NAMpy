Statistical extension scope and upstream validation
===================================================

The statistical extensions reuse NAMpy's identified bases and penalties, with
new loss functions or structured designs. Their reference pages describe the
supported inputs and outputs. The comparisons below match the mathematical
problem before comparing numbers; shared methodology alone is not numerical
parity with an entire package.

Implemented scope
-----------------

* ``ExpectileGAM``: one level, unconstrained ordinary spline bases, fixed
  smoothing and validation-based tuning through scikit-learn. LAWS includes
  convergence diagnostics and objective-decreasing step control. Its 0.5
  expectile matches the existing Gaussian penalized least-squares fit.
* ``QuantileGAM``: one quantile, either exact penalized pinball regression or
  a smooth fixed-learning-rate ELF loss. Smoothing is fixed or chosen by
  validation. This implementation does not include qgam's learning-rate
  calibration, calibrated uncertainty, automatic qgam smoothing selection,
  or jointly noncrossing quantiles. Smooth ELF fits have approximation bias
  controlled by their explicit response-scale bandwidth.
  Exact fitting uses a sparse epigraph formulation: HiGHS for unpenalized
  linear programs and OSQP for penalized convex quadratic programs. Constraint
  storage grows with observations times basis width, rather than observations
  squared. Stopping tolerances and KKT residuals are explicit; no artificial
  penalty is added to make the Hessian positive definite.
* ``student_lss``, ``beta_lss`` and ``nb_lss``: two jointly fitted distributional
  predictors with analytic mixed derivatives through order four. These extend
  the existing GAMLSS engine. They support fixed smoothing and joint Laplace
  ML/REML derivatives; R gamlss's RS backfitting conventions are not substituted
  for the joint engine. Student-t degrees of freedom are fixed, not fitted.
* Joint coefficient simulation and finite-grid simultaneous bands for linear
  functionals, curve differences, derivative regions and integrated effects.
  The full joint covariance is retained. Pointwise and smoothing-adjusted
  inference are explicit options; unavailable covariance is an error.
* Single-event right-censored survival data, including delayed entry, can be
  split into piecewise exponential Poisson rows. Exposure, offsets and event
  boundaries are explicit. Hazard integration provides cumulative hazards and
  survival curves. Recurrent events, competing risks, time-varying covariate
  reconstruction and censoring-adjusted prediction scores are outside this API.
* Distributed-lag nonlinear effects use explicit exposure histories, fitted
  exposure-basis callbacks, lag bases, tensor penalties and cumulative contrasts.
  Series boundaries and missing lags propagate correctly. This is a cross-basis
  construction API; it does not replicate all of dlnm's automatic basis choices,
  attributable-risk tools or plotting interfaces.

Design compilation and numerical boundary checks
------------------------------------------------

``compile_design`` performs no coefficient fit. A solver-independent compiled
design owns the basis and penalty, response/weight validation, term coefficient
indices, formula preprocessing and known offsets. Separate fit-time offsets
are never reused automatically on new data. Formula and smooth-owned offsets
are evaluated directly, without recovering them by subtracting predictions.
Supplying both ``X`` and ``data`` is an error. Scoring normalizes single-column
targets and rejects multi-output or mismatched targets.

Tests verify independent compilation while ``GAM.fit`` is disabled, explicit
and formula offsets, known soap boundaries, linked smoothing IDs, zero/fixed
penalties, serialization and cross-validation. A live R mgcv snapshot compares
the compiled-design fit, new-data predictions, penalty energy and EDF. A 10,000
row exact-quantile regression and sparse matrix storage checks guard against
reintroducing observation-squared allocations.

The NB2 likelihood uses stable divided logarithms and a Stirling difference
near its Poisson limit. Likelihood evaluation and reporting use the same
expression, with analytic mixed derivatives through order four. Boundary
fixtures compare counts up to 10,000 and dispersion down to 1e-16 against R
and independent 100-digit arithmetic; derivative tests also use high precision.
R gamlss.dist intentionally substitutes a Poisson approximation below dispersion
1e-4. NAMpy retains the smooth exact NB2 likelihood instead. In three extreme
tail cases (mean 10,000, dispersion 1e-12, counts 1/5/10), this R version's
``stats::dnbinom`` differs from high precision by approximately 5e-5. These
cases retain strict high-precision checks and are recorded separately from
the R comparisons; that rounding error is not reproduced in NAMpy.

Independent numerical comparisons
---------------------------------

The checked-in fixture
``tests/reference_fixtures/extensions/neighbors.json`` contains the input data,
full-precision upstream results, package versions and exact upstream git commits.
It is generated by actually executing R and the indicated upstream source, not
by saving NAMpy's answers. Tests run without installing R; regeneration needs R.

.. list-table::
   :header-rows: 1

   * - Surface
     - Independent reference
     - Matching conventions / tolerance
   * - Expectile coefficients, levels 0.2, 0.5, 0.8
     - Unmodified ``expectreg/R/asyregpen.lsfit.R``
     - Identical weighted design; upstream penalty root corresponds to S/2;
       absolute coefficient tolerance 2e-10
   * - Exact quantile coefficients
     - R ``quantreg::rq.fit``
     - Identical design, observation weights and levels; 2e-7
   * - Penalized exact quantiles
     - R ``quadprog::solve.QP``
     - Identical slack-variable problem; reference adds 1e-9 numerical ridge
       for strict positive definiteness; 2e-5
   * - Smooth quantile coefficients
     - Unmodified qgam ``elf.R`` and ``log1pexp.R`` with independent R BFGS
     - sigma=1, constant co=bandwidth; likelihood constants do not affect
       coefficients; 2e-6. This validates ELF fitting, not qgam calibration.
   * - New distributional-family densities and unpenalized joint MLEs
     - Installed R ``gamlss.dist`` densities with independent joint R BFGS
     - TF: nu=df and sigma is t scale; BE: sigma=1/sqrt(1+phi);
       NBI: sigma=alpha. Coefficients 2e-5, per-row log density 1e-5
   * - New family first through fourth mixed predictor derivatives
     - Independent PyTorch automatic differentiation in tests only
     - Same full log density, no finite-difference derivative implementation;
       2e-8 absolute and 1e-8 relative
   * - Joint penalized EDF/covariance and outer REML derivatives
     - Independently differentiated joint information / finite differences
     - Cross-parameter blocks retained; EDF 1e-6, covariance 2e-6;
       REML gradient 2e-4 and Hessian 2e-3
   * - Simultaneous interval critical value and limits
     - Unmodified gratia ``simultaneous-intervals.R``
     - Identical coefficient deviations, full covariance and grid; R's type-8
       quantile; 1e-13. RNG implementations are not compared by seed alone.
   * - Nonlinear cross-basis and grouped series histories
     - Installed R ``dlnm::crossbasis``
     - Exposure-major column order, identical polynomial and lag-linear bases,
       missing leading histories and group boundaries; 1e-13
   * - Survival exposure/event splitting
     - R ``survival::survSplit``, the kernel pammtools uses
     - Identical cut points, delayed entry and events at cuts. This is a kernel
       comparison; the full pammtools package is not installed or claimed equivalent.

The corresponding projects are `expectreg <https://github.com/cran/expectreg>`_,
`qgam <https://github.com/mfasiolo/qgam>`_,
`gamlss.dist <https://github.com/cran/gamlss.dist>`_,
`gratia <https://github.com/gavinsimpson/gratia>`_,
`pammtools <https://github.com/adibender/pammtools>`_ and
`dlnm <https://github.com/cran/dlnm>`_. Only reference sources are downloaded
for comparisons; NAMpy's runtime does not depend on these R packages.

EDF audit
---------

The general-family engine already calculates joint coefficient EDF, testing EDF
and smoothing-uncertainty-adjusted EDF. The new ``joint_edf`` utility specifies
conditional influence EDF in identifiable unconstrained coordinates:
``F = solve(information + penalty, information)``, total EDF ``trace(F)``,
and testing EDF ``diag(2*F - F@F)``. It is not a replacement for the existing
``edf2`` adjustment or for constrained-model calculations.

Tests check unpenalized full rank, ridge calculations, large ordinary and
shrinkage penalties, invariance of total EDF under nonsingular basis changes,
nonzero cross-parameter information and consistency of term EDF plus intercepts
with model EDF. The existing normal/Gamma GAMLSS semantics and general-family
mgcv parity suites also pass. These checks did not establish a general EDF bug;
no speculative change to existing EDF/AIC conventions was made.

Reproducing the comparisons
---------------------------

Clone the projects listed above into an external directory, using the exact
commit IDs in the fixture when reproducing this snapshot. Install ``mgcv``,
``jsonlite``, ``quantreg``, ``quadprog``, ``dlnm``, ``survival`` and ``gamlss.dist``
in R. expectreg, qgam and gratia are sourced only for the numerical kernels,
avoiding unrelated plotting and training dependencies.
The Python fixture generator additionally needs ``mpmath`` from the development
extra for its independent 100-digit boundary references.

.. code-block:: bash

   python tests/parity/generate_neighbor_references.py \
     --upstream-dir /path/to/upstream-clones \
     --r-library /path/to/R-library
   python -m pytest tests/extensions -q
   python -m pytest tests/public_api/test_gamlss_estimator.py \
     tests/public_api/test_remaining_gamlss_families.py \
     tests/parity/test_gamlss_public_semantic_parity.py \
     tests/families/test_general_family_mgcv_parity.py -q

Workflow and likelihood additions
---------------------------------

Subject-trajectory resampling retains every record of each sampled independent
subject and assigns fresh occurrence IDs. Independent refits validate the
statistical bootstrap; the same shared prediction/effect contract is exercised
with native and gradient neural models. Failure draws are recorded rather than
replaced. Temporal statistical PEM effects match new mgcv ti() captures.

Cox third/fourth likelihood tensors match independent automatic differentiation;
outer REML derivatives match numerical checks used only in tests. New mgcv
captures validate smoothing endpoints, baseline q/a standard errors and raw
Schoenfeld/score processes. Separate direct-baseline captures isolate two source
1.9-4 stratum/offset reporting quirks. Stratified automatic Newton remains
unsupported; stratified BFGS passes the fitted endpoint gate. Pointwise Wald
intervals include baseline and coefficient variability conditional on smoothing;
bootstrap inference re-estimates both.

Adaptive qgam checks use installed qgam 2.0.1 and mgcv 1.9-4: initial Gaussian
variance/penalty EDF, residual SHASH bandwidth, scale-profile REML criteria,
posterior/sandwich prediction variances and final calibrated functions. A flat
oversmoothing limit does not identify a unique smoothing coordinate. A nonlinear
capture checks an identified finite smoothing parameter. The ELF adapter uses
the existing extended-family PIRLS engine with exact outer derivatives; other
families retain their declared source numerical staging. Independent selected-
smoothing median coverage simulations exercise complete fits, without dropping
failed draws or claiming universal coverage.

BCT/BCPE finite moments match independent integrals of GAMLSS densities. Analytic
existence gates distinguish divergent moments from unsupported finite numerical
ranges. Neural BCCG/ZAGA/BEINF/beta-binomial use normalized differentiable losses
and the same shared inference laws. They do not implement GAMLSS RS/backfitting.

Normal observation composition checks exact/left/right/interval likelihoods,
fixed truncation and far-tail probabilities. Analytic coefficient derivatives
through order four match Torch, outer REML derivatives match test-only numerical
checks, and an independent R normalized-likelihood optimizer validates fitted
endpoints. This is separate from mgcv cnorm. Persisted distributed-lag marginal
bases replay the established dlnm column-order reference; cumulative contrasts
reuse full statistical coefficient covariance.
