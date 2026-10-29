Fisher information diagnostics
==============================

Fisher information describes the local sensitivity of a model's conditional
probability distribution to its fitted parameters. NAMpy provides the same
inspection records for neural and statistical estimators, with numerical
likelihood calculations owned by their respective backends. Diagnostics are
opt-in and do not change training loss, optimization, stopping or checkpoint
selection.

The returned quantity is **mean information per reference observation**, in the
reported parameter coordinates. The information for ``n`` independent replicated
observations is ``n`` times this mean; sampling variance of that total is
``n**2`` times sampling variance of the mean. A term trace depends on parameter
scaling and model representation. It is not a causal feature-importance score,
and raw neural and spline coefficient diagonals need not be comparable.

Configure and collect
---------------------

Constructor controls own fit-time collection::

    from nampy import FisherDiagnostics, GradientTrainingConfig, LinRegClassifier

    diagnostics = FisherDiagnostics(
        estimator="exact", n_inputs=32, every_n_epochs=5,
        group_by="term", random_state=7,
    )
    model = LinRegClassifier(
        training=GradientTrainingConfig(max_epochs=20),
        fisher_diagnostics=diagnostics,
    ).fit(X_train, y_train, offset=train_offset)

``fisher_history_`` contains requested initial, periodic and final snapshots;
``fisher_information_`` describes the selected fitted weights, after checkpoint
selection or averaging. ``fisher_config_`` records effective controls. The
fixed fit-time probe is selected before preprocessing and reused across stages.
Evaluation uses fitted normalization and projection, with dropout disabled.

Known offsets must be supplied again for post-fit reference rows::

    information = model.fisher_information(X_reference, offset=reference_offset)

Offsets are validated and sliced in raw predictor coordinates. They change the
conditional law being evaluated but are held fixed during parameter derivatives.
If fitting used explicit offsets, omitting them from this method raises an
error. Observed labels are not accepted by the post-fit information method.

Neural preprocessing uses bounded batches and restores module modes, gradients,
registered buffer values, parameter flags and random-number-generator states.
Completed fits retain prediction state and compact diagnostics, not training
observations, probes or callbacks. Unsupported schedules, likelihoods, precision
or execution routes raise explicitly rather than silently changing estimators.

Choose the estimator
---------------------

``FisherDiagnostics(estimator=...)`` supports three distinct calculations:

.. list-table:: Information estimators
   :header-rows: 1

   * - Estimator
     - Calculation
     - Output-sampling error
   * - ``exact``
     - Integrate the declared likelihood analytically, including its parameter transforms
     - No output Monte Carlo error is calculated
   * - ``score_outer_product``
     - Sample outcomes from the current model and average squared parameter scores
     - Estimated from repeated model draws at each input
   * - ``negative_hessian``
     - Sample outcomes and average negative log-probability Hessian diagonals
     - Estimated from repeated model draws at each input

Samples are detached before differentiation. The score estimator does not use
observed training labels as substitutes for model outcomes. Finite-sample
Hessian estimates can be negative; NAMpy preserves their signs. Its eligibility
requires a declared smooth architecture and supported family transforms.
Known nonsmooth activations such as ReLU and LeakyReLU are rejected for that
estimator, while their eligible first-order score route remains available.

Exact conditional information uses small output-coordinate information matrices
and parameter derivatives, or exact enumeration of categorical outcomes. It does
not allocate a full parameter-by-parameter information matrix. The sampled
Hessian route computes actual diagonal entries without allocating that matrix,
but requires repeated second-derivative passes: its time cost can be substantial
even though memory is bounded. Start with a small model and probe.

Neural coverage
----------------

The neural adapters declare normalized likelihoods for default MSE regression
with an explicitly specified ``regression_scale``, binary/multiclass classifiers,
and Normal, Poisson, Categorical, Gamma, inverse-Gamma, beta, lognormal and
Weibull LSS families. Student-t and negative-binomial LSS currently support
sampled score/Hessian information but do not declare exact integration.
Negative-binomial clamping regimes are rejected. Robust Normal is eligible only
when its robustness modification is disabled. Custom families cannot inherit
these declarations merely by subclassing a supported family.

MSE alone does not define observation noise. Supply its known standard deviation
through ``FisherDiagnostics(regression_scale=...)``; NAMpy does not infer that
scale from training residuals or validation outcomes. Statistical Gaussian
``scale`` instead denotes response variance. Reported held-fixed metadata
separates such quantities from fitted parameter coordinates.

Statistical coverage
---------------------

Statistical GAM information uses expected likelihood derivatives and compiled
coefficient/design mappings, with mean normalization. Smoothing penalties and
coefficient covariance matrices are excluded. Supported ordinary and
multi-predictor likelihoods declare their own exact or sampled capabilities;
periodic GAM collection is unavailable because solver iterations do not share
neural epoch semantics. Statistical collection is final-only, on fitted training
rows; it has no validation split or initial snapshot. The result records the
actual reference split rather than the neural configuration's default.

Quasi-likelihood families do not define a normalized response sampling law and
are rejected. Their working curvature is not relabelled Fisher information.
Source-specific families, observation context, dependence and parameter
constraints retain explicit capability checks; support for fitting or prediction
does not by itself establish Fisher eligibility.

Sampling uncertainty
---------------------

``input_sampling="fixed"`` is the default. For sampled estimators,
``output_sampling_variance`` describes a single score-square or Hessian-diagonal
sample, averaged over reference inputs. ``estimator_variance`` and
``conditional_standard_error`` describe the error of the mean information due
only to finite output draws. ``n_draws`` is the number of draws **per input**.
For exact integration these Monte Carlo fields are ``None``, not estimates of
input-population variability.

``input_sampling="iid"`` treats reference inputs as an independent population
sample. ``input_sampling="without_replacement"`` additionally requires the
finite population size through ``population_size`` when the supplied rows are a
subset of a larger finite table. Omitting it in post-fit calls treats all supplied
rows as the finite population; fit-time probes use the actual selected split
size. Both provide ``input_estimator_variance``,
``total_estimator_variance`` and ``total_standard_error``. The input component
subtracts estimated output noise and is an unbiased component estimate that can
be negative in a finite Monte Carlo run; it is not clipped. Total variance is
nonnegative. Exact integration can still have nonzero input-sampling error.

Group traces retain covariance between coordinates by estimating each draw's
summed contribution. Variance of a group trace is different from the sum of
coordinate variances. ``record_diagonal="never"`` retains group summaries
without parameter-sized result arrays; ``"final"`` retains coordinate diagonals
only for the final snapshot and ``"all"`` for every requested snapshot.

These fields describe error in **estimating information**, conditional on the
fitted model. They do not describe model-refitting uncertainty, coefficient
confidence intervals, response prediction intervals or smoothing uncertainty.
Inverting diagonal entries does not supply a valid general covariance estimate.

Examples and references
-----------------------

``examples/example_fisher_diagnostics.py`` demonstrates exact, score and
Hessian calculations with a neural classifier and an analytic statistical GAM.
Both use public estimators and explicitly aligned offsets.

The estimator distinction and derivative/moment variance analysis follow
Soen and Sun, `On the Variance of the Fisher Information for Deep Learning
<https://arxiv.org/abs/2107.04205>`_ (NeurIPS 2021). The input/output decomposition
and diagonal-estimator comparisons follow their `Trade-Offs of Diagonal Fisher
Information Matrix Estimators <https://arxiv.org/abs/2402.05379>`_ (NeurIPS 2024).
These papers' regularity assumptions are not automatic guarantees for every
NAMpy architecture, observation process or family.

Full publication citations are [Soen2021]_ and [Soen2024]_ in
:doc:`/references`. Source-level paper links are kept with the shared Fisher
contracts and both numerical backends.
