Prediction and Explanation Contracts
====================================

Backend-neutral objects provide one representation for GAM and neural
predictions. Explanation functions operate on that shared representation.

.. currentmodule:: nampy.contracts

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FeatureSchema
   AdditivePrediction
   AdditiveTermMetadata
   EnsembleAdditivePrediction

.. currentmodule:: nampy.inspection

.. autosummary::
   :toctree: generated/
   :nosignatures:

   center_additive_prediction
   explain_additive_prediction
   term_importance_table
   FisherDiagnostics
   FisherInformationResult
   FisherGroupSummary
   FisherSnapshot


Term metadata
-------------

``AdditivePrediction.term_metadata`` maps each term to its source features,
distribution parameter (when applicable), and kind. ``global`` and ``contextual``
terms must not be interpreted as unary feature curves. GAMLSS parameter
identity is stored separately from its displayed ``parameter:term`` label.
Centering preserves this metadata. Manually constructed prediction objects can omit metadata; label-based
interpretation is then only a fallback.


Conditional distribution contracts
----------------------------------

``nampy.distributions.DistributionSpec`` declares a law's parameterization, named
parameters, required rowwise context and probability operations.
``nampy.distributions.DistributionLaw`` is the structural protocol for validating
those inputs and evaluating the declared operations. Neither contract imports
a fitting backend.

``nampy.distributions.DistributionPrediction`` stores a snapshot of a law and
its row-aligned inputs. It supplies CDF, quantile, sampling and moment methods
with explicit output axes, and rejects unavailable operations. The current
estimator adapters support normal, Gamma, Student-t, beta, negative binomial,
lognormal, Weibull, statistical beta-binomial, zero-adjusted Gamma and inflated beta; see
:doc:`../user_guide/gamlss` for parameter mappings, usage and limitations.

The shared ``quantile_residuals``, ``prepare_worm_plot``, ``prepare_centiles``
and ``centile_calibration`` functions live in ``nampy.diagnostics``. They
consume probability operations or prepared residuals rather than estimators.
``nampy.plotting`` owns the corresponding renderers and plotting conveniences;
these do not alter the statistical backend's residual or QQ methods.

``AtomMixtureLaw`` composes a scalar continuous law with fixed atoms and named
rowwise atom probabilities. Available operations are derived from the base
law's capabilities. Log probability reports mass at atoms and density elsewhere;
``cdf_left`` excludes atoms at the query value. This reusable probability
composition supplies no fitting, derivative or inference capabilities by itself.
An optional named continuous weight preserves small weights from coupled
transforms instead of subtracting the atom probabilities from one. The default
parameterization and remainder-weight behavior are unchanged.

Distribution comparison
-----------------------

``DistributionSpec.atom_locations`` declares fixed atoms for mixed-law scoring.
Unknown atom metadata still permits inference, but prevents comparison under an
assumed common mixed reference measure.

``DistributionFitStatus`` records backend-owned convergence or completion
evidence. Statistical convergence, completed neural training and unknown status
are distinct. Probability support alone does not make a fit eligible to rank.

``nampy.distributions.compare_distributions`` fits named estimator templates on
explicit training data and scores identical held-out observations through
``DistributionPrediction.log_prob``. ``DistributionCandidate`` holds per-model
fit/prediction arguments and optional external status readers. Results retain
failed candidates, warnings, timing, row counts and optional unweighted centile
calibration. ``DistributionComparison.to_frame()`` returns the ranking table;
its ``models`` property exposes independently fitted clones. No AIC or EDF is
computed across backends. See :doc:`../user_guide/gamlss` for eligibility, weights,
reference measures and validation-split restrictions.

``nampy.distributions.BCCGLaw`` is a backend-neutral probability-only law for
positive responses. It supplies density, CDF/tails, quantiles, sampling and
response moments through ``DistributionPrediction``. Divergent means return
infinity; variance is NaN for an infinite mean and infinity for a finite mean
with divergent second moment. Statistical ``bccg_lss`` supports fixed smoothing
and automatic joint REML with Newton, BFGS and EFS, reusing the generic covariance
adapters. Neural fitting remains unavailable.
Its ``mu`` is not generally the response mean or exact
median; it is distinct from mgcv's existing censored ``bcg`` likelihood. See
:doc:`../user_guide/gamlss` for parameterization and numerical limits.

``nampy.distributions.BCTLaw`` and ``BCPELaw`` supply positive scalar
probabilities through the same ``DistributionPrediction`` interface. Both use
named ``mu, sigma, nu, tau`` arrays, including mixed rowwise powers and shapes.
They declare log density, CDF/left CDF, survival/log tails, quantiles and sampling.
Moments and fitting in either backend are unavailable. BCT uses Student-t scale
and degrees of freedom; BCPE uses a variance-one power-exponential latent law.
At zero power these are log-t/log-PE laws, respectively. See the user guide for
normalization, upstream discrepancies and floating-point limitations.

Predictor coordinates
----------------------

``AdditivePrediction`` contains no response predictions. Its ``link`` and every
term have shape ``(n, K)`` with K actual additive predictors, named by
``predictor_names``. Natural parameters and response probabilities may have
different widths and are queried through their estimator methods or
``DistributionPrediction``. Centering preserves predictor names and reconstruction.
Ensemble component variability describes these predictors, not response predictions.
