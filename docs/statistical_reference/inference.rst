Inference conventions
=====================

Inference concerns the penalized coefficient estimate and the fitted family.
For multi-parameter models, coefficients are stacked in predictor order;
covariance includes cross-predictor blocks. Natural-parameter uncertainty
requires the joint transformation Jacobian, rather than transforming each
standard error independently.

Covariance and intervals
------------------------

The Gaussian identified-design case makes the covariance distinction
concrete. Let K=X-transpose W X + S and let phi be response dispersion.
Conditioning on smoothing parameters, a Gaussian penalty/prior approximation
and the sampling covariance of the penalized estimate give, respectively,

.. math::

   V_B=\phi K^{-1},\qquad
   V_F=\phi K^{-1}X^\mathsf TWXK^{-1}.

The former includes uncertainty in penalized directions under the local
Gaussian-prior interpretation. The latter propagates sampling variation
through a biased/shrunk estimator; it does not remove shrinkage bias.
These are illustrative Gaussian formulas in consistent scale coordinates,
not a promise that all families use one common Hessian. General-family and
shape-constrained routes use their own curvature and coefficient maps.

For a linear functional a-transpose theta, variance is a-transpose V a.
For a nonlinear natural parameter h(eta), the delta method uses the Jacobian:

.. math::

   \operatorname{Cov}\{h(\widehat\eta)\}\approx
   J_h B_{\rm new}V_\theta B_{\rm new}^\mathsf TJ_h^\mathsf T.

Cross-parameter covariance matters when h combines several predictors.
The law of total variance motivates smoothing-uncertainty corrections by
including variation of coefficients as estimated log smoothing parameters
change. The supported correction is a local approximation, not exact
integration over every smoothing value.

``vcov(freq=False, unconditional=False)`` returns Bayesian penalized covariance
conditional on the selected smoothing parameters. ``freq=True`` requests the
frequentist sampling covariance of the penalized estimate, where supported.
These matrices answer different questions even when based on the same fit.
``unconditional=True`` requests a smoothing-adjusted Bayesian matrix. It is not
an observation-noise adjustment, and does not change the fitted coefficients.

``vcov(unconditional=True)`` falls back to conditional Bayesian covariance when
the corrected matrix is absent. Prediction emits ``UserWarning`` ("Smoothness uncertainty corrected covariance
not available") and falls back to conditional Bayesian covariance when the
correction is absent; do not infer that a requested adjustment was available
merely because fitting succeeded. ``sp_vcov()`` returns covariance of free log
smoothing parameters, or ``None`` when fitted curvature is unavailable. Family,
criterion and derivative support determine availability; see :doc:`/api/gam_advanced`.

Prediction SE is obtained from the coefficient covariance and the prediction
Jacobian. A prediction plus/minus 1.96 SE is an approximate pointwise interval;
it is not a simultaneous band over an entire curve. Use
:doc:`extensions/linear_functional_interval` for simulation-based joint coverage.
Conditional response SD (for example Gaussian sigma) describes variability of
new observations and is distinct from either coefficient or fitted-mean SE.

Effective degrees of freedom
----------------------------

Let F denote the fitted influence map in coefficient coordinates. Per-coefficient
``edf`` is its diagonal; summing within a smooth yields term EDF, and summing all
coefficients, including intercept and parametric terms, yields total EDF. EDF
can be fractional and differs from the number of basis coefficients.

``edf1()`` returns ``2*diag(F)-diag(F@F)`` in coefficient order. It is the upper
EDF convention used in supported approximate significance calculations, rather
than the complexity count in every information criterion. ``edf2`` is the
available smoothing-adjusted complexity vector computed during postprocessing;
it can be absent, and its sum is bounded by the supported upper-EDF correction.

.. list-table::
   :header-rows: 1

   * - Report
     - Complexity convention
   * - Term / total EDF
     - Sum of the corresponding ordinary per-coefficient edf values.
   * - ``loglik()``
     - Unpenalized likelihood at the penalized estimate. Reconstructed from the family AIC bookkeeping using ordinary EDF, not by inserting edf1 into the likelihood.
   * - ``aic()``
     - Effective parameter count uses sum(edf2) when available, otherwise total EDF. Add estimated scale for non-general families, cap coefficient EDF at coefficient count, then add extended-family estimated nuisance parameters. AIC is -2 loglik + 2 times this count, with family-specific stored-AIC bookkeeping.
   * - ``bic()``
     - Same effective parameter count as AIC, multiplied by log(n_obs). n_obs excludes rows with zero prior weight.
   * - Approximate smooth tests
     - Family and rank-dependent reference distribution; see :doc:`methods/summary` and :doc:`methods/anova`. EDF is not automatically an integer chi-square degree of freedom.

General multi-predictor families model scale through distribution parameters;
they do not receive the extra ordinary-family estimated-scale count. Quasi
families have no normalized likelihood: a numerical EDF does not make AIC valid.
Compare information criteria only for the same response observations, weights
and compatible likelihood definitions.

For every argument and a complete example, see :doc:`uncertainty`,
:doc:`methods/vcov`, :doc:`methods/sp_vcov`, :doc:`methods/edf1`,
:doc:`methods/aic` and :doc:`methods/bic`. The definitions above follow
``nampy.gam.inference.loglik`` and the fitted covariance postprocessing code.
