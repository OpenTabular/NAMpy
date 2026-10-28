Response laws, links and likelihoods
====================================

A basis describes how an input effect varies; a family describes how the
response varies conditional on those effects. The two choices answer different
questions. A very flexible mean function cannot turn a Poisson variance model
into a negative-binomial variance model. Each :doc:`family page <index>` gives
its own law, parameterization, target encoding and output contract.

Mean models and exponential families
------------------------------------

A one-parameter exponential-dispersion family has density or mass function

.. math::

   p(y\mid\vartheta,\phi)=\exp\left\{
   \frac{y\vartheta-b(\vartheta)}{a(\phi)}+c(y,\phi)\right\},
   \qquad E[Y]=b'(\vartheta),\quad
   \operatorname{Var}(Y)=a(\phi)b''(\vartheta).

The canonical parameter vartheta is not generally the response mean mu.
A link connects mu to the additive predictor, g(mu)=eta. Gaussian identity,
Poisson log and binomial logit are familiar examples. A canonical link sets
eta equal to the canonical parameter; other supported links are possible.
Parameter domains still have to hold for every prediction.

On an identity mean link, a term difference is a difference in conditional
means. On a log mean link, exp(delta) is a conditional mean ratio. On a logit
link, it is an odds ratio. A centered curve is a contribution to eta; applying
the inverse link to that curve alone does not generally yield its contribution
to the response mean. Reconstruct the whole eta first.

Penalized likelihood and working least squares
----------------------------------------------

At fixed smoothing parameters, a likelihood GAM minimizes

.. math::

   -\sum_i w_i\log p(y_i\mid\mu_i,\phi)
   +\tfrac12\theta^\mathsf TS_\lambda\theta.

For ordinary exponential-family mean models, penalized iteratively reweighted
least squares (PIRLS) builds a local quadratic approximation. Ignoring rank
and numerical stabilization details, one step uses

.. math::

   z_i=\eta_i+(y_i-\mu_i)g'(\mu_i),\qquad
   W_i=\frac{w_i}{\phi V(\mu_i)\{g'(\mu_i)\}^2},
   \qquad (X^\mathsf TWX+S_\lambda)\theta_{\rm new}
   =X^\mathsf TW(z-o).

Phi can be absorbed into penalty coordinates in a solver's implementation.
The displayed equations explain working responses and curvature; they are not
the universal solver for every family. Extended and general families can
require direct derivatives, nuisance updates and joint Hessians.

Weights require the family's interpretation. They may represent trial totals,
likelihood replication or prior precision under a documented convention.
Multiplying all weights can change the balance between data fit and a fixed
penalty. An offset is a known predictor contribution, not a response weight.
For counts, a log-exposure offset makes expected counts proportional to exposure.

Several distribution parameters
-------------------------------

Distributional regression supplies one additive predictor for each modeled
parameter, for example location and scale:

.. math::

   \eta^{(q)}=X^{(q)}\theta^{(q)}+o^{(q)},\qquad
   \psi_q=h_q(\eta^{(q)}),\qquad
   \ell=\sum_i\log p(y_i\mid\psi_{i1},\ldots,\psi_{iQ}).

Although designs can be parameter-specific, the likelihood usually couples
their coefficients: changing a scale alters the information about a location.
The joint Hessian therefore includes cross-parameter blocks. Fitting several
independent mean GAMs is not equivalent to fitting a joint location-scale law.

Keep three arrays distinct: raw/link predictors eta, natural parameters psi,
and low-level family response outputs. They can differ in width and meaning.
Multinomial models have K reference logits but K+1 probabilities. A source
Gaussian location-scale family can report inverse scale where the public
adapter reports standard deviation. The page-specific column contract governs
interpretation; a column named scale in another library is insufficient.

Support, mixtures and incomplete observations
---------------------------------------------

Response support is part of the model: a Gamma law has no point mass at zero,
a beta law excludes exact endpoints, and a count law is discrete. A Tweedie
law with power between one and two combines a zero mass and positive density.
Zero-inflated and hurdle laws assign zeros through different mechanisms.

Censoring contributes an interval or survival probability. Truncation instead
conditions the entire law on sample inclusion. Neither is reproduced by
replacing a missing response with its detection threshold. Heavy-tailed laws
can have valid densities but nonexistent means or variances; moment-based
outputs require their stronger existence conditions.

Losses and quasi-likelihoods
----------------------------

A quasi-family specifies variance and estimating equations without a
normalized density. Quantile and expectile losses target distributional
functionals without requiring a complete response law. Their scores remain
useful, but are not interchangeable with normalized log likelihood or
likelihood-based AIC. The neural robust-normal objective also changes the
training loss when robustification is enabled.

The neural backend combines an architecture with a response objective and
optimizes network parameters, rather than applying the GAM PIRLS/REML solver.
Shared distribution names do not imply identical transforms or estimation.

Further reading
---------------

* R's `generalized linear model family definitions
  <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/family.html>`_.
* Wood, Pya and Säfken (2016), `Smoothing Parameter and Model Selection for
  General Smooth Models <https://doi.org/10.1080/01621459.2016.1180986>`_.
* Rigby and Stasinopoulos (2005), `Generalized additive models for location,
  scale and shape <https://doi.org/10.1111/j.1467-9876.2005.00510.x>`_.
  This motivates distributional modeling; NAMpy's joint fitting engine and
  supported parameterizations are defined in its own reference contracts.
