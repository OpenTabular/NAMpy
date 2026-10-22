Additive Predictors and Identifiability
=======================================

NAMpy represents a link-scale prediction as named contributions,

.. math::

   \eta(x)=\beta_0+\sum_j f_j(x_j)+\sum_{S\in\mathcal I}f_S(x_S)+o(x).

The :class:`nampy.contracts.AdditivePrediction` object stores ``link``,
``response``, ``terms``, ``intercept``, and ``offset``.
``validate_additive_reconstruction()`` checks the accounting identity above.
It does not establish that each term is uniquely determined or reliably
estimated.

Centering removes constant-shift ambiguity relative to a stated reference
dataset. Main effects and interactions can still exchange structure when they
overlap, and correlated predictors can make recovered curves unstable. Report
these questions separately:

* Do the returned components reconstruct the prediction?
* Which data distribution defines centering?
* Which inputs may each term depend on?
* Are the estimated effects stable across samples or reasonable model choices?

The :doc:`../user_guide/interpretability` guide shows the corresponding API.
