Additive Predictors and Identifiability
=======================================

For a scalar response, NAMpy represents a link-scale prediction as named
contributions:

.. math::

   \eta(x) \equiv g\!\left(\mathbb{E}[Y \mid x]\right)
   =\beta_0+\sum_j f_j(x_j)+\sum_{S\in\mathcal I}f_S(x_S)+o(x).

Here :math:`g` is the link, :math:`\beta_0` the intercept, :math:`f_j` a
main effect, :math:`f_S` an interaction over the named feature subset
:math:`S`, and :math:`o` an optional known offset. Omit the corresponding
terms when there are no interactions or offset.

The conditional mean is :math:`g^{-1}\{\eta(x)\}`. Contributions add on the
link scale; applying a nonlinear inverse link to each term separately and
summing the results does not reconstruct the response prediction.

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
