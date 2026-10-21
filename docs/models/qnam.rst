QNAM
====

QNAM predicts several conditional quantiles with additive feature and
interaction networks. For ordered levels :math:`\tau_1 < \cdots < \tau_K`,
each term produces a free first value and positive increments,

.. math::

   q_{j,1}(x_j)=a_{j,1}(x_j), \qquad
   q_{j,k}(x_j)=q_{j,1}(x_j)
   + \sum_{r=2}^{k}\operatorname{softplus}\{a_{j,r}(x_j)\}.

The same transform is applied to the intercept and explicit interactions.
Their sum is therefore nondecreasing across quantile levels, preventing
crossing by construction.

.. code-block:: python

   from nampy.models import QNAMLSS

   model = QNAMLSS(
       quantiles=[0.1, 0.5, 0.9],
       interaction_degree=2,
       monotone_transform="softplus",
   ).fit(X_train, y_train)

   quantiles = model.predict(X_test)
   components = model.predict_components(X_test, center=True)

QNAM is distributional-only. Each component has one output per requested
quantile, so effect plots should retain the quantile labels.

Available estimator
-------------------

* :class:`nampy.models.QNAMLSS`

References
----------

* Agarwal et al. (2021), `Neural Additive Models
  <https://arxiv.org/abs/2004.13912>`_.
* Koenker and Bassett (1978), `Regression Quantiles
  <https://doi.org/10.2307/1913643>`_.

