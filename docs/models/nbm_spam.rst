NBM-SPAM
========

NBM-SPAM composes two representations. An NBM first learns nonlinear unary
concept scores, then a SPAM head combines those scores with low-rank polynomial
terms,

.. math::

   z_j = b_j(x_j), \qquad
   \eta(x) = p(z_1, \ldots, z_p).

This separates nonlinear feature representation from polynomial interaction
construction. It is useful when interactions between learned concept scores
are more plausible than polynomials of the raw transformed columns.

.. code-block:: python

   from nampy.models import NBMSPAMRegressor

   model = NBMSPAMRegressor(
       num_bases=100,
       ranks=[64, 32],
       output_penalty=1e-4,
       regularization_scale=1e-4,
   ).fit(X_train, y_train)

   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)

The returned components follow the architecture's concept and polynomial
blocks. They are not equivalent to an independently fitted function of each
raw feature.

Available estimators
--------------------

* :class:`nampy.models.NBMSPAMRegressor`
* :class:`nampy.models.NBMSPAMClassifier`
* :class:`nampy.models.NBMSPAMLSS`

References
----------

* Radenovic, Dubey, and Mahajan (2022), `Neural Basis Models for
  Interpretability <https://arxiv.org/abs/2205.14120>`_.
* Dubey, Radenovic, and Mahajan (2022), `Scalable Interpretability via
  Polynomials <https://arxiv.org/abs/2205.14108>`_.

