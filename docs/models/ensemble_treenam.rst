EnsembleTreeNAM
===============

EnsembleTreeNAM trains several complete TreeNAM learners jointly and averages
their outputs,

.. math::

   \eta(x) = \frac{1}{M}\sum_{m=1}^{M} \eta_m(x).

Term contributions are averaged with the same rule, so the additive
reconstruction remains available. This is a jointly optimized ensemble; it is
not bagging and does not fit learners sequentially as boosting would.

.. code-block:: python

   from nampy.models import EnsembleTreeNAMRegressor

   model = EnsembleTreeNAMRegressor(
       num_estimators=5,
       tree_depth=4,
       interaction_degree=2,
   ).fit(X_train, y_train)

   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)

Use the ensemble when averaging several differentiable trees per term is worth
the added training and memory cost.

Available estimators
--------------------

* :class:`nampy.models.EnsembleTreeNAMRegressor`
* :class:`nampy.models.EnsembleTreeNAMClassifier`
* :class:`nampy.models.EnsembleTreeNAMLSS`

