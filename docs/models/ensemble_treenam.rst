EnsembleTreeNAM
===============

When to use it
--------------

Use EnsembleTreeNAM when averaging several differentiable trees per term is
worth the additional training and memory cost.

Model structure
---------------

EnsembleTreeNAM trains several complete TreeNAM learners jointly and averages
their outputs,

.. math::

   \eta(x) = \frac{1}{M}\sum_{m=1}^{M} \eta_m(x).

Term contributions are averaged with the same rule, so the additive
reconstruction remains available. This is a jointly optimized ensemble; it is
not bagging and does not fit learners sequentially as boosting would.

Fit and inspect
---------------

.. code-block:: python

   from nampy.models import EnsembleTreeNAMRegressor

   model = EnsembleTreeNAMRegressor(
       num_estimators=5,
       tree_depth=4,
       interaction_degree=2,
   ).fit(X_train, y_train)

   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   figures = model.plot_terms(X_test, center=True)

Interpretation and limits
-------------------------

Term contributions are averages over jointly fitted learners. They do not
represent bootstrap variation or a confidence interval.

Available estimators
--------------------

* :class:`nampy.models.EnsembleTreeNAMRegressor`
* :class:`nampy.models.EnsembleTreeNAMClassifier`
* :class:`nampy.models.EnsembleTreeNAMLSS`

References
----------

This NAMpy implementation jointly trains additive tree components. The cited
methods provide related additive and tree methodology, rather than introducing
the exact ``EnsembleTreeNAM`` estimator.

* Agarwal et al. (2021), `Neural Additive Models: Interpretable Machine
  Learning with Neural Nets <https://arxiv.org/abs/2004.13912>`_.
* Chang, Caruana, and Goldenberg (2022), `NODE-GAM: Neural Generalized
  Additive Model for Interpretable Deep Learning
  <https://arxiv.org/abs/2106.01613>`_.
