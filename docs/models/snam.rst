SNAM
====

When to use it
--------------

Use SNAM when the analysis needs feature-wise neural effects together with
group-level shrinkage of whole terms.

Model structure
---------------

SNAM adds group sparsity to the NAM architecture. If :math:`\theta_j` contains
all parameters of feature network :math:`j`, training adds

.. math::

   \lambda \sum_j \lVert \theta_j \rVert_2

to the predictive loss. Entire feature or interaction subnetworks can
therefore shrink toward zero, providing model-level feature selection rather
than sparsity within individual weight matrices.

Fit and inspect
---------------

.. code-block:: python

   from nampy.models import SNAMRegressor

   model = SNAMRegressor(
       group_lasso_lambda=1e-3,
       group_lasso_include_interactions=True,
       interaction_degree=2,
   ).fit(X_train, y_train)

   components = model.predict_components(X_test, center=True)
   group_norms = model.model.get_group_norms()
   selected = model.model.selected_groups(threshold=1e-4)
   figures = model.plot_terms(X_test, center=True)

Interpretation and limits
-------------------------

The group-lasso coefficient controls selection strength and should be chosen
with held-out data. A small fitted group norm is evidence of shrinkage, not a
formal hypothesis test.

Available estimators
--------------------

* :class:`nampy.models.SNAMRegressor`
* :class:`nampy.models.SNAMClassifier`
* :class:`nampy.models.SNAMLSS`

Reference
---------

Xu et al. (2022), `Sparse Neural Additive Model: Interpretable Deep Learning
with Feature Selection via Group Sparsity
<https://arxiv.org/abs/2202.12482>`_.

