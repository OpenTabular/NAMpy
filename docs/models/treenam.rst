TreeNAM
=======

TreeNAM assigns one differentiable decision tree to each main effect and,
optionally, to each explicit interaction. Soft routing makes the leaf values
and split parameters trainable by gradient descent,

.. math::

   \eta(x) = \beta_0 + \sum_j T_j(x_j)
   + \sum_{S \in \mathcal{I}} T_S(x_S).

The tree depth controls the number of routing decisions and therefore the
piecewise complexity of each term. Evaluation can retain soft routing or use
hard routing when requested.

.. code-block:: python

   from nampy.models import TreeNAMRegressor

   model = TreeNAMRegressor(
       tree_depth=4,
       tree_temperature=1.0,
       interaction_degree=2,
   ).fit(X_train, y_train)

   components = model.predict_components(X_test, center=True)
   figures = model.plot_terms(X_test)

Available estimators
--------------------

* :class:`nampy.models.TreeNAMRegressor`
* :class:`nampy.models.TreeNAMClassifier`
* :class:`nampy.models.TreeNAMLSS`

References
----------

TreeNAM combines the additive decomposition of Agarwal et al. (2021) with
differentiable oblivious-tree components related to `NODE-GAM
<https://arxiv.org/abs/2106.01613>`_.

