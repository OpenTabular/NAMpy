TreeNAM
=======

When to use it
--------------

Use TreeNAM when each named term should be represented by a compact,
differentiable decision tree.

Model structure
---------------

TreeNAM assigns one differentiable decision tree to each main effect and,
optionally, to each explicit interaction. Soft routing makes the leaf values
and split parameters trainable by gradient descent. Each tree supplies one
term in the :ref:`shared decomposition <additive-predictor-interpretation>`.

The tree depth controls the number of routing decisions and therefore the
piecewise complexity of each term. Evaluation can retain soft routing or use
hard routing when requested.

Fit and inspect
---------------

.. code-block:: python

   from nampy.models import TreeNAMRegressor

   model = TreeNAMRegressor(
       tree_depth=4,
       tree_temperature=1.0,
       interaction_degree=2,
   ).fit(X_train, y_train)

   components = model.predict_components(X_test, center=True)
   figures = model.plot_terms(X_test)

Interpretation and limits
-------------------------

Report whether evaluation uses soft or hard routing, since switching to hard
routing changes predictions.

Available estimators
--------------------

* :class:`nampy.models.TreeNAMRegressor`
* :class:`nampy.models.TreeNAMClassifier`
* :class:`nampy.models.TreeNAMLSS`

References
----------

This NAMpy implementation adapts additive and differentiable-tree components;
the papers below do not introduce the exact ``TreeNAM`` classes.

* Agarwal et al. (2021), `Neural Additive Models: Interpretable Machine
  Learning with Neural Nets <https://arxiv.org/abs/2004.13912>`_.
* Chang, Caruana, and Goldenberg (2022), `NODE-GAM: Neural Generalized
  Additive Model for Interpretable Deep Learning
  <https://arxiv.org/abs/2106.01613>`_.
