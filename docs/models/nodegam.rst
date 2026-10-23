NodeGAM
=======

When to use it
--------------

Use NodeGAM when differentiable oblivious trees and sparse selectors are the
desired representation of main effects and optional pairs.

Model structure
---------------

NodeGAM builds an additive predictor from ensembles of differentiable oblivious
decision trees. Sparse feature selectors assign a tree to a feature, and
differentiable binning routes observations to leaves. With pairwise structure
enabled, a tree may depend on two selected features. The resulting main effects
and interactions follow the :doc:`shared additive decomposition
<../concepts/additive_predictors>`.

Temperature annealing sharpens the selectors during training. NAMpy also
supports the architecture's masked-pretraining path.

Fit and inspect
---------------

.. code-block:: python

   from nampy.models import NodeGAMRegressor

   model = NodeGAMRegressor(
       num_trees=200,
       num_layers=2,
       depth=3,
       interaction_degree=2,
   ).fit(X_train, y_train)

   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   figures = model.plot_terms(X_test, center=True)

Interpretation and limits
-------------------------

The selector settings and interaction regularization determine how sharply
tree contributions can be attributed to individual features or pairs.

Available estimators
--------------------

* :class:`nampy.models.NodeGAMRegressor`
* :class:`nampy.models.NodeGAMClassifier`
* :class:`nampy.models.NodeGAMLSS`

Reference
---------

Chang, Caruana, and Goldenberg (2022), `NODE-GAM: Neural Generalized Additive
Model for Interpretable Deep Learning
<https://arxiv.org/abs/2106.01613>`_.

