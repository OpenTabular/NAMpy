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
and interactions follow the :ref:`shared additive decomposition <additive-predictor-interpretation>`.

Temperature annealing sharpens the selectors during training. NAMpy also
supports the architecture's masked-pretraining path.

Fit and inspect
---------------

.. code-block:: python


   from nampy.neural.configs.nodegam_config import DefaultNodeGAMConfig
   from nampy.models import NodeGAMRegressor

   model = NodeGAMRegressor(architecture_config=DefaultNodeGAMConfig(num_trees=200, num_layers=2, depth=3, interaction_degree=2)).fit(X_train, y_train)

   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   figures = model.plot_terms(X_test, center=True)

.. _nodegam-masked-pretraining:

Optional masked reconstruction
-------------------------------

NodeGAM can first reconstruct masked transformed inputs, then transfer compatible
representation weights into supervised fitting. Configure both stages through
``GradientTrainingConfig``; preprocessing is fitted on training rows and reused
for validation. This is an initialization strategy, not additional labeled data
or a guarantee of better held-out predictions.

.. code-block:: python

   from nampy import GradientTrainingConfig

   pretrained = NodeGAMRegressor(
       architecture_config=DefaultNodeGAMConfig(
           num_trees=8, num_layers=1, depth=2,
       ),
       training=GradientTrainingConfig(
           pretrain_epochs=1, pretrain_max_steps=5,
           pretraining_ratio=0.15, pretraining_noise=0.1,
           max_epochs=10, batch_size=32, random_state=7,
       ),
   ).fit(X_train, y_train, X_val=X_validation, y_val=y_validation)
   transferred_keys = pretrained.pretrained_keys_
   prediction = pretrained.predict(X_test)

``pretrain_max_steps`` caps reconstruction updates independently of supervised
``max_epochs``. Pretraining uses main-effect structure; only compatible state is
transferred, recorded in ``pretrained_keys_``. The reconstruction head is not a
supervised prediction head. Other architectures must explicitly declare masked
pretraining support; unsupported requests raise rather than silently skipping it.
See :doc:`/user_guide/training` for checkpoint and validation controls.

Interpretation and limits
-------------------------

The selector settings and interaction regularization determine how sharply
tree contributions can be attributed to individual features or pairs.

``NodeGAMLSS`` uses the tree-based architecture to model distribution
parameters. For the distributional extension, see De, Thielmann, and Säfken
(2024), cited below.

Available estimators
--------------------

* :class:`nampy.models.NodeGAMRegressor`
* :class:`nampy.models.NodeGAMClassifier`
* :class:`nampy.models.NodeGAMLSS`

References
----------

Chang, Caruana, and Goldenberg (2022), `NODE-GAM: Neural Generalized Additive
Model for Interpretable Deep Learning
<https://arxiv.org/abs/2106.01613>`_.


De, Thielmann, and Säfken (2024), `NODE-GAMLSS: Interpretable Uncertainty
Modelling via Deep Distributional Regression
<https://openreview.net/forum?id=sWvTLlRkzy>`_. NeurIPS 2024 Workshop on Bayesian
Decision-Making and Uncertainty.
