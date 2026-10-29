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


   from dataclasses import replace
   from nampy.neural.configs.snam_config import DefaultSNAMConfig
   from nampy.neural.configuration import GradientTrainingConfig
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split

   rng = np.random.default_rng(7)
   X = pd.DataFrame({"x": np.linspace(-1, 1, 160), "z": rng.normal(size=160)})
   y = np.sin(3 * X["x"]) + 0.2 * X["z"] + rng.normal(0, 0.1, len(X))
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

   from nampy.models import SNAMRegressor

   model = SNAMRegressor(architecture_config=DefaultSNAMConfig(layer_sizes=[16], group_lasso_lambda=0.001, group_lasso_include_interactions=True, interaction_degree=2)).set_params(training=replace(SNAMRegressor(architecture_config=DefaultSNAMConfig(layer_sizes=[16], group_lasso_lambda=0.001, group_lasso_include_interactions=True, interaction_degree=2)).training or GradientTrainingConfig(), max_epochs=20, random_state=7)).fit(X_train, y_train)

   components = model.predict_components(X_test, center=True)
   group_norms = model.group_norms()
   selected = [name for name, norm in group_norms.items() if norm > 1e-4]
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

