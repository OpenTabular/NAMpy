QNAM
====

When to use it
--------------

Use QNAM when several conditional quantiles should change additively without
crossing one another.

Model structure
---------------

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

Fit and inspect
---------------

.. code-block:: python


   from dataclasses import replace
   from nampy.neural.configs.qnam_config import DefaultQNAMConfig
   from nampy.neural.configuration import GradientTrainingConfig
   from nampy.neural.configuration import ObjectiveConfig
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split

   rng = np.random.default_rng(7)
   X = pd.DataFrame({"x": np.linspace(-1, 1, 160), "z": rng.normal(size=160)})
   y = np.sin(3 * X["x"]) + 0.2 * X["z"] + rng.normal(0, 0.1, len(X))
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

   import matplotlib.pyplot as plt
   from nampy.models import QNAMLSS

   model = QNAMLSS(architecture_config=DefaultQNAMConfig(layer_sizes=[16], monotone_transform='softplus'), objective=ObjectiveConfig(distributional_kwargs={'quantiles': [0.1, 0.5, 0.9]})).set_params(training=replace(QNAMLSS(architecture_config=DefaultQNAMConfig(layer_sizes=[16], monotone_transform='softplus'), objective=ObjectiveConfig(distributional_kwargs={'quantiles': [0.1, 0.5, 0.9]})).training or GradientTrainingConfig(), max_epochs=20, random_state=7)).fit(X_train, y_train)

   quantiles = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   order = np.argsort(X_test["x"].to_numpy())
   fig, ax = plt.subplots()
   for column, level in enumerate([0.1, 0.5, 0.9]):
       ax.plot(X_test["x"].to_numpy()[order], components.terms["x"][order, column],
               label=f"Quantile {level}")
   ax.set(xlabel="x", ylabel="Centered contribution")
   ax.legend()

Interpretation and limits
-------------------------

QNAM is distributional-only. Each component has one output per requested
quantile, so effect plots should retain the quantile labels. The example
plots each column explicitly because ``plot_terms`` skips multi-column terms.
Training uses pinball loss, summed over requested levels and averaged over
rows. ``evaluate`` reports this quantity as ``pinball_loss``. A finite set of conditional quantiles does not
specify a complete probability density, and non-crossing does not establish
calibration.

Available estimator
-------------------

* :class:`nampy.models.QNAMLSS`

References
----------

* Agarwal et al. (2021), `Neural Additive Models
  <https://arxiv.org/abs/2004.13912>`_.
* Koenker and Bassett (1978), `Regression Quantiles
  <https://doi.org/10.2307/1913643>`_.

