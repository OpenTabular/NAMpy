SIAN
====

When to use it
--------------

Use SIAN when sparse higher-order interaction discovery is part of the
modeling question.

Model structure
---------------

SIAN represents the predictor as a sparse collection of main effects and
higher-order interaction functions,

.. math::

   \eta(x) = \beta_0 + \sum_{S \in \mathcal{T}} f_S(x_S),

where :math:`\mathcal{T}` contains selected feature subsets. NAMpy can fit the
terms with independent networks or a block-masked network that retains
term-specific inputs. The optional interaction-selection workflow scores
candidates with an Archipelago-style detector before fitting the additive
terms.

Fit and inspect
---------------

.. code-block:: python

   from nampy.models import SIANRegressor

   model = SIANRegressor(
       interaction_degree=3,
       execution_mode="block_masked",
       l1_regularization=5e-5,
   ).fit(X_train, y_train)

   components = model.predict_components(X_test, center=True)
   effects = model.explain_terms(X_test)
   figures = model.plot_terms(X_test, center=True)

Interpretation and limits
-------------------------

When a residual network is enabled, its unrestricted contribution is
predictive but should not be interpreted as a named additive feature effect.

Available estimators
--------------------

* :class:`nampy.models.SIANRegressor`
* :class:`nampy.models.SIANClassifier`
* :class:`nampy.models.SIANLSS`

Reference
---------

Enouen and Liu (2023), `Sparse Interaction Additive Networks via Feature
Interaction Detection and Sparse Selection
<https://arxiv.org/abs/2209.09326>`_.

