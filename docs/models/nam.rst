NAM
===

When to use it
--------------

Start with NAM when each reported main effect should be learned by its own
neural network and depend only on its named feature.

Model structure
---------------

NAM learns each feature's contribution with a separate network,

.. math::

   f_j(x_j)=h_j(x_j;w_j).

These contributions follow the :ref:`shared decomposition <additive-predictor-interpretation>`.

NAMpy can add explicitly requested interaction subnetworks
:math:`f_S(x_S)`. A reported main effect depends only on its named source
feature; an interaction depends only on the variables in its term label. The
feature networks can use ordinary multilayer perceptrons or the configured
feature-layer variants, with output and parameter regularization available
separately.

Fit and inspect
---------------

.. code-block:: python

   from nampy.models import NAMRegressor

   model = NAMRegressor(
       layer_sizes=[128, 64, 32],
       interaction_degree=2,
       dropout=0.1,
   ).fit(X_train, y_train)

   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   figures = model.plot_terms(X_test)

Interpretation and limits
-------------------------

Use NAM when feature-wise nonlinear functions are desired without imposing a
spline basis. Explicit interactions should be chosen deliberately because
their contributions have a different interpretation from main effects.

Available estimators
--------------------

* :class:`nampy.models.NAMRegressor`
* :class:`nampy.models.NAMClassifier`
* :class:`nampy.models.NAMLSS`

Reference
---------

Agarwal et al. (2021), `Neural Additive Models: Interpretable Machine
Learning with Neural Nets <https://arxiv.org/abs/2004.13912>`_.

