GPNAM
=====

When to use it
--------------

Use GPNAM for smooth additive effects built from a fixed random-feature
approximation to a Gaussian process.

Model structure
---------------

GPNAM approximates an additive Gaussian process with random Fourier features.
For feature :math:`x_j`, a fixed map such as

.. math::

   \phi_j(x_j) = \sqrt{2/M}\cos(\omega_j x_j + b_j)

approximates a stationary kernel, and the additive predictor is linear in the
mapped features. NAMpy uses the inverse-normal frequency grid and independently
permuted phase grids described by the reference implementation. Only the
additive coefficients and intercept are fitted. Pairwise terms provide the
GP-NA2M extension.

Fit and inspect
---------------

.. code-block:: python


   from nampy.neural.configs.gpnam_config import DefaultGPNAMConfig
   from nampy.neural.configuration import FixedDesignTrainingConfig
   from nampy.models import GPNAMRegressor

   model = GPNAMRegressor(architecture_config=DefaultGPNAMConfig(rff_num_feat=200, kernel_width='auto', interaction_degree=2), training=FixedDesignTrainingConfig(ridge=0.05)).fit(X_train, y_train)

   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   figures = model.plot_terms(X_test, center=True)

Interpretation and limits
-------------------------

GPNAM is useful when smooth, kernel-like feature functions are desired with a
fixed nonlinear representation and a linear coefficient solve.

Available estimators
--------------------

* :class:`nampy.models.GPNAMRegressor`
* :class:`nampy.models.GPNAMClassifier`
* :class:`nampy.models.GPNAMLSS`

Reference
---------

Zhang, Barr, and Paisley (2024), `Gaussian Process Neural Additive Models
<https://arxiv.org/abs/2402.12518>`_.

