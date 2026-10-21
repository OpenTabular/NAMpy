SplineNAM
=========

When to use it
--------------

Use SplineNAM when differentiable training should retain explicit
one-dimensional cubic-spline feature functions.

Model structure
---------------

SplineNAM replaces each feature network with a learnable cubic-spline layer.
For basis functions :math:`B_r`, a main effect is

.. math::

   f_j(x_j)=\sum_{r=1}^{K}\theta_{jr}B_r(x_j).

Knots can remain fixed or be learned, an identification transform can center
the basis, and a smoothness penalty regularizes adjacent spline coefficients.
Explicit interactions apply a spline to the summed member-feature input, so
they should be interpreted according to that one-dimensional projection rather
than as an unrestricted tensor-product surface.

Fit and inspect
---------------

.. code-block:: python

   from nampy.models import SplineNAMRegressor

   model = SplineNAMRegressor(
       n_knots=12,
       learn_knots=False,
       identify=True,
       smoothing=1e-3,
   ).fit(X_train, y_train)

   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   figures = model.plot_terms(X_test)

Interpretation and limits
-------------------------

SplineNAM currently provides a regression estimator. For likelihood-based
smoothing selection and statistical inference, use the statistical
:doc:`GAM <gam>` backend.

Available estimator
-------------------

* :class:`nampy.models.SplineNAMRegressor`

Reference
---------

Scardapane et al. (2019), `Learning activation functions from data using cubic
spline interpolation <https://doi.org/10.1007/978-3-319-95098-3_7>`_.

