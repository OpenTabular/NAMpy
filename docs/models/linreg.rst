LinReg
======

When to use it
--------------

Use LinReg as an additive linear baseline under the same preprocessing and
component interface as the neural architectures.

Model structure
---------------

LinReg is the linear member of NAMpy's estimator family. After preprocessing,
each source feature contributes its own linear block,

.. math::

   f_j(x_j) = w_j^\top z_j,

where :math:`z_j` is the transformed block for feature :math:`j`. A numeric
feature usually contributes one column; a categorical feature can contribute a
group of encoded columns. Keeping the blocks separate gives the same additive
component interface used by the nonlinear architectures.

Fit and inspect
---------------

.. code-block:: python

   from nampy.models import LinRegRegressor

   model = LinRegRegressor().fit(X_train, y_train)
   prediction = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   importance = model.term_importance(X_test)
   figures = model.plot_terms(X_test, center=True)

Interpretation and limits
-------------------------

LinReg is useful as a transparent baseline and as a check that a nonlinear
model is learning more than preprocessing and linear effects. Encoded
categorical blocks still contain several coefficients even though they are
reported under one source feature.

Available estimators
--------------------

* :class:`nampy.models.LinRegRegressor`
* :class:`nampy.models.LinRegClassifier`
* :class:`nampy.models.LinRegLSS`

