GAM
===

The statistical engine compiles formulas, constructs smooth bases and penalties,
fits coefficients, selects smoothing parameters, and supplies statistical
prediction and inference. For scikit-learn workflows use :doc:`estimators/GAMRegressor`,
:doc:`estimators/GAMClassifier` or :doc:`estimators/GAMLSS`.

Signature
---------

.. autoclass:: nampy.gam.GAM
   :no-index:
   :no-members:
   :no-inherited-members:

The constructor accepts its modeling controls through ``**kwargs``. Unknown
names raise ``TypeError``. The estimator parameter tables describe the common
modeling options with effective defaults. ``GAM`` itself defaults to
``optimize_smoothing=False``; the statistical estimator wrappers default to
REML selection.

.. include:: /_generated/reference/GAM_core_parameters.inc

Legacy adapter inputs
---------------------

The explicit signature also accepts ``cat_feature_info=None`` and
``num_feature_info=None`` (optional metadata mappings) and ``num_classes=1``
(integer). These are compatibility inputs; the statistical engine derives its
feature structure and response behavior from data, formulas and the family.
They do not replace categorical formula terms or select a classification family.
``family`` defaults to ``None`` (Gaussian identity); see the family reference
for accepted strings, option mappings and family instances.

Inputs
------

Use ``formula='y ~ s(x, bs="tp", k=12)'`` and ``fit(data=data)`` with a
DataFrame containing response and predictor columns. ``family`` accepts a name,
options mapping or independent family instance. Formula collections supply
multiple predictors for supported general families.

``fit(X, y)`` supports the array interface where the selected terms and family
permit it. Prior weights, known offsets, knots and lower smoothing bounds can
also be supplied at fitting. Prediction data must use the fitted column schema.
See :doc:`methods/fit` for every fit argument and :doc:`methods/predict` for
output scales, filtering, missing rows and standard errors.

Returns and fitted state
------------------------

Construction returns an unfitted ``GAM``. ``fit`` returns the fitted instance.
``gam_result_`` contains compiled model, numerical solution and fit summary;
``fit_result()`` exports a presentation record. ``predict`` produces arrays,
``summary`` returns a structured summary, and diagnostic/plot methods return
reports or plotting payloads described on their individual pages.

Example
-------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from nampy.gam import GAM

   data = pd.DataFrame({'x': np.linspace(-1, 1, 100)})
   data['y'] = np.sin(3 * data.x)
   model = GAM(formula='y ~ s(x, bs="tp", k=10)',
               family='gaussian', optimize_smoothing=True,
               smoothing_method='reml').fit(data=data)
   prediction, se = model.predict(data, return_se=True)
   summary = model.summary()

:doc:`All public methods <methods/index>` · :doc:`Families <families/index>` ·
:doc:`Smooths <smooths/index>` · :doc:`Controls <controls/index>`

Executable example and output shapes
------------------------------------

.. literalinclude:: /_examples/reference/statistical_reference/GAM.py
   :language: python

:download:`Download the complete example </_examples/reference/statistical_reference/GAM.py>`

The example predicts ``n=96`` observations. With ``return_se=True``, the output
is a tuple of two ``numpy.ndarray[float64]`` vectors of shape ``(96,)``.
The symbolic shape is ``(n,)`` for each vector.
