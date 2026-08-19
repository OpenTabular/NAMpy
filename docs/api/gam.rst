GAM (mgcv-parity backend)
=========================

The classical GAM backend is a Python reimplementation of R's ``mgcv``,
targeting behavioral parity to machine precision wherever feasible.

High-level model
----------------

.. currentmodule:: nampy.gam

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GAM

.. code-block:: python

   from nampy.gam import GAM

   model = GAM(
       formula="y ~ s(x0) + s(x1, k=15)",
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   )
   model.fit(data=df)
   predictions = model.predict(df, type="response")
   model.summary()
   model.plot()

Scikit-learn adapters
---------------------

.. currentmodule:: nampy.models

The adapters wrap :class:`nampy.gam.GAM` without adding any numerics; they
default to automatic REML smoothing selection (mgcv's ``gam()`` behavior).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GAMRegressor
   GAMClassifier

.. code-block:: python

   from nampy.models import GAMRegressor

   estimator = GAMRegressor(k=10)
   estimator.fit(X, y)
   estimator.score(X, y)                     # R^2
   components = estimator.predict_components(X)
   se = estimator.standard_errors(X)

Fit core
--------

.. currentmodule:: nampy.gam

Low-level entry points for the solver stage:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fit_model_core
   solve_fit
   FitCoreSolution

Shared contracts
----------------

.. currentmodule:: nampy._contracts

Backend-neutral feature and prediction contracts used by the GAM adapters
and neural estimators:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FeatureSchema
   AdditivePrediction
