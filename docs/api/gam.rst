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

Shape-constrained smooths
-------------------------

The same ``GAM`` class accepts the SCOP-spline basis codes ported from R's
``scam`` package.  For example, ``bs="mpi"`` is monotone increasing and
``bs="tecxcv"`` is a bivariate convex/concave surface.  See
:doc:`../user_guide/shape_constrained_gams` for the complete basis list,
automatic smoothing, derivatives, linear functionals, and AR(1) support.

.. currentmodule:: nampy.gam.diagnostics

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SmoothDerivativeResult

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

.. currentmodule:: nampy.contracts

Backend-neutral feature and prediction contracts used by the GAM adapters
and neural estimators:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FeatureSchema
   AdditivePrediction
