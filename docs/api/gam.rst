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

Spherical smooths
-----------------

Use ``s(latitude, longitude, bs="sos")`` for an isotropic smooth on a
sphere.  Coordinates are supplied in degrees, latitude first and longitude
second.  The default ``m=0`` is the second-order Wendelberger spline; integer
orders from ``-2`` through ``4`` select the upstream Duchon or Wahba kernel
branches.  SOS margins in ``te`` and ``ti`` must be grouped with ``d=2``.
The array API cannot express this joint two-coordinate term, so SOS models use
the formula interface.  The ``m=-1`` null space is four-dimensional; combining
that order with an ``fs`` factor smooth is rejected because the corresponding
upstream penalty split is LAPACK-orientation dependent.  Upstream SOS smooths
also do not define derivative matrices, and NAMpy does not expose the
hemisphere-specific ``plot.gam`` schemes 0 and 1 through its generic plotter.

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
   GAMLSS

.. code-block:: python

   from nampy.models import GAMRegressor

   estimator = GAMRegressor(k=10)
   estimator.fit(X, y)
   estimator.score(X, y)                     # R^2
   components = estimator.predict_components(X)
   se = estimator.standard_errors(X)

Distributional GAMs use a distinct estimator because ``predict`` returns a
matrix of natural distribution parameters rather than one regression response
or class label:

.. code-block:: python

   from nampy.models import GAMLSS

   estimator = GAMLSS(
       family="normal",
       formula={"mu": "y ~ s(x)", "sigma": "~ s(z)"},
   ).fit(data)
   parameters = estimator.predict(data)  # columns follow parameter_names_
   point = estimator.predict_point(data)

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
