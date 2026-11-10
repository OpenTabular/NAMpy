:orphan:

Statistical GAM API
===================

NAMpy's statistical backend implements formula, smooth-construction, fitting,
prediction, and inference conventions familiar from R's ``mgcv``. The
:ref:`howto-user-guide-statistical-gams` page works through formula specification, fitting, prediction and diagnosis
with reproducible data.

The detailed :doc:`statistical reference </statistical_reference/index>` follows
estimator setup, formulas, smooths, families, fitting and inference in order.

High-level model
----------------

The lower-level ``GAM`` defaults to fixed smoothing. Set
``optimize_smoothing=True`` to estimate smoothing parameters. The estimator
adapters below select smoothness with REML by default.

.. currentmodule:: nampy.gam

:doc:`Canonical estimator contracts </statistical_reference/estimators/index>`

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
sphere. Supply latitude first and longitude second, both in degrees.

* **Basis.** ``m=0`` selects the second-order Wendelberger spline. Integer
  orders from ``-2`` through ``4`` select the upstream Duchon or Wahba branches.
* **Composition.** Use the formula interface and group ``sos`` tensor margins
  in ``te`` or ``ti`` with ``d=2``. The four-dimensional ``m=-1`` null space
  makes its ``fs`` penalty split LAPACK-dependent, so that fit is guarded.
* **Prediction and display.** Upstream supplies no derivative matrices.
  ``model.plot(scheme=0)`` and ``scheme=1`` show rotated hemispheres. Set
  ``theta`` and ``phi`` to change the view.

Shape-constrained smooths
-------------------------

The same ``GAM`` class accepts SCOP-spline basis codes defined by R's ``scam``
package. For example, ``bs="mpi"`` is monotone increasing and
``bs="tecxcv"`` is a bivariate convex/concave surface.  See
:ref:`howto-user-guide-shape-constrained-gams` for the complete basis list,
automatic smoothing, derivatives, linear functionals, and AR(1) support.

Scikit-learn adapters
---------------------

.. currentmodule:: nampy.models

The adapters wrap :class:`nampy.gam.GAM` without adding any numerics; they
default to automatic REML smoothing selection (mgcv's ``gam()`` behavior).

:doc:`Canonical estimator contracts </statistical_reference/estimators/index>`

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
   parameters = estimator.predict_parameters(data)  # columns follow parameter_names_
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

Controls and neighborhood cross-validation
--------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GAMControl
   gam_control
   normalize_nei

Shared contracts
----------------

.. currentmodule:: nampy.contracts

Backend-neutral feature and prediction contracts used by the GAM adapters
and neural estimators:

See :doc:`contracts` for ``FeatureSchema`` and ``AdditivePrediction``.
