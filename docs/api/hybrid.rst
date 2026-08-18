Hybrid backends (experimental)
==============================

.. currentmodule:: nampy.hybrid

The hybrid package composes the mgcv-parity GAM backend with the neural
backend. It is the only package that imports both sides.

.. warning::

   Hybrid models are **not mgcv fits**. The GAM stage of
   :class:`GAMPlusNeural` is exact mgcv parity, but the composite has no
   mgcv counterpart; :class:`HybridJointRegressor` optimizes compiled GAM
   coefficients with Torch under **fixed** smoothing parameters — there is
   no smoothing selection, and results will not and should not match
   ``GAM``/mgcv. Hybrid models never appear in the parity test suites.

Two-stage composition
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GAMPlusNeural

Fits a GAM baseline (automatic REML), freezes it, and trains a neural
correction on the same response with the GAM link prediction as a fixed
per-sample offset:

.. code-block:: python

   from nampy.hybrid import GAMPlusNeural
   from nampy.models import LinRegRegressor

   hybrid = GAMPlusNeural(
       "y ~ s(x0)",
       LinRegRegressor(),
       family="gaussian",
   )
   hybrid.fit(df, neural_features=["x3"])
   hybrid.predict(df)                        # inverse_link(eta_gam + eta_nn)

Joint Torch training of compiled GAM terms
------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   CompiledGAMTerms
   CompiledGAMTermsModule
   HybridAdditiveNet
   HybridJointRegressor

``CompiledGAMTerms`` reuses the exact mgcv-parity basis construction,
constraint absorption, and penalty matrices — compiled standalone from a
formula or lifted read-only from a fitted ``GAM`` — and turns them into
Torch parameters and buffers with a ``sum(lam_k * b' S_k b)`` penalty:

.. code-block:: python

   from nampy.hybrid import HybridJointRegressor
   from nampy.neural.modules import LinReg
   from nampy.neural.configs import DefaultLinRegConfig

   estimator = HybridJointRegressor(
       "y ~ s(x0, k=8)",
       LinReg,
       DefaultLinRegConfig,
       lam=[0.5],
   )
   estimator.fit(df, neural_features=["x3"])
