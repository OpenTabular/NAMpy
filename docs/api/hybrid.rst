Hybrid backends
===============

.. currentmodule:: nampy.hybrid

The hybrid package composes the mgcv-parity GAM backend with the neural
backend. It is the only package that imports both sides.

.. warning::

   Hybrid models are **not mgcv fits**. The GAM stage of the residual
   estimators is exact mgcv parity, but the composite has no mgcv
   counterpart; the ``GAMNet`` estimators optimize compiled GAM
   coefficients with Torch under **fixed** smoothing parameters — there is
   no smoothing selection, and results will not and should not match
   ``GAM``/mgcv. Hybrid models never appear in the parity test suites.

Residual composition
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GAMResidualRegressor
   GAMResidualClassifier

A GAM baseline is fitted first (automatic REML), frozen, and a neural
correction is trained on the same response with the GAM link prediction as
a fixed per-sample offset — the network learns only what the smooth
additive baseline missed:

.. code-block:: python

   from nampy.hybrid import GAMResidualRegressor
   from nampy.models import LinRegRegressor

   hybrid = GAMResidualRegressor(
       "y ~ s(x0)",
       LinRegRegressor(),
       family="gaussian",          # or "poisson" (log link)
   )
   hybrid.fit(df, neural_features=["x3"], val_data=df_val)
   hybrid.predict(df)              # inverse_link(eta_gam + eta_nn)
   hybrid.predict_components(df)   # gam:/nn:-prefixed term contributions
   hybrid.plot(df)                 # GAM and neural curves, one renderer

``GAMResidualClassifier`` composes on the logit scale (binomial GAM stage,
0/1 response) and adds ``predict_proba``/``decision_function``. Both
estimators clone the passed neural template (never mutate it), so
``sklearn.model_selection.cross_val_score`` works directly.

Joint training of compiled GAM terms
------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   CompiledGAMTerms
   CompiledGAMTermsModule
   GAMNet
   GAMNetRegressor
   GAMNetClassifier

``CompiledGAMTerms`` reuses the exact mgcv-parity basis construction,
constraint absorption, and penalty matrices — compiled standalone from a
formula, or lifted read-only from a fitted ``GAM`` together with its
REML-selected smoothing parameters (the recommended way to choose them) —
and turns them into Torch parameters and buffers with a
``sum(lam_k * b' S_k b)`` penalty:

.. code-block:: python

   from nampy.gam import GAM
   from nampy.hybrid import GAMNetRegressor
   from nampy.neural.modules import LinReg
   from nampy.neural.configs import DefaultLinRegConfig

   baseline = GAM(formula="y ~ s(x0, k=8)", family="gaussian",
                  optimize_smoothing=True, smoothing_method="reml")
   baseline.fit(data=df)

   estimator = GAMNetRegressor(
       neural_model_class=LinReg,
       neural_config_class=DefaultLinRegConfig,
       gam_source=baseline,        # lifts compiled terms + REML lambdas
   )
   estimator.fit(df, neural_features=["x3"], val_data=df_val)

``GAMNetClassifier`` trains the combined logits (compiled terms plus
neural terms) through binary cross-entropy, with label encoding and
``predict_proba``.
