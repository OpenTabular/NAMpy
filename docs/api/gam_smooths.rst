Smooth Terms
============

Smooth terms construct the basis and penalty matrices used by the statistical
GAM backend. Users normally select them by code inside a formula instead of
instantiating the runtime classes directly:

.. code-block:: python

   formula = (
       'y ~ s(x, bs="tp", k=12) '
       '+ s(day, bs="cc", k=10) '
       '+ ti(latitude, longitude, bs=c("cr", "cr"))'
   )

The :ref:`gam-smooth-bases` guide lists every basis code, its purpose, and a
minimal formula. The classes below form the extension API for custom smooths.

Base contract and registry
--------------------------

.. currentmodule:: nampy.gam.smooths

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseSmoothTerm
   ByState
   make_smooth_term
   register_smooth

One-dimensional and metric smooths
----------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ThinPlateSplineTerm
   CubicSplineTerm
   PSplineTerm1D
   DerivativeBSplineTerm1D
   DuchonSplineTerm
   GaussianProcessTerm
   AdaptiveSmoothTerm
   SphericalSplineTerm

Tensor and structured smooths
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TensorProductSplineTerm
   InteractionTensorProductSplineTerm
   AlternativeTensorProductSplineTerm
   RandomEffectTerm
   MarkovRandomFieldTerm
   FSmoothInteractionTerm
   SZSmoothInteractionTerm
   ShapeConstrainedPSplineTerm

Construction helpers
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   resolve_by_state
   apply_numeric_by
   linear_functional_basis
   build_penalty_definition

