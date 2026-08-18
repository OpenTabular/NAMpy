Base Models
===========

Low-level PyTorch model implementations.

.. currentmodule:: nampy.neural.modules

These are the underlying PyTorch models. Most users should use the high-level
:mod:`nampy.models` instead, which provide a scikit-learn compatible interface.

Base Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseModel
   ~nampy.neural.training.TaskModel

Model Implementations
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NAM
   GPNAM
   NBM
   NATT
   NAMformer
   LinReg
   QNAMBase
   SNAM
   TreeNAM
   EnsembleTreeNAM
   SplineNAM
   NodeGAM

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MultiModelWrapper
