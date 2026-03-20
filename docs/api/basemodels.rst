Base Models
===========

Low-level PyTorch model implementations.

.. currentmodule:: nampy.basemodels

These are the underlying PyTorch models. Most users should use the high-level
:mod:`nampy.models` instead, which provide a scikit-learn compatible interface.

Base Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseModel
   TaskModel

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
   NodeGAM

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MultiModelWrapper
