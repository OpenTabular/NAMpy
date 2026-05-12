Models
======

High-level model classes that follow the scikit-learn API.

.. currentmodule:: nampy.models

Regression Models
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NAMRegressor
   GPNAMRegressor
   NBMRegressor
   NATTRegressor
   NAMformerRegressor
   LinRegRegressor
   TreeNAMRegressor
   SplineNAMRegressor
   SparseNAMRegressor
   NodeGAMRegressor

Classification Models
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NAMClassifier
   GPNAMClassifier
   NBMClassifier
   NATTClassifier
   NAMformerClassifier
   LinRegClassifier
   SplineNAMClassifier
   NodeGAMClassifier

Distributional Regression Models (LSS)
---------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NAMLSS
   GPNAMLSS
   NBMLSS
   NATTLSS
   NAMformerLSS
   LinRegLSS
   SplineNAMLSS
   NodeGAMLSS

Other Models
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   QNAM

Base Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SklearnBaseRegressor
   SklearnBaseClassifier
   SklearnBaseLSS
