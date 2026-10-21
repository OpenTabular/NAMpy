Estimators
==========

High-level statistical and neural model classes with ``fit`` and ``predict``
interfaces.

.. currentmodule:: nampy.models

Statistical GAMs
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GAMRegressor
   GAMClassifier
   GAMLSS

Regression Models
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NAMRegressor
   SIANRegressor
   GPNAMRegressor
   IGANNRegressor
   NBMRegressor
   NBMSPAMRegressor
   SPAMRegressor
   NATTRegressor
   NAMformerRegressor
   LinRegRegressor
   TreeNAMRegressor
   EnsembleTreeNAMRegressor
   SNAMRegressor
   SplineNAMRegressor
   NodeGAMRegressor

Classification Models
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NAMClassifier
   SIANClassifier
   GPNAMClassifier
   IGANNClassifier
   NBMClassifier
   NBMSPAMClassifier
   SPAMClassifier
   NATTClassifier
   NAMformerClassifier
   LinRegClassifier
   TreeNAMClassifier
   EnsembleTreeNAMClassifier
   SNAMClassifier
   NodeGAMClassifier

Distributional Regression Models (LSS)
---------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NAMLSS
   SIANLSS
   GPNAMLSS
   IGANNLSS
   NBMLSS
   NBMSPAMLSS
   SPAMLSS
   NATTLSS
   NAMformerLSS
   LinRegLSS
   TreeNAMLSS
   EnsembleTreeNAMLSS
   SNAMLSS
   NodeGAMLSS

Specialized Models
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   QNAMLSS

Ensembles and Architecture Factories
------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NeuralEnsemble
   NeuralEstimatorFamily
   estimator_family

Base Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NeuralRegressor
   NeuralClassifier
   NeuralLSS
