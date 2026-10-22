Estimators
==========

High-level statistical and neural estimators. Each generated page shows the
constructor, fitted attributes documented by the class, essential inherited
methods, and related gallery examples. Begin with the :doc:`modeling guide
<../user_guide>` if you are choosing an approach rather than looking up an
exact signature.

Across estimator families, ``fit`` accepts pandas data frames or array-like
inputs as documented by the class. Regression ``predict`` returns point
predictions, classifier ``predict_proba`` returns class probabilities, and LSS
``predict`` returns natural distribution parameters in ``parameter_names_``
order. ``predict_components`` returns link-scale additive pieces rather than
another point-prediction format.

.. currentmodule:: nampy.models

Statistical GAMs
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: autosummary/estimator_class.rst

   GAMRegressor
   GAMClassifier
   GAMLSS

Regression Models
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: autosummary/estimator_class.rst

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
   :template: autosummary/estimator_class.rst

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
   :template: autosummary/estimator_class.rst

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
   :template: autosummary/estimator_class.rst

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
