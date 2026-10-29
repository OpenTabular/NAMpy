Neural Modules
==============

These modules support architecture registration, training objectives, data
loading, interaction discovery, and custom model development. Fitted analyses
normally use the estimators in :doc:`models`.

Architecture registry
---------------------

.. currentmodule:: nampy.neural.registry

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NeuralArchitecture
   architectures
   get_architecture
   register_architecture

Training objectives
-------------------

.. currentmodule:: nampy.neural.objectives

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NeuralObjective
   RegressionObjective
   BinaryObjective
   MulticlassObjective
   DistributionObjective
   classification_objective

Training module
---------------

.. currentmodule:: nampy.neural.task

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TaskModule

Data module
-----------

.. currentmodule:: nampy.neural.data

.. autosummary::
   :toctree: generated/
   :nosignatures:

   NAMpyDataModule
   NAMpyDataset

Interaction selection
---------------------

.. currentmodule:: nampy.neural.interaction_selection

.. autosummary::
   :toctree: generated/
   :nosignatures:

   InteractionSearchConfig
   InteractionSelectionResult
   InteractionScore
   ArchipelagoDetector
   select_interactions
   interaction_frontier
   fit_reference_model

Architecture contracts
----------------------

.. currentmodule:: nampy.neural.contracts

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FixedLinearDesignProvider
   NativeTrainingProvider
   harvest_penalties
   validate_feature_names

Fixed-design solver
-------------------

.. currentmodule:: nampy.neural.linear_solver

.. autosummary::
   :toctree: generated/
   :nosignatures:

   solve_fixed_linear_regression


See :doc:`neural_training` for the shared training signature and controls.
