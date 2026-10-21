Prediction and Explanation Contracts
====================================

Backend-neutral objects provide one representation for GAM and neural
predictions. Explanation functions operate on that shared representation.

.. currentmodule:: nampy.contracts

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FeatureSchema
   AdditivePrediction
   EnsembleAdditivePrediction

.. currentmodule:: nampy.explanations

.. autosummary::
   :toctree: generated/
   :nosignatures:

   center_additive_prediction
   explain_additive_prediction
   term_importance_table

