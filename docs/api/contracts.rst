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
   AdditiveTermMetadata
   EnsembleAdditivePrediction

.. currentmodule:: nampy.explanations

.. autosummary::
   :toctree: generated/
   :nosignatures:

   center_additive_prediction
   explain_additive_prediction
   term_importance_table


Term metadata
-------------

``AdditivePrediction.term_metadata`` maps each term to its source features,
distribution parameter (when applicable), and kind. ``global`` and ``contextual``
terms must not be interpreted as unary feature curves. GAMLSS parameter
identity is stored separately from its displayed ``parameter:term`` label.
Centering preserves this metadata. Older or manually constructed prediction
objects can omit metadata; label-based interpretation is then only a fallback.
