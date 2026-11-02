Probability laws
================

These objects evaluate a declared probability law without fitting a model.
Estimator families choose a law and supply its parameters; the pages under
:doc:`statistical families </statistical_reference/families/index>` and
:doc:`neural families </neural_reference/families/index>` record that fitting
contract. :ref:`howto-user-guide-distributional-probabilities` shows how selected
parameterizations map onto these operations.

.. currentmodule:: nampy.distributions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DistributionLaw
   DistributionSpec
   DistributionPrediction
   ObservationTarget
   TruncatedLaw
   observation_log_prob
   AtomMixtureLaw
   NormalLaw
   StudentTLaw
   GammaLaw
   LogNormalLaw
   WeibullLaw
   BetaLaw
   BetaBinomialLaw
   NegativeBinomialLaw
   BCCGLaw
   BCTLaw
   BCPELaw
