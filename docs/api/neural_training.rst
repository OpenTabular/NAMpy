Neural Training Contract
========================

This is the shared training surface used by gradient-trained estimators.
Architecture-specific native solvers use their documented stage/linear controls.

The distributional objective follows the NAMLSS framework of Thielmann et al.
(2024), with registered families defining parameter transforms and loss.

.. include:: /_generated/reference/neural_fit_contract.inc

.. automethod:: nampy.models._neural.NeuralEstimatorBase.fit

References
----------

Thielmann, Kruse, Kneib, and Säfken (2024), `Neural Additive Models for
Location Scale and Shape: A Framework for Interpretable Neural Regression
Beyond the Mean <https://proceedings.mlr.press/v238/frederik-thielmann24a.html>`_.
AISTATS, PMLR 238:1783–1791.
