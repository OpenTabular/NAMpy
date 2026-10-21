Distributional Regression
==========================

A mean model assumes that the conditional expectation is the main quantity
that changes with predictors. Distributional regression is useful when
variability, skewness, tail behavior, or another parameter also changes.

For parameter :math:`\theta_k`, NAMpy fits an additive predictor
:math:`\eta_k` and applies the parameter's inverse link,

.. math::

   \eta_k(x)=\beta_{0k}+\sum_j f_{jk}(x_j),\qquad
   \theta_k(x)=h_k^{-1}\{\eta_k(x)\}.

Terms add on the :math:`\eta_k` scale. Natural parameters generally do not add
after a nonlinear inverse link. Conditional standard deviation describes
observation-level predictive spread. It is different from a standard error
for an estimated mean and from variation across repeated fits.

Continue with :doc:`../tutorials/changing_variability` for a fitted example or
:doc:`../user_guide/gamlss` for formulas, links, and prediction contracts.
