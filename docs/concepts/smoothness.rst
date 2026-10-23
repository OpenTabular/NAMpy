Smoothness and Basis Dimension
==============================

The basis dimension ``k`` gives a smooth enough capacity to represent a range
of curves. The smoothing penalty determines how much of that capacity the fit
uses. Increasing ``k`` therefore does not instruct the model to fit a more
irregular curve; it raises the ceiling on representational complexity.

For a statistical GAM, each smooth has a basis representation,

.. math::

   f_j(x)=\sum_{r=1}^{K_j}b_{jr}(x)\theta_{jr}.

The coefficients balance fit against roughness by minimizing

.. math::

   -\ell(\theta)+\frac{1}{2}\sum_m\lambda_m\theta^\mathsf T S_m\theta.

The matrices :math:`S_m` encode penalties, and the smoothing parameters
:math:`\lambda_m` control their strength. Effective degrees of freedom
summarize the flexibility left after penalization and need not be integers.

For statistical GAMs, fixed smoothing parameters are useful when a penalty is
known in advance. GCV, UBRE, ML, REML, and the supported related criteria
estimate smoothing from the data. After fitting, inspect effective degrees of
freedom, residual patterns, and ``k_check()``. A basis-dimension diagnostic is
a warning tool rather than a universal acceptance test.

See :doc:`../quickstart` for a complete example and
:ref:`statistical-gam-tools` for the available bases and fitting controls.
