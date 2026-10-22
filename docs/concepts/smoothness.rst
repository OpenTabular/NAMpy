Smoothness and Basis Dimension
==============================

The basis dimension ``k`` gives a smooth enough capacity to represent a range
of curves. The smoothing penalty determines how much of that capacity the fit
uses. Increasing ``k`` therefore does not instruct the model to fit a more
irregular curve; it raises the ceiling on representational complexity.

For statistical GAMs, fixed smoothing parameters are useful when a penalty is
known in advance. GCV, UBRE, ML, REML, and the supported related criteria
estimate smoothing from the data. After fitting, inspect effective degrees of
freedom, residual patterns, and ``k_check()``. A basis-dimension diagnostic is
a warning tool rather than a universal acceptance test.

See the :doc:`first GAM example
<../auto_examples/statistical/plot_first_gam>` for a complete analysis and
:ref:`statistical-gam-tools` for the available bases and fitting controls.
