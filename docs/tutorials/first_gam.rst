Fit and diagnose a GAM
======================

.. include:: ../_includes/first_gam_walkthrough.inc

The fitted model captures the temperature relationship and daily cycle, and the residual and basis checks do not indicate a clear problem in this example.

If a smooth appeared too restrictive, we would increase its basis dimension and refit. If residual structure remained after allowing sufficient flexibility, we would reconsider the model terms or response distribution.
