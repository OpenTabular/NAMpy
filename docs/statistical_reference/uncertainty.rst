Prediction uncertainty
======================

Prediction standard errors describe uncertainty about the estimated predictor
or mean. They are different from the conditional spread of a future response.
For example, Gaussian response SD includes observation noise, while a mean
prediction SE concerns the fitted mean function.

Select the uncertainty convention
---------------------------------

.. list-table::
   :header-rows: 1

   * - Entry point
     - Type and default
     - Output type and shape
     - Meaning
   * - ``model.predict(X, return_se=True)``
     - ``X``: DataFrame/ndarray ``(n, d)``; ``return_se: bool=False``
     - ``tuple[numpy.ndarray, numpy.ndarray]``; two arrays with ``n`` rows
     - Prediction and pointwise standard error in the requested prediction scale.
   * - ``model.predict(..., unconditional=True)``
     - ``unconditional: bool=False``
     - Same output shape as the selected prediction route
     - Requests available smoothing-parameter uncertainty corrections.
   * - ``model.vcov()``
     - ``freq: bool=False``; ``unconditional: bool=False``
     - ``numpy.ndarray[float64]``, ``(p, p)``
     - Bayesian penalized coefficient covariance conditional on smoothing.
   * - ``model.vcov(freq=True)``
     - ``freq: bool=False``
     - ``numpy.ndarray[float64]``, ``(p, p)``
     - Frequentist coefficient covariance for supported fits.
   * - ``model.sp_vcov()``
     - ``edge_correct: bool=True``; ``reg: float=0.001``
     - ``numpy.ndarray[float64]`` or ``None``
     - Covariance of free log smoothing parameters, possibly with scale.

Here ``n`` is prediction rows, ``d`` input columns, and ``p`` fitted coefficients.
For scalar-family response/link predictions, prediction and SE have shape
``(n,)``. Multi-predictor, term and matrix predictions follow their documented
family/type-specific shapes. The :doc:`methods/predict`, :doc:`methods/vcov`
and :doc:`methods/sp_vcov` pages list every argument and return convention.

Interpretation and availability
-------------------------------

A normal approximation such as prediction plus/minus ``1.96 * se`` is a
pointwise approximate 95% interval on that prediction scale. It is not a
simultaneous confidence band. Under nonlinear links, use the correct scale and
transformation rather than assuming symmetric response intervals.

Corrected covariance depends on the fitted family's derivative support and the
smoothing-selection route. ``vcov(unconditional=True)`` uses the corrected
matrix when present and otherwise retains conditional Bayesian covariance.
Prediction performs its own availability checks. Inspect the fitting support
matrix before interpreting a corrected result.

The private ``Vb.corr``-style matrix correction is one component of this
calculation, not a separate fitting feature or user-facing callable. Its
implementation contract is under :doc:`internals/covariance_correction`.

Runnable example
----------------

.. literalinclude:: /_examples/reference/statistical_reference/uncertainty.py
   :language: python

:download:`Download the complete example </_examples/reference/statistical_reference/uncertainty.py>`
