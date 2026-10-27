Coefficient covariance correction internals
===========================================

``Vb.corr`` (also searched as ``vb.corr``) is an internal correction for
uncertainty induced by estimated smoothing parameters. NAMpy implements the
dense ``w is NULL`` branch in ``_vb_corr_root``. It is a numerical building block,
with no public ``vb.corr()`` function.

How this relates to the public API
----------------------------------

The public workflow and a complete example live in
:doc:`/statistical_reference/uncertainty`. This page documents the private
matrix routine for numerical extension work; ordinary prediction does not
require calling it.

``unconditional=True`` requests a correction when the fitted route supplies the
required derivative and curvature information. ``vcov`` falls back to conditional
Bayesian covariance if its corrected matrix is absent. Prediction has its own
availability checks; requesting unconditional uncertainty does not make every
family/design combination supported. ``sp_vcov`` concerns free log smoothing
parameters, not coefficients. See :doc:`/statistical_reference/methods/vcov`, :doc:`/statistical_reference/methods/predict`
and the :doc:`fitting support matrix </api/gam_advanced>`.

Internal signature
------------------

.. autofunction:: nampy.gam.fit.solvers.general_family.newton._vb_corr_root
   :no-index:

Parameters
----------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``X_root``
     - Two-dimensional root of the unpenalized information matrix, with
       ``p`` columns in fitted coefficient coordinates.
   * - ``L``
     - Optional mapping from free log smoothing coordinates to penalty log
       parameters. ``None`` uses direct coordinates.
   * - ``lsp0``
     - Optional fixed offsets for mapped log penalty strengths.
   * - ``S_blocks``
     - List of penalty matrices in the compatible block coordinate system.
   * - ``off``
     - Optional one-based starting coefficient positions for local penalty blocks.
   * - ``rho``
     - One-dimensional free log smoothing parameter vector; may contain the
       final scale coordinate when ``scale_est=True``.
   * - ``Vr``
     - Covariance of those log parameters, aligned with ``rho``.
   * - ``scale_est``
     - Whether to strip the final scale coordinate and its covariance/mapping
       entries before computing the smoothing correction; default False.
   * - ``penalty_matrix``
     - Optional full penalization matrix. Required when exact ``dH_blocks``
       are supplied, and must match the information-matrix dimensions.
   * - ``dH_blocks``
     - Optional exact information derivatives per log parameter, replacing
       penalty-only derivative construction.

Returns
-------

A dense NumPy ``(p, p)`` covariance correction matrix. Empty smoothing coordinates
return a zero matrix. Callers combine this component with other uncertainty
terms and fit scale; it is not the complete unconditional covariance by itself.
Invalid matrix contracts raise ``ValueError`` or numerical shape/solve errors.
The private function is documented for implementation traceability and can change
with the backend.

Input types, shapes and defaults
--------------------------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type and shape
     - Default
   * - ``X_root``
     - ``numpy.ndarray[float64]``, ``(m, p)``
     - Required
   * - ``L``
     - ``numpy.ndarray[float64]``, ``(n_penalties, r)`` or ``None``
     - Required keyword; ``None`` selects direct coordinates
   * - ``lsp0``
     - ``numpy.ndarray[float64]``, ``(n_penalties,)`` or ``None``
     - Required keyword; ``None`` uses zero log offsets
   * - ``S_blocks``
     - ``list[numpy.ndarray]``; each penalty is square in its block coordinates
     - Required keyword
   * - ``off``
     - ``list[int]`` with one one-based start per local block, or ``None``
     - Required keyword
   * - ``rho``
     - ``numpy.ndarray[float64]``, ``(r,)``
     - Required keyword
   * - ``Vr``
     - ``numpy.ndarray[float64]``, ``(r, r)``
     - Required keyword
   * - ``scale_est``
     - ``bool``
     - ``False``
   * - ``penalty_matrix``
     - ``numpy.ndarray[float64]``, ``(p, p)`` or ``None``
     - ``None``
   * - ``dH_blocks``
     - ``list[numpy.ndarray]``, one ``(p, p)`` array per log parameter, or ``None``
     - ``None``

Runnable numerical example
--------------------------

.. literalinclude:: /_examples/reference/statistical_reference/internals/covariance_correction.py
   :language: python

:download:`Download the complete example </_examples/reference/statistical_reference/internals/covariance_correction.py>`
