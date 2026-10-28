Validation reports
==================

These reports distinguish interface execution from numerical agreement.
Finite outputs alone do not establish convergence, calibration or parity.
Each report states its inputs, acceptance conditions and remaining limits.

.. list-table:: Report purposes
   :header-rows: 1

   * - Report
     - Purpose
   * - Statistical interface checks
     - Exercise selected formulas, fitting, prediction, constraints and diagnostics.
   * - Neural architecture checks
     - Inspect compression, basis construction, local expansions and reconstruction.
   * - Neural task and distribution checks
     - Enforce task output shapes, domains, losses and additive reconstruction.
   * - Optimizer route checks
     - Record finite fits, stopping diagnostics, warnings and expected rejections.
   * - Numerical comparisons
     - Document upstream fixtures, tolerances and independent reference calculations.

.. toctree::
   :maxdepth: 1

   Statistical interface checks <notebooks/02_statistical_gams>
   Neural architecture checks <notebooks/03_neural_models>
   Neural task and distribution checks <notebooks/04_distributional_models>
   Optimizer route checks <optimizer_matrix>
   Statistical extension numerical comparisons <neighbor_parity>

See :doc:`/architecture` for versioned upstream fixtures, independent
reference calculations and the scope of numerical agreement.
