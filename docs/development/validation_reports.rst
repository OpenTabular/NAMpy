Validation Reports
==================

These reports exercise architecture and task interfaces with small budgets.
They verify prediction shapes, output domains, and promised reconstruction.
They are not performance benchmarks. Some scores use fitting rows and are
labeled accordingly. Learning workflows with separate test data live in the
:doc:`tutorials </tutorials/index>`.

.. toctree::
   :maxdepth: 1

   pretab_migration
   documentation_validation
   optimizer_matrix
   notebooks/02_statistical_gams
   notebooks/03_neural_models
   notebooks/04_distributional_models

The optimizer matrix is backed by
``tests/optimization/test_mgcv_remaining_routes.py``; prediction and metadata
contracts are checked by ``tests/neural/test_documented_component_contracts.py``.

Numerical snapshots
-------------------

``model.parity_snapshot(new_data)`` exports coefficients, predictions, smoothing parameters, and optional covariance summaries for numerical validation. It supports comparisons against reference implementations.
