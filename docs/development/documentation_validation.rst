Documentation Contracts
=======================

The documentation is checked against the public estimator interfaces rather
than maintained as a separate capability inventory.

Sources of truth
----------------

* Architecture tasks and input restrictions come from ``nampy.neural.registry``.
* Effective neural constructor defaults combine configuration dataclasses,
  registry overrides, and the instantiated estimator's ``get_params``.
* Distribution aliases and parameter order come from the family registry;
  point/spread definitions remain family-specific.
* ``AdditivePrediction.term_metadata`` records parameter identity and known
  feature dependence separately from a display label.
* Criterion/optimizer boundaries are exercised by
  ``tests/optimization/test_mgcv_remaining_routes.py`` and summarized in the
  :doc:`optimizer reference <mgcv_optimizer_boundaries>`.

Verified corrections
--------------------

.. list-table::
   :header-rows: 1

   * - Contract
     - Verification
   * - NAMformer exposes its all-feature transformer contribution and sums
       scalar and multi-output terms without mixing output channels.
     - ``test_namformer_reports_complete_global_contribution``
   * - A GAMLSS parameter prefix does not turn a univariate smooth into an
       interaction; centering preserves metadata.
     - ``test_parameter_identity_is_not_an_interaction``
   * - Global/contextual terms are not summarized as marginal feature curves.
     - ``test_global_term_is_not_presented_as_a_marginal_curve``
   * - The custom attention example accepts numerical-only, categorical-only,
       and mixed encoded blocks, attends across source tokens, and preserves
       both residual additions.
     - ``test_documented_custom_attention_handles_feature_blocks``
   * - LSS evaluation accepts pandas targets as well as arrays.
     - ``test_lss_evaluation_accepts_pandas_targets``
   * - Generated training descriptions separate optimizer controls from Trainer
       options and mark variadic keywords optional.
     - ``tests/docs/test_public_reference.py``
   * - Standalone SNAM, regression plotting, and attention examples produce
       their documented outputs.
     - ``tests/docs/test_example_semantics.py``
   * - Validation term statuses require observed finite metadata-typed
       components and successful reconstruction.
     - ``test_validation_report_checks_observed_term_kinds``
   * - Quantile and tree examples reserve disjoint test rows.
     - ``test_detailed_examples_reserve_disjoint_test_rows``
   * - Quantile evaluation reports pinball loss, matching its direct calculation
       and the negative score used for model selection.
     - ``test_quantile_evaluation_reports_pinball_loss``
   * - Sparse group norms are available on a stable estimator method.
     - ``test_group_norms_public_surface``

Reproduce the checks
--------------------

.. code-block:: bash

   python -m pytest -q tests/docs tests/neural/test_documented_component_contracts.py
   python docs/scripts/check_short_examples.py
   python docs/scripts/execute_notebooks.py
   python docs/scripts/execute_notebooks.py docs/development/notebooks/*.ipynb
   python -m sphinx -E -a -b html docs docs/_build/html -W --keep-going
   python docs/scripts/check_documentation.py
   python docs/scripts/check_sidebar_navigation.py

The separate notebook CI job executes clean kernels and passes the resulting
notebooks to the HTML build. Execution notes record package versions, runtime,
and source commit. The HTML footer identifies the package and source commit.
The structural checker checks local links and anchors, redirect destinations,
obsolete search entries, sidebar numbering, and saved notebook execution records.
Numerical regression tests verify reconstruction, parameter identity, and
extension input shapes. These checks do not prove calibration or effect
identifiability for a user's dataset.

Before publishing
-----------------

Check representative generated NAM regression, classification, and LSS pages,
plus GPNAM and IGANN. Compare resolved defaults, optimizer descriptions,
scheduler choices, native-solver restrictions, and objective-specific outputs.
Use the executed reports for evidence; registry support is a separate claim.

After publication, compare the deployed footer's source revision with the
released commit and inspect a corrected contract and report in that build.
A successful local build alone does not verify the deployed revision.
