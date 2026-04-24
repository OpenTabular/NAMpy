Important distinction:

  - Pure sign flip / orthogonal rotation of basis should not change fitted behavior if all downstream algebra is transformed consistently.
  - But if orientation leaks into initialization, truncation, pivoting, penalty block construction, warm starts, finite-difference derivatives, line
    search, or threshold branches, then optimizer path can change.
  - Once optimizer path changes, sp, coefficients, trace rows, and sometimes even final fitted values can move from finite-precision effects.

  So blanket “invariants only everywhere” is too broad. Better policy:

  - Raw basis / constructor parity: invariants only.
  - Eigenspace-derived intermediate objects: invariants or canonicalized form.
  - Initialization state and optimization traces: exact parity if mgcv uses those quantities in control flow.
  - Final behavior surfaces: exact parity for fit, predict, EDF, smoothing params, covariance, diagnostics.

  Good rule:

  - If object is mathematically non-unique, compare invariants.
  - If object influences later branch decisions or user-visible results, do not hide differences behind invariant-only checks.

  So your concern is correct:

  - if basis orientation can change initial values and then final values, that path is behavior-sensitive, not “representation-only”.
  - On those paths, invariants-only tests would hide real parity bugs.

  Best target:

  - enforce invariants-only for raw representation layers,
  - keep exact mgcv parity for optimization and final behavior layers,
  - only relax exact checks where upstream object is genuinely defined only up to orthogonal transform.






  Findings

  1. Supported optimization traces are still incomplete. tests/optimization/test_mgcv_joint_branch_trace_parity.py still carries explicit known-gap
     xfails for gamma joint-scale trace parity and negbin joint-theta trace labeling. Those are supported behavior paths, so the suite itself says
     “not finished”.
  2. Current suite contracts prove ownership, not exhaustiveness. tests/_taxonomy_registry.py and tests/regressions/test_gam_test_suite_contracts.py
     guarantee each high-level mark has some owner file, but they do not prove every supported leaf path is covered. Missing proof includes
     combinations like smooth × family × method × select × by/id/pc × weights/offset/newdata/prediction surface.
  3. “Invariants-only parity” is not centrally enforced. Raw-representation coverage is spread across tests/smooths/
     test_mgcv_raw_constructor_parity.py, tests/smooths/test_mgcv_smoothcon_parity.py, and snapshot files, but there is no single policy helper/
     marker that says: “this object is non-unique, compare invariants only.” So the suite is cleaner, but policy is still convention, not
     enforcement.
  4. Full-object parity across optimization lifecycle is still partial. nampy/gam/parity/trace.py and tests/optimization/
     test_mgcv_outer_optimization_parity.py now compare much richer trace rows and outer_info, but there is still no single exhaustive supported-
     surface rule that every optimizer/family branch must pass both:
      - full normalized optimization object parity
      - full final fitted-object parity
        across the whole supported matrix