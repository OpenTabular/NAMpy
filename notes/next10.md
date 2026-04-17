Important context:

  - Worktree dirty already: nampy/gam/fit/solvers/irls_core.py, next9.md.
  - Biggest likely non-mgcv dead-code bucket in smoothing_selection/optimize/heuristics/*.
  - general_fit5.py already says finite-difference outer fallback removed, but still keeps dead fallback functions. Strong remove candidate.
  - Many family/model/API hits look like Vulture false positives from public surface, dynamic dispatch, future mgcv contracts. Not safe delete blind.

  Bucket Plan

  1. Remove first: clear non-mgcv dead code.
      - nampy/gam/smoothing_selection/optimize/heuristics/rollback.py
      - nampy/gam/smoothing_selection/optimize/heuristics/stabilize.py
      - Reason: helper functions not referenced anywhere in nampy/gam, and comments/behavior are NAMpy rescue heuristics, not upstream mgcv control flow.
  2. Remove next: dead finite-difference / fallback remnants that violate parity policy.
      - nampy/gam/fit/solvers/general_fit5.py
      - Candidates: _penalty_logdet_terms, _finite_difference_general_fit5_gradient, _finite_difference_general_fit5_hessian_from_gradient, _finite_difference_general_fit5_hessian, dead locals like reduced_start.
      - Reason: file now hard-errors when analytic outer derivatives absent; leftover fallback code only confuses parity story.
  3. Trim low-risk local dead vars/properties after wave 1.
      - design locals/properties: nampy/gam/design/constructed.py, nampy/gam/design/constructors.py, nampy/gam/design/structures.py
      - solver locals: nampy/gam/fit/linalg/stacked_qr.py, nampy/gam/fit/penalized_system.py
      - parity/helper dead functions: nampy/gam/parity/snapshots.py, nampy/gam/runtime/factory.py, nampy/gam/specs/smooth.py
  4. Review carefully, maybe keep:
      - family capability flags and abstract contracts in nampy/gam/families/family_base.py
      - family class attrs in nampy/gam/families/exponential.py and nampy/gam/families/gamlss.py
      - public/user API methods in nampy/gam/model/api.py
      - smooth metadata in nampy/gam/smooths/smooth_base.py, tensor/categorical smooth classes
      - reason: static analysis often misses class attrs, abstract protocol hooks, public methods, test-only access, parity scaffolding.
  5. Likely separate later cleanup, not first pass:
      - nampy/gam/model/gam_solve.py
      - nampy/gam/smoothing_selection/criteria
      - nampy/gam/smoothing_selection/reparam.py
      - These interact with parity-critical optimizer/reparameterization paths. Need upstream mgcv cross-check before deletion.

  Validation Plan

  - Wave 1 heuristics/fallback deletion:
      - pytest tests/test_mgcv_trace_parity.py -k "reml or optimizer or negbin" -v
  - general_fit5.py cleanup:
      - pytest tests/test_gamlss_families.py -k "gam_fit5" -v
  - design/runtime small trims:
      - pytest tests/test_mgcv_snapshot_parity.py -k "<affected area>" -v
  - No broad suite unless one deletion crosses multiple subsystems.

  Guardrails

  - Do not delete anything only because Vulture says unused.
  - Keep code if it mirrors upstream mgcv structure, even if currently dormant.
  - Prefer whole-file/module removal for dead heuristic helpers. Better than half-keeping ghost machinery.
  - For parity-sensitive files, name exact upstream references before edit. Likely key refs:
      - mgcv/R/gam.fit3.r for outer Newton / smoothing logic
      - mgcv/R/gam.fit4.r for gam.fit5

  If you want, next step I do bucket 1 only: remove dead heuristic modules and dead general_fit5 fallback remnants, then run smallest targeted tests.