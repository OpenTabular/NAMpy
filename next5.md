Findings

  1. Biggest non-mgcv surface = outer smoothing optimizer. After main solve,
     code runs local rollback/stabilize/refine/snap/accept passes that comments
     themselves call heuristic or non-mgcv. See nampy/gam/smoothing_selection/
     optimize/driver.py:1438, nampy/gam/smoothing_selection/optimize/
     heuristics/stabilize.py:1, nampy/gam/smoothing_selection/optimize/
     heuristics/rollback.py:1. Structure parity weak here even when numeric
     endpoint sometimes close.
  2. General-family outer derivatives still not full parity. general_fit5 keeps
     zero log|S|_+ derivatives and records finite-difference Hessian fallback
     for some families. See nampy/gam/fit/solvers/general_fit5.py:110, nampy/
     gam/fit/solvers/general_fit5.py:145, nampy/gam/smoothing_selection/
     criteria/dispatch.py:60, nampy/gam/smoothing_selection/criteria/
     dispatch.py:134. Means core gam.fit5 inner loop mirrors mgcv; outer
     derivative plumbing still partly local approximation.
  3. Factor-smooth preprocess has explicit fallback rewrite, not direct
     upstream structure. FactorSmoothInteractionSpec can be rewritten into base
     s(...) plus metadata fs_base_by_fallback. See nampy/gam/formula/
     preprocess.py:212. Useful shim. Not clean mgcv mirror.
  4. Linked id= support only pools compatible 1D cubic-regression s() terms.
     Other shared-id terms get skipped with warning. See nampy/gam/design/
     linked_basis.py:113. So parity exists for covered subset, not general mgcv
     linked-basis structure.
  5. Public diagnostics/API contain deliberate NAMpy-only layers. gam_check()
     splits mgcv_comparable from nampy_specific; summary text is custom
     “General Smooth Model Summary”; extra API/features live outside parity
     suites. See nampy/gam/diagnostics/k_check.py:150, nampy/gam/diagnostics/
     summary.py:52, tests/test_new_features.py:1, nampy/gam/model/api.py:1. Not
     bad. But not mgcv structure parity.
  6. Stale repo guidance exists. CLAUDE.md says tests/legacy_mgcv_*.py remain,
     but grep found no such files. That doc stale. Same for old “stale factory
     guard” wording in tests comment, not live code path. See CLAUDE.md:74.

  What looks parity-structured

  - Core GAMLSS/gam.fit5 porting strong. See nampy/gam/fit/solvers/
    gam_fit5.py:1, nampy/gam/fit/solvers/general_fit5.py:1, nampy/gam/fit/
    solvers/gamlss_utils.py:1.
  - Gaussian linear algebra parity-minded. See nampy/gam/fit/linalg/
    stacked_qr.py:1.
  - Many family and smooth modules explicitly mirror upstream mgcv routines.
    Search hits strong in families/gamlss.py, tensor smooths, factor smooths,
    random effects.
  - Test strategy parity-first. Snapshot/output/trace/pc-id/known-gap files all
    compare against R mgcv. Known gaps tracked explicitly in tests/
    test_mgcv_known_gaps.py:19.

  Heuristics / legacy / non-mgcv summary

  - Heuristics: yes, many. Mostly outer optimization rescue logic.
  - Legacy: yes, mainly “legacy infinite-sp rollback heuristic” in optimizer
    comments/path, not huge dead-file graveyard.
  - Stale code/docs: some stale guidance/docs; no obvious large dead legacy
    module set in nampy/gam.
  - Non-mgcv: yes, mainly public API sugar, diagnostics packaging, feature-
    smoke tests, optimizer rescue logic, subset implementations/fallback
    rewrites.

  No tests run. Review only.