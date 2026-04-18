## Exact parity (or near-exact)

  - Core snapshot assertions through the exact helper:
      - [tests/_mgcv_snapshot_parity_shared.py](/home/ad32/projects/package/NAMpy/
        tests/_mgcv_snapshot_parity_shared.py) (calls to
        _assert_exact_mgcv_snapshot_parity)
      - tests/smooths/test_mgcv_pc_id_parity.py (exact helper usage for fixed-sp
        constructions and selected REML cases)
      - tests/smooths/test_mgcv_smoothcon_parity.py (several exact basis/
        constraint construction checks)
  - Optimization/trace parity:
      - tests/optimization/test_mgcv_newton_exact_parity.py (outer score/history
        exactness)
      - tests/optimization/test_mgcv_trace_parity.py (trace and optimizer state
        parity at high precision)
  - Smoothness postprocess derivative/math parity:
      - tests/optimization/test_gam_gaussian_smoothness_postprocess_parity.py
        (tight derivative/value assertions for Gaussian smoothness postprocess)

  ## Tight parity (meaningful but intentionally relaxed)

  - Most non-exact mainline fit/output checks:
      - tests/_mgcv_snapshot_parity_shared.py (pred/criterion/edf tolerance set
        per scenario, often moderate but deliberate)
      - tests/smooths/test_mgcv_pc_id_parity.py (some linked/REML/restrict cases
        are tight but not exact; known gp mismatch path noted)
      - tests/smooths/test_mgcv_smoothcon_parity.py (basis and penalty behavior
        tight; full model-level REML sometimes relaxed)
      - tests/parity/test_mgcv_additional_scenarios.py and tests/parity/
        test_mgcv_parity.py (mixed CaseSpec strictness, targeted not always exact)
  - Shared assertion utility driving this layer:
      - [tests/mgcv_parity_utils.py](/home/ad32/projects/package/NAMpy/tests/
        mgcv_parity_utils.py) (_assert_basic_mgcv_parity and requested-shared
        helpers often use 1e-4–1e-3 scale tolerances for selected metrics)

  ## Loose parity (broad coverage, low-precision gates)

  - Matrix-heavy matrix/criterion permutation tests where regressions can be
    masked:
      - tests/_mgcv_snapshot_parity_shared.py (bulk of its cases still use basic
        parity tolerances)
      - tests/parity/test_mgcv_output_parity.py (fs/sz/mrf output/SE style checks
        are looser)
      - tests/diagnostics/test_mgcv_k_check_parity.py (diagnostic parity partial,
        with representative skip paths)
  - Environment- or performance-gated behavior often avoids stricter assertions:
      - tests requiring Rscript and some heavier combinations in known-gap/failing
        suites

  ## Explicitly not fully tested / marked gaps

  - Triaged gaps:
      - [tests/test_mgcv_known_gaps.py](/home/ad32/projects/package/NAMpy/tests/
        test_mgcv_known_gaps.py)
      - [tests/test_mgcv_parity_failing_and_warnings.py](/home/ad32/projects/
        package/NAMpy/tests/test_mgcv_parity_failing_and_warnings.py)
      - Notes file identifies current blockers: notes/MGCV_PARITY_TODO.md (e.g.
        P0/P1+/P3 items)
  - Skipped or intentionally incomplete test behavior:
      - tests/diagnostics/test_mgcv_k_check_parity.py has an explicit skip for fs
        k_check parity
      - Negative binomial estimated-theta parity and some fs/sz and specific t2
        variants are intentionally triaged/loosened

  ## Code paths currently “not parity yet” despite being in gam

  - Parsing/front-end:
      - [nampy/gam/formula/parse.py](/home/ad32/projects/package/NAMpy/nampy/gam/
        formula/parse.py) (limited syntax: no dot terms, limited subtraction/term
        syntax, etc.)
  - Formula/spec build restrictions:
      - [nampy/gam/specs/build.py](/home/ad32/projects/package/NAMpy/nampy/gam/
        specs/build.py)
      - [nampy/gam/specs/modeling.py](/home/ad32/projects/package/NAMpy/nampy/gam/
        specs/modeling.py)
  - Smooth/spec limits and unsupported branches:
      - [nampy/gam/smooths/categorical/factor_smooth.py](/home/ad32/projects/
        package/NAMpy/nampy/gam/smooths/categorical/factor_smooth.py) (fs/sz
        breadth constrained)
      - [nampy/gam/smoothing_selection/criteria/pirls*.py](/home/ad32/projects/
        package/NAMpy/nampy/gam/smoothing_selection/criteria/)
      - [nampy/gam/fit/solvers](/home/ad32/projects/package/NAMpy/nampy/gam/fit/
        solvers) and .../optimize/postfit contain NotImplemented/explicitly
        unsupported behavior
  - Diagnostics surface is partial:
      - k.check, residual-style postfit parity, and some advanced diagnostics
        remain incomplete vs mgcv.

  ## Practical read

  - “Exact” is mostly for fixed-sp/optimizer-internal surfaces.
  - “Tight” is often decent for regression detection but still broad enough to
    miss ordering/numerical-path changes.
  - “Loose” is currently useful for smoke-level parity, not for behavior drift
    detection.
  - A significant amount of behavior is intentionally excluded via triage files or
    unsupported parser/smooth branches.