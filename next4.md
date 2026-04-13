 1. Finish exact analytic derivative port
  Biggest missing piece. Current blocker.

  What still wrong:

  - nampy/gam/smoothing_selection/criteria/pirls_deriv.py:1 still contains old
    analytic formulas built around:
      - static state.X_fix
      - static state.Z_rand
      - _laplace_lambda_vector()
      - _lambda_group_indices()
      - M = Z'WZ + diag(lam_vec)
  - That structure belongs to old NAMpy local mixed-model rewrite.
  - Upstream mgcv exact path use current-sp gam.reparam(UrS, sp, deriv) first,
    then derivative algebra on transformed objects.

  Need do:

  - Build dynamic derivative kernel from rp = gam_reparam(UrS, sp, deriv=1/2).
  - Replace all exact derivative use of lam_vec, group indices, and static
    random blocks.
  - Re-derive exact first/second derivative terms from dynamic transformed
    system actually used by objective.
  - Re-enable dispatch exact PIRLS derivative path only after this port.

  Files:

  - nampy/gam/smoothing_selection/criteria/pirls_deriv.py:1
  - maybe small helper additions in nampy/gam/smoothing_selection/
    reparam.py:286

  2. Replace fake Laplace helper layer
  laplace.py now mostly stale scaffolding.

  What still wrong:

  - nampy/gam/smoothing_selection/criteria/laplace.py:13 still defines
    _laplace_lambda_vector
  - nampy/gam/smoothing_selection/criteria/laplace.py:18 still defines
    _lambda_group_indices
  - Those encode old block-diagonal random-effect view, not canonical mgcv
    global reparam view.

  Need do:

  - Delete _laplace_lambda_vector
  - Delete _lambda_group_indices
  - Keep only helpers still truly canonical, likely
    _penalty_derivative_matrices or move that too if duplicate
  - Remove all imports/callers

  Files:

  - nampy/gam/smoothing_selection/criteria/laplace.py:1
  - nampy/gam/smoothing_selection/criteria/pirls_deriv.py:9

  3. Remove static exact mixed-model state from exact scoring path
  We still have two worlds.

  Old world:

  - cached nampy/gam/smoothing_selection/reparam.py:28
  - sl_blocks
  - sl_group_indices
  - sl_lambda_vector
  - build_penalty_reparameterized_system() static block decomposition

  New world:

  - canonical UrS
  - gam_reparam(UrS, sp)
  - dynamic current-sp design

  Need do:

  - Decide exact path canonical owner = dynamic world only
  - Delete ReparamState from exact path
  - Remove exact-criteria dependence on:
      - ensure_penalty_reparameterization_state
      - sl_blocks
      - sl_lambda_vector
      - sl_group_indices

  Files:

  - nampy/gam/smoothing_selection/reparam.py:68
  - nampy/gam/smoothing_selection/criteria/gaussian.py:40
  - nampy/gam/smoothing_selection/criteria/pirls.py:205
  - nampy/gam/smoothing_selection/criteria/pirls_deriv.py:1

  4. Match mgcv transform objects more literally
  Current port shape close, still not exact object flow.

  Still missing:

  - explicit U1 = cbind(Y, Z) style owner
  - explicit T = U1 %*% blockdiag(Qs, I_Mp) style full transform owner
  - explicit transformed rS, Sr, St, Eb objects as first-class state
  - current code computes enough to score, but not same structural data model
    as mgcv

  Need do:

  - Create canonical object mirroring mgcv names:
      - Y, Z, U1, UrS, rp, T, St, Sr, Eb, Mp
  - Make objective and derivative code consume those directly
  - Stop recomputing ad hoc pieces in separate helpers

  Files:

  - nampy/gam/smoothing_selection/reparam.py:463
  - maybe new dataclass in same file

  5. Decide fate of old builder
  nampy/gam/smoothing_selection/reparam.py:718 still exists and still builds
  static X_fix/Z_rand/sl_blocks.

  Need do one of two:

  - either keep only for non-parity legacy paths, mark clearly non-canonical
  - or replace internals fully with dynamic mgcv-style transform state
  - if goal exact structural parity, best end state: exact ML/REML criteria no
    longer depend on this builder at all

  Stale pieces likely removable after derivative port:

  - SlBlock
  - sl_group_indices
  - sl_lambda_vector
  - maybe whole static builder path if nothing else needs it

  Files:

  - nampy/gam/smoothing_selection/reparam.py:43
  - nampy/gam/smoothing_selection/reparam.py:92
  - nampy/gam/smoothing_selection/reparam.py:107
  - nampy/gam/smoothing_selection/reparam.py:718

  6. Validate against parity tests, not only patch tests
  Current tests only prove local regressions.

  Before claim exact parity need:

  - targeted mgcv parity slice hitting:
      - Gaussian REML exact score
      - PIRLS REML/ML score
      - derivative trace / optimizer path
      - multi-penalty tensor/factor-smooth/random-effect cases
  - especially:
      - tests/test_mgcv_trace_parity.py
      - tests/test_mgcv_snapshot_parity.py
      - maybe tests/test_mgcv_known_gaps.py narrow exact cases

  Need do in order:

  1. add one focused parity case for dynamic reparam score path
  2. add one focused parity case for derivative/outer step path
  3. only then broader nearby slice if needed

  Practical execution order
  Best order:

  1. Port analytic PIRLS derivative kernel to dynamic gam_reparam(UrS, sp)
  2. Re-enable exact derivative dispatch
  3. Delete _laplace_lambda_vector and _lambda_group_indices
  4. Remove exact-path use of static ReparamState/sl_blocks
  5. Introduce canonical mgcv-named transform state (U1, T, Sr, St, Eb)
  6. Run narrow mgcv parity tests for score and derivative traces
  7. Delete leftover legacy exact-path scaffolding

  Short truth:

  - hardest missing part = analytic derivative port
  - until that done, structural code parity with mgcv not achieved
  - objective path close now
  - derivative path still main gap