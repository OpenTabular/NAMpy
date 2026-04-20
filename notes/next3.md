Main fix:

  - In nampy/gam/smoothing_selection/reparam.py:166, _term_penalty_components()
    ~~no longer rejects overlapping null-space penalties when they live inside
    one connected primary-penalty component. That was old false restriction,
    unlike mgcv gam.reparam() flow.~~
  - In nampy/gam/smoothing_selection/reparam.py:238 and nampy/gam/model/
    gam_solve.py:80, error text now only rejects cross-component null-space
    coupling, not generic overlap.
  - In nampy/gam/model/gam_solve.py:428, stale one-penalty-per-term helper no
    longer throws on multi-penalty terms; it combines same-term penalty blocks
    instead.
  - Added narrow regression in tests/test_gam_mgcv_patch_regressions.py:255 for
    overlapping null-space penalties on one term.

  Upstream refs used:


  Result:

  - Minimal regression passed.
  - Parity likely improved for overlapping multi-penalty ML/REML reparam cases.
  - Remaining unresolved surface: null-space penalties that couple multiple
    disconnected primary support components still rejected.

• No. Bug fixed, parity not exact yet.

  Why not exact:

  - mgcv path: build global UrS in total-penalty range with
    totalPenaltySpace(), then call gam.reparam(UrS, sp, deriv), then C routine
    C_get_stableS.
  - NAMpy path still: per-term / per-component decomposition in nampy/gam/
    smoothing_selection/reparam.py:166, then local eigendecompositions via
    reparameterize_smooth(), then assemble X_fix / Z_rand.
  - So behavior closer now, but code path still not same code architecture as
    mgcv/R/gam.fit3.r.

  Big remaining non-parity pieces:

  - nampy/gam/smoothing_selection/reparam.py:309
    build_penalty_reparameterized_system() still term-local Python rewrite, not
    UrS -> gam.reparam() port.
  - nampy/gam/smoothing_selection/reparam.py:283
    can_use_exact_gaussian_ml_reml() still restricts exact Gaussian to one
    primary per connected component. mgcv gam.reparam() not have this Python-
    side structural gate.
  - nampy/gam/smoothing_selection/reparam.py:217 still rejects null-space
    penalties spanning multiple disconnected components. That is still local
    simplification.
  - No port of C_get_stableS logic. That biggest gap if goal is code-match.

  Stale / legacy / old code here:

  - ~~null_map in nampy/gam/smoothing_selection/reparam.py:231 looks legacy.
    Built, mostly unused, likely leftover from older one-primary/one-null
    design.~~
  - ~~_one_penalty_per_term_matrices in nampy/gam/model/gam_solve.py:428 name
    stale now. Function no longer “one penalty per term”; now sums many.~~
  - can_use_simple_ml_reml_structure itself is legacy-conservative concept.
    mgcv does not gate this way before gam.reparam().
  - Comments/docstrings in reparam.py still describe “conservative structural
    gate” / local component logic. That is NAMpy-specific scaffold, not
    upstream model.
