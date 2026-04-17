For strict parity, need port whole stack, not only gradients.

  Need from mgcv:

  1. Outer optimizer exactly

  - mgcv/R/gam.fit3.r::newton()
  - convergence mask uconv.ind
  - Hessian eigenvalue modification
  - qerror test
  - step halving rules
  - steepest-descent branch for indefinite Hessian
  - score scaling
  - accepted-step trace semantics
  - edge correction behavior if/when supported

  2. Exact ML/REML/LAML objective pieces

  - same criterion definition per family/backend
  - same scale.as.sp treatment for Gaussian REML/LAML
  - same gamma, Mp, n.true, weighted df logic
  - same penalty logdet terms and derivatives
  - same L %*% lsp + lsp0 transformed smoothing-param geometry if applicable

  3. Gaussian derivative algebra

  - exact profiled/joint derivatives used by gam.fit3
  - exact weighted RSS / penalty derivatives
  - exact logdet(X'WX + S) derivatives
  - exact logdet(S) derivatives
  - same null-space / rank accounting

  4. PIRLS derivative blocks for non-Gaussian families

  - exact first and second derivatives of criterion wrt log sp
  - same Laplace terms
  - same working-weight / working-response derivative propagation
  - same scale treatment for Gamma and others
  - same theta treatment for NegBin if estimated jointly

  5. Inner fitting loops

  - gam.fit3-style PIRLS/Newton control flow
  - step truncation / divergence checks
  - same convergence test
  - same start carry-forward semantics
  - no local extra polish or rescue branches

  6. Extended/general family fitting

  - mgcv/R/gam.fit4.r::gam.fit5
  - gam.fit5.post.proc
  - family-specific ll, score, Hessian contracts
  - exact ldetS, ldetS1, ldetS2 plumbing

  7. Theta/scale auxiliary updates

  - mgcv/R/efam.r::estimate.theta()
  - any joint theta-scale optimization paths actually used upstream
  - exact warning / failure behavior on unsupported combos

  8. Initialization

  - initial.spg / mgcv-style default sp init
  - same mustart / etastart / coef warm starts
  - same null model anchors

  9. Rank / reparam / penalty structure

  - gam.reparam, ldetS, null-space dimensions
  - exact penalty block ordering
  - exact fixed/free sp masking semantics
  - exact linked id= and shared-penalty geometry

  10. Postfit quantities used by optimizer/tests

  - outer.info
  - edf, edf1, edf2
  - scale estimate
  - optimizer trace rows / iteration counts / messages

  Practical rule:

  - For each parity-sensitive path, port trio:
      1. objective value
      2. gradient/Hessian
      3. optimizer consuming them

  If one piece missing, better raise unsupported than add heuristic.

  If you want, next step I can make concrete port checklist by backend:

  - Gaussian exact
  - Gaussian dynamic/random-effect
  - PIRLS Laplace families
  - general gam.fit5
  - NegBin joint theta REML/LAML


## Port Sequence: Strict mgcv Parity One Slice at a Time

  ### Summary

  Port parity-sensitive optimizer/fitting paths in this order:

  1. Gaussian REML/LAML outer optimization, including dynamic/random-effect
     cases.
  2. Negative-binomial REML/LAML with estimate_theta=True.
  3. Non-Gaussian pirls_laplace_dynamic ML/REML/LAML paths.
  4. general_fit5 outer derivative surfaces for GAMLSS/general families that
     still raise.
  5. Remaining postfit/trace/reporting parity gaps tied to the above.

  Default chosen: start with Gaussian first. This is the smallest high-signal
  unit already covered by strong trace tests and it establishes the exact
  mgcv::newton() contract to reuse elsewhere.

  ### Implementation Changes

  #### 1. Gaussian REML/LAML as reference slice

  - Treat mgcv/R/gam.fit3.r::newton() as the behavioral spec for all outer
    Newton logic.
  - Keep one canonical Newton implementation and remove backend-specific
    rescue behavior that is not present upstream.
  - Port exact state/flow, not only formulas:
    score.scale, uconv.ind, tiny-gradient masking, eigenvalue modification,
    qerror, step-halving, steepest-descent branch, accepted-step selection,
    convergence flags, and final optimizer status strings.
  - Make Gaussian exact and Gaussian dynamic/random-effect use the same
    upstream-shaped optimizer contract; differences should only be in
    objective/derivative providers, not fallback strategy.
  - Mirror upstream scale.as.sp handling exactly for Gaussian REML/LAML,
    including the reported criterion and trace rows.
  - Acceptance criteria:
    tests/
    test_mgcv_trace_parity.py::TestMgcvTraceParity::test_gaussian_reml_trace_
    matches_mgcv_endpoint
    plus one focused random-effect/dynamic Gaussian parity slice if not
    already present.

  #### 2. NegBin joint theta REML/LAML

  - Port the upstream-supported joint (log sp, log theta) path instead of
    keeping the current explicit NotImplementedError.
  - Use mgcv/R/efam.r::estimate.theta() and the relevant gam.fit3/outer-loop
    interactions as the primary spec.
  - Preserve exact theta update timing relative to coefficient updates,
    deviance recomputation, and convergence checks.
  - Remove any remaining local-only theta rescue heuristics; unsupported
    combinations must still raise explicitly.
  - Add targeted tests that move the current “known gap” cases into real
    parity tests one case at a time.
  - Acceptance criteria:
    replace the tracked-gap tests in tests/test_mgcv_known_gaps.py for one-
    smooth, then two-smooth cases.

  #### 3. Non-Gaussian pirls_laplace_dynamic

  - Port exact objective, gradient, and Hessian surfaces for dynamic/random-
    effect structures where current code still relies on incomplete coverage.
  - Reuse the same canonical outer Newton implementation from step 1; only
    derivative providers differ.
  - Ensure working-weight, working-response, Laplace logdet, scale, and null-
    space terms follow upstream ordering exactly.
  - Keep unsupported family/structure combinations explicit until their exact
    derivative blocks exist.
  - Acceptance criteria:
    add one focused binomial dynamic/random-effect parity slice and one
    focused Poisson slice before broadening.

  #### 4. general_fit5 parity completion

  - For each general family still lacking full outer derivative support, port
    the exact gam.fit5-compatible outer derivative blocks from upstream
    instead of exposing “strict parity requires analytic derivatives” errors.
  - Keep ldetS, ldetS1, ldetS2, predictor layout, and score/Hessian plumbing
    aligned with mgcv/R/gam.fit4.r::gam.fit5.
  - Do not add numerical outer differentiation for missing families.
  - Acceptance criteria:
    convert family-specific NotImplementedError expectations in tests/
    test_mgcv_gamlss_core.py into exact derivative parity tests case by case.

  #### 5. Trace and postfit cleanup

  - After each slice, align outer_info, accepted-step history, edf/edf1/edf2,
    and status/warning surfaces with upstream behavior for that slice.
  - Any parity-reporting logic should consume canonical optimizer state, not
    reconstruct local approximations after the fact.

  ### Test Plan

  For each slice, use smallest sufficient coverage only:

  - Gaussian first:
    tests/
    test_mgcv_trace_parity.py::TestMgcvTraceParity::test_gaussian_reml_trace_
    matches_mgcv_endpoint
    plus one dynamic/random-effect Gaussian trace or endpoint slice.
  - NegBin theta:
    one currently-gapped estimate_theta=True case at a time, promoting from
    known_gaps into real parity coverage.
  - PIRLS dynamic:
    one exact file/function-level binomial case, then one Poisson case.
  - general_fit5:
    one family at a time from tests/test_mgcv_gamlss_core.py or the matching
    family parity file.
  - After each implementation:
    run ruff check and python -m py_compile on touched files only.

  ### Public/API Effects

  - No new public API required.
  - Behavior changes:
    currently unsupported parity-sensitive paths will either become exact
    upstream ports or remain explicit NotImplementedError.
  - Optimizer internals should expose canonical trace/state consistently so


  - We will not change vendored mgcv; upstream code is read-only behavioral
    spec.
  - We will not use JAX or other autodiff in production parity paths.
  - We will implement one backend slice fully before starting the next; no
    mixed partial ports across slices.
  - If a slice cannot be ported exactly with current repo structure, it
    remains unsupported until the exact port is possible.