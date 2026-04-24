Cause in upstream mgcv, not NAMpy.

  Fail path:

  1. Failing case builds perfect separation data: y = 1[x > 0] in tests/
     test_mgcv_parity_failing_and_warnings.py:55.
  2. mgcv::gam(..., family=binomial(), method="REML") goes into outer optimizer
     newton() in mgcv/R/gam.fit3.r:1290.
  3. At iteration i = 173, optimizer has:
      - lsp = 2.438297 (sp ~= 11.45)
      - grad = 0
      - hess = -8.881784e-16
      - pdef = FALSE
      - sd.unused = TRUE

  Why crash:

  - In newton(), steepest-descent fallback computes Sstep <- -grad /
    max(abs(grad)) in mgcv/R/gam.fit3.r:1410.
  - Here grad = 0, so this becomes 0/0 = NaN.
  - Then fallback step makes lsp3 <- lsp + step = NaN in same branch around
    mgcv/R/gam.fit3.r:1575.
  - That NaN reaches gam.reparam(), which calls .C(C_get_stableS,
    sp=as.double(exp(lsp)), ...) in mgcv/R/gam.fit3.r:36.
  - exp(NaN) means foreign-call arg 3 (sp) is NaN, giving exact error: "NA/NaN/
    Inf in foreign function call (arg 3)".

  Pinpoint:

  - Root bug is upstream mgcv Newton fallback logic for indefinite Hessian +
    zero gradient.
  - Numerical noise leaves Hessian slightly negative (-8.88e-16), so code
    enters indefinite-path fallback even though gradient is exactly zero.
  - Fallback formula does not guard zero-gradient case, so it manufactures NaN.


  No tests run. No files changed.


› okay, fix this bug


• Explored
  └ Search MGCV_LIB_PATH|library\(mgcv\)|R_LIBS|install.packages|R CMD INSTALL|
           mgcv_snapshot.R in tests

• Ran git status --short
  └  M PARITY_SUMMARY.md
     M nampy/gam/basis/tensor.py
    … +30 lines (ctrl + t to view transcript)
    ?? tests/test_gamlss_families.py
    ?? tests/test_mgcv_parity_failing_and_warnings.py

• Waited for background terminal

• Explored
  └ Search MGCV_LIB_PATH|library\(mgcv\)|R_LIBS|install.packages|R CMD INSTALL|
           mgcv_snapshot.R in tests

───────────────────────────────────────────────────────────────────────────────

• Patch will stay minimal in upstream mgcv/R/gam.fit3.r: guard zero-gradient
  steepest-descent fallback, and treat 1D near-flat negative curvature at
  convergence as roundoff instead of true indefiniteness. Then validate on
  exact separation reproduction.

• Edited mgcv/R/gam.fit3.r (+13 -2)
    1455      d[ind] <- 1/d[ind]
    1456 +
    1457 +    ## In the 1-D case near convergence, tiny negative curvature can
          be
    1458 +    ## pure roundoff. Treat that as flat rather than indefinite, othe
          rwise
    1459 +    ## the steepest-descent fallback below can generate NaN from 0/0.
    1460 +    if (length(d) == 1 &&
    1461 +        max(abs(grad1)) <= score.scale * conv.tol &&
    1462 +        max(abs(hess1)) <= score.scale * .Machine$double.eps^.5) {
    1463 +      indef <- FALSE
    1464 +      pdef <- TRUE
    1465 +    }
    1466
         ⋮
    1468      Nstep[uconv.ind] <- -drop(U%*%(d*(t(U)%*%grad1))) # (modified) Ne
    1459 -
    1460 -    Sstep <- -grad/max(abs(grad)) # steepest descent direction

───────────────────────────────────────────────────────────────────────────────

• Validation running with patched upstream newton() injected into loaded mgcv
  namespace. Looking for exact separation case to finish without foreign-call
  NaN.

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› i reverted your changes, but so the bug is in mgcv and not in nampy? why is
  that? dont make any changes. i do not want to make changes to mgcv


• Yes. This specific failure is in mgcv, not in NAMpy.

  Why:

  - Test fails while building expected, not actual.
  - actual = _fit_nampy_snapshot(...) completes first.
  - Crash happens in _run_mgcv_snapshot(...), which shells out to Rscript and
    calls mgcv::gam(...).
  - Stack trace is entirely inside upstream mgcv:
    estimate.gam -> gam.outer -> newton -> gam.fit3 -> gam.reparam

  Why NAMpy not cause:

  - NAMpy is only generating data and formula, then asking R mgcv for reference
    output.
  - Same data/model reproduced direct in R, outside pytest and outside NAMpy
    fit path, and mgcv still crashes.
  - Root issue is mgcv outer optimization on perfect binomial separation data:
    it reaches zero gradient plus numerically tiny negative Hessian, enters
    fallback code, and produces NaN smoothing parameter input for gam.reparam.

  So:
  - Test harness impact: NAMpy parity test cannot get reference snapshot
  - NAMpy fit itself: not source of this crash

  If you do not want to touch mgcv, then fix has to be on NAMpy/test side:

  - mark this case as known mgcv failure,
  - skip/xfail this parity case,
  - or replace dataset with near-separation instead of exact separation

  No changes made.


› so can we change the data such that mgcv does not crash?


• Yes.

  mgcv is crashing because the current data is exactly separable:

  y = (x > 0.0).astype(int)

  That makes binomial fit saturate. If you want parity coverage without
  touching upstream mgcv, change the case to be near-separable instead of
  perfectly separable.

  Good options:

  - Add a few label flips near boundary.
  - Generate probabilities from a steep logistic curve, then sample Bernoulli.
  - Add small noise before thresholding.

  Best minimal change:

  - Keep same shape, but flip a small number of labels for points with smallest
    |x|.

  Example idea:

  x = rng.normal(size=n)
  y = (x > 0.0).astype(int)

  boundary = np.argsort(np.abs(x))[:8]
  y[boundary[:4]] = 0
  y[boundary[4:]] = 1

  Better statistical version:

  x = rng.normal(size=n)

  Why this works:

  - No exact separation.
  - Still strong binomial signal.
  - mgcv REML optimizer should stay finite.

  Tradeoff:

  - ~~Case no longer tests exact-separation behavior.~~
  - It becomes a “hard binomial smooth” case instead of a true separation case.

  If you want, I can suggest a concrete replacement dataset that stays
  maximally close to original while avoiding the mgcv crash.


› yes, do that


• Change only failing-case data factory. Keep shape close to original exact-
  threshold setup, but break perfect separation by flipping a few labels
  nearest x = 0, where boundary ambiguity is most natural.

• Edited tests/test_mgcv_parity_failing_and_warnings.py (+5 -0)
    60 +    # Keep this case near-separable, but avoid exact separation that tr
        iggers
    61 +    # an upstream mgcv REML optimizer NaN in the 1-D smoothing paramete
        r path.
    63 +    y[boundary[:4]] = 1
    64 +    y[boundary[4:]] = 0
    65      return pd.DataFrame({"y": y, "x": x})
