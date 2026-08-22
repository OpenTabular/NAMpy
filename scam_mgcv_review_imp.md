not all guarded boundaries are unsupported upstream. The guards fall into three categories.

   Guarded boundary                                  Upstream status                                   Interpretation
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Transformed SCAM coefficients + automatic LSS     Not supported by SCAM. mgcv supports LSS with     The combined feature would be a NAMpy
   smoothing                                         REML, but without SCAM coefficient                extension, not a direct port.
                                                     transformations.
  ────────────────────────────────────────────────  ────────────────────────────────────────────────  ─────────────────────────────────────────────────
   AR(1) with ML/REML/LAML                           Supported by mgcv::bam, including the             This is a genuine generic mgcv parity gap.
                                                     correlation-determinant correction. SCAM only
                                                     combines known AR(1) with GCV/UBRE.
  ────────────────────────────────────────────────  ────────────────────────────────────────────────  ─────────────────────────────────────────────────
   Derivatives of matrix/linear-functional terms     No direct derivative API in either upstream.      Keeping this guarded is correct until the
                                                     mgcv can encode a derivative as a linear          desired derivative semantics are explicitly
                                                     functional during fitting, which is different.    defined.
  ────────────────────────────────────────────────  ────────────────────────────────────────────────  ─────────────────────────────────────────────────
   SCAM derivatives at new data                      Not supported by derivative.scam.                 The guard mirrors upstream exactly.
  ────────────────────────────────────────────────  ────────────────────────────────────────────────  ─────────────────────────────────────────────────
   Linear-functional terms for additional            Broadly supported by mgcv for s() and te()        This is a genuine generic mgcv parity gap.
   ordinary bases                                    terms.
  ────────────────────────────────────────────────  ────────────────────────────────────────────────  ─────────────────────────────────────────────────
   Shape-constrained linear functionals without a    Not provided as a documented shape-preserving     The guard is appropriate.
   SCAM *By basis                                    SCAM surface.

  The details:

  - SCAM performs automatic smoothing through GCV/UBRE. Even its EFS implementation minimizes GCV/UBRE rather than REML (upstreams/scam/man/
    scam.Rd:50). It does not provide SCAM-constrained multi-predictor LSS fitting. mgcv does support general/LSS families with multiple predictors and
    REML (upstreams/mgcv/man/family.mgcv.Rd:38), but not combined with SCAM’s nonlinear coefficient transformation.

  - mgcv::bam explicitly applies the AR(1) whitening transform and corrects the REML criterion for its determinant (upstreams/mgcv/R/bam.r:478,
    upstreams/mgcv/R/bam.r:795). Therefore our AR(1)+REML guard represents unfinished generic mgcv functionality, not an upstream limitation.

  - scam::derivative.scam has no newdata argument, only handles one-dimensional smooths, and evaluates against object$model (upstreams/scam/R/
    derivative.scam.r:10, upstreams/scam/R/derivative.scam.r:47). Our training-data-only behavior is faithful.

  - mgcv’s summation convention supports matrix-valued covariates and weights for s() and te() terms, including integrals and finite-difference
    derivative functionals (upstreams/mgcv/man/linear.functional.terms.Rd:8). SCAM reuses this mechanism for its constrained *By bases (upstreams/scam/
    man/linear.functional.terms.Rd:8).

  Other documented guards:

  - efs, optim, nlm, nlm.fd, and coefficient BFGS are supported by SCAM upstream, so these are real remaining parity gaps.
  - select=True and arbitrary user constraint matrices are not exposed by the SCAM API, so those are not missing SCAM ports.
  - Bivariate shape-constrained bases are supported upstream, but dedicated shape-preserving bivariate by variants are not documented. Guarding that
    combination is prudent.

  So the two clearest upstream-backed next implementations are:

  1. Generic AR(1) ML/REML determinant handling.
  2. Extending mgcv-style linear-functional aggregation across additional ordinary s() and te() bases.

  The transformed-LSS and matrix-derivative boundaries require new design work beyond either upstream.