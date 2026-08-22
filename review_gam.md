 ## Overall assessment

  Against the vendored mgcv 1.9-4 and scam 1.2-22 sources, NAMpy has integrated the difficult central GAM machinery very well. It is no longer merely
  “GAM-like”: the supported path mirrors substantial parts of gam.setup, gam.side, gam.fit3, gam.fit5, smoothing selection, prediction, and post-fit
  inference.

  The remaining gap is mainly breadth: mgcv contains many specialized bases, families, optimizers, large-data facilities, mixed models, and auxiliary
  tools that NAMpy has not ported.

  This assessment concerns the current working tree, including its uncommitted GAM changes.

  ## Capability map

   Area                       Current state                                                 Assessment
  ━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Formula/specification      s, te, ti, interactions, factors, transforms, offsets, by,    Strong within the declared Python formula grammar
                              id, pc, fx, sp, m, xt, select, formula lists
  ─────────────────────────  ────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
   Smooth constructors        cr, cs, cc, ps, tp, ts, re, fs, sz, te, ti                    Strong coverage of the most commonly used bases
  ─────────────────────────  ────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
   Identifiability            smoothCon-style construction, constraint absorption,          One of the strongest parts
                              gam.side, linked smooths, point constraints
  ─────────────────────────  ────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
   Ordinary fitting           Gaussian, PIRLS, rank handling, reparameterization,           Closely aligned with gam.fit3
                              signed-weight QR
  ─────────────────────────  ────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
   General families           gam.fit5, Sl.setup, gaulss, gammals                           Deep but narrow family coverage
  ─────────────────────────  ────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
   Smoothing selection        GCV, UBRE/AIC, ML, REML, LAML where supported; Newton,        Strong core, incomplete method breadth
                              BFGS, EFS
  ─────────────────────────  ────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
   Extended families          Negative binomial, betar, ocat, tw                            Substantial recent progress
  ─────────────────────────  ────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
   Prediction                 Link, response, terms, iterms, lpmatrix, SEs, offsets,        Strong declared surface
                              newdata
  ─────────────────────────  ────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
   Inference                  EDF, covariance variants, summary, ANOVA, likelihood          Strong and unusually complete
                              criteria, residuals
  ─────────────────────────  ────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
   Diagnostics                Concurvity, k.check, data-producing gam_check, plotting       Good; graphical ecosystem incomplete
                              preparation
  ─────────────────────────  ────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
   SCAM                       Broad univariate and bivariate constructors, transformed      Constructor coverage is excellent; fitting breadth remains
                              fitting, GCV/UBRE BFGS                                        narrower than upstream
  ─────────────────────────  ────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────
   Large data/mixed models    No bam, gamm, jagam                                           Not integrated

  ## What has been integrated especially well

  ### 1. Smooth construction and penalty semantics

  NAMpy implements the central bases most users actually encounter:

  - Cubic regression and shrinkage: cr, cs
  - Cyclic cubic: cc
  - P-spline: ps
  - Thin-plate regression and shrinkage: tp, ts
  - Random effects and factor smooths: re, fs, sz
  - Tensor products: te, ti

  This follows upstream routines such as smooth.construct.*, Predict.matrix.*, and smoothCon in upstreams/mgcv/R/smooth.r.

  Particularly good details include:

  - penalty scaling and ordering;
  - tensor marginal construction order;
  - np=TRUE tensor reparameterization;
  - linked id= bases with pooled covariates;
  - factor and ordered-factor by handling;
  - pc= point constraints;
  - model-level select=TRUE null-space penalties;
  - mathematically appropriate invariant tests where raw eigenvector orientation is not unique.

  The tensor test matrix is extensive, including all ordered pairs of the six supported tensor marginal bases.

  ### 2. Identifiability and gam.side

  The implementation covers much more than simple column centering:

  - sum-to-zero constraints;
  - R-compatible QR absorption;
  - nested and repeated smooth dependence;
  - penalty-aware side-condition decisions;
  - parametric-span checks;
  - exemptions for random-effect and factor smooths;
  - zero-width terms;
  - point constraints.

  These map to upstream gam.side, smoothCon, augment.smX, and related setup code in upstreams/mgcv/R/mgcv.r.

  This is important because many Python GAM libraries implement bases but not the exact identifiability model. NAMpy has gone considerably further.

  ### 3. The fitting engine

  The ordinary fitting route is a substantive port of gam.fit3 rather than a generic penalized IRLS solver:

  - upstream-style PIRLS state transitions;
  - current-SP reparameterization;
  - pivoted QR rank handling and zero-fill restoration;
  - noncanonical-link Newton/Fisher branching;
  - negative or signed working-weight handling;
  - gdi1-style derivative accumulation;
  - upstream rank tolerance;
  - Gaussian stacked-QR post-fit overwrite.

  The relevant upstreams are upstreams/mgcv/R/gam.fit3.r and upstreams/mgcv/src/gdi.c.

  The general-family route also includes meaningful ports of:

  - gam.fit5;
  - Sl.setup;
  - Sl.repara;
  - ldetS;
  - gam.fit5.post.proc;
  - gamlss.etamu, gamlss.gH, and trind.generator.

  That is technically difficult territory and is well beyond basic GAM support.

  ### 4. Smoothing-parameter selection

  The following are integrated:

  - GCV;
  - UBRE/Cp/AIC where scale semantics permit;
  - ML and REML;
  - LAML for supported general families;
  - exact outer Newton;
  - upstream BFGS;
  - EFS;
  - initial smoothing parameter logic;
  - joint estimation of smoothing parameters with scale or family parameters.

  Joint outer problems include:

  - Gaussian scale;
  - Gamma scale;
  - negative-binomial theta;
  - beta precision;
  - ordered-category cut points;
  - Tweedie power and scale.

  The strict Newton, BFGS, and EFS traces are compared against upstream behavior, not merely final fitted values.

  ### 5. Families

  Well-covered families currently include:

  - Gaussian: identity, log, inverse
  - Binomial: logit, probit, cloglog, cauchit, log
  - Poisson: log, identity, square root
  - Gamma: log, identity, inverse
  - Negative binomial: fixed and estimated theta
  - betar: fixed and estimated precision
  - ocat: fixed and estimated cut points
  - tw: fixed and jointly optimized power/scale
  - gaulss
  - gammals

  The Tweedie implementation is especially deep: it ports the series calculations from the upstream C routines, not merely a generic SciPy likelihood:
  nampy/gam/families/tweedie.py.

  ### 6. Post-fit behavior

  NAMpy has integrated a surprisingly broad portion of the fitted gam object contract:

  - Bayesian Vp;
  - frequentist Ve;
  - smoothing-uncertainty corrected Vc;
  - EDF, EDF1, EDF2;
  - sp.vcov;
  - gam.vcomp;
  - log likelihood, AIC and BIC;
  - single- and multi-model ANOVA;
  - parametric and smooth significance tests;
  - random-effect mixture tests;
  - deviance, Pearson, working, response, and constrained-model residuals;
  - concurvity;
  - k.check;
  - summary.gam-style output;
  - much of the plot.gam data preparation.

  Prediction covers link, response, terms, iterms and lpmatrix surfaces, including standard errors and formula offsets.

  ### 7. SCAM integration

  The SCAM constructor port is broad:

  - Monotonic, decreasing, convex, concave and combinations
  - Positive, increasing-positive and decreasing-positive smooths
  - Cyclic positive smooths
  - Locally constrained smooths
  - Numeric-by constrained bases
  - All listed upstream bivariate constrained classes

  NAMpy also has:

  - positive coefficient transformations;
  - default exponential and SCAM not.exp-style softplus transformations;
  - fixed-SP constrained Newton fitting;
  - GCV/UBRE values and gradients;
  - bfgs_gcv.ubre;
  - constrained covariance and EDF;
  - derivative estimation;
  - Gaussian AR(1) with known correlation.

  The source lives under nampy/gam/smooths/shape and nampy/gam/splines/shape, against the vendored upstreams/scam/R.

  ### 8. Fit lifecycle and result ownership

  The major issues from the earlier review have now been addressed:

  - Fresh per-fit session/workspace: nampy/gam/model/session.py:58
  - Transactional refitting
  - Per-fit family cloning
  - Predictor-aware coefficient coordinates: nampy/gam/compiler/structures.py:47
  - Multi-predictor term identity
  - Defensive result snapshots: nampy/gam/fit/result_builders.py:88
  - Schema-versioned persistence

  These are important integrations even though they are Python architecture rather than direct mgcv ports.

  ## What is only partially integrated

  ### Formula language

  The formula layer implements useful R-like semantics but is not an R formula evaluator. Missing or different constructs include:

  - cbind(success, failure) binomial responses;
  - factor(), ordered(), poly(), cut(), interaction();
  - %in%;
  - R’s ^ operator—Python ** is used;
  - ordered-factor parametric contrasts;
  - shared linear-predictor components such as 1 + 2 ~ ....

  The shared-predictor case is correctly guarded because upstream shares one coefficient block; cloning the term would define a different model.

  ### General-family prediction and construction

  gaulss and gammals are strong vertical implementations, but the general-family framework is not yet generally extensible to arbitrary upstream
  families.

  Remaining limitations include:

  - terms= and exclude= filters for multiple linear predictors;
  - non-reparameterized Sl blocks;
  - real smooth constructors emitting nonlinear Sl blocks;
  - broad multi-smooth general-family inference and diagnostic coverage;
  - additional shared coefficient structures.

  ### Optimizer parity

  optim uses SciPy L-BFGS-B rather than the exact R stats::optim implementation. Endpoint behavior is generally constrained, but flat-boundary traces
  can differ.

  Also incomplete:

  - nlm;
  - magic as an identifiable performance-iteration route;
  - exact negative-binomial estimated-theta ML with optim;
  - automatic optimizer coverage for every SCAM selection route.

  ### Plotting and checking

  The numerical/data phase of plot.gam is substantially ported, but R graphics behavior is not:

  - graphics device state;
  - exact layouts;
  - contour legend behavior;
  - vis.gam;
  - derivative plots;
  - qq.gam;
  - fully graphical gam.check.

  That distinction is reasonable for a Python package, but it remains an upstream gap.

  ### AR(1)

  Implemented:

  - known rho;
  - Gaussian identity;
  - fixed smoothing and GCV;
  - independent sections through ar_start.

  Not implemented:

  - estimated correlation;
  - ML/REML/LAML correlated likelihood;
  - non-Gaussian or non-identity models.

  ## What remains absent

  ### Smooth bases and penalty types

  Major missing mgcv smooth constructors include:

  - t2
  - cyclic P-splines cp
  - B-splines bs
  - Duchon splines ds
  - Gaussian-process smooths gp
  - adaptive smooths ad
  - Markov random fields mrf
  - spherical smooths sos
  - soap-film smooths so, sf, sw
  - sc and scad
  - broad matrix covariate/linear-functional support
  - paraPen
  - user-provided constraint matrices and absorb.cons=FALSE

  Of these, t2, mrf, ds, gp, adaptive smooths, and paraPen would probably provide the highest practical value.

  ### Families

  Absent ordinary or quasi families:

  - quasi
  - quasipoisson
  - quasibinomial
  - inverse Gaussian

  Absent extended families:

  - scat
  - ziP
  - cnorm
  - clog
  - cpois
  - bcg

  Absent general or special families:

  - multinom
  - ziplss
  - gevlss
  - twlss
  - gumbls
  - shash
  - cox.ph and cox.pht
  - multivariate normal mvn
  - grouped families gfam

  The vendored qgam functionality is also not integrated into nampy/gam; there is no ELF-based additive quantile regression path.

  ### Selection criteria and fit controls

  Still absent:

  - GACV;
  - NCV and QNCV;
  - selectable P-ML and P-REML;
  - known-scale scale= workflows;
  - nei=;
  - smoothing selection for purely parametric formulas;
  - much of the full gam.control surface.

  ### Large-data, mixed-model and external-model facilities

  Not integrated:

  - bam and bam.update;
  - discrete fitting;
  - chunked prediction/fitting;
  - gamm and its nlme correlation/random-effect integration;
  - jagam;
  - JAGS model generation;
  - gam.mh;
  - ginla;
  - smooth2random as a public mixed-model bridge.

  These are large standalone projects, not small missing functions.

  ### Prediction and auxiliary APIs

  Missing prediction arguments include:

  - block.size;
  - newdata.guaranteed;
  - R-style na.action;
  - direct unconditional=;
  - iterms.type.

  Other missing upstream surfaces include:

  - vis.gam;
  - qq.gam;
  - public influence and Cook’s-distance APIs;
  - gam.sandwich;
  - a functional gam() entry point;
  - simulation helpers such as gamSim.

  ## Recommended order for further integration

  1. Finish gaps inside the currently supported surface:
      - general-family terms=/exclude=;
      - shared linear-predictor components;
      - non-unit-weight binomial AIC;
      - exact guarded optim combinations;
      - parametric-only smoothing-selection behavior.

  2. Add high-value smooths:
      - t2;
      - mrf;
      - ds;
      - gp;
      - ad;
      - paraPen.

  3. Expand families:
      - inverse Gaussian and quasi families;
      - scat and ziP;
      - multinom and ziplss;
      - survival and multivariate families only if they fit NAMpy’s product goals.

  4. Expand selection:
      - known-scale workflows;
      - NCV/QNCV;
      - GACV;
      - broader SCAM ML/REML support.

  5. Treat bam, gamm, jagam, and quantile GAMs as separate milestones. Each changes architecture substantially enough that it should not be folded
     casually into the ordinary GAM implementation.
