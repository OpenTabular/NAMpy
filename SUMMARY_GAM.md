Now I have a complete and verified picture. Here is the full gap
  analysis:

  ---
  What's Implemented vs. What Remains

  A. Smooth Basis Types

  ┌────────────────────┬──────────────┬─────────────────────────┐
  │       Basis        │    Status    │          Notes          │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ cr cubic           │ Done         │ Full support            │
  │ regression         │              │                         │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ cs cubic           │              │                         │
  │ regression +       │ Done         │ Full support            │
  │ shrinkage          │              │                         │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ cc cyclic cubic    │ Partial      │ pc= (point constraint)  │
  │                    │              │ not supported for cc    │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ ps P-spline        │ Partial      │ pc= not supported; pc=  │
  │                    │              │ works for cr only       │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ tp thin-plate      │ Done         │ Multivariate supported  │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ ts thin-plate +    │ Done         │ Full support            │
  │ shrinkage          │              │                         │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ gp Gaussian        │ Partial      │ pc= not supported       │
  │ process            │              │                         │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ re random effect   │ Done         │ Full support            │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ mrf Markov Random  │ Done         │ Full support            │
  │ Field              │              │                         │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ fs factor smooth   │ Partial      │ pc= for factor-by not   │
  │                    │              │ supported               │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ sz smooth-varying  │ Done         │ Full support            │
  │ coefficient        │              │                         │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │                    │              │ Marginals restricted to │
  │ te tensor product  │ Partial      │  cr/cs/cc only (no ps,  │
  │                    │              │ tp, gp)                 │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ ti tensor          │ Partial      │ Same marginal           │
  │ interaction        │              │ restriction as te       │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │                    │              │ Same marginal           │
  │ t2 tensor ANOVA    │ Partial      │ restriction; no by=     │
  │                    │              │ support                 │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ cp cyclic P-spline │ Not          │                         │
  │                    │ implemented  │                         │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ bs raw B-spline    │ Not          │                         │
  │                    │ implemented  │                         │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ ds Duchon splines  │ Not          │                         │
  │                    │ implemented  │                         │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ sos splines on     │ Not          │                         │
  │ sphere             │ implemented  │                         │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ ad adaptive        │ Not          │                         │
  │                    │ implemented  │                         │
  ├────────────────────┼──────────────┼─────────────────────────┤
  │ so/sf soap film    │ Not          │                         │
  │                    │ implemented  │                         │
  └────────────────────┴──────────────┴─────────────────────────┘

  Key restriction: Tensor product marginals (te, ti, t2) only accept
  cr, cs, cc. mgcv supports any combination including ps, tp, gp,
  etc.

  ---
  B. Distribution Families

  ┌────────────────────────┬─────────────┬──────────────────────┐
  │         Family         │   Status    │        Notes         │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ gaussian               │ Done        │ Closed-form + PIRLS, │
  │                        │             │  GCV/REML/ML         │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ binomial               │ Done        │ PIRLS, exact REML/ML │
  │                        │             │  derivatives         │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ poisson                │ Done        │ PIRLS, exact REML/ML │
  │                        │             │  derivatives         │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ gamma                  │ Done        │ PIRLS, exact REML/ML │
  │                        │             │  derivatives         │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ negbin (negative       │ Partial     │ Fixed θ only — no θ  │
  │ binomial)              │             │ estimation           │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ inverse.gaussian       │ Not         │                      │
  │                        │ implemented │                      │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ quasi, quasibinomial,  │ Not         │                      │
  │ quasipoisson           │ implemented │                      │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ Tweedie                │ Not         │                      │
  │                        │ implemented │                      │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ ocat (ordered          │ Not         │                      │
  │ categorical)           │ implemented │                      │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ betar (beta            │ Not         │                      │
  │ regression)            │ implemented │                      │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ scat (scaled t)        │ Not         │                      │
  │                        │ implemented │                      │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ GAMLSS families        │ Not         │ Multi-predictor      │
  │ (gaulss, gevlss,       │ implemented │ architecture not     │
  │ gammalss, …)           │             │ present              │
  ├────────────────────────┼─────────────┼──────────────────────┤
  │ multinom (multinomial) │ Not         │                      │
  │                        │ implemented │                      │
  └────────────────────────┴─────────────┴──────────────────────┘

  ---
  C. Smoothing Parameter Methods

  ┌───────────────┬────────────┬─────────────────────────────────┐
  │    Method     │   Status   │              Notes              │
  ├───────────────┼────────────┼─────────────────────────────────┤
  │               │ Done       │ GCV for Gaussian fully          │
  │ GCV.Cp        │ (Gaussian) │ implemented; GLM GCV criterion  │
  │               │  / Partial │ exists but less tested          │
  │               │  (GLM)     │                                 │
  ├───────────────┼────────────┼─────────────────────────────────┤
  │               │            │ Exact REML for Gaussian;        │
  │ REML          │ Done       │ Laplace-approximate REML for    │
  │               │            │ GLMs                            │
  ├───────────────┼────────────┼─────────────────────────────────┤
  │ ML            │ Done       │ Same coverage as REML           │
  ├───────────────┼────────────┼─────────────────────────────────┤
  │               │            │ Infrastructure (pirls_reml_deri │
  │ P-REML / P-ML │ Partial    │ vative_blocks.py) exists;       │
  │               │            │ limited testing                 │
  ├───────────────┼────────────┼─────────────────────────────────┤
  │ GACV.Cp       │ Not implem │                                 │
  │               │ ented      │                                 │
  ├───────────────┼────────────┼─────────────────────────────────┤
  │ REML with     │ Not implem │ Needed for bam()                │
  │ discrete=TRUE │ ented      │                                 │
  ├───────────────┼────────────┼─────────────────────────────────┤
  │ bam()-specifi │            │                                 │
  │ c methods     │ Not implem │                                 │
  │ (fREML,       │ ented      │                                 │
  │ bam-GCV)      │            │                                 │
  ├───────────────┼────────────┼─────────────────────────────────┤
  │ Fixed         │            │                                 │
  │ smoothing     │ Done       │ Full support                    │
  │ (fx=TRUE,     │            │                                 │
  │ sp=)          │            │                                 │
  └───────────────┴────────────┴─────────────────────────────────┘

  ---
  D. Formula / Model Features

  ┌───────────────┬───────────────┬─────────────────────────────┐
  │    Feature    │    Status     │            Notes            │
  ├───────────────┼───────────────┼─────────────────────────────┤
  │ Numeric by=   │ Done          │ Full support                │
  │ smooths       │               │                             │
  ├───────────────┼───────────────┼─────────────────────────────┤
  │ Factor by=    │ Done          │ Full support                │
  │ smooths       │               │                             │
  ├───────────────┼───────────────┼─────────────────────────────┤
  │ Linked basis  │               │ cr/cs basis sharing done;   │
  │ id=           │ Partial       │ k mismatches harmonized to  │
  │               │               │ max k (matches mgcv)        │
  ├───────────────┼───────────────┼─────────────────────────────┤
  │ offset(...)   │ Done          │ Full support                │
  │ terms         │               │                             │
  ├───────────────┼───────────────┼─────────────────────────────┤
  │ Prior/case    │               │ prior_weights_ attribute    │
  │ weights       │ Not           │ exists but weights are not  │
  │ (weights=)    │ implemented   │ plumbed through the formula │
  │               │               │  API                        │
  ├───────────────┼───────────────┼─────────────────────────────┤
  │ Fixed fx=TRUE │ Done          │                             │
  │  per term     │               │                             │
  ├───────────────┼───────────────┼─────────────────────────────┤
  │ sp= supplied  │ Done          │                             │
  │ smoothing     │               │                             │
  ├───────────────┼───────────────┼─────────────────────────────┤
  │ Point         │               │ Only cr/cs supported; cc,   │
  │ constraints   │ Partial       │ ps, gp, fs, tensor:         │
  │ pc=           │               │ NotImplementedError         │
  ├───────────────┼───────────────┼─────────────────────────────┤
  │ select=TRUE   │               │ Only univariate bases;      │
  │ (selection    │ Partial       │ raises NotImplementedError  │
  │ penalty)      │               │ for te, ti, t2, re, fs, sz, │
  │               │               │  mrf                        │
  ├───────────────┼───────────────┼─────────────────────────────┤
  │ pc= for       │ Not           │                             │
  │ tensor terms  │ implemented   │                             │
  ├───────────────┼───────────────┼─────────────────────────────┤
  │ m= penalty    │ Done for ps;  │                             │
  │ order         │ Partial for   │                             │
  │               │ gp            │                             │
  └───────────────┴───────────────┴─────────────────────────────┘

  ---
  E. Prediction and Inference

  Feature: predict(type="response")
  Status: Done
  Notes:
  ────────────────────────────────────────
  Feature: predict(type="link")
  Status: Done
  Notes:
  ────────────────────────────────────────
  Feature: predict(type="terms")
  Status: Done
  Notes: Via predict_feature_vals()
  ────────────────────────────────────────
  Feature: predict(newdata=...)
  Status: Done
  Notes:
  ────────────────────────────────────────
  Feature: predict(se.fit=TRUE)
  Status: Partial
  Notes: lpmatrix() method exposed; standard errors not computed
    directly in predict API
  ────────────────────────────────────────
  Feature: Confidence intervals
  Status: Not implemented
  Notes: Need Vp + lpmatrix wiring in predict API
  ────────────────────────────────────────
  Feature: EDF per term
  Status: Done
  Notes:
  ────────────────────────────────────────
  Feature: Bayesian covariance Vp
  Status: Done
  Notes:
  ────────────────────────────────────────
  Feature: Frequentist covariance Vf
  Status: Done
  Notes: Available as gam.Vf_; not exposed in predict
  ────────────────────────────────────────
  Feature: p-values for smooth terms
  Status: Not implemented
  Notes: Requires Wood's reference distribution machinery
  ────────────────────────────────────────
  Feature: summary.gam()
  Status: Partial
  Notes: print_summary() exists; no p-value table, no parametric term

    tests
  ────────────────────────────────────────
  Feature: plot.gam()
  Status: Partial
  Notes: plot() method exists; limited vs. mgcv
  ────────────────────────────────────────
  Feature: gam.check()
  Status: Not implemented
  Notes: No basis-adequacy / residual diagnostics
  ────────────────────────────────────────
  Feature: concurvity()
  Status: Not implemented
  Notes:
  ────────────────────────────────────────
  Feature: anova.gam() / model comparison
  Status: Not implemented
  Notes:
  ────────────────────────────────────────
  Feature: Smoothing-param uncertainty correction (Wood 2016)
  Status: Not implemented
  Notes: Corrected covariance Vc not computed

  ---
  F. Advanced / Specialized

  ┌───────────────────────────────────────────┬─────────────────┐
  │                  Feature                  │     Status      │
  ├───────────────────────────────────────────┼─────────────────┤
  │ bam() (big-data GAM)                      │ Not implemented │
  ├───────────────────────────────────────────┼─────────────────┤
  │ gamm() (GAM mixed models via nlme/lme4)   │ Not implemented │
  ├───────────────────────────────────────────┼─────────────────┤
  │ jagam() (JAGS/Bayesian interface)         │ Not implemented │
  ├───────────────────────────────────────────┼─────────────────┤
  │ Negative binomial θ estimation            │ Not implemented │
  ├───────────────────────────────────────────┼─────────────────┤
  │ Tweedie p estimation                      │ Not implemented │
  ├───────────────────────────────────────────┼─────────────────┤
  │ Multi-predictor / distributional (GAMLSS) │ Not implemented │
  ├───────────────────────────────────────────┼─────────────────┤
  │ Nested / hierarchical smooths             │ Not implemented │
  └───────────────────────────────────────────┴─────────────────┘

  ---
  Summary by Priority

  Straightforward to add (infrastructure exists):
  - pc= for ps, cc, gp, fs — code is stubbed with
  NotImplementedError, needs the point-constraint absorption path
  connected
  - select=TRUE for tensor terms — architecture supports it but
  explicitly blocked
  - Prior/case weights via formula — prior_weights_ already wired
  through fit
  - Standard errors in predict API — lpmatrix() + Vp_ already
  available; just need to wire
  - Confidence intervals — follow from SE
  - Negative binomial θ estimation — P-IRLS update step

  Medium complexity:
  - Tensor marginals beyond cr/cs/cc (ps, tp, gp) — requires
  extending te/ti/t2 to accept any RuntimeTerm as marginal           
  - p-values for smooth terms — Wood's reference distribution; needs
  Vp, EDF, and a χ²/F-test wrapper                                   
  - summary.gam() with p-values — depends on p-values above          
  - gam.check() diagnostics — residual checks + k-index tests
  - Cyclic P-spline (cp basis)                                       
                                                                     
  Large scope:                                                       
  - bam() big-data path — completely separate fitting algorithm      
  - gamm() — requires integration with a mixed-model library         
  - GAMLSS / multi-predictor families — different model class, not
  the single-predictor GAM                                           
  - Remaining exotic bases (ds, sos, ad, so) — each needs its own    
  basis primitives 