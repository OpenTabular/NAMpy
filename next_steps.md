## Remaining gaps

### GeneralFamily (multi-predictor) — not yet implemented

| mgcv name | Predictors                              | Notes                       |
|-----------|-----------------------------------------|-----------------------------|
| twlss     | 3 (mean, log-scale, logit-power)        | Tweedie LS                  |
| multinom  | k-1 (log-odds)                          | Multinomial logistic        |
| mvn       | p (multivariate normal)                 | Shared covariance           |

**Implemented:** gaulss, gammals, ziplss, gevlss, shashlss, gam_fit5 backend, multi-predictor assembly.

---

### ExtendedFamily (single-predictor, LAML) — none yet implemented

| mgcv name | Notes                                               |
|-----------|-----------------------------------------------------|
| tw        | Tweedie with estimated power via profile lik        |
| scat      | Scaled t — heavy tails                              |
| ocat      | Ordered categorical with threshold estimation       |
| cnorm     | Censored normal                                     |
| ziP       | Zero-inflated Poisson (simpler, single linear pred) |
| cox.ph    | Cox PH — no intercept, special partial lik          |

---

### Inference gaps

| Feature                        | Notes                                |
|--------------------------------|--------------------------------------|
| predict(type="iterms")         | Per-term contribution with SE — type="terms" + return_se exists but "iterms" not a distinct type |
| predict(exclude=...)           | Drop named terms from prediction     |
| Posterior simulation (rmvn)    | Draw from multivariate posterior     |
| Simultaneous credible bands    | Not just pointwise                   |
| influence.gam                  | Cook's distance style                |

**Implemented:** predict(type="terms"), predict(type="link"), predict(type="response"), se.fit/return_se on all types, vcov, logLik, AIC, BIC, gam_check (k-check + Q-Q), concurvity.

---

### Priority order

1. tw — most commonly used extended family
2. scat, ocat — frequently needed
3. twlss — completes *lss family set
4. predict(exclude=...) + type="iterms" — low-effort inference completeness
5. Posterior simulation (rmvn) — needed for simultaneous bands
6. multinom — needs k-1 predictor generalization
7. mvn — shared covariance, niche
8. cox.ph — special partial likelihood, separate backend needed
