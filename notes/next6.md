FULL CODE PARITY (algorithm mirrors upstream)
                                                                               
  ┌────────────────────┬────────────────────────────────┬───────────────────┐ 
  │      Concept       │           NAMpy file           │    mgcv source    │  
  ├────────────────────┼────────────────────────────────┼───────────────────┤ 
  │ PIRLS loop +       │ fit/solvers/irls_core.py       │ gam.fit3          │  
  │ step-halving    │                                  │                  │    
  ├─────────────────┼──────────────────────────────────┼──────────────────┤    
  │ Penalty reparam │ smoothing_selection/reparam.py + │ gam.reparam      │    
  │ eterization     │  pirls_deriv.py                  │                  │    
  ├─────────────────┼──────────────────────────────────┼──────────────────┤    
  │ Constraint      │ constraints/absorption.py        │ absorb.constrain │    
  │ absorption      │                                  │ t                │    
  ├─────────────────┼──────────────────────────────────┼──────────────────┤  
  │ Predictor       │ constraints/identifiability.py   │ gam.fit3 side    │    
  │ identifiability │                                  │ conditions       │    
  ├─────────────────┼──────────────────────────────────┼──────────────────┤  
  │ Tensor product  │ basis/tensor.py                  │ tensor.penalty   │    
  │ penalties       │                                  │                  │  
  ├─────────────────┼──────────────────────────────────┼──────────────────┤  
  │ CR/CS/CC basis  │ smooths/univariate/cubic_regress │ smooth.construct │
  │                │ n.py                               │ t.cr            │    
  ├────────────────┼────────────────────────────────────┼─────────────────┤
  │ TP/TS basis    │ smooths/univariate/thin_plate.py   │ smooth.construc │    
  │                │                                    │ t.tp + tprs.c   │    
  ├────────────────┼────────────────────────────────────┼─────────────────┤
  │ PS basis       │ smooths/univariate/pspline.py      │ smooth.construc │    
  │                │                                    │ t.ps            │    
  ├────────────────┼────────────────────────────────────┼─────────────────┤
  │ Gaussian       │ fit/solvers/gaussian_exact.py      │ magic /         │    
  │ Gaussian solver (QR)  │ fit/solvers/gaussian_exact.py  │ magic /       │   
  │                       │                                │ Gaussian path │
  ├───────────────────────┼────────────────────────────────┼───────────────┤   
  │ Covariance (Vp, Vf,   │ fit/covariance.py              │ vcov.gam      │   
  │ Vc)                   │                                │               │
  ├───────────────────────┼────────────────────────────────┼───────────────┤   
  │                       │                                │ gam.fit3      │   
  │ Exact ML/REML         │ smoothing_selection/criteria/p │ implicit      │
  │ derivatives           │ irls_deriv.py                  │ function      │   
  │                       │                                │ theorem       │
  ├───────────────────────┼────────────────────────────────┼───────────────┤
  │ Summary output        │ diagnostics/summary.py         │ summary.gam   │
  ├───────────────────────┼────────────────────────────────┼───────────────┤
  │ Residual types        │ diagnostics/residuals.py       │ residuals.gam │   
  ├───────────────────────┼────────────────────────────────┼───────────────┤
  │ Predictions (response │ predict/predictions.py         │ predict.gam   │   
  │ /link/terms/lpmatrix) │                                │               │
  └───────────────────────┴────────────────────────────────┴───────────────┘   
                  
  ---
  PARTIAL — concept exists, but incomplete vs upstream
                                                                               
  ┌────────────────────────┬─────────────────────────┬─────────────────────┐
  │        Concept         │           Gap           │   mgcv reference    │   
  ├────────────────────────┼─────────────────────────┼─────────────────────┤
  │ General family PIRLS   │ 40% done — framework    │ gam.fit5            │
  │ (general_fit5.py)      │ only, not integrated    │                     │
  ├────────────────────────┼─────────────────────────┼─────────────────────┤   
  │ Tensor ti with        │ Margin centering not   │ smooth.construct.ti  │ 
  │ mc=TRUE               │ aligned                │                      │    
  ├───────────────────────┼────────────────────────┼──────────────────────┤    
  │ Tensor t2 with        │ Penalty                │                      │ 
  │ full=FALSE           │ parameterization        │ smooth.construct.t2  │    
  │                      │ differs                 │                      │ 
  ├──────────────────────┼─────────────────────────┼──────────────────────┤    
  │ Negbin theta joint   │ ~0.2 log-unit gap in    │ negbin extended      │ 
  │ REML                 │ theta; outer loop       │ family               │    
  │                      │ differs                 │                      │ 
  ├──────────────────────┼─────────────────────────┼──────────────────────┤    
  │ Factor smooth sz     │ Penalty differs from    │ smooth.construct.fs  │ 
  │                      │ mgcv                    │                      │    
  ├──────────────────────┼─────────────────────────┼──────────────────────┤ 
  │                      │ Works but API differs,  │                      │    
  │ MRF smooth           │ not mgcv-compatible     │ smooth.construct.mrf │    
  │                      │ spec                    │                      │
  ├──────────────────────┼─────────────────────────┼──────────────────────┤    
  │ NCV / QNCV criteria  │ Stubs in criteria/ — no │ gam.fit3 NCV path    │
  │                      │  actual implementation  │                      │    
  ├──────────────────────┼─────────────────────────┼──────────────────────┤
  │ Laplace              │ Stubs in                │ gam.fit3 Laplace     │    
  │ approximation        │ criteria/laplace.py     │ path                 │
  │                      │ only                    │                      │
  └──────────────────────┴─────────────────────────┴──────────────────────┘

  ---
  MISSING — zero implementation, reference code only in vendored mgcv
                                                                               
  ┌────────────────────────────────────┬────────────────────────┬─────────┐ 
  │              Concept               │  mgcv basis/function   │ Effort  │    
  ├────────────────────────────────────┼────────────────────────┼─────────┤ 
  │ magic() C adaptive step-size       │ magic.c                │ Medium  │    
  ├────────────────────────────────────┼────────────────────────┼─────────┤ 
  │ Shape-constrained smooths (sc,     │ smooth.construct.sc    │ High    │    
  │ scad, sos, ds)                     │ etc.                   │         │    
  ├────────────────────────────────────┼────────────────────────┼─────────┤    
  │ SOAP manifold smoothing (so, sf,   │ soap.c                 │ Very    │    
  │ sw)                                │                        │ High    │    
  ├────────────────────────────────────┼────────────────────────┼─────────┤
  │ Adaptive splines (ad)              │ smooth.construct.ad    │ High    │    
  ├────────────────────────────────────┼────────────────────────┼─────────┤    
  │ Cyclic P-spline (cp)               │ smooth.construct.cp    │ Low     │
  ├────────────────────────────────────┼────────────────────────┼─────────┤    
  │ gamm() mixed effects path          │ gamm.r                 │ High    │
  ├────────────────────────────────────┼────────────────────────┼─────────┤    
  │ bam() big-data path                │ bam.r                  │ Very    │
  │                                    │                        │ High    │    
  └────────────────────────────────────┴────────────────────────┴─────────┘
                                                                               
  ---             
  PRIORITY ORDER for completion (concepts, not new features)
                                                                               
  1. general_fit5.py — GAMLSS PIRLS is 40% done; complete it or remove stubs.
  This is gam.fit5 equivalent. Core concept, not new feature.                  
  2. ti with mc=TRUE and t2 with full=FALSE — penalty lifting incomplete. Small
   targeted fix.                                                               
  3. Negbin theta outer REML — locate exact difference vs mgcv outer loop in
  negbin extended family source.                                               
  4. NCV/QNCV criteria — stubs exist, upstream logic in gam.fit3 NCV section.
  Port or delete stubs.                                                        
  5. cp basis — simplest missing basis (cyclic P-spline). Low effort, completes
   the basis set.                                                              
  6. magic() step-size — not ported from C. Affects optimizer robustness on
  hard problems.                                                               
                  
  ---                                                                          
  Bottom line: Core pipeline (~90% of use) has good code-level parity. Main
  incomplete concepts are general_fit5 (GAMLSS PIRLS), tensor edge cases, and  
  NCV criteria — all partially started, none finished. Recommended: complete or
   cleanly remove partial stubs before adding anything new.