Structure is in good shape now. The layering is clean and consistent: 12 flat architecture files + components/, thin models/ wrappers (each model file ~25 lines wiring architecture + config into shared Regressor/Classifier/LSS bases — this is the best-designed part of the neural side), config-per-model uniform after the de-sharing, GAM subsystem deep and ledgered. Ownership rules hold. What remains, roughly in order of importance:

1. Neural side has no numerical validation against reference implementations. GAM side is machine-precision-vs-mgcv; neural side has smoke + contract tests only. Nothing checks NodeGAM against the official nodegam package, or NAM against google-research/nam outputs. Users will assume paper fidelity. Either add one seeded-comparison test per adapted model or state "reimplementations, not verified ports" in the README.
2. additive_trees.py is still the roughest file (~700 lines): upstream-adapted style with side-channel attribute passing (self.prev_feature_selectors, self.feature_selectors set in get_feature_selection_values and read in forward), bare asserts for user errors, a print() warning in convert_onehot_vector_to_integers (should be warnings.warn), and the x[0]*1e10 + x[1]*1e5 sort-key hack in build_terms_frame.
3. extract_additive_terms is unreachable from the public API — the whole extraction machinery we just cleaned exists, but NodeGAMRegressor doesn't expose it and it doesn't feed nampy/plotting/. Wire it up or it's dead weight.
4. Two live TODOs (natt.py:70, namformer.py:63): their feature_dropout is elementwise dropout, semantically different from interactions.apply_feature_dropout used by NAM/QNAM/TreeNAM. Same kwarg name, different meaning — worth unifying or renaming.
5. Old findings check: RUNTIME_TERM_INTERFACE_CHECKLIST still defined-but-unenforced (smooth_base.py:303); smooth_base.py still 870 lines carrying non-base helpers. ByVariableState alias and module-level matplotlib import are fixed/gone.
6. Per project ledger, GAM's plot() is still not a plot.gam port, and the parametric-column aliasing side-condition gap is open.

What to add — research findings

Neural models that fit your modules/ + components/ pattern:

- ProtoNAM (ACM TKDD 2025) — prototype-based shape functions, current SOTA-ish NAM variant (paper, journal)
- SIAN — Sparse Interaction Additive Networks — feature-interaction detection + sparse selection; slots directly into your existing interaction_degree machinery (paper)
- IGANN — boosted ELM-based additive nets, popular in the IS community, smooth shape functions (paper, IGANN Sparse)
- GAMformer (NeurIPS 2024, Mueller/Nori/Caruana/Hutter) — TabPFN-style in-context GAM estimation in one forward pass; would be a headline differentiator but needs pretrained weights (arxiv, OpenReview)
- Newer/optional: Regionally Additive Models, Neural Additive Experts (2026, context-gated additivity control)

Bigger win — a new task flavor, not a new architecture: survival. PyGAM and interpretml both lack survival support — documented gap. CoxNAM, CRISP-NAM (competing risks, 2025), and DNAMite all show demand; dnamite is the only package there. A <Model>Survival flavor beside Regressor/Classifier/LSS reuses your whole stack. Pairs perfectly with cox.ph family in the GAM backend — mgcv has it, so you get survival on both backends symmetrically.

GAM backend (mgcv-parity) candidates, from the mgcv surface you haven't ported:

1. More families — tw (Tweedie), betar, ocat, ziplss, scat, shash, cox.ph. Cheapest wins; your family/fit plumbing already handles extended + general families.
2. bam() — big-data GAM with discretized covariates; the single most-used mgcv feature you lack, and the main reason people can't leave R.
3. Shape constraints — monotone smooths via pcls/mono.con (or scam-style); no Python equivalent exists.
4. qgam-style quantile GAMs — separate R package but same machinery, big applied audience.
5. gamm() last — nlme dependency makes it the hardest port.

My priority order: survival flavor (CoxNAM + cox.ph) → Tweedie/betar/ocat families → SIAN or ProtoNAM → bam(). Survival exploits your unique position — nobody else has both a strict mgcv port and a NAM zoo in one package; every alternative covers only one side.

Sources: dnamite paper (Python GAM landscape), NAM overview, ProtoNAM, GAMformer, CRISP-NAM, CoxNAM, DNAMite, NAMLSS, SIAN, IGANN, mgcv reference, GP-NAM, Hierarchical NAM forecasting