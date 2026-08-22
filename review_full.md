## Executive assessment

  Of the 31 repositories under upstreams, NAMpy currently has:

  - 2 deep behavioral ports: mgcv, scam
  - 3 strong algorithmic integrations: GP-NAM, IGANN, SIAN
  - 5 partial/core-component integrations: Google NAM, NODE-GAM, NODE, entmax, NBM-SPAM
  - 3 historical or secondary NAM references: Nick Frosst NAM, AMR-NAM, anfreth-NAMpy
  - 1 comparison mirror: cran-mgcv
  - 17 reference-only repositories with no substantive implementation

  The best work is concentrated in the mgcv/SCAM ports and the GP-NAM, IGANN, and SIAN implementations. The biggest weakness is that the repository’s
  documentation and generated model API can make partial integrations and NAMpy-native extensions look more upstream-faithful than they are.

  The maintained UPSTREAM_LEDGER.md is a good start, but it is not fully consistent with the code or THIRD_PARTY_NOTICES.md.

  ## Complete upstream matrix

  ### Deep behavioral ports

   Upstream                 mgcv
   What is integrated well  The deepest integration. Formula/spec compilation, smooth construction, tensor products, identifiability, smoothing-
                            parameter selection, multiple families, prediction, inference, and diagnostics follow vendored R/C control flow for the
                            supported subset. Parity tests are extensive and appropriately distinguish behavioral parity from arbitrary eigenvector
                            orientation.
   What remains             Many mgcv surfaces remain absent: t2, cp, bs, ds, gp, adaptive, MRF, soap-film and spherical smooths; paraPen; substantial
                            family breadth; GACV, NCV and PML; and high-level facilities such as bam, gamm and jagam. The current scope is documented
                            in GAM_IMPLEMENTED.md and GAM_NOT_IMPLEMENTED.md.
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Upstream                 scam
   What is integrated well  A serious port of SCAM 1.2-22: released univariate and bivariate constrained bases, prediction matrices, constrained Newton
                            fitting, coefficient transforms, GCV/UBRE selection, inference and Gaussian-identity AR(1). This is much more than
                            “inspired by SCAM.”
   What remains             It is not the whole package. Upstream EFS and alternative optimizer paths are incomplete, as are richer checking/
                            visualization methods and broader AR(1)/non-Gaussian combinations. These should remain explicitly unsupported rather than
                            approximated.

  ### Strong algorithmic integrations

   Upstream                 gpnam
   What is integrated well  Paper-aligned inverse-normal frequency grids, phase grids, automatic bandwidth selection, conjugate-gradient fixed-design
                            fitting, and GP-NA2M pairwise terms. Tests cover equations and seeded reference behavior. NAMpy correctly follows the paper
                            where the released implementation appears to have phase artifacts.
   What remains             No general GP posterior covariance or uncertainty surface. GPNAMLSS is a useful NAMpy extension, but it is not upstream GP-
                            NAM and should be labeled as such everywhere.
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Upstream                 igann
   What is integrated well  Native L1 initialization, feature-wise ELM construction, masked random matrices, ridge boosting of pseudo-responses,
                            categorical reference coding, and optional ABESS-based sparse selection. Direct equation/seeded-draw tests are a strong
                            choice.
   What remains             The upstream interactive approximation is not ported. Regression and binary classification are upstream-aligned; multiclass
                            and LSS variants are NAMpy extensions. Pairwise interaction support remains absent.
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Upstream                 sian
   What is integrated well  Archipelago/FIS interaction detection, fractional-heredity candidate construction, hierarchical interaction handling,
                            block-masked networks, model compression and checkpoint round-trips. This has unusually good direct upstream equivalence
                            tests.
   What remains             Upstream’s training/checkpoint code is itself incomplete in places, so NAMpy necessarily uses shared infrastructure. That
                            boundary should remain documented. SIAN’s upstream repository appears to lack a license, which is a provenance risk
                            requiring review.

  ### Substantial or component-level integrations

   Upstream               What is integrated well                                         What remains
  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   google-research NAM    ExU and centered-ReLU feature networks, adaptive hidden         This is not an exact TensorFlow training-loop port and there
                          widths, output/L2 penalties and balancing concepts are          is no original-compatibility mode. Direct end-to-end
                          incorporated into the shared NAM implementation.                numerical comparison is much weaker than for GP-NAM, IGANN
                                                                                          and SIAN.
  ─────────────────────  ──────────────────────────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   nodegam                Core GAM/GA2M oblivious-tree ideas, selectors, additive term    No full reproduction of upstream training schedules,
                          extraction and purification are represented in NodeGAM.         checkpoint averaging or complete experimental interface.
                          Shared persistence and reconstruction are useful                Tests mostly establish local invariants rather than complete
                          improvements.                                                   upstream numerical parity.
  ─────────────────────  ──────────────────────────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   node                   The ODST and dense oblivious-tree block concepts are adapted    This is not a full NODE package port. Optimizers, data
                          into reusable components. This is good under-abstraction:       pipelines, training recipes and benchmark behavior are
                          the useful primitive was extracted without copying the          outside the integration.
                          entire repository.
  ─────────────────────  ──────────────────────────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   entmax                 Sparsemax, entmax-1.5, entmoid-style activations and            The generic-alpha/bisection family, upstream losses and the
                          temperature behavior are reusable building blocks for the       rest of the entmax package API are not ported. This should
                          tree models.                                                    be described as adapted activation functions, not entmax
                                                                                          integration as a whole.
  ─────────────────────  ──────────────────────────────────────────────────────────────  ──────────────────────────────────────────────────────────────
   nbm-spam               nampy/neural/architectures/nbm.py implements the shared-        This is under-audited. NAMpy changes important
                          basis, n-ary concept architecture and tuple-based term          parameterization details: term weights, normalization,
                          construction.                                                   classifier bias/global intercept handling and optional GLU/
                                                                                          skip behavior. There are no direct upstream numerical tests.
                                                                                          ConceptNBMNarySparse, SPAM polynomial/tensor penalties,
                                                                                          proximal fitting, and the NBM-SPAM hybrid are absent. The
                                                                                          ledger should say “partial integration,” not merely “audit
                                                                                          pending.”

  ### Historical and secondary NAM lineage

   Upstream                             Current relationship                                     What remains
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   nickfrosst-neural_additive_models    Audited as an overlapping early NAM implementation.      If retained, document exactly which conventions were
                                        No dedicated compatibility mode.                         adopted and which were superseded by Google NAM/
                                                                                                 shared NAMpy behavior.
  ───────────────────────────────────  ───────────────────────────────────────────────────────  ───────────────────────────────────────────────────────
   amr-nam                              Secondary API/default reference. Shared NAM              No separate faithful AMR-NAM surface or reference
                                        capabilities incorporate some of its ideas               suite.
                                        compositionally.
  ───────────────────────────────────  ───────────────────────────────────────────────────────  ───────────────────────────────────────────────────────
   anfreth-nampy                        Historical ancestor for formula-driven NAM/NAMLSS/       A migration/provenance audit is needed. There is no
                                        NATT concepts, preprocessing, distributional models      compatibility test suite. The old repository’s SNAM
                                        and plotting. The present package is a broad PyTorch     meant structural/spline NAM, while current SNAM means
                                        redesign rather than a drop-in continuation.             sparse group-lasso NAM; that naming collision is
                                                                                                 particularly hazardous.

  ### Source comparison mirror

   Upstream     Current relationship                                                 What remains
  ━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   cran-mgcv    Used for source/version comparison. The separately vendored          Establish one documented canonical source. cran-mgcv is in the
                upstreams/mgcv is the actual behavioral specification.               manifest while upstreams/mgcv is tracked separately, which
                                                                                     weakens automated provenance verification.

  ### Reference-only or not substantively integrated

   Upstream                   Current state      Missing capability
  ━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   protonam                   Not integrated.    Prototype activations and hierarchical/layerwise prototype explanations.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   ticl                       Not integrated.    GAMformer/MotherNet-style pretrained in-context additive modeling, priors, checkpoint loading and
                                                 extracted binned GAMs. Current nampy/neural/architectures/namformer.py is an ordinary trained
                                                 transformer with additive heads; it is not upstream GAMformer.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   effector                   Not integrated.    Model-agnostic PDP, derivative PDP, ALE, RHALE, SHAP-DP, heterogeneity and regional explanations.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   regional-rhale             Not integrated.    Regional partitioning and heterogeneous feature-effect discovery.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   nae                        Not integrated.    Context-gated mixtures of additive experts and the associated additivity controls.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   coxse                      Not integrated.    Cox survival objectives, CoxSE/CoxSENAM/CoxNAM and survival-specific explanation behavior.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   crisp-nam                  Not integrated.    Competing-risk NAMs, cause-specific hazards, cumulative incidence, IPCW and time-dependent
                                                 evaluation.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   dnamite                    Not integrated.    Discrete/bin-based additive survival models, pseudo-values, IPCW, ranked probability and Cox losses.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   hnam-demand-forecasting    Not integrated.    Hierarchical/grouped, time-varying demand forecasting. The cloned default branch does not contain the
                                                 principal HNAM implementation; it refers to another fork/branch, so the vendored source is
                                                 insufficient for a faithful port.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   pygam                      Reference only.    No direct API or implementation integration. It may offer sklearn-style usability ideas, but must not
                                                 become the numerical specification for nampy.gam; that role belongs to mgcv.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   interpret                  Not integrated.    EBM bagged boosting, automatic interaction selection, model editing and the local/global explanation
                                                 ecosystem.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   qgam                       Not integrated.    ELF likelihood, calibrated learning rates, qgam/mqgam, and tuneLearn. Current QNAM is a neural
                                                 pinball/monotone-quantile model and should not be presented as a qgam port.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   nam-fs                     Not integrated.    NAM-FS feature selection and its NA2M/NB2M selection procedures. Current sparse SNAM is a different
                                                 group-lasso approach.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   la-nam                     Not integrated.    Linearized Laplace posteriors, marginal-likelihood hyperparameter fitting, subnetwork covariance and
                                                 uncertainty-driven feature/interaction selection.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   gaminet-pytorch            Not integrated.    Its staged main-effect/interaction selection, heredity constraints, marginality and pruning
                                                 procedure. SIAN interaction detection is not equivalent.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   honam                      Not integrated.    HONAM’s representation-vector construction, dedicated higher-order modules and local/global
                                                 interaction interpretation.
  ─────────────────────────  ─────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   neuralgam                  Not integrated.    Local-scoring/backfitting training, independently configurable term subnetworks, per-term histories
                                                 and Monte Carlo epistemic uncertainty. Current NAM uses joint optimization.

  ## What NAMpy has done particularly well

  ### 1. It has a strong shared neural architecture layer

  The central nampy/neural/registry.py and common objective/task system avoid reproducing every upstream’s incompatible data loader and training
  harness. Preprocessing, objectives, distributions, persistence, additive decomposition and estimator generation are shared effectively.

  That is valuable reuse, especially for GP-NAM, IGANN and SIAN, where the algorithm-specific mathematics remains visible instead of being buried in
  model-specific boilerplate.

  ### 2. It distinguishes extensions correctly in some of the strongest integrations

  GP-NAM and IGANN make a reasonably clear distinction between upstream behavior and NAMpy-added multiclass/LSS support. This is the right pattern:
  reproduce the upstream core, then mark broader task support as a NAMpy extension.

  That pattern needs to be applied consistently to every generated estimator family.

  ### 3. GAM architecture is much better disciplined than the average Python GAM reimplementation

  The GAM subsystem has sensible ownership boundaries:

  - Runtime terms own basis semantics.
  - Design assembly is largely basis-agnostic.
  - Fit and prediction transformations are paired.
  - Constraints and smoothing-parameter selection have dedicated subsystems.
  - Unsupported behavior generally fails explicitly.

  The folder structure is deeper than a small library would need, but it is justified by mgcv’s complexity. Collapsing it would probably reduce
  traceability to upstream.

  ### 4. The upstream collection is reproducible

  All 30 manifest-managed clones passed:

  python3 scripts/verify_upstreams.py

  Their configured commit prefixes matched. The separately tracked upstreams/mgcv is the one exception to this manifest mechanism.

  ## Cross-cutting problems

  ### Upstream fidelity is not encoded in the public model registry

  The registry describes tasks and capabilities, but not provenance. Consequently, users can see generated regression, classification or LSS variants
  without knowing whether each is:

  - directly supported upstream,
  - a faithful NAMpy port,
  - a NAMpy extension around an upstream core,
  - historically inspired,
  - or entirely NAMpy-native.

  Add fields such as:

  provenance
  integration_tier
  upstream_supported_tasks
  nampy_extensions
  reference_tests
  upstream_commit

  This metadata should generate documentation and prevent extension variants from being mistaken for upstream parity.

  ### Several names imply relationships that do not exist

  The most important cases are:

  - NAMformer is not TiCL/GAMformer.
  - QNAM is not qgam.
  - Current SNAM is not historical anfreth SNAM.
  - Current sparse SNAM is not NAM-FS.
  - SIAN, GamiNet and HONAM all concern interactions, but implement different algorithms.

  These do not necessarily require immediate renaming, but their documentation should state the distinction on the first line.

  ### Documentation and notices disagree with the code

  THIRD_PARTY_NOTICES.md describes Google NAM and GP-NAM as reference-only even though their equations/components are substantively adapted. It also
  omits the partial NBM integration.

  More seriously, direct behavioral ports of GPL-family mgcv and SCAM code coexist with an MIT package declaration in pyproject.toml. The notices
  acknowledge the SCAM concern but do not provide equally clear treatment for mgcv.

  This is a material licensing/provenance issue—not a correctness bug, and not legal advice—but it should be reviewed before distribution. SIAN’s
  missing upstream license and NBM-SPAM’s non-commercial licensing also deserve explicit attention if source-derived material is retained.

  ### The ledger is useful but too manual

  UPSTREAM_LEDGER.md should be generated from or validated against machine-readable metadata. Its current NBM status already demonstrates how textual
  status can fall behind implementation.

  A per-upstream record should include:

  - exact commit and license,
  - implementation owner/files,
  - integration tier,
  - adapted versus reference-only status,
  - supported upstream behavior,
  - NAMpy extensions,
  - reference/parity tests,
  - unresolved audit findings.

  ### Testing depth is highly uneven

  The focused upstream integration slice produced:

  35 passed, 1 skipped

  for GP-NAM, IGANN, SIAN and shared upstream integration tests. The skipped path was optional ABESS functionality.

  The evidence hierarchy is roughly:

  1. Strong: GP-NAM, IGANN and SIAN direct equation/reference tests.
  2. Strong but subsystem-specific: mgcv and SCAM parity suites.
  3. Moderate: NODE-GAM, NODE, entmax and Google NAM component/invariant tests.
  4. Weak: NBM, historical NAM lineage, NAMformer/TiCL relationship.
  5. None by design: the 17 reference-only repositories.

  The test was run under Python 3.10.12, while the project declares Python 3.11–3.12, so it is supporting evidence rather than a complete supported-
  environment validation.

  ## Recommended integration roadmap

  ### Priority 0: make provenance truthful

  Before adding more upstream algorithms:

  1. Reconcile the manifest, ledger, registry and third-party notices.
  2. Establish one canonical mgcv vendored source.
  3. Give every public architecture an integration tier.
  4. Separate upstream-supported tasks from NAMpy-generated extensions.
  5. Resolve the GPL, missing-license and non-commercial-license questions.

  ### Priority 1: finish the integrations that already exist

  The highest-value technical work is:

  1. Add direct upstream/reference tests for NBM.
  2. Audit and document the historical anfreth-NAMpy migration.
  3. Strengthen end-to-end reference coverage for Google NAM and NODE-GAM.
  4. Clearly distinguish NAMformer from GAMformer/TiCL.
  5. Complete the explicitly scoped mgcv and SCAM gaps without heuristic substitutes.

  ### Priority 2: choose coherent capability families

  Trying to port all 17 remaining repositories would produce a sprawling and incoherent API. Select a product direction:

  - Explainability: Effector, Regional-RHALE, InterpretML.
  - Survival: CoxSE, CRISP-NAM, DNAMite.
  - Feature/interaction selection: NAM-FS, GamiNet, HONAM.
  - Expert/time-series models: NAE and HNAM.
  - In-context/pretrained GAMs: TiCL/GAMformer.

  Each family requires new shared contracts. Survival, for example, needs censoring, time-dependent predictions and survival metrics; adding a Weibull
  distribution alone does not provide a survival modeling framework.

  ## Bottom line

  NAMpy is not yet a comprehensive integration of all repositories under upstreams/. It is currently:

  - a substantial mgcv/SCAM behavioral-port project,
  - a strong shared neural additive framework,
  - a faithful home for a handful of algorithms,
  - and a research catalog for many unimplemented candidates.

  The right next step is not to claim broader integration. It is to formalize fidelity levels, finish auditing the partial ports, correct provenance/
  licensing records, and then deliberately choose which missing capability family belongs in the package.