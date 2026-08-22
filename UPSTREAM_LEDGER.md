# Upstream capability ledger

The repositories listed below are local, reference-only clones described by
[`upstreams/manifest.json`](upstreams/manifest.json). They are intentionally
ignored by Git so the checkout does not redistribute unrelated source trees.
The tracked `upstreams/mgcv` tree is the exception: it is the released R/C
specification used by the classical GAM backend.

| Local clone | Capability to audit | Integration target | Status |
| --- | --- | --- | --- |
| `nodegam` | NODE-GAM, sparse tree selectors, GAM/GA2M terms | `NodeGAM`, additive-tree term extraction | Ported core; focused regression added |
| `google-research` | Google Research NAM | `NAM`, preprocessing and term semantics | ExU, adaptive widths, penalties, and balancing adapted; focused regression added |
| `nickfrosst-neural_additive_models` | Standalone Google NAM fork | `NAM` compatibility checks | Audited as a standalone copy; no separate compatibility mode |
| `amr-nam` | Package-style NAM training | `NAM` API and training defaults | Audited as a secondary API reference; shared capabilities integrated compositionally |
| `protonam` | Prototypical NAM explanations | new prototype/anchor term API | Candidate, not yet exposed |
| `sian` | Sparse interaction additive networks | interaction selection and regularization | Archipelago/FIS, higher-order SIAN, and block-masked execution integrated |
| `igann` | Interpretable generalized additive neural net | `IGANN`, native ELM boosting and nonlinear sparsity | Core and IGANN-Sparse selection integrated |
| `ticl` | GAMformer, MotherNet, TabFlex | `NAMformer`/pretrained tabular path | Reference clone; audit pending |
| `effector` | Global and regional effects | shared explanation/region API | Candidate, not yet exposed |
| `regional-rhale` | Regional RHALE effect estimation | regional explanation utilities | Candidate, not yet exposed |
| `nae` | Neural additive experts and context gating | expert-gated additive models | Candidate, not yet exposed |
| `coxse` | CoxNAM, CoxSE, DeepSurv comparison code | survival LSS/family integration | Candidate, not yet exposed |
| `crisp-nam` | Competing-risks NAM | competing-risks survival family | Candidate, not yet exposed |
| `dnamite` | Discretized additive regression/classification/survival | binning and survival-specific heads | Candidate, not yet exposed |
| `anfreth-nampy` | Historical NAMpy implementation | compatibility and migration audit | Reference clone; audit pending |
| `gpnam` | Gaussian-process NAM | `GPNAM`, fixed RFF basis and convex fitting | Paper RFF construction, auto bandwidths, CG solve, and GP-NA2M integrated |
| `hnam-demand-forecasting` | Hierarchical NAM forecasting | grouped/time-series additive terms | Candidate, not yet exposed |
| `pygam` | Python classical GAM | `nampy.gam` public/API comparison | Reference clone; audit pending |
| `interpret` | EBM and additive explanations | generic term storage/plotting | Candidate, not yet exposed |
| `cran-mgcv` | Released CRAN mgcv mirror | source/version comparison | Reference clone; parity source |
| `qgam` | Quantile GAM | quantile smoothing/family behavior | Candidate, not yet exposed |
| `scam` | Shape-constrained additive models | explicit shape constraints | All released univariate/bivariate bases, constrained fitting, GCV/UBRE, inference, and AR(1) integrated |
| `nbm-spam` | Neural Basis Models and SPAM | `NBM`, `SPAM`, `NBMSPAM` | Dense/sparse NBM, SPAM, and hybrid integrated; direct parity coverage added |
| `node` | Original oblivious decision trees | shared tree primitives | Adapted component |
| `entmax` | Entmax/sparsemax activations | sparse selector primitives | Adapted component |
| `nam-fs` | NAM feature selection and interactions | feature selection/interaction regularizers | Candidate, not yet exposed |
| `la-nam` | Laplace uncertainty for NAM | posterior uncertainty API | Candidate, not yet exposed |
| `gaminet-pytorch` | Structured GAM and interaction networks | interaction architecture | Candidate, not yet exposed |
| `honam` | Higher-order NAMs | configurable higher-order terms | Candidate, not yet exposed |
| `neuralgam` | R neural GAM backfitting/uncertainty | neural GAM comparison | Candidate, not yet exposed |

## Selection rules

Generic behavior belongs in shared contracts and components: additive output
keys, feature preprocessing, target-shape handling, contribution objects,
interaction naming, and distribution-family registration. Paper-specific
behavior belongs in a model module/configuration with an explicit capability
flag and focused tests; it should not silently change the semantics of the
existing NAM/GAM surfaces.

Before adopting a feature, record the upstream file/function, license, target
API, and a parity or invariant test here. Do not copy an entire upstream tree
into the package. Refresh and verify the local references with:

```bash
python3 scripts/fetch_upstreams.py
python3 scripts/verify_upstreams.py
```

The untracked research clones are temporary development references. They are
used to audit behavior and run direct parity tests, but they are not runtime
dependencies and are not shipped in the Python distribution. The intended
end state is to remove these clones after the relevant behavior is represented
by durable NAMpy tests, fixtures, provenance records, and citations.

## NBM-SPAM adaptation references

NBM uses the released topology as its ordinary configuration defaults; there
is no separate compatibility mode. The generalized activation, normalization,
GLU, residual, feature-dropout, distributional, and alternate-featurizer
options remain explicit NAMpy extensions.

| Upstream source | Upstream symbol | NAMpy target | Focused validation |
| --- | --- | --- | --- |
| `nbm_spam/models/concept_nbm.py` | `ConceptNBMNary` | `neural.architectures.nbm.NBM`, dense `conv1d` path | direct feature/output/gradient/penalty parity |
| `nbm_spam/models/concept_nbm.py` | `ConceptNBMNarySparse` | `NBM(sparse=True)` | active-tuple, scatter, and output parity |
| `nbm_spam/models/concept_spam.py` | `ConceptSPAM` | `neural.architectures.spam.SPAM` | output, regularizer, and local-importance parity |
| `nbm_spam/models/concept_nbm.py` | `polynomial` branch of `ConceptNBMNary` | `neural.architectures.nbm_spam.NBMSPAM` | block assembly and output parity |
| `nbm_spam/train_tabular.py` | optimizer, output penalty, cosine scheduling | shared `neural.task.TaskModule` contracts | training-only penalties and warmup-cosine schedule tests |

## NAM adaptation references

The NAM additions intentionally remain independent configuration options; there
is no special ``original`` architecture mode.

| Upstream source | Upstream symbol | NAMpy target | Focused validation |
| --- | --- | --- | --- |
| `google-research/neural_additive_models/models.py` | `exu`, `ActivationLayer`, `FeatureNN` | `components.nam.ExU`, `CenteredReLU`, `NAMFeatureNN` | exact activation geometry and estimator fit |
| `google-research/neural_additive_models/graph_builder.py` | `create_nam_model` | train-only cardinality and adaptive first-layer widths | train-split cardinality and width tests |
| `google-research/neural_additive_models/graph_builder.py` | `feature_output_regularization`, `weight_decay` | shared additive-output and normalized L2 penalties | deterministic penalty tests |
| `google-research/neural_additive_models/graph_builder.py` | `create_balanced_dataset` | classifier balanced sampler | inverse-frequency sampler test |

## GP-NAM adaptation references

GP-NAM is a fixed-RFF point estimator. Neither the released implementation nor
NAMpy's adaptation claims GP posterior covariance. ``GPNAMLSS`` is NAMpy's
objective-driven distributional extension.

| Upstream source | Upstream symbol/equation | NAMpy target | Focused validation |
| --- | --- | --- | --- |
| GP-NAM paper, Eq. 5 and Algorithm 1 | inverse-normal frequency grid and per-dimension phase pairing | `neural.architectures.gpnam.GPNAM` | RFF grid and phase invariants |
| `gpnam/data.py` | `get_kernel_width` (`std / 24`) | fitted `kernel_widths_` | exact sample-standard-deviation comparison |
| `gpnam/trainer.py` | `Trainer.train`, `optimizer="CG"` | `neural.linear_solver.solve_fixed_linear_regression` | weighted multi-output normal-equation comparison |
| GP-NAM paper, GP-NA2M discussion | two-dimensional pairwise GP terms | GPNAM explicit/all-pairs interactions | additive reconstruction and fixed-design shape |

The released Python phase expression sorts random values rather than
permuting a uniform grid, while the MATLAB files overwrite the constructed
phase grid with a debug constant. NAMpy follows the paper's stated grid
construction and tests its invariants instead of reproducing those apparent
source artifacts. The released solvers use ridge ``1/20`` with an unpenalized
intercept even though the paper displays ``I + A``; NAMpy exposes this as the
named ``ridge=0.05`` default.

## SIAN adaptation references

SIAN's reusable pieces are detector, search, and execution components. The
public SIAN estimator supplies the paper-specific defaults and staged workflow;
the final architecture remains independent of the regression, classification,
or distributional objective.

| Upstream source | Upstream symbol/equation | NAMpy target | Focused validation |
| --- | --- | --- | --- |
| `sian/interpret/explainer2.py` | `get_archipelago_values`, triangle-marginal accumulation | `neural.interaction_selection.ArchipelagoDetector` | exact synthetic product/non-interaction scores |
| `sian/combinatorial_utils.py` | `constructHigherInteractions` | generic fractional-heredity frontier | weak/strong heredity candidate tests |
| `sian/fis/feature_interaction_selection.py` | layerwise FIS, `theta`, `tau`, `K` | detector-independent hierarchical search | known ranking and selected-term result |
| `sian/models/models.py` | `Blocksparse_Deep_Relu_GAM`, `compress`, `blocksparse` | `BlockMaskedAdditiveNetwork`, `SIAN` | direct upstream term/output parity and conversion round-trip |
| SIAN paper, Section 4.1 | `[16,12,8]` ReLU terms, Adagrad `5e-3`, L1 `5e-5` | `DefaultSIANConfig` | architecture smoke and staged estimator fit |

The current upstream `main` branch mixes the 2022 implementation with later
masked/InstaSHAP development and contains incomplete classification and
checkpoint paths. NAMpy therefore ports the stable mathematical surfaces and
uses its shared preprocessing, objective, training, checkpoint, and persistence
infrastructure rather than copying the upstream training loop.

## SCAM parity references

The implementation targets the vendored CRAN mirror at commit
``0e43f56b598cfff1b7915ae210bd9fc228025fc3`` (SCAM 1.2-22).  For this
subsystem the upstream routines are a strict behavioral specification, not
merely architectural inspiration.

| Upstream source/function | NAMpy target | Focused validation |
| --- | --- | --- |
| `R/uni.smooth.const.r`, `R/uni.smooth.const-lscop.r` smooth constructors and predict methods | `gam/splines/shape/scop.py`, `gam/smooths/shape/scop.py` | all 24 raw constructors, prediction matrices, compiler assembly, and fixed fits |
| `R/bivar.smooth.const.R`, `R/bivar.smooth.const-ti.R` | `gam/splines/shape/bivariate.py`, `gam/smooths/shape/bivariate.py` | all 17 basis, knot, penalty, mask, prediction, compiler, and fit cases |
| `R/scam.fit1.r::scam.fit.newton`, `scam.fit.post` | `gam/fit/solvers/shape_constrained.py` | Gaussian and non-Gaussian coefficient, transformed-coefficient, eta/mu, EDF, covariance, and AR(1) parity |
| `R/scam.r` positive transforms and coefficient-space conventions | `gam/coefficients/transforms.py` | exp/softplus values, first three derivatives, subset mapping, and covariance transport |
| `R/bfgs.r::bfgs_gcv.ubre`, `R/scam.fit1.r::gcv.ubre_grad` | `gam/fit/selection/criteria/shape.py`, `gam/fit/selection/optimize/shape_bfgs.py` | exact GCV/UBRE values and gradients plus selected SP, endpoint, coefficients, and score |
| `R/predict.scam.R`, `R/summary.scam.R`, `R/residuals.scam.R`, `R/derivative.scam.r` | shared prediction/summary/residual machinery and `gam/diagnostics/derivatives.py` | link/response/term fits and SEs, summary tables/scalars, six residual types, first/second derivatives and SEs |
| `R/scam.r`/`R/scam.fit1.r` `AR1.rho`, `AR.start`, and `rwMatrix` path | `GAM(ar1_rho=, ar_start=)` and constrained solver root transform | coefficients, transformed coefficients, eta, deviance, EDF, scale, covariance, boundaries, standardized residuals |

Raw representation checks use invariant comparisons only where an upstream
eigenspace or active boundary is mathematically non-unique.  All uniquely
identified constructor and fitted outputs are compared directly.

## IGANN adaptation references

IGANN uses a native stagewise optimizer: an L1 linear initialization followed
by feature-wise ELM ridge solves against Newton-style pseudo-responses. It is
not routed through Lightning gradient descent. NAMpy retains its global
train-only preprocessing invariant, whereas the released standalone ``fit``
constructs preprocessing and the initial linear model before applying its
validation split.

| Upstream source | Upstream symbol/equation | NAMpy target | Focused validation |
| --- | --- | --- | --- |
| `igann/igann.py` | `ELM_Regressor.__init__`, masked random hidden matrix | `neural.architectures.igann.IGANN.hidden_weights` | exact seeded diagonal-block draw |
| `igann/igann.py` | `ELM_Regressor.fit`, `torch_Ridge.fit` | native ELM stage solve | direct one-stage normal-equation comparison |
| `igann/igann.py` | `IGANN._loss_sqrt_hessian`, `_get_y_tilde`, `_run_optimization` | `IGANN.fit_native` | regression and binary estimator fits, retained-stage history |
| `igann/igann.py` | Lasso/logistic linear initialization and reference-coded categoricals | native initialization and grouped additive outputs | categorical reference-code and additive reconstruction tests |
| IGANN-Sparse paper, Eq. 2–3; `igann_bagged.py::_select_features` | grouped nonlinear best-subset selection with ABESS | ``sparse`` option and ``selected_features_`` | optional-dependency guard; ABESS path remains environment-dependent |
| `igann_bagged.py::IGANN_Bagged` | bootstrap replicas and prediction dispersion | generic `NeuralEnsemble(bootstrap=True)` | deterministic aligned row/weight bootstrap test |

The current stable ``igann.py`` native optimizer supports regression and binary
classification, not arbitrary distributional losses. NAMpy's multiclass and
``IGANNLSS`` surfaces are clearly identified extensions: they jointly optimize
the output coefficients over the same fixed feature-wise ELM basis using the
shared objective engine. Pairwise interactions are described as future work in
the IGANN-Sparse paper and are therefore not advertised as an IGANN capability.
The experimental ``igann_interactive.py`` interpolation wrapper is also not
ported: NAMpy already exposes exact model contributions, batched inference,
and shared plotting without replacing predictions by an interpolated
approximation.
