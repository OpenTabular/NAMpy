# NAMpy

<p align="center">
  <img src="https://raw.githubusercontent.com/OpenTabular/NAMpy/main/docs/_static/logo.png" alt="NAMpy — Interpretable Additive Modeling" width="704">
</p>

[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Development status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/OpenTabular/NAMpy)

Interpretable additive modeling in Python, from classical generalized additive
models to modern neural architectures.

NAMpy brings two complementary model families behind familiar estimator APIs:

- an experimental, strict behavioral port of R's [`mgcv`](https://cran.r-project.org/package=mgcv)
  for statistical GAMs; and
- a PyTorch/Lightning backend for neural additive, basis, spline, tree,
  attention, interaction, and distributional models.

The common idea is simple: retain a useful decomposition of a prediction into
terms while choosing the modeling machinery appropriate for the problem.

```text
prediction = intercept + main effects + optional interactions + optional offset
```

Both backends expose sklearn-style estimators, additive component predictions,
term explanations, importance tables, plotting, and persistence. They do not,
however, hide their different statistical semantics: GAM inference remains GAM
inference, while neural training remains neural training.

## Contents

- [Why NAMpy?](#why-nampy)
- [Installation](#installation)
- [Quick start: statistical GAMs](#quick-start-statistical-gams)
- [Quick start: neural additive models](#quick-start-neural-additive-models)
- [One interpretation contract](#one-interpretation-contract)
- [GAM backend](#gam-backend)
- [Neural backend](#neural-backend)
- [Preprocessing with PreTab](#preprocessing-with-pretab)
- [Training and sklearn integration](#training-and-sklearn-integration)
- [Distributional regression](#distributional-regression)
- [Ensembling and persistence](#ensembling-and-persistence)
- [Extending NAMpy](#extending-nampy)
- [Project status and documentation](#project-status-and-documentation)
- [Contributing and citation](#contributing-and-citation)



## Why NAMpy?

Additive models occupy a useful middle ground between simple linear models and
unrestricted black boxes. A model can learn nonlinear effects and selected
interactions while still answering questions such as:

- Which terms contributed to this prediction?
- What response shape did the model learn for a feature?
- Which interactions matter most?
- Does the sum of the reported terms reconstruct the link-scale prediction?

NAMpy lets users answer those questions across two backends without pretending
that every estimator is the same.


|                  | Statistical GAM backend                                                      | Neural backend                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Primary API      | `GAMRegressor`, `GAMClassifier`, low-level `GAM`                             | `NAMRegressor`, `NBMRegressor`, `NAMLSS`, and other registered families                                             |
| Numerical engine | NumPy/SciPy ports of `mgcv` algorithms                                       | PyTorch and Lightning, plus architecture-native solvers where required                                              |
| Main strengths   | Penalized smooths, smoothing selection, covariance and classical diagnostics | Large architecture catalog, flexible objectives, GPU training and distributional regression                         |
| Formula support  | Yes                                                                          | No; use feature matrices or data frames                                                                             |
| Preprocessing    | GAM-owned formula/model-frame semantics                                      | Pristine PreTab block contract                                                                                      |
| Parity policy    | Strict `mgcv` behavior where supported                                       | Preserve reference architecture/training behavior where integrated; shared APIs and PreTab may intentionally differ |




## Installation

NAMpy supports Python 3.11 and 3.12. Install the extra for the backend you need:

```bash
# Both backends and all optional model dependencies
pip install "nampy[all]"

# Statistical GAM backend only
pip install "nampy[gam]"

# Neural backend only
pip install "nampy[neural]"

# Optional nonlinear best-subset selection for IGANN-Sparse
pip install "nampy[igann-sparse]"
```

The base package contains NumPy, pandas, scikit-learn, and joblib. The `gam`
extra adds SciPy and Matplotlib. The `neural` extra adds PyTorch, Lightning,
PreTab, TorchMetrics, SciPy, plotting, and distributional metrics. Ordinary
IGANN does not require ABESS; only `IGANNRegressor(sparse=...)` does.

For development:

```bash
git clone https://github.com/OpenTabular/NAMpy.git
cd NAMpy
pip install -e ".[all,dev,docs]"
```

NumPy is currently constrained to `<=1.26.4`; see
[`pyproject.toml`](pyproject.toml) for the authoritative dependency versions.

## Quick start: statistical GAMs

The sklearn-style adapters default to automatic REML smoothing selection, which
is the usual high-level `mgcv::gam()` experience.

```python
import numpy as np
import pandas as pd

from nampy.models import GAMRegressor

rng = np.random.default_rng(7)
x = np.linspace(0.0, 1.0, 300)
data = pd.DataFrame(
    {
        "x": x,
        "group": rng.choice(["a", "b"], size=x.size),
        "y": np.sin(2 * np.pi * x) + rng.normal(scale=0.15, size=x.size),
    }
)

gam = GAMRegressor(
    formula="y ~ s(x, bs='cr', k=12) + group",
    smoothing_method="reml",
).fit(data)

predictions = gam.predict(data)
summary = gam.summary()

components = gam.predict_components(data)
components.validate_additive_reconstruction()
importance = gam.term_importance(data)
```

For array-based work, omit the formula:

```python
gam = GAMRegressor(k=10, basis="tp").fit(X_train, y_train)
y_pred = gam.predict(X_test)
y_se = gam.standard_errors(X_test)
```

Binary classification uses the corresponding adapter:

```python
from nampy.models import GAMClassifier

classifier = GAMClassifier(formula="label ~ s(age) + income").fit(
    frame, frame["label"]
)
probability = classifier.predict_proba(frame)[:, 1]
label = classifier.predict(frame)
```

Use the lower-level `nampy.gam.GAM` when you need direct access to fixed
smoothing parameters, prediction types, derivatives, covariance choices, or
advanced diagnostics. Its defaults intentionally differ from the adapters:
the raw class uses fixed smoothing unless automatic selection is requested.

```python
from nampy.gam import GAM

raw_gam = GAM(
    formula="y ~ s(x, bs='mpi', k=10)",
    family="gaussian",
    optimize_smoothing=True,
    smoothing_method="gcv",
    smoothing_optimizer="bfgs",
).fit(data=data)

derivative = raw_gam.derivative(smooth_number=1, deriv=1)
```



## Quick start: neural additive models

Neural estimators separate constructor-time model/preprocessing configuration
from fit-time training controls.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from nampy.models import NAMRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

nam = NAMRegressor(
    layer_sizes=[64, 32],
    dropout=0.1,
    numerical_method="minmax",
)
nam.fit(
    X_train,
    y_train,
    max_epochs=100,
    batch_size=128,
    patience=12,
    random_state=7,
)

y_pred = nam.predict(X_test)
r2 = nam.score(X_test, y_test)

components = nam.predict_components(X_test, center=True)
components.validate_additive_reconstruction()
importance = nam.term_importance(X_test)
figures = nam.plot_terms(X_test)
```

Classification follows the same lifecycle:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from nampy.models import NAMClassifier

Xc, yc = load_breast_cancer(return_X_y=True, as_frame=True)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, stratify=yc, random_state=7
)

classifier = NAMClassifier().fit(
    Xc_train, yc_train, max_epochs=100, class_weight="balanced"
)
labels = classifier.predict(Xc_test)
probabilities = classifier.predict_proba(Xc_test)
accuracy = classifier.score(Xc_test, yc_test)
```

To add explicit interactions, pass logical source-feature names. Several
architectures also accept an interaction degree that constructs the complete
set for that order.

```python
model = NAMRegressor(
    interactions=(("age", "income"), ("debt", "assets")),
)
```



## One interpretation contract

`predict_components()` returns a backend-neutral `AdditivePrediction` object.
Its important fields are:

- `response`: the prediction on the response scale;
- `link`: the additive prediction before the inverse link or distributional
parameter transform;
- `terms`: an ordered mapping of term name to contribution;
- `intercept`: the fitted intercept contribution;
- `offset`: an optional link-scale offset; and
- `backend`: either `"gam"` or `"neural"`.

For an ordinary scalar additive prediction, the central invariant is:

```python
components.link == (
    components.intercept
    + sum(components.terms.values())
    + (0.0 if components.offset is None else components.offset)
)
```

Use the built-in validator instead of writing that comparison yourself:

```python
components.validate_additive_reconstruction()
```

Both estimator backends provide explanation tables and importance summaries:

```python
table = model.explain_terms(X_test)
importance = model.term_importance(X_test)
interaction_importance = model.interaction_importance(X_test)
```

Neural models additionally provide `center_components()`, `plot_terms()`, and
`plot_interactions()` for centered effects, one-dimensional curves,
interaction heatmaps, and conditioned slices. GAM adapters provide `plot()`,
which renders the ported `plot.gam` data phase through the same shared renderer.
Centering changes the allocation between the intercept and terms, not the
reconstructed prediction. For LSS models, additivity holds on the raw
distribution-parameter/link scale rather than on the transformed parameter
scale.

## GAM backend

`nampy.gam` is not a loosely inspired GAM implementation. For supported
behavior, the upstream R and C sources of `mgcv` are treated as the behavioral
specification. Control flow, penalty ordering, constraints, smoothing
selection, and numerically significant factorization choices are ported as
directly as practical and exercised by parity tests.

### Supported surface

The current implementation includes:

- ordinary Gaussian, binomial, Poisson, and Gamma families, including their
documented supported noncanonical links;
- fixed or estimated negative-binomial theta, beta regression (`betar`),
ordered categorical regression (`ocat`), Tweedie (`tw`), and the
multi-predictor `gaulss` and `gammals` families on their supported routes;
- fixed smoothing and automatic GCV/Cp, ML, REML, or general-family LAML where
the selected family and fitting route support the criterion;
- `outer_newton`, `bfgs`, and `efs` smoothing optimizers, plus guarded `optim`
support and the NAMpy `lbfgsb` extension;
- `s(...)` smooths with `cr`, `cs`, `cc`, `ps`, `tp`, `ts`, `re`, `fs`, and
`sz` bases;
- `te(...)` and `ti(...)` tensor products over supported numeric marginals;
- SCAM-compatible shape-constrained smooths, including the supported
univariate and bivariate SCOP-spline classes;
- `link`, `response`, `terms`, `iterms`, and `lpmatrix` prediction modes;
- pointwise standard errors, covariance matrices, derivatives, residuals,
summaries, concurvity, basis-dimension checks, and diagnostic data; and
- matplotlib rendering of the ported `plot.gam` data phase.



### Deliberate boundaries

Unsupported inputs raise explicit errors. Important current exclusions include
`t2`, Gaussian-process, MRF, adaptive and soap smooths, `paraPen`, NCV/QNCV,
many extended/general families, several matrix-covariate cases, complete R
graphics-device behavior, and exact R endpoint behavior for every `optim`
combination.

Advanced users can consult the
[shape-constrained GAM guide](docs/user_guide/shape_constrained_gams.rst).
The supported boundary is defined by the public API, explicit unsupported-input
errors, and the committed test suite.

The stable package-level low-level surface is intentionally small:
`GAM`, `fit_model_core`, `solve_fit`, and `FitCoreSolution`. Internal GAM
modules are parity-sensitive implementation details.

## Neural backend

The neural backend uses a declarative architecture registry. Each architecture
declares its forward module, configuration dataclass, supported objectives,
preprocessing defaults, and optional native estimator lifecycle. The registry
then generates consistent regressor, classifier, and LSS estimator families.

### Model catalog


| Architecture    | Public estimators                                                             | Main idea                                                                                                 |
| --------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Linear          | `LinRegRegressor`, `LinRegClassifier`, `LinRegLSS`                            | A neural-objective-compatible linear baseline                                                             |
| NAM             | `NAMRegressor`, `NAMClassifier`, `NAMLSS`                                     | Independent feature networks with optional explicit interactions                                          |
| SIAN            | `SIANRegressor`, `SIANClassifier`, `SIANLSS`                                  | Archipelago-based sparse, arbitrary-order interaction discovery followed by additive fitting              |
| SNAM            | `SNAMRegressor`, `SNAMClassifier`, `SNAMLSS`                                  | Sparse neural additive modeling                                                                           |
| GPNAM           | `GPNAMRegressor`, `GPNAMClassifier`, `GPNAMLSS`                               | Fixed random Fourier feature shape functions; native conjugate-gradient regression or gradient objectives |
| IGANN           | `IGANNRegressor`, `IGANNClassifier`, `IGANNLSS`                               | Linear initialization plus additive ELM boosting; native regression/binary fitting where supported        |
| NBM             | `NBMRegressor`, `NBMClassifier`, `NBMLSS`                                     | Shared learned basis functions, including dense, sparse, Conv1D, einsum, and n-ary configurations         |
| SPAM            | `SPAMRegressor`, `SPAMClassifier`, `SPAMLSS`                                  | Scalable low-rank polynomial additive effects and local term importance                                   |
| NBM-SPAM        | `NBMSPAMRegressor`, `NBMSPAMClassifier`, `NBMSPAMLSS`                         | Neural basis concepts combined with polynomial interaction structure                                      |
| NATT            | `NATTRegressor`, `NATTClassifier`, `NATTLSS`                                  | Attentive tabular modeling                                                                                |
| NAMformer       | `NAMformerRegressor`, `NAMformerClassifier`, `NAMformerLSS`                   | Transformer-based additive tabular modeling                                                               |
| TreeNAM         | `TreeNAMRegressor`, `TreeNAMClassifier`, `TreeNAMLSS`                         | Tree-inspired neural additive terms                                                                       |
| EnsembleTreeNAM | `EnsembleTreeNAMRegressor`, `EnsembleTreeNAMClassifier`, `EnsembleTreeNAMLSS` | An integrated ensemble-tree NAM architecture                                                              |
| NodeGAM         | `NodeGAMRegressor`, `NodeGAMClassifier`, `NodeGAMLSS`                         | Differentiable oblivious trees with optional masked reconstruction pretraining                            |
| QNAM            | `QNAMLSS`                                                                     | Distributional-only quantile NAM                                                                          |
| SplineNAM       | `SplineNAMRegressor`                                                          | Regression-only neural spline additive model                                                              |


Most architectures support main effects, explicit interactions, and all three
objectives. Exceptions are intentional: QNAM is distributional-only,
SplineNAM is regression-only, and IGANN does not expose interaction terms.

### Architecture-specific behavior

Some models need more than a generic training loop:

- **GPNAM regression** can use the released fixed-basis conjugate-gradient
ridge solve. Classification and LSS use the common objective engine.
- **IGANN regression and binary classification** can use the released
sequential ELM optimizer. Multiclass and LSS are explicit NAMpy extensions
trained over the fixed basis with the gradient engine.
- **SIAN** can discover sparse higher-order interactions or accept an explicit
interaction set that bypasses discovery.
- **NBM** treats sparse as a configuration option—not a separate estimator
class—and supports Conv1D and arbitrary-order interactions.
- **SPAM and NBM-SPAM** are first-class registered architecture families, not
aliases for NBM configurations.
- **NodeGAM** is the architecture currently supporting masked reconstruction
pretraining through the shared fit surface.

Reference repositories cloned locally under the ignored `upstreams/` directory
exist for implementation study and fixture generation. They are neither
runtime nor test dependencies. Normal parity tests consume the committed,
versioned results under `tests/reference_fixtures/`.

## Preprocessing with PreTab

The neural backend delegates generic tabular preprocessing to
[PreTab](https://github.com/OpenTabular/PreTab) and targets the public contract in
pristine PreTab `1.0.0rc2` or newer:

- `Preprocessor.fit(X, y)` and `transform(X)`;
- dictionary outputs with `num_<feature>` and `cat_<feature>` blocks;
- block metadata from `get_feature_info(verbose=False)`; and
- constructor options supported by the installed `Preprocessor`, including
numerical/categorical methods, per-feature preprocessing, scaling, output
dimensions, degree, dtype, and random state.

High-level estimators accept PreTab constructor arguments directly:

```python
model = NAMRegressor(
    numerical_method="ple",
    categorical_method="one-hot",
    output_dim=32,
)
```

If a preprocessing name collides with an architecture parameter, use the
explicit prefix:

```python
from nampy.models import SplineNAMRegressor

model = SplineNAMRegressor(
    n_knots=8,
    preprocessor__output_dim=32,
)
```

Preprocessing is fitted on training rows only. NAM consumes one network per
PreTab source-feature block, so a one-hot categorical feature remains one
grouped categorical term. NBM, SPAM, and NBM-SPAM flatten multi-column blocks
into scalar concepts inside their architectures. GAMs do not import or use
PreTab.

NAMpy intentionally does not carry a second generic preprocessing
implementation. It therefore does not require experimental PreTab surfaces
such as atomic output-column metadata, output ordering/granularity controls,
post-encoding output ranges, generic representation parameter dictionaries,
quantile-noise controls, or TF-IDF categoricals. This preserves compatibility
with pristine PreTab but can differ from preprocessing in individual upstream
model repositories.

See [the PreTab compatibility contract](docs/user_guide/pretab_compatibility.rst)
and [the preprocessing guide](docs/user_guide/preprocessing.rst) for the exact
boundary.

## Training and sklearn integration

NAMpy estimators inherit scikit-learn's `BaseEstimator` parameter protocol and
support `get_params`, `set_params`, cloning, pipelines, and model-selection
tools. Their task-specific scoring conventions are:

- regressors: R²;
- classifiers: accuracy; and
- LSS estimators: negative mean negative log-likelihood, so larger is better.

Common neural fit controls include:

```python
model.fit(
    X_train,
    y_train,
    X_val=X_validation,       # or let val_size create a split
    y_val=y_validation,
    max_epochs=200,
    max_steps=-1,
    batch_size=256,
    patience=15,
    lr=1e-3,
    weight_decay=1e-5,
    optimizer="adamw",
    sample_weight=weights,
    random_state=7,
)
```

Additional fit options cover offsets for non-LSS objectives, class weighting,
sampling strategies, learning-rate schedules, warm starts, recent-checkpoint
averaging, explicit Lightning trainer arguments, and NodeGAM pretraining.
Prediction methods accept an independent `batch_size` for bounded-memory
inference.

Architecture-native optimizers retain their own semantics. For example,
IGANN's `n_estimators` is a boosting-stage limit rather than a Lightning epoch
count. Inspect model-specific guides before transferring hyperparameters across
architectures.

## Distributional regression

`*LSS` estimators learn every parameter of a conditional distribution instead
of only its mean. The constructor chooses the family; `predict()` returns
transformed, valid distribution parameters.

```python
from nampy.models import NAMLSS

lss = NAMLSS(
    family="normal",
    distributional_kwargs={},
)
lss.fit(X_train, y_train, max_epochs=150, patience=15)

parameters = lss.predict(X_test)
negative_nll = lss.score(X_test, y_test)
components = lss.predict_components(X_test)
components.validate_additive_reconstruction()  # raw parameter/link scale
```

Registered families are:

- continuous: `normal`, `robustnormal`, `studentt`, `gamma`, `inversegamma`,
`beta`, `lognormal`, `weibull`, `loglogistic`, and `tweedie`;
- counts: `poisson`, `negativebinom`, `zip`, `zinb`, `hurdlepoisson`, and
`hurdlenegativebinom`;
- discrete/ordered: `categorical` and `ordinal`;
- multivariate: `dirichlet` and `mvnormdiag`; and
- quantile regression: `quantile`.

Family-specific arguments such as class count, dimension, or quantile levels
belong in `distributional_kwargs`. Where possible, output dimension is inferred
from `y` during fitting.

## Ensembling and persistence

`NeuralEnsemble` fits independently cloned regressors or classifiers, with
optional bootstrapping and joblib parallelism:

```python
from nampy.models import NAMRegressor, NeuralEnsemble

ensemble = NeuralEnsemble(
    NAMRegressor(layer_sizes=[64, 32]),
    n_estimators=5,
    bootstrap=True,
    n_jobs=2,
    random_state=7,
).fit(X_train, y_train, max_epochs=100)

mean_prediction = ensemble.predict(X_test)
uncertainty = ensemble.predict_component_uncertainty(X_test)
```

The uncertainty object contains between-member standard deviations for the
response, link, intercept, and every additive term. LSS ensembling is not
provided because distribution-family parameters require family-specific
aggregation.

Fitted GAM adapters and neural estimators have a versioned persistence API:

```python
path = model.save_model("model.pkl")
restored = type(model).load_model(path)
```

These artifacts use Python pickle. Load them only from trusted sources and
recreate the training environment when long-term reproducibility matters.

## Extending NAMpy

Neural architecture and objective semantics are intentionally separate. A new
architecture normally consists of:

1. a PyTorch `nn.Module` in `nampy/neural/architectures/`;
2. a configuration dataclass in `nampy/neural/configs/`;
3. a `NeuralArchitecture` registry declaration with explicit capabilities;
4. generated estimator exports in `nampy/models/`; and
5. focused tests for forward shapes, estimator contracts, preprocessing block
  interpretation, additive reconstruction, and every advertised objective.

Shared architecture components belong in
`nampy/neural/architectures/components/`. Generic preprocessing belongs in
PreTab; NAMpy should only interpret PreTab's output blocks in a
model-specific way. GAM extensions follow a different rule: locate the exact
upstream `mgcv` R/C routine, port its control flow, and add targeted parity
tests rather than introducing a generic approximation.

See [the custom model guide](docs/user_guide/custom_models.rst) and
[architecture overview](docs/architecture.rst).

## Project status and documentation

NAMpy is under active development and is classified as beta. Before using an
advanced model or fitting route in production, verify it with data and settings
representative of the intended workload.

Repository documentation:

- [quick-start guide](docs/quickstart.rst)
- [user guide](docs/user_guide.rst)
- [model guide](docs/models/index.rst)
- [API reference](docs/api/index.rst)
- [examples](examples/)
- [tutorial notebooks](docs/notebooks/)
- [FAQ](docs/faq.rst)
- [changelog](docs/changelog.rst)

The source tree is organized around ownership boundaries:

```text
nampy/
├── contracts.py                 # backend-neutral additive results
├── explanations.py              # shared explanation tables
├── plotting/                    # shared rendering
├── gam/                         # mgcv-aligned statistical backend
├── models/                      # public sklearn-style estimators
└── neural/
    ├── architectures/           # PyTorch forward architectures
    ├── configs/                 # architecture configuration dataclasses
    ├── data/                    # PreTab-to-Torch data layer
    ├── distributions/           # LSS families and metrics
    ├── objectives.py            # task/output/loss semantics
    ├── registry.py              # architecture declarations
    └── task.py                  # Lightning training harness
```



## Contributing and citation

Bug reports, parity cases, documentation improvements, and focused model
contributions are welcome. Please read [the contributing guide](CONTRIBUTING.md)
and open an issue before beginning a large architectural change.

For GAM work, include the corresponding upstream `mgcv` function and the
smallest parity test that demonstrates the behavior. For neural models, state
which upstream implementation or paper defines the architecture and distinguish
reference behavior from NAMpy extensions.

If NAMpy supports your research, cite the software:

```bibtex
@software{nampy,
  author  = {Ananyapam De and Anton Thielmann},
  title   = {NAMpy: Interpretable Additive Modeling in Python},
  url     = {https://github.com/OpenTabular/NAMpy},
  version = {0.2.0},
  date    = {2026-08-22}
}
```

NAMpy builds on ideas and reference implementations from `mgcv`, neural
additive models, GPNAM, IGANN, NBM, SPAM, NBM-SPAM, SIAN, NodeGAM, TreeNAM,
NAMformer, NATT, SNAM, and related interpretable tabular-modeling research.
Those projects remain the primary references for their respective methods.

NAMpy is released under the [MIT License](LICENSE).
