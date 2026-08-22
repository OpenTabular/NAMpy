"""Generate the concise, public-API model notebooks in ``notebooks/``.

Run from the repository root:

    python scripts/generate_model_notebooks.py

The notebooks are deterministic documentation artifacts. Training cells are
present but disabled by default so structural documentation checks stay fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks"


def _source(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": _source(text),
    }


def code(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


DATA_CELL = r"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(7)
n = 180
X = pd.DataFrame({
    "x1": rng.uniform(-1.0, 1.0, n),
    "x2": rng.normal(size=n),
    "group": rng.choice(["a", "b", "c"], size=n),
})
y = (
    np.sin(np.pi * X["x1"])
    + 0.35 * X["x2"] ** 2
    + 0.30 * (X["group"] == "b")
    + rng.normal(0.0, 0.12, n)
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

# Set True to run the small fit and all fitted-model demonstrations.
RUN_TRAINING = False
"""


GENERIC_FIT_CELL = r"""
if RUN_TRAINING:
    model.fit(
        X_train,
        y_train,
        max_epochs=3,
        batch_size=64,
        random_state=7,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    predictions = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    metrics = model.evaluate(X_test, y_test)
    components = model.predict_components(X_test, center=True)
    components.validate_additive_reconstruction()
    display({"R2": r2, **metrics})
    display(model.term_importance(X_test).head())
"""


MODEL_SPECS = [
    {
        "filename": "02_linreg.ipynb",
        "title": "LinReg",
        "subtitle": "Additive linear baseline",
        "concept": "Each transformed feature block has its own linear map. It is the transparent baseline for deciding whether nonlinear shape functions are necessary.",
        "math": r"""
For feature blocks $x_j\in\mathbb{R}^{p_j}$,

$$
\eta(x)=\beta_0+\sum_{j=1}^{d} w_j^\top x_j.
$$

Every $w_j^\top x_j$ is returned as an additive contribution.
""",
        "import": "from nampy.models import LinRegClassifier, LinRegLSS, LinRegRegressor",
        "constructor": r"""
model = LinRegRegressor(
    intercept=True,
    numerical_method="standardization",
    categorical_method="one-hot",
)
model.get_params(deep=False)
""",
        "specific_text": "`intercept` controls the global bias. PreTab settings determine whether a source feature enters as one scalar or an expanded block.",
        "specific_code": r"""
model.set_params(intercept=False)
model.set_params(intercept=True)
if RUN_TRAINING:
    display(model.predict_components(X_test).terms.keys())
""",
        "variants": "`LinRegRegressor` uses $R^2$ scoring; `LinRegClassifier` exposes `predict_proba`; `LinRegLSS(family=...)` predicts distribution parameters.",
    },
    {
        "filename": "03_nam.ipynb",
        "title": "NAM",
        "subtitle": "Neural Additive Model",
        "concept": "NAM learns one neural shape function per encoded scalar column. Optional explicitly selected interactions extend the model without losing a term-wise decomposition.",
        "math": r"""
$$
\eta(x)=\beta_0+\sum_{j=1}^{d}f_j(x_j)
          +\sum_{S\in\mathcal I}f_S(x_S).
$$

Each $f_j$ is an independent feature network. NAMpy also supports ExU and centered-ReLU first layers.
""",
        "import": "from nampy.models import NAMClassifier, NAMLSS, NAMRegressor",
        "constructor": r"""
model = NAMRegressor(
    feature_layer="exu",
    layer_sizes=[32, 16],
    interactions=(("x1", "x2"),),
    output_regularization=1e-4,
    l2_regularization=1e-6,
    dropout=0.0,
)
model.get_params(deep=False)
""",
        "specific_text": "Use `feature_layer`, adaptive widths, or `feature_widths` to control individual shape networks. `interactions` is preferable to generating every combination.",
        "specific_code": r"""
model.set_params(
    adaptive_width=True,
    num_basis_functions=64,
    units_multiplier=2,
    feature_widths={"x1": 24},
)
if RUN_TRAINING:
    display(model.interaction_importance(X_test))
    model.plot_terms(X_test, pages=1)
""",
        "variants": "`NAMClassifier` adds probabilities and class labels. `NAMLSS(family='normal')` produces additive distribution-parameter predictors.",
    },
    {
        "filename": "04_snam.ipynb",
        "title": "SNAM",
        "subtitle": "Sparse Neural Additive Model",
        "concept": "SNAM reuses NAM shape networks and adds group-lasso regularization so entire feature or interaction subnetworks can be suppressed.",
        "math": r"""
$$
\eta(x)=\beta_0+\sum_j f_j(x_j)+\sum_{S\in\mathcal I}f_S(x_S),
\qquad
\Omega(\theta)=\lambda\sum_g\lVert\theta_g\rVert_2.
$$

Each group $g$ is one complete term network, making sparsity term-wise rather than weight-wise.
""",
        "import": "from nampy.models import SNAMClassifier, SNAMLSS, SNAMRegressor",
        "constructor": r"""
model = SNAMRegressor(
    layer_sizes=[32, 16],
    group_lasso_lambda=2e-4,
    group_lasso_include_interactions=True,
    interactions=(("x1", "x2"),),
    dropout=0.0,
)
model.get_params(deep=False)
""",
        "specific_text": "`group_lasso_lambda` controls term selection. Compare term importance across penalty strengths and use `model_complexity` for fitted parameter counts.",
        "specific_code": r"""
if RUN_TRAINING:
    display(model.term_importance(X_test))
    display(model.model_complexity())
""",
        "variants": "SNAM has `SNAMRegressor`, `SNAMClassifier`, and `SNAMLSS`; all share the same group penalty.",
    },
    {
        "filename": "05_sian.ipynb",
        "title": "SIAN",
        "subtitle": "Sparse Interaction Additive Network",
        "concept": "SIAN represents main effects and selected higher-order interactions in a block-masked ReLU network. It can discover interactions with Archipelago or accept an explicit sparse term set.",
        "math": r"""
$$
\eta(x)=\beta_0+\sum_j f_j(x_j)+
\sum_{S\in\widehat{\mathcal I}} f_S(x_S),
$$

where $\widehat{\mathcal I}$ is selected from inclusion/removal contrasts of a reference network. Main effects are always retained.
""",
        "import": "from nampy.models import SIANClassifier, SIANLSS, SIANRegressor",
        "constructor": r"""
# Explicit terms bypass discovery and keep this example inexpensive.
model = SIANRegressor(
    interactions=(("x1", "x2"),),
    layer_sizes=[16, 12, 8],
    execution_mode="block_masked",
    l1_regularization=5e-5,
)
model.get_params(deep=False)
""",
        "specific_text": "Without explicit `interactions`, configure `max_interaction_order`, thresholds, and heredity. Fitted models can switch losslessly between block-masked and independent term execution.",
        "specific_code": r"""
auto_model = SIANRegressor(
    max_interaction_order=2,
    interaction_thresholds=0.10,
    threshold_mode="fraction",
)
if RUN_TRAINING:
    display(model.selected_interactions_)
    model.compress_terms()
    compressed = model.predict(X_test)
    model.block_mask_terms()
    np.testing.assert_allclose(compressed, model.predict(X_test), rtol=1e-5, atol=1e-5)
    # For a fitted auto_model: auto_model.interaction_selection_table()
""",
        "variants": "`SIANRegressor`, `SIANClassifier`, and `SIANLSS` share interaction discovery and execution-mode controls.",
    },
    {
        "filename": "06_gpnam.ipynb",
        "title": "GPNAM",
        "subtitle": "Gaussian-process-inspired additive model",
        "concept": "GPNAM fixes a random Fourier feature map for each scalar input and estimates only additive coefficients. It approximates RBF-kernel functions but does not produce GP posterior covariance.",
        "math": r"""
$$
\phi_j(x_j)=\sqrt{\frac{2}{M}}
\left[\cos\!\left(z_mx_j/\ell_j+c_{mj}\right)\right]_{m=1}^{M},
\qquad
\eta(x)=\beta_0+\sum_j\phi_j(x_j)^\top w_j.
$$

Selected pairs add two-dimensional GP-NA2M feature maps.
""",
        "import": "from nampy.models import GPNAMClassifier, GPNAMLSS, GPNAMRegressor",
        "constructor": r"""
model = GPNAMRegressor(
    rff_num_feat=32,
    kernel_width="auto",
    solver="cg",
    ridge=0.05,
    interactions=(("x1", "x2"),),
    rff_random_state=7,
)
model.get_params(deep=False)
""",
        "specific_text": "Regression defaults to the fixed conjugate-gradient ridge solve. `basis_transform`, `basis_metadata`, and `model_complexity` expose the fitted finite basis.",
        "specific_code": r"""
if RUN_TRAINING:
    Phi = model.basis_transform(X_test)
    display(Phi.shape)
    display(model.basis_metadata())
    display(model.model_complexity())
    display(model.kernel_widths_)
""",
        "variants": "`GPNAMClassifier` and `GPNAMLSS` use gradient training. Distributional uncertainty is aleatoric, not GP posterior uncertainty.",
    },
    {
        "filename": "07_igann.ipynb",
        "title": "IGANN",
        "subtitle": "Interpretable Generalized Additive Neural Network",
        "concept": "IGANN begins with a sparse linear model and stagewise fits feature-wise extreme-learning-machine corrections. Random hidden weights stay fixed; only output coefficients are solved.",
        "math": r"""
$$
F_T(x)=\beta_0+\sum_j\beta_jx_j+
\eta\sum_{t=1}^{T}\sum_j w_{tj}^{\mathsf T}\sigma(a_{tj}x_j).
$$

The stagewise residual fit lets a feature remain linear unless nonlinear corrections improve the objective.
""",
        "import": "from nampy.models import IGANNClassifier, IGANNLSS, IGANNRegressor",
        "constructor": r"""
model = IGANNRegressor(
    n_hid=10,
    n_estimators=80,
    boost_rate=0.1,
    early_stopping=12,
    solver="native",
    sparse=0,
)
model.get_params(deep=False)
""",
        "specific_text": "`solver='native'` requests the upstream-style stagewise optimizer. `sparse>0` enables ABESS feature selection and requires the optional `igann-sparse` dependency.",
        "specific_code": r"""
if RUN_TRAINING:
    display(model.training_history())
    display(model.selected_features_)
    display(model.basis_metadata())
    display(model.model_complexity())
""",
        "variants": "Native training supports regression and binary classification. Multiclass `IGANNClassifier` and `IGANNLSS` use the fixed basis with the shared gradient engine.",
    },
    {
        "filename": "08_nbm.ipynb",
        "title": "NBM",
        "subtitle": "Neural Basis Model",
        "concept": "NBM learns shared neural basis functions for n-ary concept tuples, combines each tuple's basis responses independently, and applies a final linear output layer.",
        "math": r"""
For tuple $S$ of order $o$ and $K$ learned bases,

$$
b_o(x_S)\in\mathbb R^K,\qquad
h_S(x_S)=a_S^\top b_o(x_S)+c_S,
\qquad
\eta(x)=\beta_0+\sum_S v_Sh_S(x_S).
$$

The default grouped $1\times1$ convolution implements all $a_S$ independently.
""",
        "import": "from nampy.models import NBMClassifier, NBMLSS, NBMRegressor",
        "constructor": r"""
model = NBMRegressor(
    nary=[1, 2],
    num_bases=16,
    layer_sizes=[32, 16],
    featurizer="conv1d",
    sparse=False,
    batch_norm=True,
)
model.get_params(deep=False)
""",
        "specific_text": "Use `nary`, `order`, or `interaction_degree` to define tuples. `featurizer='conv1d'` matches upstream; `einsum` is equivalent. Sparse execution is a configuration option, not a separate class.",
        "specific_code": r"""
einsum_model = NBMRegressor(
    nary=[1, 2], num_bases=16, featurizer="einsum", sparse=False
)
sparse_model = NBMRegressor(
    nary=[1], num_bases=16, sparse=True, nary_ignore_input=0.0
)
if RUN_TRAINING:
    display({key: model.get_params(deep=False)[key] for key in ("nary", "featurizer", "sparse")})
    display(model.predict_components(X_test).terms.keys())
""",
        "variants": "NBM is available as `NBMRegressor`, `NBMClassifier`, and `NBMLSS`. PreTab emits one scalar encoded column per NBM concept.",
    },
    {
        "filename": "09_spam.ipynb",
        "title": "SPAM",
        "subtitle": "Scalable Polynomial Additive Model",
        "concept": "SPAM represents homogeneous polynomial blocks with low-rank projections, avoiding explicit construction of every polynomial coefficient tensor.",
        "math": r"""
For degree $q$ and rank $R_q$,

$$
p_q(x)=\sum_{r=1}^{R_q}\alpha_{qr}(w_{qr}^{\top}x)^q,
\qquad
\eta(x)=\beta_0+u(x)+\sum_{q=2}^{Q}p_q(x),
$$

where $u(x)$ contains unary effects and diagonal corrections.
""",
        "import": "from nampy.models import SPAMClassifier, SPAMLSS, SPAMRegressor",
        "constructor": r"""
model = SPAMRegressor(
    ranks=[16, 8],       # ranks for degrees 2 and 3
    reg_order=2,
    regularization_scale=1e-5,
    basis_l1_regularization=0.0,
    use_geometric_mean=True,
)
model.get_params(deep=False)
""",
        "specific_text": "`local_term_importance` expands the low-rank representation into sample-specific unary and distinct-variable polynomial terms.",
        "specific_code": r"""
if RUN_TRAINING:
    local = model.local_term_importance(X_test.iloc[:5], top_k=5)
    display(local[0])
    display(model.term_importance(X_test))
""",
        "variants": "`SPAMRegressor`, `SPAMClassifier`, and `SPAMLSS` share the same polynomial basis and regularization controls.",
    },
    {
        "filename": "10_nbm_spam.ipynb",
        "title": "NBM-SPAM",
        "subtitle": "Learned unary bases with low-rank polynomial heads",
        "concept": "NBM-SPAM learns unary NBM score channels, uses one segment linearly, and sends the remaining segments through degree-specific SPAM heads.",
        "math": r"""
Let $z_j^{(q)}(x_j)$ be unary NBM scores reserved for degree $q$. Then

$$
\eta(x)=\beta_0+\sum_j a_jz_j^{(1)}(x_j)
+\sum_{q=2}^{Q}\sum_{r=1}^{R_q}
\alpha_{qr}\left(\sum_jw_{qrj}z_j^{(q)}(x_j)\right)^q.
$$

Higher-order structure is created by SPAM; the NBM stage remains unary.
""",
        "import": "from nampy.models import NBMSPAMClassifier, NBMSPAMLSS, NBMSPAMRegressor",
        "constructor": r"""
model = NBMSPAMRegressor(
    num_bases=16,
    layer_sizes=[32, 16],
    ranks=[12],
    num_subnets=1,
    featurizer="conv1d",
    output_penalty=1e-4,
)
model.get_params(deep=False)
""",
        "specific_text": "`ranks` sets the SPAM head rank for each polynomial degree. `num_subnets` controls replicated unary basis score channels per polynomial.",
        "specific_code": r"""
if RUN_TRAINING:
    display({key: model.get_params(deep=False)[key] for key in ("ranks", "num_subnets", "featurizer")})
    display(model.predict_components(X_test).terms.keys())
""",
        "variants": "NBM-SPAM provides regressor, classifier, and LSS estimators with the same hybrid decomposition.",
    },
    {
        "filename": "11_treenam.ipynb",
        "title": "TreeNAM",
        "subtitle": "Additive differentiable trees",
        "concept": "TreeNAM assigns one soft neural decision tree to each feature and optional interaction, retaining a term-wise additive output.",
        "math": r"""
For a depth-$D$ soft tree,

$$
f_j(x_j)=\sum_{\ell=1}^{2^D}p_{j\ell}(x_j)v_{j\ell},
\qquad
\eta(x)=\beta_0+\sum_jf_j(x_j)+\sum_{S\in\mathcal I}f_S(x_S).
$$

$p_{j\ell}$ is the product of differentiable routing probabilities along a path.
""",
        "import": "from nampy.models import TreeNAMClassifier, TreeNAMLSS, TreeNAMRegressor",
        "constructor": r"""
model = TreeNAMRegressor(
    tree_depth=3,
    tree_lamda=1e-3,
    tree_temperature=1.0,
    use_hard_routing_in_eval=False,
    interactions=(("x1", "x2"),),
)
model.get_params(deep=False)
""",
        "specific_text": "`tree_temperature` controls soft routing; `use_hard_routing_in_eval=True` selects a single leaf path at inference. The tree penalty is included during training.",
        "specific_code": r"""
hard_routing_model = TreeNAMRegressor(
    tree_depth=3,
    tree_temperature=1.0,
    use_hard_routing_in_eval=True,
)
if RUN_TRAINING:
    display(model.interaction_importance(X_test))
    display(model.model_complexity())
""",
        "variants": "TreeNAM supports regression, classification, and LSS objectives.",
    },
    {
        "filename": "12_ensemble_treenam.ipynb",
        "title": "EnsembleTreeNAM",
        "subtitle": "Jointly trained TreeNAM ensemble",
        "concept": "EnsembleTreeNAM trains several complete TreeNAM learners jointly and averages their predictions and term contributions. It is not bootstrap bagging or boosting.",
        "math": r"""
For $M$ jointly optimized learners,

$$
\eta(x)=\frac{1}{M}\sum_{m=1}^{M}\eta_m(x),
\qquad
f_j(x_j)=\frac{1}{M}\sum_{m=1}^{M}f_{mj}(x_j).
$$
""",
        "import": "from nampy.models import (\n    EnsembleTreeNAMClassifier, EnsembleTreeNAMLSS, EnsembleTreeNAMRegressor,\n)",
        "constructor": r"""
model = EnsembleTreeNAMRegressor(
    num_estimators=3,
    aggregation="mean",
    tree_depth=3,
    tree_lamda=1e-3,
)
model.get_params(deep=False)
""",
        "specific_text": "`num_estimators` controls jointly trained learners. Use `NeuralEnsemble` instead when independently initialized or bootstrapped fitted models are required.",
        "specific_code": r"""
if RUN_TRAINING:
    display({"learners": model.get_params(deep=False)["num_estimators"]})
    components = model.predict_components(X_test)
    components.validate_additive_reconstruction()
""",
        "variants": "The joint ensemble is available for regression, classification, and LSS.",
    },
    {
        "filename": "13_nodegam.ipynb",
        "title": "NodeGAM",
        "subtitle": "Additive oblivious decision trees",
        "concept": "NodeGAM stacks differentiable oblivious trees with sparse feature selectors. GAM mode learns univariate terms; GA2M-style settings can learn pairwise terms.",
        "math": r"""
$$
\eta(x)=\beta_0+\sum_{t=1}^{T}g_t(x_{S_t}),
\qquad |S_t|\in\{1,2\}.
$$

Each tree uses the same split feature at a given depth, while sparse selector activations concentrate mass on a small feature set.
""",
        "import": "from nampy.models import NodeGAMClassifier, NodeGAMLSS, NodeGAMRegressor",
        "constructor": r"""
model = NodeGAMRegressor(
    num_trees=32,
    num_layers=2,
    depth=3,
    selector_activation="entmax15",
    bin_activation="entmoid15",
    interaction_degree=2,
    l2_interactions=1e-4,
)
model.get_params(deep=False)
""",
        "specific_text": "NodeGAM supports optional masked-reconstruction pretraining and recent-checkpoint averaging through fit-time controls.",
        "specific_code": r"""
if RUN_TRAINING:
    pretrained = NodeGAMRegressor(num_trees=32, num_layers=2, depth=3)
    pretrained.fit(
        X_train, y_train,
        pretrain_epochs=2,
        average_checkpoints=True,
        n_last_checkpoints=2,
        max_epochs=3,
        batch_size=64,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    display(pretrained.interaction_importance(X_test))
""",
        "variants": "NodeGAM has regressor, classifier, and LSS variants. Quantile preprocessing controls are ordinary PreTab parameters.",
    },
    {
        "filename": "14_natt.ipynb",
        "title": "NATT",
        "subtitle": "Neural attentive tabular additive model",
        "concept": "NATT models numerical features with separate subnetworks while categorical tokens are contextualized by a transformer and mapped to additive contributions.",
        "math": r"""
$$
\eta(x)=\beta_0+\sum_{j\in\mathrm{num}}f_j(x_j)
+g\!\left(\operatorname{Transformer}(E(x_{\mathrm{cat}}))\right)
+\sum_{S\in\mathcal I}f_S(x_S).
$$
""",
        "import": "from nampy.models import NATTLSS, NATTClassifier, NATTRegressor",
        "constructor": r"""
model = NATTRegressor(
    d_model=16,
    n_layers=2,
    n_heads=2,
    transformer_dim_feedforward=32,
    head_layer_sizes=(16,),
    layer_sizes=[16, 8],
    attn_dropout=0.0,
)
model.get_params(deep=False)
""",
        "specific_text": "`d_model`, `n_heads`, and `n_layers` control categorical attention; `layer_sizes` controls independent numerical shape networks.",
        "specific_code": r"""
if RUN_TRAINING:
    terms = model.predict_components(X_test).terms
    display({name: value.shape for name, value in terms.items()})
    model.plot_terms(X_test, pages=1)
""",
        "variants": "NATT supports regression, classification, and LSS objectives.",
    },
    {
        "filename": "15_namformer.ipynb",
        "title": "NAMformer",
        "subtitle": "Transformer-enhanced additive model",
        "concept": "NAMformer contextualizes all feature tokens, adds a global CLS-token head, and retains token-level and optional explicit interaction contributions.",
        "math": r"""
$$
h=\operatorname{Transformer}(E(x)),\qquad
\eta(x)=g(h_{\mathrm{CLS}})+\sum_jf_j(e_j)
+\sum_{S\in\mathcal I}f_S(e_S)+\beta_0.
$$

The global CLS term means the model is decomposed but not purely univariate-additive.
""",
        "import": "from nampy.models import NAMformerClassifier, NAMformerLSS, NAMformerRegressor",
        "constructor": r"""
model = NAMformerRegressor(
    d_model=16,
    n_layers=2,
    n_heads=2,
    transformer_dim_feedforward=32,
    head_layer_sizes=(16,),
    interactions=(("x1", "x2"),),
    attn_dropout=0.0,
)
model.get_params(deep=False)
""",
        "specific_text": "The CLS head captures unrestricted global context. Token heads and explicitly configured interaction heads remain available through `predict_components`.",
        "specific_code": r"""
if RUN_TRAINING:
    components = model.predict_components(X_test)
    display(components.terms.keys())
    display(model.interaction_importance(X_test))
""",
        "variants": "NAMformer provides regression, classification, and LSS estimators.",
    },
    {
        "filename": "16_qnam.ipynb",
        "title": "QNAM",
        "subtitle": "Noncrossing additive quantile regression",
        "concept": "QNAM predicts several conditional quantiles while constraining every intercept and term contribution to be nondecreasing across ordered quantile levels.",
        "math": r"""
For $\tau_1<\cdots<\tau_K$,

$$
q_{\tau_k}(x)=\beta_{0k}+\sum_j f_{jk}(x_j),
\qquad
q_{\tau_1}(x)\le\cdots\le q_{\tau_K}(x).
$$

Positive transformed increments enforce the ordering at every additive component.
""",
        "import": "from nampy.models import QNAMLSS",
        "constructor": r"""
quantiles = [0.1, 0.5, 0.9]
model = QNAMLSS(
    layer_sizes=[32, 16],
    monotone_transform="softplus",
    min_increment=0.0,
    dropout=0.0,
    distributional_kwargs={"quantiles": quantiles},
)
model.get_params(deep=False)
""",
        "specific_text": "QNAM is intentionally distributional-only. Set quantile levels with the estimator's `distributional_kwargs`; `predict` returns one ordered column per level.",
        "specific_code": r"""
if RUN_TRAINING:
    model.fit(
        X_train, y_train,
        max_epochs=3,
        batch_size=64,
        random_state=7,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    predicted_quantiles = model.predict(X_test)
    assert np.all(np.diff(predicted_quantiles, axis=1) >= -1e-7)
    score = model.score(X_test, y_test)
    display(model.evaluate(X_test, y_test))
    model.predict_components(X_test).validate_additive_reconstruction()
""",
        "variants": "QNAM exposes `QNAMLSS` only because its output contract is an ordered quantile distribution.",
        "custom_fit": True,
    },
    {
        "filename": "17_spline_nam.ipynb",
        "title": "SplineNAM",
        "subtitle": "Additive cubic spline network",
        "concept": "SplineNAM replaces feature MLPs with trainable cubic spline layers. Scalar preprocessing and optional identifiability centering keep each learned shape interpretable.",
        "math": r"""
$$
f_j(x_j)=\sum_{k=1}^{K}\theta_{jk}B_{jk}(x_j),
\qquad
\eta(x)=\beta_0+\sum_jf_j(x_j),
$$

with optional roughness control proportional to squared adjacent coefficient differences.
""",
        "import": "from nampy.models import SplineNAMRegressor",
        "constructor": r"""
model = SplineNAMRegressor(
    n_knots=10,
    learn_knots=False,
    identify=True,
    smoothing=1e-3,
    interactions=(("x1", "x2"),),
)
model.get_params(deep=False)
""",
        "specific_text": "`n_knots` controls basis resolution, `learn_knots` makes locations trainable, and `identify` centers shapes. SplineNAM requires scalar transformed features.",
        "specific_code": r"""
if RUN_TRAINING:
    display(model.term_importance(X_test))
    display(model.model_complexity())
    model.plot_terms(X_test, pages=1)
""",
        "variants": "The current public surface is `SplineNAMRegressor`; classification and LSS are deliberately unsupported.",
    },
]


def neural_notebook(spec: dict) -> dict:
    slug = Path(spec["filename"]).stem
    cells = [
        markdown(
            f"# {spec['title']}: {spec['subtitle']}\n\n{spec['concept']}",
            f"{slug}-title",
        ),
        markdown(f"## Model\n\n{spec['math']}", f"{slug}-math"),
        markdown(
            """## Shared estimator API

All neural estimators use `fit`, `predict`, `score`, `evaluate`, and
`predict_components`. The component result reconstructs predictions on the link
scale and supports shared term-importance and plotting utilities. Constructor
options such as `numerical_method` and `categorical_method` are forwarded to
PreTab and are fitted on training rows only.
""",
            f"{slug}-api",
        ),
        code(DATA_CELL, f"{slug}-data"),
        markdown("## Construct the estimator", f"{slug}-construct-title"),
        code(
            f"{spec['import']}\n\n{spec['constructor']}",
            f"{slug}-construct",
        ),
        markdown(
            """## Fit and inspect

Enable `RUN_TRAINING` above for a short demonstration. Real work should use a
larger validation set, enough epochs, and early stopping.
""",
            f"{slug}-fit-title",
        ),
    ]
    if not spec.get("custom_fit"):
        cells.append(code(GENERIC_FIT_CELL, f"{slug}-fit"))
    cells.extend(
        [
            markdown(
                f"## Model-specific controls\n\n{spec['specific_text']}",
                f"{slug}-specific-title",
            ),
            code(spec["specific_code"], f"{slug}-specific"),
            markdown(
                f"## Task variants and limits\n\n{spec['variants']}",
                f"{slug}-variants",
            ),
        ]
    )
    return notebook(cells)


def overview_notebook() -> dict:
    return notebook(
        [
            markdown(
                r"""# NAMpy model notebooks

These notebooks cover every registered neural architecture, the mgcv-aligned
GAM backend, and independent neural ensembles. They use a common estimator
surface while documenting each model's distinct mathematics and controls.
""",
                "overview-title",
            ),
            markdown(
                r"""## Shared additive contract

Most models expose a link-scale decomposition

$$
\eta(x)=\beta_0+\sum_t f_t(x_t)+o,
$$

where $o$ is an optional offset. `predict_components()` returns response,
link, terms, intercept, and offset and can validate this reconstruction.
""",
                "overview-contract",
            ),
            code(
                r"""
from nampy.neural.registry import architectures

registry = architectures()
{
    name: sorted(spec.capabilities)
    for name, spec in registry.items()
}
""",
                "overview-registry",
            ),
            markdown(
                """## Choosing a model

- **Statistical smooths:** GAM.
- **Independent neural shapes:** NAM; add group sparsity with SNAM.
- **Sparse discovered interactions:** SIAN.
- **Fixed bases:** GPNAM or IGANN.
- **Learned bases/polynomials:** NBM, SPAM, or NBM-SPAM.
- **Differentiable trees:** TreeNAM, EnsembleTreeNAM, or NodeGAM.
- **Attention:** NATT or NAMformer.
- **Ordered quantiles:** QNAM.
- **Trainable splines:** SplineNAM.
- **Baseline:** LinReg.
- **Between-fit variability:** NeuralEnsemble.
""",
                "overview-choice",
            ),
            markdown(
                """## Running the notebooks

From a repository checkout, install `nampy[all,docs]` and JupyterLab, start it
in the project root, and open `notebooks/`. Each notebook defaults to
`RUN_TRAINING = False`; set it to `True` to execute its short fit.
""",
                "overview-run",
            ),
        ]
    )


def gam_notebook() -> dict:
    return notebook(
        [
            markdown(
                """# GAM: mgcv-aligned generalized additive models

NAMpy's classical GAM backend aims to reproduce `mgcv` behavior for bases,
penalties, constraints, smoothing selection, prediction, and inference.
""",
                "gam-title",
            ),
            markdown(
                r"""## Model

For an exponential-family response,

$$
g\{\mathbb E(Y\mid x)\}=\eta(x)
=\beta_0+\sum_j f_j(x_j)+\sum_r z_r\beta_r,
$$

with basis representation $f_j(x)=B_j(x)\theta_j$ and roughness penalty
$\lambda_j\theta_j^\top S_j\theta_j$. REML, ML, GCV, or UBRE selects
smoothing parameters where supported.
""",
                "gam-math",
            ),
            code(
                r"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 180
data = pd.DataFrame({
    "x1": rng.uniform(-1.0, 1.0, n),
    "x2": rng.normal(size=n),
})
data["y"] = (
    np.sin(np.pi * data["x1"])
    + 0.3 * data["x2"]
    + rng.normal(0.0, 0.12, n)
)
RUN_TRAINING = False
""",
                "gam-data",
            ),
            markdown(
                """## Formula adapter

`GAMRegressor` and `GAMClassifier` provide sklearn-style methods while exposing
the fitted parity backend as `gam_`.
""",
                "gam-adapter-title",
            ),
            code(
                r"""
from nampy.models import GAMClassifier, GAMRegressor

model = GAMRegressor(
    formula="y ~ s(x1, k=10, bs='cr') + x2",
    optimize_smoothing=True,
    smoothing_method="reml",
)
model.get_params(deep=False)

if RUN_TRAINING:
    model.fit(data)
    predictions = model.predict(data)
    r2 = model.score(data, data["y"])
    metrics = model.evaluate(data, data["y"])
    components = model.predict_components(data)
    components.validate_additive_reconstruction()
    display(model.summary())
    display(model.term_importance(data))
""",
                "gam-adapter",
            ),
            markdown(
                """## GAM-specific functions

Use `standard_errors`, `lpmatrix`, `summary`, and `plot` for statistical
inference and smooth inspection. The raw `GAM` surface additionally exposes
prediction types, residuals, derivatives, and parity snapshots.
""",
                "gam-specific-title",
            ),
            code(
                r"""
if RUN_TRAINING:
    se = model.standard_errors(data)
    Xp = model.lpmatrix(data)
    terms = model.gam_.predict(data, type="terms")
    residuals = model.gam_.residuals(type="deviance")
    derivative = model.gam_.derivative(data, smooth_number=1, deriv=1)
    snapshot = model.gam_.parity_snapshot(data)
    plots = model.plot(pages=1, se=True)
""",
                "gam-specific",
            ),
            markdown(
                """## Direct backend and classification

Use `nampy.gam.GAM` when you need the raw mgcv-shaped interface. The adapters
are preferable for sklearn workflows. `GAMClassifier` supports binary targets
and adds `predict_proba` and `decision_function`.
""",
                "gam-variants-title",
            ),
            code(
                r"""
from nampy.gam import GAM

raw = GAM(
    formula="y ~ s(x1, k=10, bs='cr') + x2",
    family="gaussian",
    optimize_smoothing=True,
    smoothing_method="reml",
)

binary = (data["y"] > data["y"].median()).astype(int)
classifier = GAMClassifier(k=8, basis="cr")
if RUN_TRAINING:
    raw.fit(data=data)
    link = raw.predict(data, type="link")
    classifier.fit(data[["x1", "x2"]], binary)
    probabilities = classifier.predict_proba(data[["x1", "x2"]])
""",
                "gam-variants",
            ),
            markdown(
                """## Practical limits

Prefer formulas when term structure matters. Unsupported mgcv arguments raise
explicitly rather than degrading to approximations. Treat raw basis columns as
representation-dependent when eigenspaces are not uniquely oriented.
""",
                "gam-limits",
            ),
        ]
    )


def neural_ensemble_notebook() -> dict:
    return notebook(
        [
            markdown(
                """# NeuralEnsemble: independent fitted-model ensembles

`NeuralEnsemble` clones and independently fits any NAMpy neural regressor or
classifier, optionally on bootstrap samples. It is distinct from the jointly
trained `EnsembleTreeNAM` architecture.
""",
                "ensemble-title",
            ),
            markdown(
                r"""## Model

$$
\widehat y(x)=\frac1M\sum_{m=1}^{M}\widehat y_m(x),
\qquad
s_t(x)=\operatorname{sd}_m\{f_{m,t}(x)\}.
$$

The first quantity is the ensemble prediction; $s_t$ measures between-member
variation of an additive term, not calibrated posterior uncertainty.
""",
                "ensemble-math",
            ),
            code(DATA_CELL, "ensemble-data"),
            code(
                r"""
from nampy.models import NAMRegressor, NeuralEnsemble

base = NAMRegressor(layer_sizes=[24, 12], dropout=0.0)
model = NeuralEnsemble(
    base,
    n_estimators=3,
    random_state=7,
    n_jobs=1,
    bootstrap=True,
)
model.get_params(deep=False)
""",
                "ensemble-construct",
            ),
            markdown(
                """## Fit, predict, and inspect uncertainty

Fit parameters after `y` are forwarded to every cloned member. Each member owns
its preprocessing and fitted architecture.
""",
                "ensemble-fit-title",
            ),
            code(
                r"""
if RUN_TRAINING:
    model.fit(
        X_train, y_train,
        max_epochs=3,
        batch_size=64,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    predictions = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    components = model.predict_components(X_test, center=True)
    uncertainty = model.predict_component_uncertainty(X_test, center=True)
    components.validate_additive_reconstruction()
    display({"R2": r2, "members": uncertainty.n_estimators})
    display({name: values.mean() for name, values in uncertainty.term_std.items()})
""",
                "ensemble-fit",
            ),
            markdown(
                """## Limits

The generic ensemble accepts regressors and classifiers. LSS aggregation is
family-specific and therefore rejected. Classification adds `predict_proba`.
""",
                "ensemble-limits",
            ),
        ]
    )


README = """# Model notebooks

Concise conceptual and API tutorials for every supported NAMpy model. Start
with `00_overview.ipynb`; training cells are disabled by default.

| Notebook | Model |
|---|---|
| `01_gam.ipynb` | mgcv-aligned GAM and sklearn adapters |
| `02_linreg.ipynb` | LinReg |
| `03_nam.ipynb` | NAM |
| `04_snam.ipynb` | SNAM |
| `05_sian.ipynb` | SIAN |
| `06_gpnam.ipynb` | GPNAM |
| `07_igann.ipynb` | IGANN |
| `08_nbm.ipynb` | NBM |
| `09_spam.ipynb` | SPAM |
| `10_nbm_spam.ipynb` | NBM-SPAM |
| `11_treenam.ipynb` | TreeNAM |
| `12_ensemble_treenam.ipynb` | jointly trained EnsembleTreeNAM |
| `13_nodegam.ipynb` | NodeGAM |
| `14_natt.ipynb` | NATT |
| `15_namformer.ipynb` | NAMformer |
| `16_qnam.ipynb` | QNAM |
| `17_spline_nam.ipynb` | SplineNAM |
| `18_neural_ensemble.ipynb` | independent NeuralEnsemble |

From the repository root, install the complete environment and launch Jupyter:

```bash
pip install -e ".[all,docs]" jupyterlab
jupyter lab notebooks
```
"""


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    generated = {
        "00_overview.ipynb": overview_notebook(),
        "01_gam.ipynb": gam_notebook(),
        **{spec["filename"]: neural_notebook(spec) for spec in MODEL_SPECS},
        "18_neural_ensemble.ipynb": neural_ensemble_notebook(),
    }
    for filename, payload in generated.items():
        path = OUTPUT / filename
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text(README, encoding="utf-8")
    print(f"generated {len(generated)} notebooks in {OUTPUT}")


if __name__ == "__main__":
    main()
