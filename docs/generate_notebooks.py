"""Generate the theory-first model notebooks in ``docs/notebooks/``.

Sphinx invokes this module before reading its sources, and the checked-in
notebooks are verified against it by the documentation tests.

The notebooks are deterministic documentation artifacts. Training cells are
present but disabled by default so structural documentation checks stay fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
OUTPUT = DOCS / "notebooks"


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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2563EB", "#F97316", "#10B981", "#8B5CF6", "#EF4444"]

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
RUN_TRAINING = bool(globals().get("RUN_TRAINING", False))
"""


DATA_VISUAL_CELL = r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
axes[0].scatter(X["x1"], y, s=22, alpha=0.65, color=COLORS[0], edgecolor="none")
axes[0].set(title="Response across x1", xlabel="x1", ylabel="y")

group_order = ["a", "b", "c"]
group_values = [y[X["group"].to_numpy() == level] for level in group_order]
boxes = axes[1].boxplot(group_values, tick_labels=group_order, patch_artist=True)
for patch, color in zip(boxes["boxes"], COLORS, strict=False):
    patch.set_facecolor(color)
axes[1].set(title="Response by group", xlabel="group", ylabel="y")
fig.suptitle("Synthetic mixed-feature example", fontweight="bold")
plt.show()
plt.close(fig)
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
    components = model.predict_components(X_test)
    explanation = model.explain_terms(X_test, max_bins=24, center=True)
    importance = model.term_importance(X_test, center=True)
    interaction_importance = model.interaction_importance(X_test, center=True)
    display({"R2": r2, **metrics})
    display(importance)
    display(explanation.head(12))
"""


GENERIC_VISUAL_CELL = r"""
if RUN_TRAINING:
    observed = np.asarray(y_test).reshape(-1)
    fitted = np.asarray(predictions).reshape(-1)
    residual = observed - fitted

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    axes[0].scatter(observed, fitted, s=28, alpha=0.75, color=COLORS[0])
    lo = min(observed.min(), fitted.min())
    hi = max(observed.max(), fitted.max())
    axes[0].plot([lo, hi], [lo, hi], "--", color="#334155", linewidth=1.2)
    axes[0].set(title="Observed vs predicted", xlabel="Observed", ylabel="Predicted")

    axes[1].scatter(fitted, residual, s=28, alpha=0.75, color=COLORS[1])
    axes[1].axhline(0.0, color="#334155", linestyle="--", linewidth=1.2)
    axes[1].set(title="Residual pattern", xlabel="Predicted", ylabel="Residual")

    importance_plot = (
        importance.groupby("term", as_index=False)["importance"]
        .mean()
        .sort_values("importance")
    )
    axes[2].barh(importance_plot["term"], importance_plot["importance"], color=COLORS[2])
    axes[2].set(title="Mean absolute contribution", xlabel="Link-scale importance")
    fig.suptitle("Predictive fit and global explanation", fontweight="bold")
    plt.show()
    plt.close(fig)

    main_effects = explanation.loc[
        (explanation["term_type"] == "main") & (explanation["output"] == 0)
    ].copy()
    main_effects["numeric_value"] = pd.to_numeric(main_effects["value"], errors="coerce")
    main_effects = main_effects.dropna(subset=["numeric_value"])
    if not main_effects.empty:
        numeric_terms = set(main_effects["term"])
        top_term = next(
            term
            for term in importance_plot.sort_values("importance", ascending=False)["term"]
            if term in numeric_terms
        )
        curve = main_effects.loc[main_effects["term"] == top_term].sort_values("numeric_value")
        if not curve.empty:
            fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
            ax.plot(curve["numeric_value"], curve["contribution"], color=COLORS[3], linewidth=2.4)
            ax.scatter(curve["numeric_value"], curve["contribution"], s=20, color=COLORS[3])
            ax.axhline(0.0, color="#64748B", linewidth=1)
            ax.set(title=f"Binned explanation: {top_term}", xlabel=top_term, ylabel="Contribution")
            plt.show()
            plt.close(fig)

    term_figures = model.plot_terms(X_test, center=True, rug=True, pages=1)
    interaction_terms = interaction_importance["term"].astype(str).tolist()
    interactions_are_raw_columns = all(
        all(
            part in X_test.columns
            and pd.api.types.is_numeric_dtype(X_test[part])
            for part in term.split(":")
        )
        for term in interaction_terms
    )
    if interaction_terms and interactions_are_raw_columns:
        model.plot_interactions(X_test)
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
    numerical_preprocessing="standardization",
    categorical_preprocessing="one-hot",
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
    components.validate_additive_reconstruction(rtol=1e-5, atol=1e-6)
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
    score = model.score(X_test, np.asarray(y_test))
    display(model.evaluate(X_test, np.asarray(y_test)))
    components = model.predict_components(X_test)
    explanation = model.explain_terms(X_test, max_bins=24, center=True)
    importance = model.term_importance(X_test, center=True)
    display(importance)

    order = np.argsort(X_test["x1"].to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].scatter(
        X_test["x1"], y_test, s=24, alpha=0.45, color="#64748B", label="Observed"
    )
    for index, (quantile, color) in enumerate(zip(quantiles, COLORS, strict=False)):
        axes[0].plot(
            X_test["x1"].to_numpy()[order],
            predicted_quantiles[order, index],
            color=color,
            linewidth=2,
            label=f"q={quantile}",
        )
    axes[0].set(title="Ordered conditional quantiles", xlabel="x1", ylabel="y")
    axes[0].legend(frameon=False, ncol=2)

    interval_width = predicted_quantiles[:, -1] - predicted_quantiles[:, 0]
    axes[1].hist(interval_width, bins=12, color=COLORS[1], alpha=0.85)
    axes[1].set(title="80% interval width", xlabel="Upper − lower quantile", ylabel="Rows")
    fig.suptitle("Quantile fit and predictive spread", fontweight="bold")
    plt.show()
    plt.close(fig)

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


MODEL_THEORY = {
    "02_linreg.ipynb": r"""
The linear baseline is additive because each logical feature block contributes
through its own affine map. For scalar numerical inputs this is ordinary
multiple linear regression; one-hot or otherwise expanded PreTab blocks enter
as grouped linear effects. With a non-Gaussian objective the same linear
predictor is passed through the family link, giving the usual generalized
linear-model interpretation.

The important inductive bias is constant marginal effect on the link scale:
$\partial\eta/\partial x_j=w_j$. Consequently, a linear model is a useful
calibration point for every nonlinear architecture below. If a flexible model
does not materially outperform it, the additional shape complexity may not be
supported by the data.
""",
    "03_nam.ipynb": r"""
A NAM replaces each scalar coefficient with an independently parameterized
neural function. Because the feature networks do not exchange hidden state,
the output remains exactly decomposable. Identifiability is not automatic:
adding a constant to one shape and subtracting it from another leaves the
prediction unchanged. Centered component reporting fixes a reference gauge for
interpretation without changing predictions.

The architecture trades unrestricted multivariate approximation for readable
one-dimensional effects. Explicit interaction networks relax that assumption
only where requested. ExU first layers can represent sharp changes in slope,
while ordinary ReLU layers offer a more conventional smooth piecewise-linear
parameterization.
""",
    "04_snam.ipynb": r"""
SNAM applies a group penalty to all parameters belonging to one additive term.
Unlike element-wise $\ell_1$ regularization, group lasso removes or retains an
entire feature function, so sparsity aligns with the model's semantic units.
The nonsmooth penalty competes with predictive loss: larger values produce
fewer active terms but can shrink useful shapes.

The selection result is conditional on preprocessing, network capacity, and
optimization. It should therefore be treated as structured regularization,
not as a classical hypothesis test. Stability across seeds or resamples is a
more informative diagnostic than a single active-term set.
""",
    "05_sian.ipynb": r"""
SIAN represents candidate subsets of variables with masked network blocks and
uses a sparse selected frontier rather than fitting the full combinatorial
interaction lattice. Heredity rules constrain which higher-order terms can be
considered after lower-order evidence is found. Explicit term sets bypass
selection and turn SIAN into a known-structure additive interaction model.

Each selected subset owns one contribution, so the final predictor is still a
sum of named terms. The interaction detector and the final term estimator are
distinct stages: selection identifies a sparse support, while block-masked or
independent execution estimates the functions on that support.
""",
    "06_gpnam.ipynb": r"""
For a stationary RBF kernel, Bochner's theorem represents the kernel as an
expectation over sinusoidal features. GPNAM samples a finite deterministic
random-feature approximation per input and estimates additive ridge
coefficients in that basis. Kernel width controls the frequency scale and
therefore how rapidly a learned shape may vary.

The resulting estimator is finite-dimensional and computationally cheaper
than an exact Gaussian process. It borrows the GP function-space bias but does
not return a GP posterior or epistemic covariance. Pairwise GP-NA2M blocks use
the same construction on selected two-dimensional inputs.
""",
    "07_igann.ipynb": r"""
IGANN starts from a sparse linear predictor and repeatedly fits weak nonlinear
feature-wise corrections to current residuals or gradients. Each correction
uses an extreme-learning-machine block: hidden weights are random and fixed,
while output weights are obtained from a regularized linear solve. The boost
rate shrinks each stage before it enters the ensemble.

This produces a useful hierarchy. Features remain linear when nonlinear stages
offer little gain, and become nonlinear only through accumulated corrections.
Early stopping controls functional complexity, while the optional sparse stage
controls how many atomic features may participate.
""",
    "08_nbm.ipynb": r"""
NBM separates representation learning from concept-specific combination.
Shared neural basis functions are evaluated for every concept tuple, after
which independent linear weights assemble each tuple's contribution. Sharing
reduces the parameter cost relative to a separate deep network per feature,
while the final sum retains a named additive decomposition.

Unary NBM is a learned-basis additive model. Higher `nary` orders create
explicit interactions, and sparse execution can omit inactive tuples. The
Conv1D and einsum featurizers are two algebraically equivalent implementations
of the same grouped combination.
""",
    "09_spam.ipynb": r"""
SPAM approximates high-order polynomial coefficient tensors with sums of
rank-one factors. A rank-$R$ degree-$q$ block therefore scales with $R$ times
the input dimension instead of the full $d^q$ tensor. Unary and diagonal
corrections separate lower-order effects from distinct-variable interactions.

The low-rank factorization is efficient but its factors are not individually
identifiable: permutations and compensating rescalings can describe the same
polynomial. Interpretation should use the reconstructed local terms exposed by
`local_term_importance`, rather than raw factor weights.
""",
    "10_nbm_spam.ipynb": r"""
NBM-SPAM first learns nonlinear unary coordinates and then applies SPAM's
low-rank polynomial machinery to those coordinates. It can therefore express
interactions between learned feature shapes rather than only between raw
inputs. Separate subnet segments reserve channels for the linear and
higher-degree heads.

The decomposition is hierarchical: the NBM stage defines the coordinate
system and the SPAM stage defines polynomial interaction structure. Increasing
the number of bases, subnet replicas, degree, or rank expands different parts
of the model and should be tuned separately.
""",
    "11_treenam.ipynb": r"""
TreeNAM uses one differentiable decision tree for each additive term. Internal
nodes produce soft routing probabilities and leaves store output values, so a
term is the probability-weighted mean of its leaves. Temperature controls the
continuum between diffuse routing and near-discrete partitions.

Soft routing supports gradient training and smooth transitions; hard routing
at evaluation produces a piecewise-constant explanation but changes the
inference rule. Explicit interaction trees consume only their named feature
subset, preserving the distinction between main effects and interactions.
""",
    "12_ensemble_treenam.ipynb": r"""
This architecture places several complete TreeNAM learners inside one module
and minimizes loss for their mean prediction. All learners see the same
batches and are updated by one optimizer, so they may co-adapt; this is not
equivalent to fitting independent estimators and averaging afterwards.

Term contributions and tree penalties are averaged with the predictions. The
construction can reduce sensitivity to one tree parameterization, but it does
not by itself estimate between-fit uncertainty or create bootstrap diversity.
Use `NeuralEnsemble(TreeNAMRegressor(...))` for independent or bagged members.
""",
    "13_nodegam.ipynb": r"""
NodeGAM builds the predictor from layers of differentiable oblivious trees. An
oblivious tree uses the same selected feature and threshold rule across all
nodes at a given depth, enabling efficient vectorized evaluation. Sparse
selector activations concentrate each tree on one feature or a small feature
pair.

Restricting selectors to unary inputs yields a GAM-like decomposition; allowing
pairs gives a GA2M-like model. Because later layers consume transformed
representations, the architecture is more coupled than a collection of fully
independent NAM subnetworks even though term extraction remains available.
""",
    "14_natt.ipynb": r"""
NATT treats numerical and categorical inputs differently. Numerical variables
retain separate shape subnetworks, while categorical embeddings are processed
as tokens by self-attention. Attention lets a categorical contribution depend
on the context supplied by other categorical tokens.

The numerical part is strictly feature-additive; the contextual categorical
part is decomposed into reported token contributions but is not a set of
context-free univariate functions. Head count, embedding dimension, and depth
control contextual capacity and must satisfy the usual transformer dimension
constraints.
""",
    "15_namformer.ipynb": r"""
NAMformer embeds every feature as a token and contextualizes the complete token
set with a transformer. Token heads produce named contributions, explicit
interaction heads add selected terms, and a CLS-token head captures global
information not assigned to one feature.

This makes the output decomposable but not a classical additive model: a token
contribution may depend on other inputs through attention, and the CLS term is
multivariate. Interpretation is therefore conditional and prediction-specific,
closer to an additive accounting identity than to isolated marginal functions.
""",
    "16_qnam.ipynb": r"""
QNAM estimates several conditional quantiles with the pinball loss. Rather than
fitting each level independently, it parameterizes ordered increments with a
positive transform. Cumulative summation guarantees noncrossing quantiles for
the intercept and every additive term, hence also for their total.

The quantile dimension is part of the output contract. Wider intervals describe
conditional outcome dispersion, not parameter uncertainty. Dense quantile
grids increase output size, and the minimum-increment control trades strict
separation against the ability of adjacent levels to coincide.
""",
    "17_spline_nam.ipynb": r"""
SplineNAM expands each scalar input in a compact cubic B-spline basis and learns
the coefficients by gradient descent. Local basis support makes shapes cheaper
and more structured than general MLP subnetworks. A difference penalty on
neighboring coefficients discourages rapid oscillation.

Unlike statistical GAM smoothing selection, the smoothing weight is a neural
training hyperparameter rather than a REML/GCV-estimated variance component.
Identifiability centering fixes each term's additive constant. Learnable knots
increase flexibility but make the basis itself part of the nonconvex fit.
""",
}


MODEL_REFERENCES = {
    "02_linreg.ipynb": "- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*, Chapter 3.",
    "03_nam.ipynb": "- [Agarwal et al. (2021), Neural Additive Models](https://proceedings.nips.cc/paper_files/paper/2021/hash/251bd0442dfcc53b5a761e050f8022b8-Abstract.html).",
    "04_snam.ipynb": "- [Xu et al. (2022), Sparse Neural Additive Models](https://arxiv.org/abs/2202.12482).",
    "05_sian.ipynb": "- [Tsang et al. (2022), Sparse Interaction Additive Networks](https://arxiv.org/abs/2209.09326).",
    "06_gpnam.ipynb": "- [Gaussian Process Neural Additive Models (2024)](https://arxiv.org/abs/2402.12518).",
    "07_igann.ipynb": "- [Kraus et al. (2023), Interpretable Generalized Additive Neural Networks](https://doi.org/10.1016/j.ejor.2023.06.032).",
    "08_nbm.ipynb": "- [Radenovic et al. (2022), Neural Basis Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/37da88965c016dca016514df0e420c72-Abstract-Conference.html).",
    "09_spam.ipynb": "- [Scalable Polynomial Additive Models (2022)](https://arxiv.org/abs/2205.14108).",
    "10_nbm_spam.ipynb": "- [Neural Basis Models](https://arxiv.org/abs/2205.14120).\n- [Scalable Polynomial Additive Models](https://arxiv.org/abs/2205.14108).",
    "11_treenam.ipynb": "- NAMpy TreeNAM implementation notes and the differentiable soft-tree formulation used by `NeuralDecisionTree`.",
    "12_ensemble_treenam.ipynb": "- NAMpy composition of jointly trained TreeNAM learners; this is not presented as a separate published method.",
    "13_nodegam.ipynb": "- [Chang et al. (2021), Neural Generalized Additive Model](https://arxiv.org/abs/2106.01613).",
    "14_natt.ipynb": "- [Neural Attentive Tabular Transformer](https://openreview.net/forum?id=TdJ7lpzAkD).",
    "15_namformer.ipynb": "- [NAMformer (2025)](https://arxiv.org/abs/2504.08712).",
    "16_qnam.ipynb": "- Koenker, *Quantile Regression*; NAMpy's monotone-increment parameterization enforces noncrossing outputs.",
    "17_spline_nam.ipynb": "- Eilers and Marx (1996), Flexible smoothing with B-splines and penalties.",
}


MODEL_TASK_VARIANTS = {
    "02_linreg.ipynb": "task_variants = {\"regression\": LinRegRegressor(), \"classification\": LinRegClassifier(), \"distributional\": LinRegLSS()}",
    "03_nam.ipynb": "task_variants = {\"regression\": NAMRegressor(), \"classification\": NAMClassifier(), \"distributional\": NAMLSS()}",
    "04_snam.ipynb": "task_variants = {\"regression\": SNAMRegressor(), \"classification\": SNAMClassifier(), \"distributional\": SNAMLSS()}",
    "05_sian.ipynb": "task_variants = {\"regression\": SIANRegressor(), \"classification\": SIANClassifier(), \"distributional\": SIANLSS()}",
    "06_gpnam.ipynb": "task_variants = {\"regression\": GPNAMRegressor(), \"classification\": GPNAMClassifier(), \"distributional\": GPNAMLSS()}",
    "07_igann.ipynb": "task_variants = {\"regression\": IGANNRegressor(), \"classification\": IGANNClassifier(), \"distributional\": IGANNLSS()}",
    "08_nbm.ipynb": "task_variants = {\"regression\": NBMRegressor(), \"classification\": NBMClassifier(), \"distributional\": NBMLSS()}",
    "09_spam.ipynb": "task_variants = {\"regression\": SPAMRegressor(), \"classification\": SPAMClassifier(), \"distributional\": SPAMLSS()}",
    "10_nbm_spam.ipynb": "task_variants = {\"regression\": NBMSPAMRegressor(), \"classification\": NBMSPAMClassifier(), \"distributional\": NBMSPAMLSS()}",
    "11_treenam.ipynb": "task_variants = {\"regression\": TreeNAMRegressor(), \"classification\": TreeNAMClassifier(), \"distributional\": TreeNAMLSS()}",
    "12_ensemble_treenam.ipynb": "task_variants = {\"regression\": EnsembleTreeNAMRegressor(), \"classification\": EnsembleTreeNAMClassifier(), \"distributional\": EnsembleTreeNAMLSS()}",
    "13_nodegam.ipynb": "task_variants = {\"regression\": NodeGAMRegressor(), \"classification\": NodeGAMClassifier(), \"distributional\": NodeGAMLSS()}",
    "14_natt.ipynb": "task_variants = {\"regression\": NATTRegressor(), \"classification\": NATTClassifier(), \"distributional\": NATTLSS()}",
    "15_namformer.ipynb": "task_variants = {\"regression\": NAMformerRegressor(), \"classification\": NAMformerClassifier(), \"distributional\": NAMformerLSS()}",
    "16_qnam.ipynb": "task_variants = {\"distributional_quantiles\": model}",
    "17_spline_nam.ipynb": "task_variants = {\"regression\": model}",
}


def neural_notebook(spec: dict) -> dict:
    slug = Path(spec["filename"]).stem
    theory = MODEL_THEORY[spec["filename"]]
    references = MODEL_REFERENCES[spec["filename"]]
    task_variants = MODEL_TASK_VARIANTS[spec["filename"]]
    cells = [
        markdown(
            f"# {spec['title']}: {spec['subtitle']}\n\n{spec['concept']}",
            f"{slug}-title",
        ),
        markdown(f"## Model in one view\n\n{spec['math']}\n\n{theory}", f"{slug}-theory"),
        code(DATA_CELL, f"{slug}-data"),
        code(DATA_VISUAL_CELL, f"{slug}-data-visual"),
        markdown("## Fit the model", f"{slug}-construct-title"),
        code(
            f"{spec['import']}\n\n{spec['constructor']}",
            f"{slug}-construct",
        ),
    ]
    if not spec.get("custom_fit"):
        cells.append(code(GENERIC_FIT_CELL, f"{slug}-fit"))
        cells.extend(
            [
                markdown(
                    "## Evaluate and explain",
                    f"{slug}-visual-title",
                ),
                code(GENERIC_VISUAL_CELL, f"{slug}-visual"),
            ]
        )
    cells.extend(
        [
            markdown(
                f"## Model-specific controls\n\n{spec['specific_text']}",
                f"{slug}-specific-title",
            ),
            code(spec["specific_code"], f"{slug}-specific"),
            markdown(
                f"## Supported task variants\n\n{spec['variants']}",
                f"{slug}-variants",
            ),
            code(task_variants, f"{slug}-task-variants"),
            markdown(
                f"## References\n\n{references}",
                f"{slug}-references",
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
import matplotlib.pyplot as plt
import pandas as pd
from nampy.neural.registry import architectures

registry = architectures()
capability_map = {
    name: sorted(spec.capabilities)
    for name, spec in registry.items()
}
capability_map
""",
                "overview-registry",
            ),
            code(
                r"""
rows = [
    {"model": model, "task": task}
    for model, tasks in capability_map.items()
    for task in tasks
]
coverage = (
    pd.DataFrame(rows)
    .assign(supported=1)
    .pivot(index="model", columns="task", values="supported")
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
image = ax.imshow(coverage.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(coverage.columns)), coverage.columns, rotation=25, ha="right")
ax.set_yticks(range(len(coverage.index)), coverage.index)
ax.set_title("Registered architecture capabilities", fontweight="bold")
for row in range(coverage.shape[0]):
    for column in range(coverage.shape[1]):
        if coverage.iloc[row, column]:
            ax.text(column, row, "●", ha="center", va="center", color="#0F172A")
plt.show()
plt.close(fig)
""",
                "overview-visual",
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
in the project root, and open `docs/notebooks/`. Each notebook defaults to
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
                """# GAM: theory, fitting, inference, and constrained effects

This notebook develops one coherent energy-system case study while introducing
NAMpy's strict `mgcv`-aligned GAM backend. We model continuous demand, event
counts, binary alerts, positive costs, and distributional location/scale using
the same covariates. Training is disabled by default so the notebook remains a
fast documentation artifact; set the flags in the data cell to execute fits.
""",
                "gam-title",
            ),
            markdown(
                r"""## 1. Additive predictors and exponential-family responses

For an exponential-family response, the conditional mean $\mu_i$ is connected
to an additive predictor through a link $g$:

$$
g(\mu_i)=\eta_i=\beta_0+\sum_j f_j(x_{ij})
+\sum_r z_{ir}\beta_r+o_i.
$$

The offset $o_i$ has a fixed coefficient of one. Each smooth is represented in
a finite basis,

$$
f_j(x)=\sum_{k=1}^{K_j}B_{jk}(x)\beta_{jk}
=\mathbf B_j(x)^\top\boldsymbol\beta_j,
$$

so fitting is finite-dimensional. A roughness penalty
$\lambda_j\boldsymbol\beta_j^\top\mathbf S_j\boldsymbol\beta_j$ discourages
wiggly functions. The basis dimension $K_j$ sets an upper limit on complexity;
the smoothing parameter $\lambda_j$ determines how much of that capacity is
used.
""",
                "gam-math",
            ),
            markdown(
                r"""## 2. Penalized likelihood, identifiability, and uncertainty

For fixed smoothing parameters, coefficients maximize

$$
\ell_p(\boldsymbol\beta)
=\ell(\boldsymbol\beta)
-\frac12\boldsymbol\beta^\top
\left(\sum_j\lambda_j\mathbf S_j\right)\boldsymbol\beta.
$$

PIRLS supplies the local weighted least-squares problem for ordinary families.
Smooths commonly contain an unpenalized null space (for example constants or
linear trends), so side conditions remove overlap with the intercept and other
terms. This is why a reported smooth is an identified contribution rather than
an arbitrary basis coefficient vector.

The influence matrix $\mathbf A$ maps working observations to fitted values.
Its trace gives the model effective degrees of freedom (EDF); term-wise EDF
measures how much flexibility survives penalization. NAMpy exposes Bayesian
and frequentist covariance choices. Pointwise standard errors quantify
coefficient uncertainty conditional on the fitted model and, when requested,
can include smoothing-parameter uncertainty.
""",
                "gam-estimation-theory",
            ),
            markdown(
                r"""## 3. Selecting smoothness

GCV and UBRE/AIC estimate prediction risk, whereas ML and REML optimize a
Laplace-approximated marginal likelihood. REML usually offers stable default
behavior for Gaussian and many generalized responses because fixed effects are
accounted for when estimating smoothness. Outer iteration alternates a full
coefficient fit with updates to log smoothing parameters.

NAMpy ports the supported `mgcv` optimizer routes rather than treating their
names as interchangeable aliases:

- `outer_newton`: derivative-based outer Newton iteration;
- `bfgs`: quasi-Newton updates;
- `efs`: extended Fellner-Schall updates where supported; and
- `optim`: the `optim`-compatible route.

An optimizer is not a modeling assumption by itself: family, criterion,
penalty structure, and optimizer jointly determine whether a route is
supported. Unsupported combinations raise explicitly.
""",
                "gam-smoothing-theory",
            ),
            markdown(
                """## Primary references

- [Wood (2011), *Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models*](https://doi.org/10.1111/j.1467-9868.2010.00749.x). This is the main reference for stable ML/REML smoothness selection and outer iteration.
- [Pya and Wood (2015), *Shape constrained additive models*](https://doi.org/10.1007/s11222-013-9448-7). This develops the constrained P-spline parameterization, smoothness selection, and inference used in the shape-constrained section.

NAMpy follows the upstream `mgcv` and `scam` implementations as its behavioral
specification; the papers explain the statistical construction.
""",
                "gam-references",
            ),
            code(
                r"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2563EB", "#F97316", "#10B981", "#8B5CF6", "#EF4444"]

rng = np.random.default_rng(7)
n = 240
time = np.arange(n)
hour = time % 24
temperature = 12 + 9 * np.sin(2 * np.pi * time / n) + rng.normal(0, 1.8, n)
humidity = np.clip(65 - 1.1 * temperature + rng.normal(0, 5, n), 15, 95)
site = pd.Categorical(rng.choice(["north", "central", "south"], size=n))
exposure = rng.uniform(80, 160, n)

site_effect = pd.Series(site).map(
    {"north": 1.4, "central": 0.0, "south": -1.0}
).astype(float).to_numpy()
daily_cycle = 2.2 * np.sin(2 * np.pi * hour / 24)
temperature_effect = 0.055 * (temperature - 18) ** 2
mean_demand = 18 + daily_cycle + temperature_effect + site_effect

data = pd.DataFrame({
    "time": time,
    "hour": hour,
    "temperature": temperature,
    "humidity": humidity,
    "site": site,
    "exposure": exposure,
    "log_exposure": np.log(exposure),
})
data["demand"] = mean_demand + rng.normal(0, 0.8, n)

event_rate = np.exp(-4.2 + 0.035 * (temperature - 18) ** 2 + 0.008 * humidity)
data["events"] = rng.poisson(exposure * event_rate)
alert_probability = 1 / (1 + np.exp(-(-3.2 + 0.12 * (temperature - 22))))
data["alert"] = rng.binomial(1, alert_probability)

mean_cost = np.exp(2.2 + 0.018 * humidity + 0.025 * np.maximum(temperature - 20, 0))
data["cost"] = rng.gamma(shape=8.0, scale=mean_cost / 8.0)
data["efficiency"] = np.clip(
    rng.beta(8 + 0.12 * temperature, 4 + 0.03 * humidity), 1e-4, 1 - 1e-4
)

data["dose"] = np.linspace(0, 1, n)
shape_mean = 4 + 7 * (1 - np.exp(-4 * data["dose"]))
data["shape_y"] = shape_mean + rng.normal(0, 0.25, n)
data["shape_count"] = rng.poisson(np.exp(0.4 + 1.2 * data["dose"]))

RUN_TRAINING = bool(globals().get("RUN_TRAINING", False))
RUN_EXTENDED_FITS = bool(globals().get("RUN_EXTENDED_FITS", False))
""",
                "gam-data",
            ),
            code(
                r"""
site_colors = {"north": COLORS[0], "central": COLORS[1], "south": COLORS[2]}
fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)

axes[0, 0].plot(data["time"], data["demand"], color=COLORS[0], linewidth=1.4)
axes[0, 0].set(title="Demand through time", xlabel="Hour index", ylabel="Demand")

for level in data["site"].cat.categories:
    subset = data["site"] == level
    axes[0, 1].scatter(
        data.loc[subset, "temperature"],
        data.loc[subset, "demand"],
        s=22,
        alpha=0.65,
        color=site_colors[str(level)],
        label=str(level),
    )
axes[0, 1].set(title="Nonlinear temperature response", xlabel="Temperature", ylabel="Demand")
axes[0, 1].legend(frameon=False)

axes[1, 0].scatter(data["humidity"], data["events"] / data["exposure"], s=22, alpha=0.65, color=COLORS[3])
axes[1, 0].set(title="Event rate and exposure", xlabel="Humidity", ylabel="Events / exposure")

axes[1, 1].scatter(data["dose"], data["shape_y"], s=20, alpha=0.45, color=COLORS[4])
axes[1, 1].plot(data["dose"], shape_mean, color="#0F172A", linewidth=2.2, label="Generating shape")
axes[1, 1].set(title="Monotone saturation experiment", xlabel="Dose", ylabel="Response")
axes[1, 1].legend(frameon=False)

fig.suptitle("One energy-system story, several response types", fontweight="bold")
plt.show()
plt.close(fig)
""",
                "gam-data-visual",
            ),
            markdown(
                """## 4. First model: Gaussian demand with cyclic and ordinary smooths

Demand has a nonlinear temperature response, a 24-hour cycle, and site-level
differences. A cubic regression spline estimates the temperature shape; a
cyclic cubic spline forces midnight and the end of the day to join smoothly;
the categorical site enters parametrically.

`GAMRegressor` and `GAMClassifier` provide sklearn-style methods while exposing
the fitted parity backend as `gam_`.
""",
                "gam-adapter-title",
            ),
            code(
                r"""
from nampy.models import GAMClassifier, GAMRegressor

model = GAMRegressor(
    formula=(
        "demand ~ s(temperature, k=10, bs='cr') "
        "+ s(hour, k=8, bs='cc') + site"
    ),
    optimize_smoothing=True,
    smoothing_method="reml",
)
model.get_params(deep=False)

if RUN_TRAINING:
    model.fit(data)
    predictions = model.predict(data)
    r2 = model.score(data, data["demand"])
    metrics = model.evaluate(data, data["demand"])
    components = model.predict_components(data)
    components.validate_additive_reconstruction()
    explanation = model.explain_terms(data, max_bins=30)
    importance = model.term_importance(data)
    display(model.summary())
    display(importance)
    display(explanation.head(12))
""",
                "gam-adapter",
            ),
            code(
                r"""
if RUN_TRAINING:
    observed = data["demand"].to_numpy()
    residual = observed - predictions
    importance_plot = importance.sort_values("importance")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    axes[0].plot(data["time"], observed, color="#94A3B8", linewidth=1.2, label="Observed")
    axes[0].plot(data["time"], predictions, color=COLORS[0], linewidth=1.8, label="Fitted")
    axes[0].set(title="Demand fit", xlabel="Hour index", ylabel="Demand")
    axes[0].legend(frameon=False)

    axes[1].scatter(predictions, residual, s=24, alpha=0.65, color=COLORS[1])
    axes[1].axhline(0, color="#334155", linestyle="--", linewidth=1)
    axes[1].set(title="Residual pattern", xlabel="Fitted", ylabel="Residual")

    axes[2].barh(importance_plot["term"], importance_plot["importance"], color=COLORS[2])
    axes[2].set(title="Term importance", xlabel="Mean absolute contribution")
    fig.suptitle("Fit quality and additive explanation", fontweight="bold")
    plt.show()
    plt.close(fig)
""",
                "gam-adapter-visual",
            ),
            markdown(
                r"""## 5. Smooth construction gallery

Different bases encode different boundary behavior and penalty null spaces.
The choice should follow the covariate geometry rather than be treated as a
generic tuning label.

| Term | Role |
|---|---|
| `cr`, `cs` | cubic regression spline; `cs` adds null-space shrinkage |
| `cc` | cyclic cubic spline for periodic covariates |
| `ps` | P-spline with difference penalties |
| `tp`, `ts` | thin-plate regression spline; `ts` adds shrinkage |
| `te(...)` | scale-invariant tensor product including main-effect directions |
| `ti(...)` | tensor interaction with marginal main-effect directions removed |
| `re` | random-effect/ridge-penalized factor block |
| `fs`, `sz` | factor-specific smooths with different identifiability structures |

Tensor products construct marginal bases first and combine their columns and
penalties. `te` is suitable for a whole surface; `ti` is useful when marginal
main effects are already present.
""",
                "gam-smooth-gallery-title",
            ),
            code(
                r"""
from nampy.gam import GAM

smooth_examples = {
    "cubic": GAM(formula="demand ~ s(temperature, bs='cr', k=10)"),
    "shrinkage_cubic": GAM(formula="demand ~ s(temperature, bs='cs', k=10)"),
    "cyclic": GAM(formula="demand ~ s(hour, bs='cc', k=8)"),
    "p_spline": GAM(formula="demand ~ s(temperature, bs='ps', k=10)"),
    "thin_plate": GAM(formula="demand ~ s(temperature, bs='tp', k=10)"),
    "shrinkage_thin_plate": GAM(formula="demand ~ s(temperature, bs='ts', k=10)"),
    "tensor_surface": GAM(
        formula="demand ~ te(temperature, humidity, bs=['cr', 'ps'], k=[6, 6])"
    ),
    "tensor_interaction": GAM(
        formula=(
            "demand ~ s(temperature, bs='cr', k=8) "
            "+ s(humidity, bs='ps', k=8) "
            "+ ti(temperature, humidity, bs=['cr', 'ps'], k=[6, 6])"
        )
    ),
    "random_effect": GAM(formula="demand ~ s(site, bs='re')"),
    "factor_smooth": GAM(
        formula="demand ~ s(site, temperature, bs='fs', k=5, xt='cr')"
    ),
    "sum_to_zero_factor_smooth": GAM(
        formula="demand ~ s(site, temperature, bs='sz', k=5, xt='cr')"
    ),
}
sorted(smooth_examples)
""",
                "gam-smooth-gallery",
            ),
            markdown(
                """## 6. Criteria and outer optimizers

The following objects demonstrate the supported public optimizer names. In
practice, compare converged endpoints and diagnostics rather than choosing the
optimizer with the smallest iteration count. `fixed` smoothing is also useful
for reproducible sensitivity analysis and constructor-level parity work.
""",
                "gam-optimizers-title",
            ),
            code(
                r"""
optimizer_models = {
    name: GAM(
        formula="demand ~ s(temperature, bs='cr', k=10)",
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="reml",
        smoothing_optimizer=name,
    )
    for name in ("outer_newton", "bfgs", "efs", "optim")
}

fixed_model = GAM(
    formula="demand ~ s(temperature, bs='cr', k=10)",
    family="gaussian",
    optimize_smoothing=False,
    smoothing_method="fixed",
    smoothing_params=[0.7],
)

if RUN_EXTENDED_FITS:
    fitted_optimizers = {
        name: candidate.fit(data=data) for name, candidate in optimizer_models.items()
    }
    optimizer_summary = pd.DataFrame(
        {
            "optimizer": name,
            "criterion": candidate.fit_result(include_covariances=False).criterion_value,
            "log_sp": np.log(candidate.fit_result(include_covariances=False).smoothing_params[0]),
        }
        for name, candidate in fitted_optimizers.items()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    axes[0].bar(optimizer_summary["optimizer"], optimizer_summary["criterion"], color=COLORS[:4])
    axes[0].set(title="Converged REML criterion", ylabel="Criterion")
    axes[1].bar(optimizer_summary["optimizer"], optimizer_summary["log_sp"], color=COLORS[:4])
    axes[1].set(title="Selected smoothness", ylabel="log smoothing parameter")
    plt.show()
    plt.close(fig)
""",
                "gam-optimizers",
            ),
            markdown(
                r"""## 7. Families and links in the same application

The linear predictor always has additive structure, but the family determines
the conditional distribution and the inverse-link interpretation.

| Response | Example family/link | Interpretation |
|---|---|---|
| continuous demand | Gaussian/identity | additive changes in the mean |
| event count | Poisson/log | additive log-rate; exposure enters as an offset |
| binary alert | Binomial/logit | additive log odds |
| positive cost | Gamma/log | additive log mean with mean-dependent variance |
| overdispersed count | negative binomial/log | Poisson-like mean with extra dispersion |
| proportion | beta regression/logit | conditional mean in $(0,1)$ plus precision |
| positive mass with zeros | Tweedie/log | compound Poisson-Gamma mean |
| ordered category | ordered categorical | thresholded latent predictor |

Family choice is not merely a transformation of `y`: it determines variance,
deviance, likelihood, working weights, residuals, and sometimes additional
parameters optimized jointly with smoothness.
""",
                "gam-families-title",
            ),
            code(
                r"""
family_models = {
    "gaussian": GAM(
        formula="demand ~ s(temperature, bs='cr', k=10) + site",
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="reml",
    ),
    "poisson_offset": GAM(
        formula=(
            "events ~ s(temperature, bs='cr', k=9) "
            "+ s(humidity, bs='ps', k=8) + offset(log_exposure)"
        ),
        family="poisson",
        optimize_smoothing=True,
        smoothing_method="ubre",
        smoothing_optimizer="bfgs",
    ),
    "binomial": GAM(
        formula="alert ~ s(temperature, bs='cr', k=9) + site",
        family={"name": "binomial", "link": "logit"},
        optimize_smoothing=True,
        smoothing_method="reml",
    ),
    "gamma": GAM(
        formula="cost ~ s(humidity, bs='cr', k=9) + s(temperature, bs='ps', k=8)",
        family={"name": "gamma", "link": "log"},
        optimize_smoothing=True,
        smoothing_method="reml",
    ),
    "negative_binomial": GAM(
        formula="events ~ s(temperature, bs='cr', k=9) + offset(log_exposure)",
        family={"name": "nb"},
        optimize_smoothing=True,
        smoothing_method="reml",
    ),
    "tweedie": GAM(
        formula="cost ~ s(temperature, bs='cr', k=9)",
        family={"name": "tweedie", "link": "log"},
        optimize_smoothing=True,
        smoothing_method="reml",
    ),
    "beta": GAM(
        formula="efficiency ~ s(temperature, bs='cr', k=9)",
        family={"name": "betar", "link": "logit", "theta": 12.0},
        optimize_smoothing=True,
        smoothing_method="reml",
    ),
}

if RUN_EXTENDED_FITS:
    fitted_families = {
        name: candidate.fit(data=data) for name, candidate in family_models.items()
    }
    response_columns = {
        "gaussian": "demand",
        "poisson_offset": "events",
        "binomial": "alert",
        "gamma": "cost",
        "negative_binomial": "events",
        "tweedie": "cost",
        "beta": "efficiency",
    }
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), constrained_layout=True)
    for ax, (name, candidate) in zip(axes.flat, fitted_families.items(), strict=False):
        observed = data[response_columns[name]].to_numpy()
        fitted = np.asarray(candidate.predict(data, type="response")).reshape(-1)
        ax.scatter(observed, fitted, s=18, alpha=0.55, color=COLORS[0])
        lo, hi = min(observed.min(), fitted.min()), max(observed.max(), fitted.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#475569", linewidth=1)
        ax.set(title=name.replace("_", " "), xlabel="Observed", ylabel="Fitted")
    axes.flat[-1].axis("off")
    fig.suptitle("Response-scale checks across families", fontweight="bold")
    plt.show()
    plt.close(fig)
""",
                "gam-families",
            ),
            markdown(
                r"""## 8. Multi-linear-predictor GAMs

General families can own more than one additive predictor. For Gaussian
location-scale modeling, NAMpy's `gaulss` family uses one predictor for the
conditional mean and another for the scale parameter. Each predictor has its
own formula, design matrix, smooths, penalties, and offsets, while joint
likelihood derivatives couple their estimation.

This is different from fitting two unrelated GAMs: both predictors describe
one response distribution and their covariance follows from one joint fit.
""",
                "gam-multipredictor-title",
            ),
            code(
                r"""
location_scale = GAM(
    formula=[
        "demand ~ s(temperature, bs='cr', k=9) + s(hour, bs='cc', k=8) + site",
        "~ s(temperature, bs='cr', k=7)",
    ],
    family="gaulss",
    optimize_smoothing=True,
    smoothing_method="ml",
    smoothing_optimizer="outer_newton",
)

gamma_location_scale = GAM(
    formula=[
        "cost ~ s(temperature, bs='cr', k=8) + s(humidity, bs='ps', k=8)",
        "~ 1",
    ],
    family="gammals",
    optimize_smoothing=True,
    smoothing_method="ml",
    smoothing_optimizer="outer_newton",
    select=True,
)

if RUN_EXTENDED_FITS:
    location_scale.fit(data=data)
    mean_and_scale, mean_and_scale_se = location_scale.predict(
        data, type="response", return_se=True
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    axes[0].scatter(data["demand"], mean_and_scale[:, 0], s=22, alpha=0.6, color=COLORS[0])
    axes[0].set(title="Location predictor", xlabel="Observed demand", ylabel="Estimated mean")
    axes[1].scatter(data["temperature"], mean_and_scale[:, 1], s=22, alpha=0.6, color=COLORS[3])
    axes[1].set(title="Scale predictor", xlabel="Temperature", ylabel="Estimated scale")
    fig.suptitle("One response, two linked additive predictors", fontweight="bold")
    plt.show()
    plt.close(fig)
""",
                "gam-multipredictor",
            ),
            markdown(
                r"""## 9. Shape-constrained GAM: monotone saturation

Engineering knowledge says the response in the dose experiment should never
decrease. An unconstrained smooth can violate that assumption in sparse regions.
The `mpi` basis uses a nonlinear coefficient transform whose positive
increments enforce monotonic increase, while a P-spline penalty still controls
roughness.

Pya and Wood's construction changes the coefficient parameterization rather
than clipping predictions after fitting. Consequently, prediction,
derivatives, covariance transport, and diagnostics operate on a genuinely
shape-valid fitted function.
""",
                "gam-shape-title",
            ),
            code(
                r"""
monotone_fixed = GAM(
    formula="shape_y ~ s(dose, bs='mpi', k=10)",
    family="gaussian",
    optimize_smoothing=False,
    smoothing_method="fixed",
    smoothing_params=[0.5],
)

monotone_selected = GAM(
    formula="shape_count ~ s(dose, bs='mpi', k=10)",
    family="poisson",
    optimize_smoothing=True,
    smoothing_method="ubre",
    smoothing_optimizer="bfgs",
)

if RUN_EXTENDED_FITS:
    monotone_fixed.fit(data=data)
    shape_derivative = monotone_fixed.derivative(smooth_number=1, deriv=1)
    assert np.min(shape_derivative.derivative) >= -1e-7
    shape_fit, shape_se = monotone_fixed.predict(data, type="response", return_se=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    axes[0].scatter(data["dose"], data["shape_y"], s=18, alpha=0.4, color="#64748B")
    axes[0].plot(data["dose"], shape_fit, color=COLORS[0], linewidth=2.2)
    axes[0].fill_between(data["dose"], shape_fit - 2 * shape_se, shape_fit + 2 * shape_se, color=COLORS[0], alpha=0.18)
    axes[0].set(title="Monotone fitted response", xlabel="Dose", ylabel="Response")
    axes[1].plot(data["dose"], shape_derivative.derivative, color=COLORS[1], linewidth=2.2)
    axes[1].axhline(0, color="#334155", linestyle="--", linewidth=1)
    axes[1].set(title="Estimated first derivative", xlabel="Dose", ylabel="Derivative")
    plt.show()
    plt.close(fig)
""",
                "gam-shape",
            ),
            markdown(
                """## 10. Prediction, inference, and diagnostics

Use `standard_errors`, `lpmatrix`, `summary`, and `plot` for statistical
inference and smooth inspection. The raw `GAM` surface additionally exposes
prediction types, residuals, derivatives, ANOVA, `k_check`, `gam_check`,
concurvity, covariance choices, and parity snapshots.
""",
                "gam-specific-title",
            ),
            code(
                r"""
if RUN_TRAINING:
    se = model.standard_errors(data)
    Xp = model.lpmatrix(data)
    terms = model.gam_.predict(data, type="terms")
    response_with_se = model.gam_.predict(data, type="response", return_se=True)
    residuals = model.gam_.residuals(type="deviance")
    derivative_model = GAM(
        formula="demand ~ s(temperature, bs='ps', k=10)",
        family="gaussian",
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[0.7],
    ).fit(data=data)
    derivative = derivative_model.derivative(data, smooth_number=1, deriv=1)
    summary = model.gam_.summary()
    k_diagnostics = model.gam_.k_check()
    check_report = model.gam_.gam_check()
    concurvity = model.gam_.concurvity()
    covariance = model.gam_.vcov(freq=False, unconditional=True)
    snapshot = model.gam_.parity_snapshot(data)
    plots = model.plot(pages=1, se=True)

    from scipy import stats

    fitted = np.asarray(predictions).reshape(-1)
    deviance_residuals = np.asarray(residuals).reshape(-1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].scatter(fitted, deviance_residuals, s=22, alpha=0.65, color=COLORS[0])
    axes[0, 0].axhline(0, color="#334155", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Deviance residuals", xlabel="Fitted", ylabel="Residual")

    stats.probplot(deviance_residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Normal Q–Q check")

    k_values = k_diagnostics["k_index"].to_numpy(dtype=float)
    axes[1, 0].barh(k_diagnostics.index.astype(str), k_values, color=COLORS[2])
    axes[1, 0].axvline(1, color="#334155", linestyle="--", linewidth=1)
    axes[1, 0].set(title="Basis-dimension diagnostic", xlabel="k-index")

    concurvity_values = np.asarray(concurvity["values"], dtype=float)
    image = axes[1, 1].imshow(concurvity_values, vmin=0, vmax=1, cmap="magma", aspect="auto")
    axes[1, 1].set_xticks(range(len(concurvity["labels"])), concurvity["labels"], rotation=25, ha="right")
    axes[1, 1].set_yticks(range(len(concurvity["measure_names"])), concurvity["measure_names"])
    axes[1, 1].set_title("Concurvity")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.suptitle("GAM diagnostic dashboard", fontweight="bold")
    plt.show()
    plt.close(fig)
""",
                "gam-specific",
            ),
            markdown(
                """## 11. Direct backend, adapters, and classification

Use `nampy.gam.GAM` when you need the raw mgcv-shaped interface. The adapters
are preferable for sklearn workflows. `GAMClassifier` supports binary targets
and adds `predict_proba` and `decision_function`.
""",
                "gam-variants-title",
            ),
            code(
                r"""
raw = GAM(
    formula="demand ~ s(temperature, k=10, bs='cr') + site",
    family="gaussian",
    optimize_smoothing=True,
    smoothing_method="reml",
)

classifier = GAMClassifier(
    formula="alert ~ s(temperature, k=9, bs='cr') + site",
    basis="cr",
    smoothing_method="reml",
)
if RUN_TRAINING:
    raw.fit(data=data)
    link = raw.predict(data, type="link")
    raw_terms = raw.predict(data, type="terms", return_se=True)
    classifier.fit(data, data["alert"])
    probabilities = classifier.predict_proba(data)
    labels = classifier.predict(data)
    probability_frame = pd.DataFrame(
        {"temperature": data["temperature"], "alert": data["alert"], "probability": probabilities[:, 1]}
    )
    probability_frame["temperature_bin"] = pd.qcut(
        probability_frame["temperature"], q=10, duplicates="drop"
    )
    calibration = probability_frame.groupby("temperature_bin", observed=True).agg(
        temperature=("temperature", "mean"),
        observed_rate=("alert", "mean"),
        predicted_rate=("probability", "mean"),
    )
    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    ax.plot(calibration["temperature"], calibration["observed_rate"], "o-", color=COLORS[1], label="Observed")
    ax.plot(calibration["temperature"], calibration["predicted_rate"], "o-", color=COLORS[0], label="Predicted")
    ax.set(title="Alert probability across temperature", xlabel="Temperature", ylabel="Probability")
    ax.legend(frameon=False)
    plt.show()
    plt.close(fig)
""",
                "gam-variants",
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
                r"""## Ensemble in one view

$$
\widehat y(x)=\frac1M\sum_{m=1}^{M}\widehat y_m(x),
\qquad
s_t(x)=\operatorname{sd}_m\{f_{m,t}(x)\}.
$$

The first quantity is the ensemble prediction; $s_t$ measures between-member
variation of an additive term, not calibrated posterior uncertainty.

Independent refits can vary because of initialization, stochastic optimization,
data resampling, and early stopping. Averaging reduces variance when member
errors are not perfectly correlated. With `bootstrap=False`, diversity comes
from member seeds; with `bootstrap=True`, each member also receives a sampled
training set and aligned sample weights or offsets.

Regression predictions are averaged on the response scale. Additive terms,
links, and intercepts are averaged on the link scale so the mean decomposition
remains internally coherent. For classification, class probabilities are
averaged before selecting a label. The reported standard deviations summarize
between-member disagreement only: they are neither confidence intervals nor a
Bayesian posterior.
""",
                "ensemble-math",
            ),
            code(DATA_CELL, "ensemble-data"),
            code(DATA_VISUAL_CELL, "ensemble-data-visual"),
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

# The same wrapper accepts a neural classifier. LSS aggregation is rejected
# because distribution parameters require family-specific aggregation rules.
from nampy.models import NAMClassifier

classifier_ensemble = NeuralEnsemble(
    NAMClassifier(layer_sizes=[24, 12], dropout=0.0),
    n_estimators=3,
    random_state=7,
)
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
    components = model.predict_components(X_test)
    uncertainty = model.predict_component_uncertainty(X_test, center=True)
    components.validate_additive_reconstruction(rtol=1e-5, atol=1e-6)
    from nampy.explanations import explain_additive_prediction, term_importance_table

    explanation = explain_additive_prediction(X_test, components, max_bins=24)
    importance = term_importance_table(components)
    display({"R2": r2, "members": uncertainty.n_estimators})
    display(importance)
    display(explanation.head(12))
    display({name: values.mean() for name, values in uncertainty.term_std.items()})

    observed = np.asarray(y_test).reshape(-1)
    fitted = np.asarray(predictions).reshape(-1)
    member_predictions = np.stack(
        [member.predict(X_test) for member in model.estimators_], axis=0
    )
    term_disagreement = {
        name: float(np.mean(values)) for name, values in uncertainty.term_std.items()
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    axes[0].scatter(observed, fitted, s=26, alpha=0.7, color=COLORS[0])
    lo, hi = min(observed.min(), fitted.min()), max(observed.max(), fitted.max())
    axes[0].plot([lo, hi], [lo, hi], "--", color="#334155", linewidth=1)
    axes[0].set(title="Observed vs ensemble", xlabel="Observed", ylabel="Predicted")

    order = np.argsort(X_test["x1"].to_numpy())
    for member_index, member_values in enumerate(member_predictions):
        axes[1].plot(
            X_test["x1"].to_numpy()[order],
            np.asarray(member_values).reshape(-1)[order],
            color=COLORS[member_index],
            alpha=0.55,
            linewidth=1,
        )
    axes[1].plot(X_test["x1"].to_numpy()[order], fitted[order], color="#0F172A", linewidth=2.3, label="Mean")
    axes[1].set(title="Member variation", xlabel="x1", ylabel="Prediction")
    axes[1].legend(frameon=False)

    disagreement = pd.Series(term_disagreement).sort_values()
    axes[2].barh(disagreement.index, disagreement.values, color=COLORS[2])
    axes[2].set(title="Term disagreement", xlabel="Mean between-member SD")
    fig.suptitle("Ensemble fit and uncertainty", fontweight="bold")
    plt.show()
    plt.close(fig)

    model.estimators_[0].plot_terms(X_test, center=True, rug=True, pages=1)
""",
                "ensemble-fit",
            ),
            markdown(
                """## Interpretation

Use independent ensembles when prediction stability or between-fit variation
matters more than the cost of fitting several complete estimators. Bootstrap
members add diversity; their spread measures fit-to-fit disagreement rather
than calibrated uncertainty. The wrapper accepts regressors and classifiers,
with classification probabilities averaged before labels are selected.
""",
                "ensemble-interpretation",
            ),
            markdown(
                """## Reference

- Breiman (1996), *Bagging Predictors*, for the classical variance-reduction
  motivation. NAMpy's wrapper also supports independent non-bootstrap refits.
""",
                "ensemble-reference",
            ),
        ]
    )


README = """# Model notebooks

Compact, visual tutorials for every supported NAMpy model. Each notebook pairs
the essential theory with a runnable fit, predictive checks, additive
explanations, term importance, and model-specific plots. Start with
`00_overview.ipynb`; training cells are disabled by default except when
explicitly enabled by the reader.

`01_gam.ipynb` is a longer energy-system case study covering penalized
likelihood, smooth construction, smoothing criteria and optimizers, response
families, multi-linear-predictor models, shape constraints, inference, and
diagnostics.

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
| `13_nodegam.ipynb` | NodeGAM |
| `14_natt.ipynb` | NATT |
| `15_namformer.ipynb` | NAMformer |
| `16_qnam.ipynb` | QNAM |
| `17_spline_nam.ipynb` | SplineNAM |
| `18_neural_ensemble.ipynb` | independent NeuralEnsemble |

From the repository root, install the complete environment and launch Jupyter:

```bash
pip install -e ".[all,docs]" jupyterlab
jupyter lab docs/notebooks
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
