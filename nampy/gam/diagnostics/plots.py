from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .._model_state import _coerce_feature_matrix, _require_fitted, _term_blocks_seq
from ..data import coerce_formula_predict_inputs


def plot_gam_terms(model, X=None, n_cols=2, figsize=None):
    _require_fitted(model)

    if X is None:
        X_plot = _coerce_feature_matrix(model, None, none_is_training=True)
        contribution_input = None
    elif bool(getattr(model, "formula_mode_", False)):
        X_plot, _, _ = coerce_formula_predict_inputs(model, X)
        contribution_input = X
    else:
        X_plot = _coerce_feature_matrix(model, X, none_is_training=True)
        contribution_input = X_plot

    contributions = model.predict_feature_vals(contribution_input)

    term_blocks = tuple(_term_blocks_seq(model))
    n_terms = len(term_blocks)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_terms / n_cols))
    if figsize is None:
        figsize = (5 * n_cols, 3.8 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for j, tb in enumerate(term_blocks):
        ax = axes[j]
        fj = np.asarray(contributions[tb.term_id]).ravel()
        feature_info = getattr(tb, "feature_info", None)
        idxs = [] if feature_info is None else list(feature_info.feature_indices)
        names = [] if feature_info is None else list(feature_info.feature_names)

        if len(idxs) == 1:
            xj = X_plot[:, idxs[0]]
            order = np.argsort(xj)
            ax.plot(xj[order], fj[order])
            ax.set_title(tb.label)
            ax.set_xlabel(names[0] if names else tb.label)
            ax.set_ylabel("term effect")
            continue

        if len(idxs) == 2:
            x1 = X_plot[:, idxs[0]]
            x2 = X_plot[:, idxs[1]]
            try:
                tcf = ax.tricontourf(x1, x2, fj, levels=20)
                fig.colorbar(tcf, ax=ax)
            except Exception:
                sc = ax.scatter(x1, x2, c=fj, s=18)
                fig.colorbar(sc, ax=ax)

            axis_names = names if len(names) == 2 else [f"x{idxs[0]}", f"x{idxs[1]}"]
            ax.set_title(tb.label)
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(axis_names[1])
            continue

        ax.text(
            0.5,
            0.5,
            f"Plot not implemented\nfor term {tb.label!r}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()

    for j in range(n_terms, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig


__all__ = ["plot_gam_terms"]
