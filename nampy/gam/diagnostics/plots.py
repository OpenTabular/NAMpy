from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .._model_state import _coerce_feature_matrix, _require_fitted


def plot_gam_terms(model, X=None, n_cols=2, figsize=None):
    _require_fitted(model)

    X_plot = _coerce_feature_matrix(model, X, none_is_training=True)
    contributions = model.predict_feature_vals(X_plot)

    n_terms = len(model.term_blocks_)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_terms / n_cols))
    if figsize is None:
        figsize = (5 * n_cols, 3.8 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for j, tb in enumerate(model.term_blocks_):
        ax = axes[j]
        fj = np.asarray(contributions[tb.term_id]).ravel()
        smooth = tb.smooth
        term = smooth.runtime

        if hasattr(term, "_feature_index") and term._feature_index is not None:
            xj = X_plot[:, term._feature_index]
            order = np.argsort(xj)
            ax.plot(xj[order], fj[order])
            ax.set_title(tb.label)
            ax.set_xlabel(getattr(term, "_feature_name", tb.label))
            ax.set_ylabel("term effect")
            continue

        if hasattr(term, "_feature_indices") and term._feature_indices is not None:
            idxs = list(term._feature_indices)

            if len(idxs) == 1:
                xj = X_plot[:, idxs[0]]
                order = np.argsort(xj)
                name = (
                    term._feature_names[0]
                    if getattr(term, "_feature_names", None) is not None
                    else f"x{idxs[0]}"
                )
                ax.plot(xj[order], fj[order])
                ax.set_title(tb.label)
                ax.set_xlabel(name)
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

                names = getattr(term, "_feature_names", [f"x{idxs[0]}", f"x{idxs[1]}"])
                ax.set_title(tb.label)
                ax.set_xlabel(names[0])
                ax.set_ylabel(names[1])
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
