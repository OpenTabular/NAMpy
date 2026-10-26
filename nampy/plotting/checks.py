"""Render backend-prepared QQ and model-check panels."""

import numpy as np


def _draw_qq(ax, data):
    x, y = np.asarray(data["theoretical"]), np.asarray(data["residuals"])
    if data["lower"] is not None:
        ax.fill_between(x, data["lower"], data["upper"], color="0.85")
    if data["rep"] and data["level"] >= 1:
        ax.plot(x, data["simulated_residuals"], color="0.85", linewidth=0.7)
    ax.scatter(x, y, s=12, color="black")
    if data["reference"] == "identity":
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(limits, limits, color="red", linewidth=1)
    ax.set(xlabel=data["xlab"], ylabel=data["ylab"])
    if "ylim" in data:
        ax.set_ylim(data["ylim"])


def render_qq(data, *, figsize=None):
    """Render quantiles, residuals and optional simulated envelope."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    _draw_qq(ax, data)
    fig.tight_layout()
    return [fig]


def render_model_check(data, *, figsize=None):
    """Render the four gam.check/scam.check panels using supplied histogram bins."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize or (9, 7))
    _draw_qq(axes[0, 0], data["qq"])
    for ax, key in [
        (axes[0, 1], "residual_vs_linear"),
        (axes[1, 1], "response_vs_fitted"),
    ]:
        panel = data[key]
        ax.scatter(panel["x"], panel["y"], s=12, color="black")
        ax.set(xlabel=panel["xlab"], ylabel=panel["ylab"], title=panel["title"])
    hist = data["histogram"]
    axes[1, 0].bar(
        hist["breaks"][:-1],
        hist["counts"],
        width=np.diff(hist["breaks"]),
        align="edge",
        color="white",
        edgecolor="black",
    )
    axes[1, 0].set(xlabel=hist["xlab"], ylabel="Frequency", title=hist["title"])
    fig.tight_layout()
    return [fig]
