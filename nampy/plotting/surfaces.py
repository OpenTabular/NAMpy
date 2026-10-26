"""Backend-neutral rendering of prepared two-predictor response surfaces."""

from __future__ import annotations

import warnings

import numpy as np


def render_prediction_surface(
    prepared,
    *,
    plot_type="persp",
    color="heat",
    col=None,
    contour_col=None,
    n_col=50,
    theta=30,
    phi=30,
    figsize=None,
    xlab=None,
    ylab=None,
    zlab=None,
    main=None,
):
    """Render a prepared grid, optionally with plus/minus SE surfaces.

    Numerical grids and intervals belong to the caller. Matplotlib supplies
    the display; R device colors and perspective projection are not reproduced
    pixel for pixel.
    """
    import matplotlib.pyplot as plt

    if plot_type not in {"persp", "contour"}:
        raise ValueError("plot_type must be 'persp' or 'contour'.")
    palettes = {
        "heat": "hot",
        "topo": "terrain",
        "cm": "cool",
        "terrain": "terrain",
        "gray": "gray",
        "bw": "gray",
    }
    if color not in palettes:
        raise ValueError("color must be heat, topo, cm, terrain, gray, or bw.")
    if int(n_col) < 2:
        raise ValueError("n_col must be at least two.")
    cmap = plt.get_cmap(palettes[color], int(n_col))
    x, y = np.asarray(prepared["x"]), np.asarray(prepared["y"])
    X, Y = np.meshgrid(x, y)
    fit = np.asarray(prepared["fit"], dtype=float).reshape(len(y), len(x))
    se = float(prepared["se"])
    if se > 0 and plot_type == "contour":
        warnings.warn(
            "No contour display with errors; drawing perspective surfaces.",
            stacklevel=2,
        )
        plot_type = "persp"
    fig = plt.figure(figsize=figsize or (6.4, 4.8))
    ax = fig.add_subplot(111, projection="3d" if plot_type == "persp" else None)
    limits = prepared["zlim"]
    if plot_type == "contour":
        if color != "bw":
            image = ax.pcolormesh(
                X, Y, fit, cmap=cmap, shading="nearest", vmin=limits[0], vmax=limits[1]
            )
            fig.colorbar(image, ax=ax)
        ax.contour(
            X, Y, fit, colors=contour_col or ("black" if color == "bw" else "tab:blue")
        )
        ax.set_title(main or prepared["zlab"])
    else:
        if se > 0:
            for key, edge in (("lower", "green"), ("fit", "black"), ("upper", "red")):
                surface = np.asarray(prepared[key], dtype=float).reshape(fit.shape)
                ax.plot_wireframe(
                    X,
                    Y,
                    surface,
                    color="gray" if color in {"gray", "bw"} and key != "fit" else edge,
                    linewidth=0.6,
                )
        else:
            kwargs = (
                {"color": col or "white"}
                if col is not None or color == "bw"
                else {"cmap": cmap}
            )
            ax.plot_surface(X, Y, fit, vmin=limits[0], vmax=limits[1], **kwargs)
        ax.view_init(elev=phi, azim=-theta)
        ax.set_zlim(*limits)
        ax.set_zlabel(zlab or prepared["zlab"])
        if main:
            ax.set_title(main)
    ax.set_xlabel(xlab or prepared["view"][0])
    ax.set_ylabel(ylab or prepared["view"][1])
    fig.tight_layout()
    return [fig]


__all__ = ["render_prediction_surface"]
