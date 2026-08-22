"""Backend-neutral renderer for prepared term-plot data.

The body is the render phase of the mgcv ``plot.gam`` port (plots.r:1042-1188),
extracted verbatim from ``nampy.gam.diagnostics.plots.render_plot_gam`` — it
consumes only the ``prepared`` dict, so any backend that synthesizes the same
``pd`` entry schema (kind "1d"/"2d"/"re"/"fs"/"sz"/"md" with x/fit/se/raw/...)
can render with it. The GAM-side wrapper delegates here.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm as _norm


def _page_layout(n_plots: int, pages: int):
    """Port of the pages/rows/cols logic (plots.r:1457-1474)."""
    if pages > n_plots:
        pages = n_plots
    if pages < 0:
        pages = 0
    if pages == 0:
        return 0, 1
    ppp = n_plots // pages
    if n_plots % pages != 0:
        ppp += 1
        while ppp * (pages - 1) >= n_plots:
            pages -= 1
    return pages, ppp


def render_term_plots(prepared, *, rug=None, pages=0, theta=30, phi=30,
                      shade_col="0.8", figsize=None):
    """Render prepared plot data with matplotlib (plots.r:1042-1188).

    ``rug=None`` falls back to ``prepared["rug_default"]`` (True when absent).
    """
    import matplotlib.pyplot as plt

    pd_list = prepared["pd"]
    trans = prepared["trans"]
    shift = prepared["shift"]
    common_ylim = prepared["ylim"]
    select = prepared["select"]

    if rug is None:
        rug = bool(prepared.get("rug_default", True))

    plots = [
        (i, P)
        for i, P in enumerate(pd_list)
        if P.get("plot_me") and (select is None or i == select)
    ]
    n_plots = len(plots)
    if n_plots == 0:
        raise ValueError("No terms to plot - nothing for plot.gam() to do.")

    pages, ppp = _page_layout(n_plots, pages)
    if pages == 0:
        ppp = n_plots
    ncols = nrows = max(1, int(math.isqrt(ppp)))
    if ncols * nrows < ppp:
        ncols += 1
    if ncols * nrows < ppp:
        nrows += 1

    figures = []
    axes_flat: list = []
    n_pages = max(1, pages)
    for _pg in range(n_pages):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize or (4.0 * ncols, 3.2 * nrows),
            squeeze=False,
        )
        figures.append(fig)
        axes_flat.extend(axes.ravel())

    for slot, (_i, P) in enumerate(plots):
        ax = axes_flat[slot]
        kind = P.get("kind")
        scheme = int(P.get("scheme", 0))
        if kind == "re":
            b = np.sort(trans(np.asarray(P["fit"], dtype=np.float64) + shift))
            nq = b.size
            q = _norm.ppf((np.arange(1, nq + 1) - 0.375) / (nq + 0.25))
            ax.plot(q, b, "o", ms=3, color="black")
            lo_q, hi_q = np.quantile(b, [0.25, 0.75])
            lo_t, hi_t = _norm.ppf([0.25, 0.75])
            slope = (hi_q - lo_q) / (hi_t - lo_t)
            inter = lo_q - slope * lo_t
            ax.axline((0.0, inter), slope=slope, color="black", lw=0.8)
            ax.set_title(P["main"])
        elif kind in {"fs", "sz"}:
            xx = P["x"]
            n1 = int(P["n"])
            nft = int(np.prod(P["nf"]))
            fit = trans(np.asarray(P["fit"], dtype=np.float64) + shift)
            for i_curve in range(nft):
                seg = fit[i_curve * n1 : (i_curve + 1) * n1]
                ax.plot(xx, seg, lw=1.2)
                if P.get("plot_ci") and "ll" in P:
                    ll = trans(P["ll"][i_curve * n1 : (i_curve + 1) * n1, 0] + shift)
                    ul = trans(P["ul"][i_curve * n1 : (i_curve + 1) * n1, 0] + shift)
                    ax.fill_between(xx, ll, ul, alpha=0.25, lw=0)
            ax.set_xlabel(P["xlab"])
            ax.set_ylabel(P["ylab"])
        elif kind == "1d":
            xx = P["x"]
            fit = trans(np.asarray(P["fit"], dtype=np.float64) + shift)
            ylimit = common_ylim
            if P.get("plot_ci"):
                ll0 = trans(P["ll"][:, 0] + shift)
                ul0 = trans(P["ul"][:, 0] + shift)
                if ylimit is None:
                    lo, hi = np.nanmin(ll0), np.nanmax(ul0)
                    if "p_resid" in P:
                        lo = min(lo, np.nanmin(trans(P["p_resid"] + shift)))
                        hi = max(hi, np.nanmax(trans(P["p_resid"] + shift)))
                    ylimit = (lo, hi)
                if scheme in {1, 2}:
                    ax.fill_between(xx, ll0, ul0, color=shade_col, lw=0)
                    if scheme == 2 and P["ll"].shape[1] > 1:
                        ax.fill_between(
                            xx,
                            trans(P["ll"][:, 1] + shift),
                            trans(P["ul"][:, 1] + shift),
                            color="tab:blue",
                            alpha=0.5,
                            lw=0,
                        )
                    ax.plot(xx, fit, color="black", lw=1.2)
                else:
                    ax.plot(xx, fit, color="black", lw=1.2)
                    ax.plot(xx, ul0, color="black", lw=0.8, ls="--")
                    ax.plot(xx, ll0, color="black", lw=0.8, ls="--")
            else:
                ax.plot(xx, fit, color="black", lw=1.2)
            if ylimit is not None:
                ax.set_ylim(*ylimit)
            if prepared["partial_resids"] and "p_resid" in P:
                ax.plot(
                    P["raw"],
                    trans(np.asarray(P["p_resid"], dtype=np.float64) + shift),
                    ".",
                    ms=2,
                    color="black",
                )
            if rug:
                raw = np.asarray(P["raw"], dtype=np.float64)
                if prepared["jit"]:
                    span = np.ptp(raw) if raw.size else 1.0
                    raw = raw + np.random.default_rng(0).uniform(
                        -span / 50.0, span / 50.0, raw.size
                    )
                ax.plot(
                    raw,
                    np.full(raw.size, ax.get_ylim()[0]),
                    "|",
                    ms=6,
                    color="black",
                )
            ax.set_xlabel(P["xlab"])
            ax.set_ylabel(P["ylab"])
            if P.get("main"):
                ax.set_title(P["main"])
        elif kind == "2d":
            n2 = int(P["n2"])
            fit = trans(np.asarray(P["fit"], dtype=np.float64) + shift)
            Z = fit.reshape(n2, n2)
            if scheme == 1:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                pos = ax.get_subplotspec()
                fig = ax.figure
                ax.remove()
                ax3 = fig.add_subplot(pos, projection="3d")
                Xg, Yg = np.meshgrid(P["x"], P["y"])
                ax3.plot_surface(
                    Xg, Yg, Z, cmap="viridis", linewidth=0.2
                )
                ax3.view_init(elev=phi, azim=-theta)
                ax3.set_xlabel(P["xlab"])
                ax3.set_ylabel(P["ylab"])
                ax3.set_title(P["main"])
                axes_flat[slot] = ax3
            elif scheme in {2, 3}:
                cmap = "gray" if scheme == 3 else "YlOrRd"
                ax.imshow(
                    Z,
                    origin="lower",
                    aspect="auto",
                    extent=(P["x"][0], P["x"][-1], P["y"][0], P["y"][-1]),
                    cmap=cmap,
                )
                ax.contour(P["x"], P["y"], Z, colors="tab:blue")
                if rug:
                    ax.plot(P["raw"]["x"], P["raw"]["y"], ".", ms=2, color="black")
                ax.set_xlabel(P["xlab"])
                ax.set_ylabel(P["ylab"])
                ax.set_title(P["main"])
            else:
                ax.contour(P["x"], P["y"], Z, colors="black", linewidths=1.5)
                if P.get("plot_ci") and "ll" in P:
                    Zl = trans(P["ll"][:, 0] + shift).reshape(n2, n2)
                    Zu = trans(P["ul"][:, 0] + shift).reshape(n2, n2)
                    ax.contour(
                        P["x"], P["y"], Zl, colors="tab:red",
                        linestyles="dashed", linewidths=0.8,
                    )
                    ax.contour(
                        P["x"], P["y"], Zu, colors="tab:blue",
                        linestyles="dotted", linewidths=0.8,
                    )
                if rug:
                    ax.plot(P["raw"]["x"], P["raw"]["y"], ".", ms=2, color="black")
                ax.set_xlabel(P["xlab"])
                ax.set_ylabel(P["ylab"])
                ax.set_title(P["main"])
        elif kind == "md":
            # md.plot (plots.r:1192-1269): panel grid of image+contour slices
            m = int(P["m"])
            nr = int(P["nr"])
            nc = int(P["nc"])
            fit = trans(np.asarray(P["fit"], dtype=np.float64) + shift)
            fit = fit.copy()
            fit[np.asarray(P["exclude"], dtype=bool)] = np.nan
            # pack panels with NA separators, mirroring md.plot
            F = np.asarray(fit, dtype=np.float64).reshape(
                (m * nr, nc * m), order="F"
            )
            f1 = np.full((nr * m + nr - 1, nc * m), np.nan)
            ii = np.tile(np.arange(m), nr) + np.repeat(np.arange(nr) * (m + 1), m)
            f1[ii, :] = F
            F2 = np.full((nr * m + nr - 1, nc * m + nc - 1), np.nan)
            jj = np.tile(np.arange(m), nc) + np.repeat(np.arange(nc) * (m + 1), m)
            F2[:, jj] = f1
            ax.imshow(F2, origin="lower", aspect="auto", cmap="YlOrRd")
            ax.set_axis_off()
            ax.set_title(P["main"])
        else:  # pragma: no cover - defensive
            ax.set_axis_off()

    for extra_ax in axes_flat[n_plots:]:
        extra_ax.set_axis_off()
    for fig in figures:
        fig.tight_layout()
    return figures


def prepared_from_contributions(
    frame, terms, *, ylab_format="f({name})", term_features=None
):
    """Build a prepared dict (1-d entries) from per-term contribution arrays.

    ``frame`` supplies the raw x values by column name; ``terms`` maps term
    name -> (n,) or (n, 1) contribution array. ``term_features`` optionally
    maps a term name to the frame column carrying its x values — labeled
    terms such as ``"gam:s(x0)"`` then render against that column while
    keeping the term name as the axis label. Unmapped interaction names
    (containing ':'), names absent from ``frame``, non-numeric columns, and
    multi-column contributions are skipped.
    """
    term_features = dict(term_features or {})
    columns = getattr(frame, "columns", ())

    pd_list = []
    n_rows = None
    for name, values in terms.items():
        column = term_features.get(name, name)
        if column not in columns or (":" in name and name not in term_features):
            continue
        try:
            x_raw = np.asarray(frame[column], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        contrib = np.asarray(values, dtype=np.float64).reshape(len(x_raw), -1)
        if contrib.shape[1] != 1:
            continue
        order = np.argsort(x_raw)
        n_rows = x_raw.size
        pd_list.append(
            {
                "kind": "1d",
                "plot_me": True,
                "x": x_raw[order],
                "fit": contrib[order, 0],
                "raw": x_raw,
                "xlab": name,
                "ylab": ylab_format.format(name=name),
                "main": "",
                "scheme": 0,
            }
        )

    if not pd_list:
        raise ValueError("No numeric 1-d terms available to plot.")

    return {
        "pd": pd_list,
        "ylim": None,
        "partial_resids": False,
        "by_resids": False,
        "shift": 0.0,
        "trans": lambda values: values,
        "jit": False,
        "select": None,
        "scale": False,
        "rug_default": bool(n_rows is None or n_rows <= 10000),
    }
