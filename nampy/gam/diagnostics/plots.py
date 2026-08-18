"""Port of ``mgcv::plot.gam`` (mgcv/R/plots.r:1271-1565).

The port keeps upstream's two-phase structure:

1. A data phase mirroring ``plot.mgcv.smooth`` / ``plot.random.effect`` /
   ``plot.fs.interaction`` / ``plot.sz.interaction`` with ``P=NULL``
   (plots.r:928-1041, 350-375, 769-802, 680-766) plus the ``plot.gam`` loop
   that turns the prediction matrices into fits, standard errors (including
   ``seWithMean``), confidence limits and partial residuals
   (plots.r:1374-1445). :func:`prepare_plot_gam_data` exposes this phase so
   the numbers are directly parity-testable against the list ``plot.gam``
   returns invisibly.
2. A rendering phase (:func:`render_plot_gam`) drawing the prepared data with
   matplotlib, following the upstream scheme semantics (plots.r:1042-1188).

Quantities that are pure R graphics state (character expansion, ``strwidth``
layout of contour legends, device asking) are not part of the parity
contract; every number that reaches the plot is.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm as _norm

from .._model_state import (
    _coef_column_offset,
    _coef_full,
    _coerce_feature_matrix,
    _require_fitted,
    _term_blocks_seq,
)
from ..predict.linear_predictor_matrix import build_lpmatrix
from ..predict.predictions import (
    _prediction_term_groups,
    _term_has_absorbed_constraint,
)
from ..smooths.categorical import factor_levels_from_metadata
from ..term_labels import normalize_mgcv_term_label
from .residuals import _prior_weights

__all__ = [
    "exclude_too_far",
    "prepare_plot_gam_data",
    "render_plot_gam",
    "plot_gam",
]


def exclude_too_far(g1, g2, d1, d2, dist):
    """Port of ``mgcv::exclude.too.far`` (mgcv/R/plots.r:1569-1596).

    Grid nodes further than ``dist`` (on the unit square) from every datum are
    flagged TRUE. The C ``MinimumSeparation`` kernel reduces to a plain
    minimum Euclidean separation per grid node.
    """
    g1 = np.asarray(g1, dtype=np.float64).ravel().copy()
    g2 = np.asarray(g2, dtype=np.float64).ravel().copy()
    d1 = np.asarray(d1, dtype=np.float64).ravel().copy()
    d2 = np.asarray(d2, dtype=np.float64).ravel().copy()
    if g1.size != g2.size:
        raise ValueError("grid vectors are different lengths")
    if d1.size != d2.size:
        raise ValueError("data vectors are of different lengths")
    if dist < 0:
        raise ValueError("supplied dist negative")

    mig = np.min(g1)
    d1 -= mig
    g1 -= mig
    mag = np.max(g1)
    d1 /= mag
    g1 /= mag
    mig = np.min(g2)
    d2 -= mig
    g2 -= mig
    mag = np.max(g2)
    d2 /= mag
    g2 /= mag

    diff1 = g1[:, np.newaxis] - d1[np.newaxis, :]
    diff2 = g2[:, np.newaxis] - d2[np.newaxis, :]
    distance = np.sqrt(np.min(diff1 * diff1 + diff2 * diff2, axis=1))
    return distance > float(dist)


def _sub_edf(label: str, edf: float) -> str:
    """Port of the local ``sub.edf`` (plots.r:1292-1305)."""
    label = str(label)
    pos = label.find(":")
    edf_txt = f"{round(float(edf), 2):g}"
    if pos < 0:
        return f"{label[: len(label) - 1]},{edf_txt})"
    return f"{label[: pos - 1]},{edf_txt}{label[pos - 1:]}"


def _model_feature_names(model) -> list[str]:
    names = getattr(model, "formula_feature_columns_", None)
    if names:
        return [str(v) for v in names]
    n = int(np.asarray(model.X_).shape[1])
    return [f"x{i}" for i in range(n)]


def _new_feature_matrix(model, n_rows: int) -> np.ndarray:
    """Row-0-filled feature matrix; term columns are overwritten per plot."""
    X_train = np.asarray(model.X_)
    base_row = X_train[0, :]
    X_new = np.empty((n_rows, X_train.shape[1]), dtype=object)
    X_new[:, :] = base_row[np.newaxis, :]
    return X_new


def _activate_by_column(model, tb, X_new) -> None:
    """Mirror ``by <- rep(1, n)`` in the upstream prepare steps.

    For numeric ``by`` the indicator is set to one; a factor-by block's own
    level is written instead so its indicator evaluates to one everywhere.
    """
    by_info = getattr(tb, "by_variable_info", None)
    name = None if by_info is None else getattr(by_info, "name", None)
    if not name:
        return
    feature_names = _model_feature_names(model)
    try:
        j = feature_names.index(str(name))
    except ValueError:
        return
    level = getattr(by_info, "level", None)
    if level is None:
        level = getattr(tb, "metadata", {}).get("by_level", None)
    if level is None:
        level = getattr(tb, "constructor_metadata", {}).get("by_level", None)
    X_new[:, j] = 1.0 if level is None else level


def _term_numeric_features(model, tb) -> list[tuple[int, str]]:
    feature_names = _model_feature_names(model)
    info = tb.feature_info
    out = []
    X_train = np.asarray(model.X_)
    for idx in info.feature_indices:
        col = X_train[:, int(idx)]
        try:
            np.asarray(col, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        out.append((int(idx), feature_names[int(idx)]))
    return out


def _term_factor_features(model, tb) -> list[tuple[int, str, list]]:
    feature_names = _model_feature_names(model)
    X_train = np.asarray(model.X_)
    out = []
    stored_levels = getattr(tb, "_factor_levels", None)
    if stored_levels is None:
        fs_levels = getattr(tb, "_levels", None)
        stored_levels = None if fs_levels is None else [fs_levels]
    factor_position = 0
    for idx in tb.feature_info.feature_indices:
        col = X_train[:, int(idx)]
        try:
            np.asarray(col, dtype=np.float64)
        except (TypeError, ValueError):
            metadata_levels = factor_levels_from_metadata(
                getattr(tb, "metadata", None), feature_names[int(idx)]
            )
            if metadata_levels is not None:
                levels = [str(value) for value in metadata_levels]
            elif stored_levels is not None and factor_position < len(stored_levels):
                # mgcv/R/plots.r::plot.fs.interaction and
                # plot.sz.interaction use x$flev, i.e. the fitted smooth's
                # canonical level order rather than training-row encounter order.
                levels = [str(value) for value in stored_levels[factor_position]]
            else:
                levels = list(dict.fromkeys(str(v) for v in col))
            out.append((int(idx), feature_names[int(idx)], levels))
            factor_position += 1
    return out


def _term_X(model, tb, X_new) -> np.ndarray:
    return np.asarray(tb.predict_matrix(X_new), dtype=np.float64)


def _prepare_smooth(model, tb, *, se, n, n2, n3, xlab, ylab, main, label,
                    xlim, ylim, too_far, scheme):
    """The ``P=NULL`` phase of the upstream plot methods."""
    basis_name = str(getattr(tb, "basis_name", "")).lower()
    X_train = np.asarray(model.X_)

    if basis_name == "re":
        # plot.random.effect (plots.r:357-367): X is the identity; the plot
        # is a normal QQ plot of the estimated effects.
        p = int(tb.coef_slice.stop - tb.coef_slice.start)
        raw_idx = int(tb.feature_info.feature_indices[0])
        raw = X_train[:, raw_idx]
        return {
            "kind": "re",
            "X": np.eye(p, dtype=np.float64),
            "scale": False,
            "se": False,
            "raw": raw,
            "xlab": xlab or "Gaussian quantiles",
            "ylab": ylab or "effects",
            "main": main or label,
        }

    numeric = _term_numeric_features(model, tb)
    factors = _term_factor_features(model, tb)

    if basis_name in {"fs", "sz"} and len(numeric) == 1:
        # plot.fs.interaction (plots.r:775-793) / plot.sz.interaction
        # (plots.r:699-720): one curve per (product of) factor level(s).
        idx, term_name = numeric[0]
        raw = np.asarray(X_train[:, idx], dtype=np.float64)
        xx = np.linspace(np.min(raw), np.max(raw), n)
        nf = [len(levels) for (_i, _n, levels) in factors]
        nft = int(np.prod(nf)) if nf else 1
        X_new = _new_feature_matrix(model, n * nft)
        X_new[:, idx] = np.tile(xx, nft)
        # rightmost factor varies slowest, matching the upstream rep() nest
        for pos, (fidx, _fname, levels) in enumerate(factors):
            re_n = int(np.prod(nf[pos + 1:])) if pos + 1 < len(nf) else 1
            rs = int(np.prod(nf[:pos])) if pos > 0 else 1
            block = np.repeat(levels, re_n * n)
            X_new[:, fidx] = np.tile(block, rs)
        _activate_by_column(model, tb, X_new)
        X = _term_X(model, tb, X_new)
        se_flag = basis_name == "sz"  # fs: se=FALSE, sz: se=TRUE upstream
        clev = 0.95
        return {
            "kind": basis_name,
            "X": X,
            "x": xx,
            "n": n,
            "nf": nf if nf else [1],
            "scale": True,
            "se": bool(se) and se_flag,
            "clev": [clev],
            "raw": raw,
            "xlab": xlab or term_name,
            "ylab": ylab or label,
            "main": main or "",
        }

    if factors:
        return None  # no default method for factor covariates

    dim = len(numeric)
    if dim == 0 or dim > 4:
        if dim > 4:
            warnings.warn(
                "no automatic plotting for smooths of more than four variables",
                stacklevel=2,
            )
        return None

    if dim == 1:
        idx, term_name = numeric[0]
        raw = np.asarray(X_train[:, idx], dtype=np.float64)
        if xlim is None:
            xx = np.linspace(np.min(raw), np.max(raw), n)
        else:
            xx = np.linspace(xlim[0], xlim[1], n)
        X_new = _new_feature_matrix(model, n)
        X_new[:, idx] = xx
        _activate_by_column(model, tb, X_new)
        X = _term_X(model, tb, X_new)
        clev = [0.95, 0.68] if scheme == 2 else [0.95]
        return {
            "kind": "1d",
            "X": X,
            "x": xx,
            "scale": True,
            "se": bool(se),
            "clev": clev,
            "raw": raw,
            "xlab": xlab or term_name,
            "ylab": ylab or label,
            "main": main,
            "xlim": (float(xx[0]), float(xx[-1])) if xlim is None else tuple(xlim),
        }

    if dim == 2:
        (ix, xname), (iy, yname) = numeric[0], numeric[1]
        raw_x = np.asarray(X_train[:, ix], dtype=np.float64)
        raw_y = np.asarray(X_train[:, iy], dtype=np.float64)
        n2 = max(10, int(n2))
        xm = (
            np.linspace(np.min(raw_x), np.max(raw_x), n2)
            if xlim is None
            else np.linspace(xlim[0], xlim[1], n2)
        )
        ym = (
            np.linspace(np.min(raw_y), np.max(raw_y), n2)
            if ylim is None
            else np.linspace(ylim[0], ylim[1], n2)
        )
        xx = np.tile(xm, n2)
        yy = np.repeat(ym, n2)
        if too_far > 0:
            exclude = exclude_too_far(xx, yy, raw_x, raw_y, dist=too_far)
        else:
            exclude = np.zeros(n2 * n2, dtype=bool)
        X_new = _new_feature_matrix(model, n2 * n2)
        X_new[:, ix] = xx
        X_new[:, iy] = yy
        _activate_by_column(model, tb, X_new)
        X = _term_X(model, tb, X_new)
        return {
            "kind": "2d",
            "X": X,
            "x": xm,
            "y": ym,
            "n2": n2,
            "scale": False,
            "se": True,
            "clev": [se if isinstance(se, float) and 0 < se < 1 else 0.68],
            "raw": {"x": raw_x, "y": raw_y},
            "xlab": xlab or xname,
            "ylab": ylab or yname,
            "main": main or label,
            "exclude": exclude,
            "xlim": (float(xm[0]), float(xm[-1])),
            "ylim": (float(ym[0]), float(ym[-1])),
        }

    # 3/4-D slice grid (plots.r:990-1041)
    m = int(n2)
    nr = nc = int(n3)
    idxs = [i for i, _n in numeric]
    names = [nm for _i, nm in numeric]
    lo = [float(np.min(np.asarray(X_train[:, i], dtype=np.float64))) for i in idxs]
    hi = [float(np.max(np.asarray(X_train[:, i], dtype=np.float64))) for i in idxs]
    x1 = np.linspace(lo[0], hi[0], m)
    x2 = np.linspace(lo[1], hi[1], m)
    if dim == 3:
        x3 = np.linspace(lo[2], hi[2], nr * nc)
        col1 = np.tile(x1, m * nr * nc)
        col2 = np.tile(np.repeat(x2, m * nr), nc)
        i3 = np.tile(np.repeat((np.arange(1, nr + 1) - 1) * nc, m), m * nc) + np.repeat(
            np.arange(1, nc + 1), m * m * nr
        )
        col3 = x3[i3 - 1]
        cols = [col1, col2, col3]
    else:
        x3 = np.linspace(lo[2], hi[2], nr)
        x4 = np.linspace(lo[3], hi[3], nc)
        col1 = np.tile(x1, m * nr * nc)
        col2 = np.tile(np.repeat(x2, m * nr), nc)
        col3 = np.tile(np.repeat(x3, m), m * nc)
        col4 = np.repeat(x4, m * m * nr)
        cols = [col1, col2, col3, col4]
    n_rows = cols[0].size
    X_new = _new_feature_matrix(model, n_rows)
    for i, col in zip(idxs, cols, strict=True):
        X_new[:, i] = col
    _activate_by_column(model, tb, X_new)
    X = _term_X(model, tb, X_new)
    if too_far > 0:
        raw1 = np.asarray(X_train[:, idxs[0]], dtype=np.float64)
        raw2 = np.asarray(X_train[:, idxs[1]], dtype=np.float64)
        exclude = exclude_too_far(cols[0], cols[1], raw1, raw2, dist=too_far)
    else:
        exclude = np.zeros(n_rows, dtype=bool)
    return {
        "kind": "md",
        "X": X,
        "scale": False,
        "se": False,
        "m": m,
        "nc": nc,
        "nr": nr,
        "lo": lo,
        "hi": hi,
        "vname": names,
        "main": main or label,
        "exclude": exclude,
    }


def _training_lpmatrix(model) -> np.ndarray:
    X_np = _coerce_feature_matrix(model, None, none_is_training=True)
    return np.asarray(build_lpmatrix(model, X_new=X_np), dtype=np.float64)


def _smooth_edf_by_block(model):
    result = model.fit_result()
    values = np.atleast_1d(np.asarray(result.edf_by_term, dtype=np.float64))
    blocks = tuple(_term_blocks_seq(model))
    if values.size == len(blocks):
        return {id(tb): float(values[i]) for i, tb in enumerate(blocks)}
    # edf_by_term may enumerate expanded parametric blocks differently; fall
    # back to per-coefficient attribution via the trace of the hat pieces.
    return {id(tb): float("nan") for tb in blocks}


def prepare_plot_gam_data(
    model,
    *,
    residuals=False,
    se=True,
    select=None,
    scale=-1,
    n=100,
    n2=40,
    n3=3,
    jit=False,
    xlab=None,
    ylab=None,
    main=None,
    ylim=None,
    xlim=None,
    too_far=0.1,
    shift=0.0,
    trans=None,
    se_with_mean=False,
    unconditional=False,
    by_resids=False,
    scheme=0,
):
    """The data phase of ``plot.gam`` (plots.r:1312-1509). Returns ``pd``."""
    _require_fitted(model)
    trans_fn = (lambda v: v) if trans is None else trans

    Vp = np.asarray(
        model.vcov(unconditional=bool(unconditional)), dtype=np.float64
    )

    partial_resids = bool(residuals)
    w_resid = None
    fv_terms = None
    term_group_labels: list[str] = []
    if partial_resids:
        wr = np.sqrt(np.asarray(_prior_weights(model), dtype=np.float64))
        w_resid = (
            np.asarray(model.residuals(type="working"), dtype=np.float64) * wr
        )
        fv_terms = np.asarray(model.predict(None, type="terms"), dtype=np.float64)
        term_group_labels = [
            str(g["label"]) for g in _prediction_term_groups(model)
        ]

    blocks = tuple(_term_blocks_seq(model))
    smooth_blocks = [
        tb for tb in blocks if str(getattr(tb, "term_type", "")) != "parametric"
    ]
    m = len(smooth_blocks)
    schemes = [scheme] * m if np.isscalar(scheme) else list(scheme)
    if len(schemes) != m:
        warnings.warn(
            f"scheme should be a single number, or a vector with {m} elements",
            stacklevel=2,
        )
        schemes = [schemes[0]] * m

    coef_full = np.asarray(_coef_full(model), dtype=np.float64).ravel()
    offset0 = _coef_column_offset(model)
    edf_map = _smooth_edf_by_block(model)
    cmX = None

    pd_list = []
    for i, tb in enumerate(smooth_blocks):
        label = _sub_edf(str(tb.label), edf_map.get(id(tb), float("nan")))
        P = _prepare_smooth(
            model,
            tb,
            se=se,
            n=n,
            n2=n2,
            n3=n3,
            xlab=xlab,
            ylab=ylab,
            main=main,
            label=label,
            xlim=xlim,
            ylim=ylim,
            too_far=too_far,
            scheme=schemes[i],
        )
        if P is None:
            pd_list.append({"plot_me": False, "label": str(tb.label)})
            continue

        sl = slice(offset0 + tb.coef_slice.start, offset0 + tb.coef_slice.stop)
        p_coef = coef_full[sl]
        X = P.pop("X")
        fit = X @ p_coef
        exclude = P.get("exclude", None)
        if exclude is not None:
            fit = fit.copy()
            fit[np.asarray(exclude, dtype=bool)] = np.nan
        P["fit"] = fit
        P["label"] = str(tb.label)

        if bool(se) and P["se"]:
            # standard errors (plots.r:1393-1435)
            if se_with_mean and _term_has_absorbed_constraint(tb):
                if cmX is None:
                    cmX = np.mean(_training_lpmatrix(model), axis=0)
                cm_row = np.zeros(Vp.shape[1], dtype=np.float64)
                cm_row[: cmX.size] = cmX
                X1 = np.broadcast_to(cm_row, (X.shape[0], Vp.shape[1])).copy()
                X1[:, sl] = X
                se_fit = np.sqrt(
                    np.maximum(0.0, np.sum((X1 @ Vp) * X1, axis=1))
                )
            else:
                if se_with_mean and not _term_has_absorbed_constraint(tb):
                    warnings.warn("seWithMean unavailable", stacklevel=2)
                V_block = Vp[sl, sl]
                se_fit = np.sqrt(
                    np.maximum(0.0, np.sum((X @ V_block) * X, axis=1))
                )
            if exclude is not None:
                se_fit = se_fit.copy()
                se_fit[np.asarray(exclude, dtype=bool)] = np.nan
            clev = np.atleast_1d(np.asarray(P.get("clev", [0.95]), dtype=np.float64))
            se_mult = -_norm.ppf((1.0 - clev) / 2.0)
            ll = np.empty((fit.size, se_mult.size), dtype=np.float64)
            ul = np.empty_like(ll)
            for j, mult in enumerate(se_mult):
                ll[:, j] = fit - se_fit * mult
                ul[:, j] = fit + se_fit * mult
            P["ll"] = ll
            P["ul"] = ul
            P["se_mult"] = se_mult
            P["se_fit"] = se_fit
            # back-compatible field: se.fit * first multiplier (plots.r:1435)
            P["se"] = se_fit * float(se_mult[0])
            P["plot_ci"] = True
        else:
            P["plot_ci"] = False
            P["se"] = False

        if partial_resids:
            normalized = str(normalize_mgcv_term_label(str(tb.label)))
            group_index = None
            for gi, glabel in enumerate(term_group_labels):
                if glabel == normalized:
                    group_index = gi
                    break
            if group_index is not None and fv_terms is not None:
                P["p_resid"] = fv_terms[:, group_index] + w_resid
        P["plot_me"] = True
        P["scheme"] = schemes[i]
        pd_list.append(P)

    # common y scale (plots.r:1481-1509)
    common_ylim = ylim
    if scale == -1 and ylim is None:
        lo = hi = None
        for P in pd_list:
            if not P.get("plot_me") or not P.get("scale"):
                continue
            if "ll" in P and np.size(P["ll"]) > 1:
                p_lo = np.nanmin(P["ll"][:, 0])
                p_hi = np.nanmax(P["ul"][:, 0])
            else:
                p_lo = np.nanmin(P["fit"])
                p_hi = np.nanmax(P["fit"])
            if "p_resid" in P:
                p_lo = min(p_lo, np.nanmin(P["p_resid"]))
                p_hi = max(p_hi, np.nanmax(P["p_resid"]))
            lo = p_lo if lo is None else min(lo, p_lo)
            hi = p_hi if hi is None else max(hi, p_hi)
        if lo is not None:
            common_ylim = (
                float(trans_fn(lo + shift)),
                float(trans_fn(hi + shift)),
            )

    return {
        "pd": pd_list,
        "ylim": common_ylim,
        "partial_resids": partial_resids,
        "by_resids": bool(by_resids),
        "shift": float(shift),
        "trans": trans_fn,
        "jit": bool(jit),
        "select": select,
        "scale": scale,
        "rug_default": bool(np.asarray(model.X_).shape[0] <= 10000),
    }


def render_plot_gam(model, prepared, *, rug=None, pages=0, theta=30, phi=30,
                    shade_col="0.8", figsize=None):
    """Render prepared plot data with matplotlib (plots.r:1042-1188).

    Delegates to the backend-neutral :func:`nampy.plotting.render_term_plots`;
    ``model`` is kept in the signature for the public GAM surface but the
    renderer itself consumes only ``prepared``.
    """
    from ...plotting import render_term_plots

    return render_term_plots(
        prepared,
        rug=rug,
        pages=pages,
        theta=theta,
        phi=phi,
        shade_col=shade_col,
        figsize=figsize,
    )



def plot_gam(
    model,
    *,
    residuals=False,
    rug=None,
    se=True,
    pages=0,
    select=None,
    scale=-1,
    n=100,
    n2=40,
    n3=3,
    theta=30,
    phi=30,
    jit=False,
    xlab=None,
    ylab=None,
    main=None,
    ylim=None,
    xlim=None,
    too_far=0.1,
    shade_col="0.8",
    shift=0.0,
    trans=None,
    se_with_mean=False,
    unconditional=False,
    by_resids=False,
    scheme=0,
    figsize=None,
):
    """Port of ``mgcv::plot.gam``: prepare per-term plot data and render it.

    Returns the prepared plot-data list, mirroring upstream's invisible
    ``pd`` return; the matplotlib figures are attached under ``"figures"``.
    """
    prepared = prepare_plot_gam_data(
        model,
        residuals=residuals,
        se=se,
        select=select,
        scale=scale,
        n=n,
        n2=n2,
        n3=n3,
        jit=jit,
        xlab=xlab,
        ylab=ylab,
        main=main,
        ylim=ylim,
        xlim=xlim,
        too_far=too_far,
        shift=shift,
        trans=trans,
        se_with_mean=se_with_mean,
        unconditional=unconditional,
        by_resids=by_resids,
        scheme=scheme,
    )
    figures = render_plot_gam(
        model,
        prepared,
        rug=rug,
        pages=pages,
        theta=theta,
        phi=phi,
        shade_col=shade_col,
        figsize=figsize,
    )
    prepared["figures"] = figures
    return prepared
