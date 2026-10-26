"""Spherical plotting coordinates from mgcv/R/plots.r."""

from __future__ import annotations

import numpy as np


def repole(lo, la, lop, lap):
    """Port of mgcv's ``repole``; all angles are in radians."""
    lo, la = np.asarray(lo, dtype=float), np.asarray(la, dtype=float)
    yp = np.sin(lap)
    xp = np.cos(lap) * np.sin(lop)
    zp = np.cos(lap) * np.cos(lop)
    ym = np.sin(lap - np.pi / 2)
    xm = np.cos(lap - np.pi / 2) * np.sin(lop)
    zm = np.cos(lap - np.pi / 2) * np.cos(lop)
    y = np.sin(la)
    x = np.cos(la) * np.sin(lo)
    z = np.cos(la) * np.cos(lo)
    d = np.sqrt((x - xp) ** 2 + (y - yp) ** 2 + (z - zp) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.pi / 2 - 2 * np.arcsin(d / 2)
        x = x - xp * np.sin(phi)
        y = y - yp * np.sin(phi)
        z = z - zp * np.sin(phi)
        d = np.sqrt((x - xm) ** 2 + (y - ym) ** 2 + (z - zm) ** 2)
        theta = (1 + np.cos(phi) ** 2 - d**2) / (2 * np.cos(phi))
        theta = np.arccos(np.clip(theta, -1, 1))
    x1, z1 = np.sin(lop + np.pi / 2), np.cos(lop + np.pi / 2)
    x0, z0 = np.sin(lop - np.pi / 2), np.cos(lop - np.pi / 2)
    d1 = np.sqrt((x - x1) ** 2 + y**2 + (z - z1) ** 2)
    d0 = np.sqrt((x - x0) ** 2 + y**2 + (z - z0) ** 2)
    return {"lo": np.where(d0 < d1, -theta, theta), "la": phi}


def lolaxy(lo, la, theta, phi):
    """Project the visible raw observations as in mgcv's ``lolaxy``."""
    er = repole(-np.asarray(lo), la, -np.pi, phi)
    longitude = er["lo"] - theta
    y = np.sin(er["la"])
    x = np.cos(er["la"]) * np.sin(longitude)
    z = np.cos(er["la"]) * np.cos(longitude)
    ind = z < 0
    return {"x": x[ind], "y": y[ind]}


def prepare_sos_plot(model, tb, *, n2=40, theta=30, phi=30, rug=True):
    """Prepare the hemisphere disk used by ``plot.sos.smooth`` schemes 0/1."""
    from .plots import _activate_by_column, _new_feature_matrix, _term_X

    theta = float(theta) * np.pi / 180 % (2 * np.pi)
    phi = float(phi) * np.pi / 180 % (2 * np.pi)
    if theta > np.pi:
        theta -= 2 * np.pi
    if phi > np.pi:
        phi -= 2 * np.pi
    if phi > np.pi / 2:
        phi = np.pi - phi
    if phi < -np.pi / 2:
        phi = -(phi + np.pi)

    indices = [int(i) for i in tb.feature_info.feature_indices]
    raw = np.asarray(model.X_)[:, indices].astype(float)
    if rug:
        raw = lolaxy(raw[:, 1] * np.pi / 180, raw[:, 0] * np.pi / 180, theta, phi)
    m = int(np.rint(n2 * 1.5))
    xm = ym = np.linspace(-1, 1, m)
    gx, gy = np.tile(xm, m), np.repeat(ym, m)
    ind = gx**2 + gy**2 <= 1
    la = np.arcsin(gy[ind])
    lo = np.arcsin(gx[ind] / np.cos(la))
    um = repole(lo, la, theta, phi)
    dat = np.column_stack((um["la"] * 180 / np.pi, um["lo"] * 180 / np.pi))
    X_new = _new_feature_matrix(model, len(dat))
    X_new[:, indices] = dat
    _activate_by_column(model, tb, X_new)
    X = _term_X(model, tb, X_new)
    lo = dat[:, 1].copy()
    lo[lo <= -177] += 360
    lo[((lo < -165) & (lo > -177)) | (np.abs(dat[:, 0]) > 80)] = np.nan
    return {
        "kind": "sos",
        "X": X,
        "scale": False,
        "se": False,
        "raw": raw,
        "xlab": "",
        "ylab": "",
        "main": "",
        "ind": ind,
        "xm": xm,
        "ym": ym,
        "lo": lo,
        "la": dat[:, 0],
        "grid_coordinates": dat,
        "theta": theta,
        "phi": phi,
    }
