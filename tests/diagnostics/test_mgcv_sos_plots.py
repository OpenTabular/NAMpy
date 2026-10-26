"""Static mgcv hemisphere grid, rotation, projection and fitted-term parity."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.diagnostics.plots import prepare_plot_gam_data
from nampy.gam.diagnostics.sos_plots import lolaxy, prepare_sos_plot, repole
from nampy.gam.model_state import (
    _coef_full,
    _term_blocks_seq,
    _term_full_coefficient_indices,
)
from tests.reference_fixtures import (
    generation_enabled,
    load_reference,
    reference_key,
    save_reference,
)


def _reference(payload, code):
    key = reference_key("sos_plots", {"payload": payload, "code": code})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as directory:
            src, inp, out = [
                Path(directory) / n for n in ("ref.R", "in.json", "out.json")
            ]
            src.write_text(
                "args <- commandArgs(TRUE)\nlibrary(mgcv)\nlibrary(jsonlite)\na <- fromJSON(args[1])\n"
                + code
                + '\nwrite_json(ans,args[2],digits=17,auto_unbox=TRUE,na="null")\n'
            )
            inp.write_text(json.dumps(payload))
            proc = subprocess.run(
                ["Rscript", str(src), str(inp), str(out)],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stderr
            save_reference("mgcv_sos_plots", key, json.loads(out.read_text()))
    return load_reference("mgcv_sos_plots", key)


@pytest.mark.parametrize(
    "pole", [(0, 0), (0.7, 0.4), (-2, -1), (0, np.pi / 2), (np.pi, -np.pi / 2)]
)
def test_sos_coordinate_functions(pole):
    lo = np.linspace(-3, 3, 27)
    la = np.linspace(-1.4, 1.4, 27)[::-1]
    ref = _reference(
        {"lo": lo.tolist(), "la": la.tolist(), "theta": pole[0], "phi": pole[1]},
        "ans <- list(repole=mgcv:::repole(a$lo,a$la,a$theta,a$phi),lolaxy=mgcv:::lolaxy(a$lo,a$la,a$theta,a$phi))",
    )
    for name, values in [
        ("repole", repole(lo, la, *pole)),
        ("lolaxy", lolaxy(lo, la, *pole)),
    ]:
        for key, value in values.items():
            np.testing.assert_allclose(value, ref[name][key], atol=2e-13, rtol=2e-13)


CASES = [
    (30, 30, 10, True, False),
    (30, 30, 40, True, False),
    (0, 90, 8, True, False),
    (180, -90, 8, True, False),
    (750, 150, 10, True, True),
    (-390, -150, 10, True, False),
    (0, 0, 9, False, True),
    (20, 450, 6, True, True),
]


@pytest.mark.parametrize("theta,phi,n2,rug,by", CASES)
def test_sos_prepared_grid_and_predictions(theta, phi, n2, rug, by):
    i = np.arange(65, dtype=float)
    dat = pd.DataFrame(
        {"la": -78 + (i * 31) % 156, "lo": -175 + (i * 71) % 350, "z": 0.7 + i / 90}
    )
    dat["y"] = (
        np.sin(dat.la * np.pi / 180)
        + 0.3 * np.cos(dat.lo * np.pi / 180)
        + 0.1 * np.sin(i)
    )
    formula = 'y ~ s(la, lo, bs="sos", k=12, sp=.7' + (", by=z)" if by else ")")
    model = GAM(
        formula=formula,
        family="gaussian",
        optimize_smoothing=False,
        smoothing_method="REML",
    ).fit(data=dat)
    tb = next(tb for tb in _term_blocks_seq(model) if tb.basis_name == "sos")
    P = prepare_sos_plot(model, tb, n2=n2, theta=theta, phi=phi, rug=rug)
    ref = _reference(
        {
            "data": dat.to_dict(orient="list"),
            "formula": formula,
            "theta": theta,
            "phi": phi,
            "n2": n2,
            "rug": rug,
        },
        r"""
d <- as.data.frame(a$data)
b <- gam(as.formula(a$formula),data=d,method="REML")
s <- b$smooth[[1]]
P <- mgcv:::plot.sos.smooth(s,data=d,n2=a$n2,theta=a$theta,phi=a$phi,rug=a$rug)
P$fit <- as.numeric(P$X %*% coef(b)[s$first.para:s$last.para])
ans <- P
""",
    )
    for name in ["ind", "xm", "ym", "lo", "la"]:
        np.testing.assert_allclose(
            P[name],
            np.asarray(ref[name], dtype=float),
            rtol=1e-10,
            atol=3e-6 if name == "lo" else 2e-12,
            equal_nan=True,
            err_msg=name,
        )
    fit = P["X"] @ _coef_full(model)[_term_full_coefficient_indices(model, tb)]
    np.testing.assert_allclose(fit, ref["fit"], atol=2e-8, rtol=2e-8)
    if rug:
        for name in ("x", "y"):
            np.testing.assert_allclose(
                P["raw"][name], ref["raw"][name], atol=2e-12, rtol=2e-12
            )
    else:
        np.testing.assert_allclose(P["raw"], dat[["la", "lo"]])
    assert P["scale"] is False and P["se"] is False
    assert P["X"].shape[0] == np.count_nonzero(P["ind"])

    for scheme in (0, 1):
        integrated = prepare_plot_gam_data(
            model, n2=n2, theta=theta, phi=phi, rug=rug, scheme=scheme
        )["pd"][0]
        np.testing.assert_allclose(integrated["fit"], ref["fit"], atol=2e-8, rtol=2e-8)
        assert integrated["kind"] == "sos"
        assert integrated["plot_ci"] is False
        np.testing.assert_array_equal(integrated["ind"], P["ind"])


@pytest.mark.parametrize("scheme", [2, 3, 4])
def test_sos_generic_scheme_dispatch(scheme):
    i = np.arange(65, dtype=float)
    dat = pd.DataFrame({"la": -78 + (i * 31) % 156, "lo": -175 + (i * 71) % 350})
    dat["y"] = np.sin(dat.la * np.pi / 180) + 0.3 * np.cos(dat.lo * np.pi / 180)
    formula = 'y ~ s(la,lo,bs="sos",k=12,sp=.7)'
    model = GAM(formula=formula, family="gaussian", optimize_smoothing=False).fit(
        data=dat
    )
    P = prepare_plot_gam_data(model, n2=11, scheme=scheme, too_far=0)["pd"][0]
    ref = _reference(
        {"data": dat.to_dict(orient="list"), "formula": formula, "scheme": scheme},
        r"""
d <- as.data.frame(a$data)
b <- gam(as.formula(a$formula),data=d,method="REML")
s <- b$smooth[[1]]
P <- mgcv:::plot.sos.smooth(s,data=d,n2=11,scheme=a$scheme,too.far=0)
P$fit <- as.numeric(P$X %*% coef(b)[s$first.para:s$last.para])
ans <- P
""",
    )
    assert P["kind"] == "2d"
    assert P["scheme"] == scheme - 2
    for name in ("x", "y", "fit"):
        np.testing.assert_allclose(P[name], ref[name], atol=2e-8, rtol=2e-8)
