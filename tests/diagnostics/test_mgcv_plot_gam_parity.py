"""plot.gam data parity: the numbers behind every plotted element vs mgcv.

``mgcv::plot.gam`` invisibly returns its per-term plot data (plots.r:1564),
which makes the whole data phase directly comparable: the x grids, the term
fits, the standard errors (with and without ``seWithMean``), the partial
residuals, and the 2-D grid with its too-far exclusion pattern. The rendering
phase is matplotlib and only smoke-tested (figure/axes structure).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from tests._paths import REPO_ROOT
from tests.reference_fixtures import (
    load_reference,
    portable_dataframe_identity,
    reference_key,
    save_reference,
)

pytestmark = [pytest.mark.surface_output]

R_SCRIPT = shutil.which("Rscript")

_R_DRIVER = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = TRUE)
mode <- args[[3]]

serialize_pd <- function(pd) {
  lapply(pd, function(P) {
    out <- list()
    for (nm in c("x", "y", "fit", "se", "p.resid", "raw")) {
      v <- P[[nm]]
      if (!is.null(v) && is.numeric(v)) out[[gsub(".", "_", nm, fixed=TRUE)]] <- as.numeric(v)
    }
    out
  })
}

png(tempfile(fileext = ".png"))
if (mode == "univariate") {
  b <- gam(y ~ s(x0, bs = "cr", k = 8) + s(x1, bs = "cr", k = 8) +
             s(f, bs = "re"),
           data = d, method = "REML")
  pd_plain <- plot(b, residuals = TRUE, seWithMean = FALSE)
  pd_mean <- plot(b, seWithMean = TRUE)
  pd_unconditional <- plot(b, unconditional = TRUE)
  payload <- list(
    plain = serialize_pd(pd_plain),
    with_mean = serialize_pd(pd_mean),
    unconditional = serialize_pd(pd_unconditional)
  )
} else if (mode == "bivariate") {
  b <- gam(y ~ s(x0, x1, bs = "tp", k = 20), data = d, method = "REML")
  pd <- plot(b)
  payload <- list(plain = serialize_pd(pd))
} else if (mode == "structured_fs") {
  b_fs <- gam(
    y ~ s(f, x0, bs = "fs", k = 5, xt = "cr", by = z,
          sp = c(0.7, 0.9, 1.1)),
    data = d,
    method = "REML"
  )
  payload <- list(
    plain = serialize_pd(plot(b_fs)),
    fitted = unname(as.numeric(fitted(b_fs)))
  )
} else if (mode == "structured_sz") {
  b_sz <- gam(
    y ~ s(f1, f2, x1, bs = "sz", k = 6),
    data = d,
    method = "REML"
  )
  payload <- list(
    plain = serialize_pd(plot(b_sz)),
    fitted = unname(as.numeric(fitted(b_sz)))
  )
} else if (mode == "factor_by") {
  b <- gam(y ~ f + s(x0, by = f, bs = "cr", k = 6),
           data = d, method = "REML")
  pd <- plot(b)
  payload <- list(plain = serialize_pd(pd))
} else {
  stop(sprintf("Unsupported plot parity mode: %s", mode))
}
dev.off()
write_json(payload, args[[2]], auto_unbox = TRUE, digits = 17, na = "null")
"""


def _run_plot_driver(data: pd.DataFrame, mode: str) -> dict:
    key = reference_key(
        "plot_gam", {"data": portable_dataframe_identity(data), "mode": mode}
    )
    cached = load_reference("mgcv", key)
    if cached is not None:
        return cached
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / "d.csv"
        json_path = tmp / "out.json"
        script_path = tmp / "plot_driver.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(_R_DRIVER, encoding="utf-8")
        subprocess.run(
            [R_SCRIPT, str(script_path), str(csv_path), str(json_path), mode],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))
        save_reference("mgcv", key, result)
        return result


def _univariate_data(seed=91, n=120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "x0": rng.uniform(-2.0, 2.0, n),
            "x1": rng.uniform(-1.5, 1.5, n),
            "f": rng.choice(np.array(["a", "b", "c"], dtype=object), n),
        }
    )
    level_effect = data["f"].map({"a": 0.5, "b": -0.2, "c": 0.1}).astype(float)
    data["y"] = (
        np.sin(1.2 * data["x0"])
        + 0.4 * data["x1"] ** 2
        + level_effect
        + rng.normal(0.0, 0.2, n)
    )
    return data


def _structured_plot_data(seed=97, n=144) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    row = np.arange(n)
    x0 = rng.uniform(-1.8, 1.8, n)
    x1 = rng.uniform(-1.5, 1.5, n)
    f = np.asarray(["a", "b", "c"])[row % 3]
    f1 = np.asarray(["u", "v", "w"])[row % 3]
    f2 = np.asarray(["left", "right"])[(row // 3) % 2]
    z = 0.8 + 0.2 * np.cos(x0)
    f_effect = np.asarray(
        [{"a": -0.25, "b": 0.1, "c": 0.3}[value] for value in f]
    )
    cell_effect = np.asarray(
        [
            {
                ("u", "left"): -0.15,
                ("u", "right"): 0.1,
                ("v", "left"): 0.2,
                ("v", "right"): -0.1,
                ("w", "left"): 0.05,
                ("w", "right"): -0.05,
            }[(left, right)]
            for left, right in zip(f1, f2, strict=True)
        ]
    )
    y = (
        z * (0.35 * np.sin(1.2 * x0) + f_effect)
        + z * (0.2 * x1**2 + cell_effect)
        + rng.normal(scale=0.18, size=n)
    )
    return pd.DataFrame(
        {
            "y": y,
            "x0": x0,
            "x1": x1,
            "z": z,
            "f": pd.Categorical(f),
            "f1": pd.Categorical(f1),
            "f2": pd.Categorical(f2),
        }
    )


def _fs_numeric_by_plot_data(seed=381, n=120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.0, 1.0, size=n)
    z = rng.uniform(0.5, 1.5, size=n)
    f = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    shifts = {"a": 0.35, "b": -0.25, "c": 0.15}
    y = z * (
        np.sin(1.4 * x0) + np.asarray([shifts[str(value)] for value in f])
    )
    y += rng.normal(0.0, 0.05, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "z": z, "f": f})


def _sz_plot_data(seed=41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = np.asarray(["a", "b", "b", "c", "c", "a", "b", "c"])
    f2 = np.asarray(["x", "x", "y", "x", "y", "y", "x", "y"])
    x1 = rng.uniform(0.0, 2.0, size=len(f1))
    base = np.sin(x1) + 0.2 * x1
    offsets = {
        ("a", "x"): 0.2,
        ("a", "y"): -0.1,
        ("b", "x"): -0.6,
        ("b", "y"): 0.4,
        ("c", "x"): 0.1,
        ("c", "y"): -0.3,
    }
    y = np.asarray(
        [base[i] + offsets[(f1[i], f2[i])] for i in range(len(f1))]
    )
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "f1": pd.Categorical(f1),
            "f2": pd.Categorical(f2),
        }
    )


def _num(values) -> np.ndarray:
    return np.asarray(
        [np.nan if v is None else float(v) for v in np.ravel(values)],
        dtype=np.float64,
    )


def test_plot_gam_univariate_and_re_data_match_mgcv():
    """1-D grids/fits/SEs/partial residuals and the re QQ effects match."""
    data = _univariate_data()
    expected = _run_plot_driver(data, "univariate")

    gam = GAM(
        family="gaussian",
        formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8) + s(f, bs="re")',
        optimize_smoothing=True,
        smoothing_method="REML",
    ).fit(data=data)

    from nampy.gam.diagnostics import prepare_plot_gam_data

    prepared_plain = prepare_plot_gam_data(
        gam, residuals=True, se_with_mean=False
    )
    prepared_mean = prepare_plot_gam_data(gam, se_with_mean=True)
    prepared_unconditional = prepare_plot_gam_data(gam, unconditional=True)

    for i in range(2):  # the two 1-D cr smooths
        P = prepared_plain["pd"][i]
        E = expected["plain"][i]
        np.testing.assert_allclose(
            np.asarray(P["x"], dtype=np.float64), _num(E["x"]), atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(P["fit"], dtype=np.float64).ravel(),
            _num(E["fit"]),
            atol=5e-6,
            rtol=5e-6,
        )
        np.testing.assert_allclose(
            np.asarray(P["se"], dtype=np.float64).ravel(),
            _num(E["se"]),
            atol=5e-6,
            rtol=5e-6,
        )
        np.testing.assert_allclose(
            np.asarray(P["p_resid"], dtype=np.float64),
            _num(E["p_resid"]),
            atol=5e-6,
            rtol=5e-6,
        )
        np.testing.assert_allclose(
            np.asarray(P["raw"], dtype=np.float64), _num(E["raw"]), atol=1e-12
        )

        P_mean = prepared_mean["pd"][i]
        E_mean = expected["with_mean"][i]
        np.testing.assert_allclose(
            np.asarray(P_mean["se"], dtype=np.float64).ravel(),
            _num(E_mean["se"]),
            atol=5e-6,
            rtol=5e-6,
        )
        # seWithMean must genuinely change the bands.
        assert not np.allclose(
            np.asarray(P_mean["se"], dtype=np.float64),
            np.asarray(P["se"], dtype=np.float64),
            atol=1e-10,
        )

        P_unconditional = prepared_unconditional["pd"][i]
        E_unconditional = expected["unconditional"][i]
        np.testing.assert_allclose(
            np.asarray(P_unconditional["se"], dtype=np.float64).ravel(),
            _num(E_unconditional["se"]),
            atol=5e-5,
            rtol=5e-5,
        )
        assert not np.allclose(
            np.asarray(P_unconditional["se"], dtype=np.float64),
            np.asarray(P["se"], dtype=np.float64),
            atol=1e-10,
        )

    # random effect: fit is the coefficient vector itself (X = identity)
    P_re = prepared_plain["pd"][2]
    E_re = expected["plain"][2]
    np.testing.assert_allclose(
        np.sort(np.asarray(P_re["fit"], dtype=np.float64)),
        np.sort(_num(E_re["fit"])),
        atol=5e-6,
        rtol=5e-6,
    )


def test_plot_gam_bivariate_grid_fit_and_exclusion_match_mgcv():
    """2-D grid coordinates, fits and the too-far NA pattern match mgcv."""
    data = _univariate_data(seed=93, n=140)
    expected = _run_plot_driver(data, "bivariate")

    gam = GAM(
        family="gaussian",
        formula='y ~ s(x0, x1, bs="tp", k=20)',
        optimize_smoothing=True,
        smoothing_method="REML",
    ).fit(data=data)

    from nampy.gam.diagnostics import prepare_plot_gam_data

    prepared = prepare_plot_gam_data(gam)
    P = prepared["pd"][0]
    E = expected["plain"][0]

    np.testing.assert_allclose(
        np.asarray(P["x"], dtype=np.float64), _num(E["x"]), atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(P["y"], dtype=np.float64), _num(E["y"]), atol=1e-10
    )
    fit_actual = np.asarray(P["fit"], dtype=np.float64).ravel()
    fit_expected = _num(E["fit"])
    assert fit_actual.shape == fit_expected.shape
    # identical too-far exclusion pattern (NA positions)
    np.testing.assert_array_equal(
        np.isnan(fit_actual), np.isnan(fit_expected)
    )
    mask = ~np.isnan(fit_expected)
    np.testing.assert_allclose(
        fit_actual[mask], fit_expected[mask], atol=5e-5, rtol=5e-5
    )


def test_plot_gam_fs_sz_numeric_by_data_match_mgcv():
    """FS/SZ curve grids, numeric-by activation, fits and SZ bands match."""
    cases = (
        (
            "fs",
            _fs_numeric_by_plot_data(),
            "structured_fs",
            'y ~ s(f, x0, bs="fs", k=5, xt="cr", by=z,'
            ' sp=[0.7, 0.9, 1.1])',
            False,
            2e-6,
        ),
        (
            "sz",
            _sz_plot_data(),
            "structured_sz",
            'y ~ s(f1, f2, x1, bs="sz", k=6)',
            True,
            6e-3,
        ),
    )

    from nampy.gam.diagnostics import prepare_plot_gam_data

    prepared = []
    references = []
    for kind, data, mode, formula, optimize_smoothing, fit_atol in cases:
        expected = _run_plot_driver(data, mode)
        gam = GAM(
            family="gaussian",
            formula=formula,
            optimize_smoothing=optimize_smoothing,
            smoothing_method="REML" if optimize_smoothing else "fixed",
        ).fit(data=data)
        actual = prepare_plot_gam_data(gam)["pd"][0]
        reference = expected["plain"][0]
        assert actual["kind"] == kind
        prepared.append(actual)
        references.append(reference)
        np.testing.assert_allclose(
            np.asarray(gam.predict(data, type="response"), dtype=np.float64),
            _num(expected["fitted"]),
            atol=fit_atol,
            rtol=fit_atol,
        )
        np.testing.assert_allclose(
            np.asarray(actual["x"], dtype=np.float64),
            _num(reference["x"]),
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"], dtype=np.float64),
            _num(reference["fit"]),
            atol=max(2e-3, fit_atol),
            rtol=max(2e-3, fit_atol),
        )
        np.testing.assert_allclose(
            np.asarray(actual["raw"], dtype=np.float64),
            _num(reference["raw"]),
            atol=1e-12,
            rtol=0.0,
        )

    assert prepared[0]["se"] is False
    np.testing.assert_allclose(
        np.asarray(prepared[1]["se"], dtype=np.float64),
        _num(references[1]["se"]),
        atol=2e-2,
        rtol=2e-2,
    )


def test_plot_gam_factor_by_curve_activation_matches_mgcv():
    """Each factor-by plot block activates its own level exactly as mgcv."""
    data = _structured_plot_data(seed=99)
    expected = _run_plot_driver(data, "factor_by")["plain"]
    gam = GAM(
        family="gaussian",
        formula='y ~ f + s(x0, by=f, bs="cr", k=6)',
        optimize_smoothing=True,
        smoothing_method="REML",
    ).fit(data=data)

    from nampy.gam.diagnostics import prepare_plot_gam_data

    prepared = prepare_plot_gam_data(gam)["pd"]
    assert len(prepared) == len(expected) == 3
    for actual, reference in zip(prepared, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual["x"], dtype=np.float64),
            _num(reference["x"]),
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(actual["fit"], dtype=np.float64),
            _num(reference["fit"]),
            atol=8e-5,
            rtol=8e-5,
        )
        np.testing.assert_allclose(
            np.asarray(actual["se"], dtype=np.float64),
            _num(reference["se"]),
            atol=8e-5,
            rtol=8e-5,
        )


def test_plot_gam_renders_figures_and_returns_plot_data():
    """Rendering smoke: figures exist, page layout honors `pages`."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    data = _univariate_data(seed=95, n=90)
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)',
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[1.0, 1.0],
    ).fit(data=data)

    out = gam.plot(pages=1)
    assert len(out["figures"]) == 1
    assert sum(1 for P in out["pd"] if P.get("plot_me")) == 2

    out_sel = gam.plot(select=1)
    assert len(out_sel["figures"]) == 1

    import matplotlib.pyplot as plt

    plt.close("all")
