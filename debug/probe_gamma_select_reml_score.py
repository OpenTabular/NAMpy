"""Probe Gamma select=TRUE REML score storage for the extended snapshot case."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.mgcv_parity_utils import (  # noqa: E402
    _fit_nampy_model,
    _make_gamma_data,
    _run_mgcv_snapshot,
)


def main() -> None:
    data = _make_gamma_data(seed=302, n=220)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    gam = _fit_nampy_model(data, formula, "gamma", "REML", select=True)
    result = gam._optim_result
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(data, formula, "gamma", "REML", select=True)
    view = actual.get("parity", {}).get("criterion_view", {})
    print("actual criterion:", repr(actual["fit"]["criterion_value"]))
    print("expected criterion:", repr(expected["fit"]["criterion_value"]))
    print("criterion diff:", actual["fit"]["criterion_value"] - expected["fit"]["criterion_value"])
    print("sp diff:", np.asarray(actual["fit"]["smoothing_params"]) - np.asarray(expected["fit"]["smoothing_params"]))
    print("view:", view)
    print("result.fun:", repr(getattr(result, "fun", None)))
    print("joint_gamma:", repr(getattr(result, "joint_gamma_reml_outer", None)))
    print("joint_log_phi:", repr(getattr(result, "joint_log_phi", None)))
    print("outer score_hist:", getattr(result, "mgcv_score_hist", None))
    print("outer_info score_hist:", getattr(result, "outer_info", {}).get("score_hist", None))


if __name__ == "__main__":
    main()
