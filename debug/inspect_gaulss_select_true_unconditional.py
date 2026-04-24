from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.families.test_general_family_mgcv_parity import _gaulss_data
from tests.mgcv_parity_utils import _fit_nampy_model, _run_mgcv_snapshot


def inspect(seed: int) -> None:
    data = _gaulss_data(seed=seed)
    gam = _fit_nampy_model(
        data,
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        "gaulss",
        "ML",
        select=True,
    )
    expected = _run_mgcv_snapshot(
        data,
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        "gaulss",
        "ML",
        select=True,
    )
    outer_info = dict(getattr(gam._optim_result, "outer_info", {}) or {})
    print(f"seed={seed}")
    print(f"outer_info keys={sorted(outer_info.keys())}")
    for key in ("hess", "hess1", "db_drho1", "lsp1"):
        value = outer_info.get(key, None)
        if value is None:
            print(f"  {key}=None")
        else:
            arr = np.asarray(value, dtype=np.float64)
            print(f"  {key}.shape={arr.shape}")
    cov = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)
    print(f"  cov.trace={float(np.trace(cov)):.16f}")
    print(f"  sp={np.asarray(gam.smoothing_params, dtype=np.float64)}")
    print(
        "  expected_sp="
        f"{np.asarray(expected['fit']['smoothing_params'], dtype=np.float64)}"
    )
    expected_cov = np.asarray(expected["fit"]["cov_unconditional"], dtype=np.float64)
    print(f"  expected_cov.trace={float(np.trace(expected_cov)):.16f}")


if __name__ == "__main__":
    inspect(11)
    inspect(13)
