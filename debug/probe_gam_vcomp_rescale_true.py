from __future__ import annotations

import numpy as np

from nampy.gam.smoothing_selection.postfit import (
    _fit_scale,
    _mgcv_penalty_rescale_factors,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _make_gaussian_data,
    _run_mgcv_gam_vcomp,
    _run_mgcv_snapshot,
)


def main() -> None:
    data = _make_gaussian_data(seed=41, n=120)
    formula = 'y ~ s(x0, bs="cr", k=8)'
    gam = _fit_nampy_model(data, formula, "gaussian", "GCV")
    expected = _run_mgcv_gam_vcomp(data, formula, "gaussian", "GCV", rescale=True)
    expected_false = _run_mgcv_gam_vcomp(
        data, formula, "gaussian", "GCV", rescale=False
    )
    snapshot = _run_mgcv_snapshot(data, formula, "gaussian", "GCV")

    print("scale", float(_fit_scale(gam)))
    print("sp", np.asarray(gam.smoothing_params, dtype=np.float64))
    print("rescale_factors", _mgcv_penalty_rescale_factors(gam))
    print("vcomp_false", gam.gam_vcomp(rescale=False))
    print("vcomp_true", gam.gam_vcomp(rescale=True))
    print("expected", expected)
    print("expected_false", expected_false)
    print("snapshot_scale", snapshot["fit"]["scale"])
    print("snapshot_sp", snapshot["fit"]["smoothing_params"])


if __name__ == "__main__":
    main()
