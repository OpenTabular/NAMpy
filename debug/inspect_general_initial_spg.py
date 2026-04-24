from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam.smoothing_selection.optimize.basics import (  # noqa: E402
    _initial_smoothing_params_mgcv_style,
)
from nampy.gam.fit.solvers.general_family_solver import (  # noqa: E402
    build_general_family_setup_state,
)
from nampy.gam.smoothing_selection.reparam import (  # noqa: E402
    build_estimate_gam_setup_state,
)
from tests.optimization.test_mgcv_outer_optimization_parity import (  # noqa: E402
    _compile_optimization_state,
    _run_mgcv_initial_spg,
)
from debug.compare_general_ncv_fixed_sp import (  # noqa: E402
    _alt_singleton_repara_x,
)
from tests.optimization.test_mgcv_ncv_qncv_parity import (  # noqa: E402
    _make_gaulss_data,
)
from tests.families.test_general_family_mgcv_parity import (  # noqa: E402
    _gevlss_data,
    _gammals_data,
    _shashlss_data,
    _ziplss_data,
)


CASES = {
    "gaulss_ncv": (
        "gaulss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        lambda: _make_gaulss_data(seed=11, n=90),
        "NCV",
    ),
    "gammals_ncv": (
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        "NCV",
    ),
    "gevlss_ncv": (
        "gevlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_data,
        "NCV",
    ),
    "shashlss_ncv": (
        "shashlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_data,
        "NCV",
    ),
    "ziplss_ncv": (
        "ziplss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _ziplss_data,
        "NCV",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id", choices=sorted(CASES))
    args = parser.parse_args()

    family, formula, data_factory, method = CASES[args.case_id]
    data = data_factory()
    gam = _compile_optimization_state(data, formula, family, method)
    y = np.asarray(gam.y_, dtype=np.float64)

    init_actual = _initial_smoothing_params_mgcv_style(gam, y)
    init_expected = _run_mgcv_initial_spg(data, formula, family, method)

    n_sp = int(np.asarray(gam.smoothing_params, dtype=np.float64).size)
    fit5_setup = build_general_family_setup_state(
        gam,
        np.ones(n_sp, dtype=np.float64),
        score_type="REML",
    )
    x_expected = np.asarray(init_expected["X_initial"], dtype=np.float64)
    x_actual = np.asarray(fit5_setup.X_initial, dtype=np.float64)
    x_alt_numpy = _alt_singleton_repara_x(gam, "numpy")
    x_alt_scipy_evr = _alt_singleton_repara_x(gam, "scipy_evr")
    x_alt_scipy_ev = _alt_singleton_repara_x(gam, "scipy_ev")
    exact_setup = build_estimate_gam_setup_state(gam)
    weights = (
        np.ones_like(y, dtype=np.float64)
        if gam.prior_weights_ is None
        else np.asarray(gam.prior_weights_, dtype=np.float64)
    )
    start_actual = np.asarray(
        gam.family.initialize(
            y,
            np.asarray(init_expected["X_initial"], dtype=np.float64),
            fit5_setup.jj,
            offset=fit5_setup.offset_list,
            weights=weights,
            E=np.asarray(exact_setup.Eb, dtype=np.float64),
        ),
        dtype=np.float64,
    )
    start_expected = np.asarray(init_expected["start"], dtype=np.float64)
    diff = start_actual - start_expected

    report = {
        "case_id": args.case_id,
        "initial_sp_actual": None
        if init_actual is None
        else np.asarray(init_actual, dtype=np.float64).tolist(),
        "initial_sp_expected": np.asarray(
            init_expected["initial_sp"], dtype=np.float64
        ).tolist(),
        "start_actual": start_actual.tolist(),
        "start_expected": start_expected.tolist(),
        "start_diff": diff.tolist(),
        "max_abs_start_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "x_max_abs_diff": float(np.max(np.abs(x_actual - x_expected))),
        "x_alt_numpy_max_abs_diff": float(np.max(np.abs(x_alt_numpy - x_expected))),
        "x_alt_scipy_evr_max_abs_diff": float(
            np.max(np.abs(x_alt_scipy_evr - x_expected))
        ),
        "x_alt_scipy_ev_max_abs_diff": float(
            np.max(np.abs(x_alt_scipy_ev - x_expected))
        ),
        "x_col0_actual": x_actual[:, 0].tolist()[:5],
        "x_col0_expected": x_expected[:, 0].tolist()[:5],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
