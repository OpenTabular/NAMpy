from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.mgcv_parity_utils import (
    _fit_nampy_snapshot,
    _run_mgcv_snapshot,
)
from tests.smooths.test_mgcv_raw_constructor_parity import CASES


def main() -> None:
    case_ids = ["ti_2d_cs_cs", "ti_2d_cs_ps", "ti_2d_ps_cs"]
    raw_cases = {case.case_id: case for case in CASES}
    for case_id in case_ids:
        raw_case = raw_cases[case_id]
        data = raw_case.data_factory()
        formula = raw_case.formula.replace("])", "], sp=[0.7, 1.3])")
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        pred_diff = np.max(
            np.abs(
                np.asarray(actual["predictions"]["response"], dtype=np.float64)
                - np.asarray(expected["predictions"]["response"], dtype=np.float64)
            )
        )
        edf_diff = np.max(
            np.abs(
                np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64)
                - np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64)
            )
        )
        dev_diff = abs(
            float(actual["fit"]["deviance"]) - float(expected["fit"]["deviance"])
        )
        print(case_id)
        print(f"  {formula}")
        print(f"  max_response_abs_diff={pred_diff:.17g}")
        print(f"  max_edf_abs_diff={edf_diff:.17g}")
        print(f"  deviance_abs_diff={dev_diff:.17g}")


if __name__ == "__main__":
    main()
