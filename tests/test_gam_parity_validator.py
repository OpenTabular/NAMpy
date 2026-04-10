from __future__ import annotations

from pathlib import Path

from mgcv_parity_utils import _make_gaussian_data

from nampy.basemodels.gam import GAM
from nampy.gam._mgcv_constants import (
    EIG_TOL_POWER,
    FAMILY_EPS,
    GAMMA_ABSTOL,
    LINK_ETA_EXP_CLIP,
    LOG_GUARD_MIN,
    PENALTY_RIDGE_REL,
    QR_TOL_SCALE,
    SMOOTHING_SCORE_ABS_FLOOR,
)
from nampy.gam.parity import build_parity_snapshot, compare, save_parity_snapshot


def test_mgcv_constants_frozen_to_1_9_1_reference():
    assert EIG_TOL_POWER == 0.8
    assert PENALTY_RIDGE_REL == 1e-6
    assert GAMMA_ABSTOL == 1e-12
    assert QR_TOL_SCALE == 1.0
    assert FAMILY_EPS == 1e-9
    assert LINK_ETA_EXP_CLIP == 700.0
    assert LOG_GUARD_MIN == 1e-300
    assert SMOOTHING_SCORE_ABS_FLOOR == 1e-8


def test_parity_validator_compare_accepts_dict_and_path(tmp_path: Path):
    data = _make_gaussian_data(n=40)
    model = GAM(
        formula='y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)',
        optimize_smoothing=False,
        smoothing_params=[1.0, 1.0],
    )
    model.fit(data=data)

    snapshot = build_parity_snapshot(model, X=data)
    report_dict = compare(model, snapshot)
    assert report_dict["passed"] is True

    snapshot_path = tmp_path / "mgcv_snapshot.json"
    save_parity_snapshot(snapshot, snapshot_path)

    report_path = compare(model, snapshot_path)
    assert report_path["passed"] is True
