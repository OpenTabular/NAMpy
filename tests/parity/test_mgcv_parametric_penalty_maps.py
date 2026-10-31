"""Static source parity for general paraPen log-smoothing-parameter maps."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.selection.reparam import build_estimate_gam_setup_state
from nampy.gam.model_state import _edf_total, _penalty_blocks_seq
from tests.reference_fixtures import load_reference, reference_key, save_reference

ROOT = Path(__file__).parents[1] / "reference_fixtures/mgcv_parametric_penalty_maps"
PROBE = Path(__file__).with_name("probes") / "mgcv_parametric_penalty_maps.R"
MAPS = {
    "dense": [[1.0, 0.5], [-0.25, 1.0], [0.75, -0.5]],
    "fractional": [[0.5], [1.5], [-0.25]],
    "constant": [[1.0, 0.0], [0.0, 0.0], [0.5, 1.0]],
    "unused": [[1.0, 0.0], [0.5, 0.0], [0.0, 0.0]],
    "onehot_unused": [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    "shared": [[1.0], [1.0], [1.0]],
    "square": [[1.0, 0.5, 0.0], [-0.25, 1.0, 0.2], [0.75, -0.5, 1.0]],
    "dependent": [[1.0, 2.0], [0.5, 1.0], [0.25, 0.5]],
}


def payload(
    name="dense", family="gaussian", state="fixed", smooth=False, optimizer="newton"
):
    data = json.loads((ROOT / "inputs.json").read_text())
    mapping = MAPS[name]
    count = len(mapping[0])
    sp = [0.7, 1.3, 0.8][:count]
    if state == "selected":
        sp = [-1.0] * count
    elif state == "mixed":
        sp[-1] = -1.0
    elif state == "zero":
        sp[0] = 0.0
    return {
        "data": data,
        "family": family,
        "L": mapping,
        "penalties": [np.diag(np.eye(3)[i]).tolist() for i in range(3)],
        "sp": sp,
        "eval_sp": [0.9, 1.2, 1.1][:count],
        "smooth": smooth,
        "optimizer": optimizer,
    }


def source_reference(inputs, probe=PROBE):
    key = reference_key(
        "para-pen-log-map-v1",
        {
            "payload": inputs,
            "source_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        },
    )
    result = load_reference("mgcv_parametric_penalty_maps", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            inp, out = [
                Path(directory) / name for name in ("inputs.json", "output.json")
            ]
            inp.write_text(json.dumps(inputs))
            run = subprocess.run(
                ["Rscript", str(probe), str(inp), str(out)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(out.read_text())
        save_reference("mgcv_parametric_penalty_maps", key, result)
    return result


def fit_python(inputs, *, control=None):
    data = pd.DataFrame(inputs["data"])
    data["y"] = data[f"y_{inputs['family']}"]
    data["B"] = list(data[["x", "u", "v"]].to_numpy())
    family = inputs["family"]
    formula = "y~B" + ("+s(z,bs='cr',k=5,sp=.4)" if inputs["smooth"] else "")
    optimizer = {"newton": "outer_newton"}.get(inputs["optimizer"], inputs["optimizer"])
    model = GAM(
        formula=formula,
        family=family,
        smoothing_method="REML",
        smoothing_optimizer=optimizer,
        optimize_smoothing=True,
        control=control or {"epsilon": 1e-10, "newton": {"conv.tol": 1e-10}},
        para_pen={
            "B": {
                "penalties": inputs["penalties"],
                "L": inputs["L"],
                "sp": inputs["sp"],
            }
        },
    ).fit(data)
    newdata = data.iloc[np.linspace(1, len(data) - 1, 13).astype(int)].copy()
    return model, data, newdata


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("state", ["fixed", "mixed", "selected"])
def test_dense_parametric_log_map_source_fits(family, state):
    inputs = payload(family=family, state=state)
    reference = source_reference(inputs)
    assert reference["ok"], reference
    model, data, newdata = fit_python(inputs)
    assert_fit(model, data, newdata, reference)


def test_mixed_gaussian_log_map_equivalent_newton_endpoint():
    inputs = payload(family="gaussian", state="mixed")
    reference = source_reference(inputs)
    # Stop one Newton step earlier on the roundoff-level objective plateau,
    # reproducing the endpoint seen with macOS's numerical libraries.
    model, data, newdata = fit_python(
        inputs, control={"epsilon": 1e-10, "newton": {"conv.tol": 1e-8}}
    )
    assert_fit(model, data, newdata, reference)
    np.testing.assert_allclose(
        model.fit_result().criterion_value, reference["score"], atol=1e-10, rtol=0
    )


@pytest.mark.parametrize(
    "name,state",
    [
        ("fractional", "fixed"),
        ("fractional", "selected"),
        ("constant", "fixed"),
        ("constant", "mixed"),
        ("constant", "selected"),
        ("unused", "fixed"),
        ("dependent", "fixed"),
        ("dense", "zero"),
    ],
)
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_general_parametric_log_map_rows_and_fixed_zero(name, state, family):
    inputs = payload(name=name, family=family, state=state)
    reference = source_reference(inputs)
    assert reference["ok"], reference
    model, data, newdata = fit_python(inputs)
    assert_fit(model, data, newdata, reference)


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_dense_parametric_log_map_with_ordinary_smooth(family):
    inputs = payload(family=family, state="selected", smooth=True)
    reference = source_reference(inputs)
    assert reference["ok"], reference
    model, data, newdata = fit_python(inputs)
    assert_fit(model, data, newdata, reference)


@pytest.mark.parametrize("name,smooth", [("shared", True), ("onehot_unused", False)])
def test_one_hot_map_source_ordering_and_unused_fixed_coordinate(name, smooth):
    inputs = payload(name=name, state="fixed", smooth=smooth)
    reference = source_reference(inputs)
    assert reference["ok"], reference
    model, data, newdata = fit_python(inputs)
    assert_fit(model, data, newdata, reference)


@pytest.mark.parametrize(
    "name,state,optimizer",
    [
        ("unused", "selected", "newton"),
        ("dependent", "selected", "newton"),
        ("shared", "selected", "efs"),
        ("dense", "selected", "efs"),
        ("square", "selected", "efs"),
    ],
)
def test_upstream_map_optimizer_classification(name, state, optimizer):
    inputs = payload(name=name, state=state, optimizer=optimizer)
    reference = source_reference(inputs)
    if name == "square":
        assert reference["ok"], reference
        X = np.asarray(reference["X"])
        y = np.asarray(inputs["data"]["y_gaussian"])
        full_penalties = []
        for S, off in zip(
            reference["setup"]["S"], reference["setup"]["off"], strict=True
        ):
            full = np.zeros((X.shape[1], X.shape[1]))
            start = int(off) - 1
            full[start : start + len(S), start : start + len(S)] = S
            full_penalties.append(full)

        def fitted_at(sp):
            penalty = sum(
                (value * S for value, S in zip(sp, full_penalties, strict=True)),
                start=np.zeros((X.shape[1], X.shape[1])),
            )
            return X @ np.linalg.solve(X.T @ X + penalty, X.T @ y)

        np.testing.assert_allclose(
            fitted_at(reference["sp"]), reference["fitted"], atol=2e-7, rtol=2e-7
        )
        assert (
            np.max(np.abs(fitted_at(reference["full_sp"]) - reference["fitted"])) > 1e-4
        )
    else:
        assert not reference["ok"], reference
        assert reference["error"]
    if optimizer == "efs":
        with pytest.raises(NotImplementedError, match="efsudr.*L/lsp0"):
            fit_python(inputs)
    else:
        with pytest.raises(ValueError, match="rank-deficient.*log-SP map"):
            fit_python(inputs)


def test_source_general_and_transformed_penalty_map_classification():
    reference = source_reference(
        payload(), PROBE.with_name("mgcv_parametric_penalty_map_rejections.R")
    )
    for name, result in reference["cases"].items():
        assert not result["ok"], (name, result)
        assert result["error"]
    inputs = payload()
    frame = pd.DataFrame(inputs["data"])
    frame["B"] = list(frame[["x", "u", "v"]].to_numpy())
    frame["y"] = (frame.y_poisson > 1).astype(int)
    pp = {"B": {"penalties": inputs["penalties"], "L": inputs["L"], "sp": inputs["sp"]}}
    with pytest.raises(ValueError, match="general-family formula-list"):
        GAM(formula=["y~B"], family=("multinom", {"K": 1}), para_pen=pp).fit(frame)
    frame["y"] = frame.y_gaussian
    with pytest.raises(NotImplementedError, match="transformed-coefficient"):
        GAM(formula="y~B+s(z,bs='mpi',k=5)", para_pen=pp).fit(frame)


def test_positive_min_sp_map_scope_is_explicit():
    inputs = payload(name="square")
    reference = source_reference(
        inputs, PROBE.with_name("mgcv_parametric_penalty_map_min_sp.R")
    )
    assert reference["parametric_only"]["ok"] and reference["with_smooth"]["ok"]
    assert np.max(np.abs(reference["parametric_only"]["H"])) > 0
    assert np.max(np.abs(reference["with_smooth"]["H"])) > 0
    frame = pd.DataFrame(inputs["data"])
    frame["y"] = frame.y_gaussian
    frame["B"] = list(frame[["x", "u", "v"]].to_numpy())
    with pytest.raises(NotImplementedError, match="full-penalty coordinates"):
        GAM(
            formula="y~B",
            min_sp=0.2,
            para_pen={
                "B": {
                    "penalties": inputs["penalties"],
                    "L": inputs["L"],
                    "sp": inputs["sp"],
                }
            },
        ).fit(frame)


def test_fixed_mapped_parameters_allow_efs_spelling_without_selection():
    inputs = payload()
    reference = source_reference(inputs)
    data = pd.DataFrame(inputs["data"])
    data["y"] = data.y_gaussian
    data["B"] = list(data[["x", "u", "v"]].to_numpy())
    model = GAM(
        formula="y~B",
        smoothing_optimizer="efs",
        smoothing_method="fixed",
        optimize_smoothing=False,
        para_pen={
            "B": {
                "penalties": inputs["penalties"],
                "L": inputs["L"],
                "sp": inputs["sp"],
            }
        },
    ).fit(data)
    np.testing.assert_allclose(
        model.predict(), reference["fitted"], atol=2e-7, rtol=2e-7
    )
    np.testing.assert_array_equal(model.smoothing_fixed_mask_, [True, True])
    np.testing.assert_array_equal(model.smoothing_params, inputs["sp"])


def test_mapped_penalty_persistence(tmp_path):
    inputs = payload(state="mixed", smooth=True)
    model, _, newdata = fit_python(inputs)
    restored = GAM.load_model(model.save_model(tmp_path / "mapped-penalties.pkl"))
    np.testing.assert_allclose(
        restored.predict(newdata), model.predict(newdata), atol=2e-12, rtol=2e-12
    )
    P, Q = (
        model.predict(newdata, type="lpmatrix"),
        restored.predict(newdata, type="lpmatrix"),
    )
    np.testing.assert_allclose(
        Q @ restored.vcov(unconditional=True) @ Q.T,
        P @ model.vcov(unconditional=True) @ P.T,
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_array_equal(restored.smoothing_params, model.smoothing_params)


@pytest.mark.parametrize("state", ["fixed", "selected"])
def test_multiple_parametric_terms_keep_local_map_columns(state):
    inputs = payload(state=state)
    inputs["data"] = json.loads((ROOT / "multiple_inputs.json").read_text())
    inputs["z_sp"] = 0.9 if state == "fixed" else -1.0
    reference = source_reference(
        inputs, PROBE.with_name("mgcv_parametric_penalty_map_multiple_terms.R")
    )
    data = pd.DataFrame(inputs["data"])
    data["y"] = data.y_gaussian
    data["B"] = list(data[["x", "u", "v"]].to_numpy())
    model = GAM(
        formula="y~B+z",
        smoothing_method="REML",
        optimize_smoothing=True,
        control={"epsilon": 1e-10, "newton": {"conv.tol": 1e-10}},
        para_pen={
            "B": {
                "penalties": inputs["penalties"],
                "L": inputs["L"],
                "sp": inputs["sp"],
            },
            "z": {"penalties": [[[2.0]]], "L": [[0.5]], "sp": [inputs["z_sp"]]},
        },
    ).fit(data)
    newdata = data.iloc[np.linspace(1, len(data) - 1, 13).astype(int)].copy()
    assert model.smoothing_params.size == 3
    assert_fit(model, data, newdata, reference)


def test_penalty_coordinate_evaluations_preserve_underlying_state(monkeypatch):
    from nampy.gam.fit.backends import solve_gaussian_given_smoothing
    from nampy.gam.fit.penalty_coordinates import penalty_coordinate_model
    from nampy.gam.fit.selection.criteria.dispatch import (
        criterion_gradient,
        criterion_hessian,
        criterion_value,
    )
    from nampy.gam.fit.selection.reparam import _canonical_penalty_space
    from nampy.gam.fit.solvers import gaussian_exact
    from nampy.gam.model_state import _compiled_model, _fit_workspace

    model, data, newdata = fit_python(payload(state="mixed"))
    result, compiled = model.gam_result_, _compiled_model(model)
    sp = model.smoothing_params.copy()
    cache = _canonical_penalty_space(model)
    metadata = repr([block.metadata for block in _penalty_blocks_seq(model)])
    prediction = model.predict(newdata).copy()
    for value in (1.2, 0.8, 1.2):
        rho = np.log([value])
        criterion_value(model, data.y.to_numpy(), rho, "reml")
        criterion_gradient(model, data.y.to_numpy(), rho, "reml")
        criterion_hessian(model, data.y.to_numpy(), rho, "reml")
        assert model.gam_result_ is result and _compiled_model(model) is compiled
        assert _fit_workspace(model).get("penalty_subspace_cache") is cache
        np.testing.assert_array_equal(model.smoothing_params, sp)
        assert (
            repr([block.metadata for block in _penalty_blocks_seq(model)]) == metadata
        )
    with pytest.raises(RuntimeError, match="view assignment"):
        with penalty_coordinate_model(model, sp) as (view, full_sp, _):
            assert full_sp.size == len(_penalty_blocks_seq(model))
            view.smoothing_params = np.empty(0)
            view.smoothing_fixed_mask_ = np.empty(0, dtype=bool)
            view.gam_result_ = None
            raise RuntimeError("view assignment")
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("inner fit failure")

        patch.setattr(gaussian_exact, "_solve_gaussian_working_predictor", fail)
        with pytest.raises(RuntimeError, match="inner fit failure"):
            solve_gaussian_given_smoothing(model, data.y.to_numpy(), sp)
    assert model.gam_result_ is result and _compiled_model(model) is compiled
    assert _fit_workspace(model).get("penalty_subspace_cache") is cache
    np.testing.assert_array_equal(model.smoothing_params, sp)
    np.testing.assert_array_equal(model.predict(newdata), prediction)
    last = _fit_workspace(model).get("pirls_last_coef")
    if last is not None:
        assert np.asarray(last).size == compiled.coefficient_transform.size


@pytest.mark.parametrize(
    "family,criterion", [("gaussian", "GCV"), ("poisson", "UBRE"), ("poisson", "NCV")]
)
def test_mapped_ordinary_selection_criteria(family, criterion):
    from nampy.gam.fit.selection.criteria.dispatch import (
        criterion_gradient,
        criterion_hessian,
        criterion_value,
    )

    inputs = payload(family=family, state="selected")
    inputs["criterion"] = criterion
    reference = source_reference(
        inputs, PROBE.with_name("mgcv_parametric_penalty_map_criteria.R")
    )
    model, data, _ = fit_python(inputs)
    rho = np.log(np.asarray(inputs["eval_sp"]))
    score = criterion_value(model, data.y.to_numpy(), rho, method=criterion.lower())
    gradient = criterion_gradient(
        model, data.y.to_numpy(), rho, method=criterion.lower()
    )
    np.testing.assert_allclose(score, reference["score"], atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(gradient, reference["gradient"], atol=2e-8, rtol=2e-8)
    if reference["hessian"] is not None:
        hessian = criterion_hessian(
            model, data.y.to_numpy(), rho, method=criterion.lower()
        )
        np.testing.assert_allclose(hessian, reference["hessian"], atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("family", ["nb", "scat"])
@pytest.mark.parametrize("state", ["fixed", "mixed", "selected"])
def test_extended_family_parametric_log_map_source_fits(family, state):
    inputs = payload(family=family, state=state)
    reference = source_reference(inputs)
    assert reference["ok"], reference
    model, data, newdata = fit_python(inputs)
    assert_fit(model, data, newdata, reference)
    np.testing.assert_allclose(
        model.family.getTheta(True), reference["theta"], atol=2e-7, rtol=2e-7
    )


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("state", ["fixed", "mixed", "selected"])
def test_parametric_log_map_analytic_score_gradient_hessian(family, state):
    from nampy.gam.fit.selection.criteria.dispatch import (
        criterion_gradient,
        criterion_hessian,
    )
    from nampy.gam.fit.selection.criteria.gaussian_dyn import (
        criterion_gradient_ml_reml_gaussian_dynamic_joint,
        criterion_hessian_ml_reml_gaussian_dynamic_joint,
        criterion_ml_reml_gaussian_dynamic_joint,
    )
    from nampy.gam.fit.selection.criteria.ml_reml import criterion_ml_reml

    inputs = payload(family=family, state=state)
    reference = source_reference(inputs)["derivatives"]
    model, data, _ = fit_python(inputs)
    free = ~model.smoothing_fixed_mask_
    rho = np.log(np.asarray(inputs["eval_sp"])[free])
    y = data["y"].to_numpy()
    if family == "gaussian":
        scale = reference["log_scale"]
        score = criterion_ml_reml_gaussian_dynamic_joint(model, y, rho, scale)
        gradient = criterion_gradient_ml_reml_gaussian_dynamic_joint(
            model, y, rho, scale
        )
        hessian = criterion_hessian_ml_reml_gaussian_dynamic_joint(model, y, rho, scale)
    else:
        score = criterion_ml_reml(model, y, rho, "reml")
        gradient = criterion_gradient(model, y, rho, "reml")
        hessian = criterion_hessian(model, y, rho, "reml")
    np.testing.assert_allclose(score, reference["score"], atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(
        gradient, np.atleast_1d(reference["gradient"]), atol=2e-8, rtol=2e-8
    )
    np.testing.assert_allclose(
        hessian,
        np.asarray(reference["hessian"]).reshape(hessian.shape),
        atol=2e-8,
        rtol=2e-8,
    )


@pytest.mark.parametrize("family", ["nb", "scat"])
def test_extended_log_map_joint_analytic_derivatives(family):
    from nampy.gam.fit.selection.criteria.pirls import family_negbin as nb
    from nampy.gam.fit.selection.criteria.pirls import family_vector_theta as vec

    inputs = payload(family=family, state="mixed")
    reference = source_reference(
        inputs, PROBE.with_name("mgcv_parametric_penalty_map_extended_derivatives.R")
    )
    model, data, _ = fit_python(inputs)
    rho = np.log(np.asarray(inputs["eval_sp"])[~model.smoothing_fixed_mask_])
    theta = np.atleast_1d(reference["theta"])
    if family == "nb":
        score = nb.criterion_ml_reml_pirls_negbin_joint(
            model, data.y.to_numpy(), rho, theta[0], "REML"
        )
        gradient = nb.criterion_gradient_ml_reml_pirls_negbin_joint(
            model, data.y.to_numpy(), rho, theta[0], "REML"
        )
        hessian = nb.criterion_hessian_ml_reml_pirls_negbin_joint(
            model, data.y.to_numpy(), rho, theta[0], "REML"
        )
        # The convenience NB criterion API orders SP first, theta last; mgcv's
        # raw gam.fit4 derivative arrays prepend theta.
        order = np.r_[gradient.size - 1, np.arange(gradient.size - 1)]
        gradient = gradient[order]
        hessian = hessian[np.ix_(order, order)]
    else:
        score = vec.criterion_ml_reml_pirls_vector_theta_joint(
            model, data.y.to_numpy(), rho, theta, "REML"
        )
        gradient = vec.criterion_gradient_ml_reml_pirls_vector_theta_joint(
            model, data.y.to_numpy(), rho, theta, "REML"
        )
        hessian = vec.criterion_hessian_ml_reml_pirls_vector_theta_joint(
            model, data.y.to_numpy(), rho, theta, "REML"
        )
    np.testing.assert_allclose(score, reference["score"], atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(gradient, reference["gradient"], atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(hessian, reference["hessian"], atol=2e-8, rtol=2e-8)


def test_tweedie_log_map_theta_scale_analytic_derivatives():
    from nampy.gam.families.tweedie import TweedieTwFamily
    from nampy.gam.fit.selection.criteria.pirls import tweedie

    inputs = payload(family="poisson", state="mixed")
    reference = source_reference(
        inputs, PROBE.with_name("mgcv_parametric_penalty_map_tweedie_derivatives.R")
    )
    data = pd.DataFrame(inputs["data"])
    data["y"] = data.y_poisson
    data["B"] = list(data[["x", "u", "v"]].to_numpy())
    family = TweedieTwFamily()
    family._disable_theta_efs = True
    model = GAM(
        formula="y~B",
        family=family,
        smoothing_method="fixed",
        optimize_smoothing=False,
        control={"epsilon": 1e-10},
        para_pen={
            "B": {
                "penalties": inputs["penalties"],
                "L": inputs["L"],
                "sp": inputs["sp"],
            }
        },
    ).fit(data)
    rho = np.log(np.asarray(inputs["eval_sp"])[~model.smoothing_fixed_mask_])
    args = (
        model,
        data.y.to_numpy(),
        rho,
        reference["theta"],
        reference["log_phi"],
        "REML",
    )
    score = tweedie.criterion_ml_reml_pirls_tweedie_joint(*args)
    gradient = tweedie.criterion_gradient_ml_reml_pirls_tweedie_joint(*args)
    hessian = tweedie.criterion_hessian_ml_reml_pirls_tweedie_joint(*args)
    np.testing.assert_allclose(score, reference["score"], atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(gradient, reference["gradient"], atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(hessian, reference["hessian"], atol=2e-8, rtol=2e-8)


def assert_fit(model, data, newdata, reference):
    prediction, se = model.predict(newdata, type="link", return_se=True)
    P = model.predict(newdata, type="lpmatrix")
    for name, actual in {
        "prediction": prediction,
        "se": se,
        "fitted": model.predict(type="link"),
        "covariance": P @ model.vcov() @ P.T,
        "freq_covariance": P @ model.vcov(freq=True) @ P.T,
        "edf": _edf_total(model),
        "scale": model.fit_result().scale,
        "aic": model.aic(),
        "loglik": model.loglik(),
        "score": model.fit_result().criterion_value,
    }.items():
        np.testing.assert_allclose(
            actual, reference[name], atol=2e-7, rtol=2e-7, err_msg=name
        )
    # Newton optimizes log(SP). At roundoff-level changes in REML, BLAS
    # differences can reject its final step: the mixed Gaussian fixture's
    # last step changes log(SP) by 3.66e-7 but REML by only 2.84e-14.
    # Allow 1e-6 in those coordinates, keeping fitted quantities above and
    # fixed-coordinate score/gradient/Hessian checks at their tighter tolerances.
    np.testing.assert_allclose(
        np.log(model.smoothing_params[~model.smoothing_fixed_mask_]),
        np.log(reference["sp"]),
        atol=1e-6,
        rtol=0,
        err_msg="optimized log smoothing parameters",
    )
    if reference["unconditional_covariance"] is not None:
        np.testing.assert_allclose(
            P @ model.vcov(unconditional=True) @ P.T,
            reference["unconditional_covariance"],
            atol=2e-7,
            rtol=2e-7,
        )
    setup = build_estimate_gam_setup_state(model)
    expected = reference["setup"]
    if expected["L"] is None:
        assert setup.L is None
    else:
        np.testing.assert_allclose(setup.L, expected["L"], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(setup.lsp0, expected["lsp0"], atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(setup.rank, np.atleast_1d(expected["rank"]))
    # Compare smooth/parametric penalty operators only after design transport.
    X, source_X = model.predict(type="lpmatrix"), np.asarray(reference["X"])
    T = np.linalg.lstsq(X, source_X, rcond=None)[0]
    np.testing.assert_allclose(X @ T, source_X, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(P @ T, reference["Xp"], atol=2e-10, rtol=2e-10)
    for actual, expected, off in zip(
        _penalty_blocks_seq(model),
        reference["setup"]["S"],
        np.atleast_1d(reference["setup"]["off"]),
        strict=True,
    ):
        target = np.zeros((X.shape[1], X.shape[1]))
        offset = int(model.fit_intercept)
        target[
            actual.coef_slice.start + offset : actual.coef_slice.stop + offset,
            actual.coef_slice.start + offset : actual.coef_slice.stop + offset,
        ] = actual.matrix
        source = np.zeros_like(target)
        start = int(off) - 1
        width = len(expected)
        source[start : start + width, start : start + width] = expected
        np.testing.assert_allclose(T.T @ target @ T, source, atol=2e-9, rtol=2e-9)
