"""Shared design compilation must not depend on any coefficient solver."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.model_selection import GridSearchCV

from nampy import ExpectileGAM, QuantileGAM
from nampy.gam import GAM, compile_design


def test_compiled_design_fit_newdata_penalty_and_edf_match_live_r_snapshot():
    from nampy.gam.extensions import joint_edf

    fixture = json.loads(
        (
            Path(__file__).parents[1] / "reference_fixtures/extensions/neighbors.json"
        ).read_text()
    )
    frame = pd.DataFrame(fixture["inputs"]["design_data"])
    design = compile_design(
        frame, formula="y~s(x,bs='cr',k=6)+offset(off)", smoothing_params=[0.3]
    )
    information = design.matrix.T @ design.matrix
    beta = np.linalg.solve(
        information + design.penalty,
        design.matrix.T @ (design.response - design.training_offset),
    )
    expected = fixture["upstream"]["design_reference"]
    assert_allclose(
        design.matrix @ beta + design.training_offset, expected["fit"], atol=1e-11
    )
    new = design.evaluate(frame.assign(x=frame.x + 0.01, off=frame.off + 3))
    assert_allclose(new.matrix @ beta + new.offset, expected["newfit"], atol=1e-11)
    assert_allclose(beta @ design.penalty @ beta, expected["penalty"], atol=1e-11)
    assert joint_edf(information, design.penalty)[1] == pytest.approx(
        expected["edf"], abs=1e-10
    )


def data():
    x = np.linspace(-1, 1, 60)
    return pd.DataFrame(
        {
            "x": x,
            "z": np.sin(np.arange(60)),
            "y": 0.4 + 0.7 * x + 0.1 * np.cos(np.arange(60)),
            "off": 0.2 * x,
        }
    )


@pytest.mark.parametrize("cls", [ExpectileGAM, QuantileGAM])
def test_loss_estimators_never_fit_a_surrogate_gaussian(monkeypatch, cls):
    def forbidden(*args, **kwargs):
        pytest.fail("Design compilation invoked a Gaussian/model solver.")

    monkeypatch.setattr(GAM, "fit", forbidden)
    frame = data()
    design = compile_design(frame, formula="y~s(x,k=5)+offset(off)")
    assert design.matrix.shape == (60, 5)
    model = cls(formula="y~s(x,k=5)+offset(off)").fit(frame)
    assert not hasattr(model, "_template_")
    assert model.fit_result_.converged
    assert_allclose(model.design_.training_offset, frame.off)


@pytest.mark.parametrize(
    "formula",
    [
        "y~x+z",
        "y~0+x",
        "y~s(x,k=6)",
        "y~s(x,bs='cr',k=6)+z",
        "y~s(x,bs='ps',k=6)+s(z,k=5)",
        "y~te(x,z,k=[4,4])",
        "y~s(x,k=5,sp=.7)",
        "y~s(x,k=5,by=z)+offset(off)",
        "y~s(x,k=5,id='shared')+s(z,k=5,id='shared')",
    ],
)
def test_basis_penalty_offsets_and_predictions_agree_with_gam(formula):
    frame = data()
    design = compile_design(
        frame, formula=formula, smoothing_params=0.3, offset=np.ones(60)
    )
    gam = GAM(formula=formula, smoothing_params=0.3).fit(frame, offset=np.ones(60))
    evaluation = design.evaluate(frame, offset=np.ones(60))
    assert_allclose(design.matrix, evaluation.matrix, atol=1e-10)
    assert_allclose(evaluation.matrix, gam.lpmatrix(frame), atol=1e-10)
    assert_allclose(design.training_offset, gam.offset_train_)
    assert_allclose(evaluation.offset, design.training_offset)
    from nampy.gam.model_state import _coef_full

    beta = np.linalg.solve(
        design.matrix.T @ design.matrix + design.penalty,
        design.matrix.T @ (frame.y.to_numpy() - design.training_offset),
    )
    assert_allclose(design.matrix @ beta, design.matrix @ _coef_full(gam), atol=2e-8)
    shifted = frame.assign(x=frame.x + 0.01, off=frame.off + 3)
    pred = design.evaluate(shifted, offset=np.full(60, 0.2))
    assert_allclose(pred.matrix, gam.lpmatrix(shifted), atol=1e-10)
    assert_allclose(
        pred.matrix @ beta + pred.offset,
        gam.predict(shifted, type="link", offset=np.full(60, 0.2)),
        atol=2e-8,
    )


@pytest.mark.parametrize("cls", [ExpectileGAM, QuantileGAM])
def test_shape_normalization_input_precedence_and_score_validation(cls):
    frame = data()
    model = cls(formula="y~x").fit(frame)
    y = frame.y.to_numpy()
    assert model.score(frame, y) == pytest.approx(model.score(frame, y[:, None]))
    column = cls(formula="~x").fit(frame, y[:, None])
    assert_allclose(model.predict(frame), column.predict(frame))
    with pytest.raises(ValueError, match="either X or data"):
        cls(formula="y~x").fit(frame, data=frame.assign(x=-frame.x))
    with pytest.raises(ValueError):
        model.score(frame, np.column_stack([y, y]))
    with pytest.raises(ValueError):
        model.score(frame, y[:-1])
    with pytest.raises(ValueError):
        model.score(frame, y, sample_weight=np.zeros(len(y)))


def test_response_free_compile_zero_penalties_array_column_order_and_pickle():
    frame = data()[["x", "z"]]
    design = compile_design(frame, k=5, smoothing_params=0)
    assert design.response is None
    assert not design.penalty.any()
    assert_allclose(design.evaluate(frame[["z", "x"]]).matrix, design.matrix)
    rebuilt = pickle.loads(pickle.dumps(design))
    assert_allclose(rebuilt.evaluate(frame).matrix, design.matrix)
    with pytest.raises(ValueError):
        design.evaluate(np.ones((60, 3)))
    for invalid in [-0.1, np.inf, np.nan]:
        with pytest.raises(ValueError):
            compile_design(frame, k=5, smoothing_params=invalid)


def test_design_rejects_transformed_and_multiple_predictors():
    frame = data()
    with pytest.raises(NotImplementedError, match="unconstrained"):
        compile_design(frame, formula="y~s(x,bs='mpi',k=5)")
    with pytest.raises(ValueError, match="one linear predictor"):
        compile_design(frame, formula=["y~x", "~z"])


def test_train_only_offsets_do_not_leak_and_formula_offsets_add():
    frame = data()
    design = compile_design(frame, formula="y~x+offset(off)", offset=np.full(60, 2.0))
    assert_allclose(design.training_offset, 2 + frame.off)
    assert_allclose(design.evaluate(frame).offset, frame.off)
    assert_allclose(design.evaluate(frame, offset=np.ones(60)).offset, frame.off + 1)
    assert_allclose(
        design.evaluate(frame.assign(off=3 * frame.off)).offset, 3 * frame.off
    )


@pytest.mark.parametrize("cls", [ExpectileGAM, QuantileGAM])
def test_loss_model_pickle_and_cross_validation(cls, tmp_path):
    frame = data()
    model = cls(formula="y~s(x,k=5)").fit(frame)
    rebuilt = pickle.loads(pickle.dumps(model))
    assert_allclose(model.predict(frame), rebuilt.predict(frame))
    saved = model.save_model(tmp_path / (cls.__name__ + ".nampy"))
    restored = cls.load_model(saved)
    assert_allclose(model.predict(frame), restored.predict(frame))
    cv = GridSearchCV(
        cls(formula="y~s(x,k=5)"), {"smoothing_params": [[0.1], [1.0]]}, cv=2
    )
    cv.fit(frame, frame.y.to_numpy())
    assert np.isfinite(cv.best_score_)


def test_known_soap_boundary_offsets_use_shared_compiler():
    from tests.smooths.test_mgcv_soap_parity import _case

    case = _case(geometry="known", by=False)
    frame = pd.DataFrame(case["data"])
    formula = f"y~s(x,z,bs='so',k=8,xt={case['xt']!r})"
    design = compile_design(
        frame, formula=formula, knots=case["knots"], smoothing_params=[0.37]
    )
    gam = GAM(formula=formula, knots=case["knots"], smoothing_params=[0.37]).fit(frame)
    new = pd.DataFrame(case["newdata"])
    from nampy.gam.model_state import _coef_full

    evaluation = design.evaluate(new)
    assert_allclose(
        evaluation.matrix @ _coef_full(gam) + evaluation.offset,
        gam.predict(new, type="link"),
        atol=1e-9,
    )
