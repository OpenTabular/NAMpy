import numpy as np
import pytest
from nampy.gam.design.constructors import construct_terms
from nampy.gam.design.structures import PenaltySpec


class _WrapperConstraintRuntime:
    label = "fake_constraint_term"
    term_id = "fake_term_id"
    basis_name = "fake"
    term_type = "smooth"
    by = None
    smoothing_id = None
    metadata = {}
    by_done = True
    constraints_absorbed = False
    constraint_kind = None
    prediction_offset = None
    _by_state = None

    def __init__(self, *, predict_coefficient_map=None):
        self.fit_constraint_matrix = np.array([[1.0, 0.0, 0.0]])
        self.predict_coefficient_map = predict_coefficient_map

    def fit(self, X, feature_names):
        self.basis_train = np.asarray(X, dtype=np.float64)
        return self

    def get_penalty_definitions(self):
        return [PenaltySpec(matrix=np.eye(3), kind="smooth")]

    def transform_new(self, X_new):
        return np.asarray(X_new, dtype=np.float64)


def test_construct_terms_uses_explicit_predict_coefficient_map():
    X = np.eye(3, dtype=np.float64)
    predict_map = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    runtime = _WrapperConstraintRuntime(predict_coefficient_map=predict_map)

    term = construct_terms(runtime, X=X, feature_names=["x0", "x1", "x2"])[0]

    assert term.fit_constraint_operator is not None
    assert term.fit_coefficient_map is not None
    assert term.predict_coefficient_map is not None
    assert np.allclose(term.predict_coefficient_map, predict_map)

    pred = term.predict_matrix(X)
    assert pred.shape == term.train_design_matrix.shape
    assert np.allclose(pred, predict_map)
    assert not np.allclose(pred, term.train_design_matrix)


def test_construct_terms_validates_predict_coefficient_map_shape():
    X = np.eye(3, dtype=np.float64)
    runtime = _WrapperConstraintRuntime(
        predict_coefficient_map=np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
    )
    with pytest.raises(ValueError, match="Predict coefficient map"):
        construct_terms(runtime, X=X, feature_names=["x0", "x1", "x2"])
