"""GAMResidual estimators: frozen mgcv-parity baseline + neural correction.

The composite is NOT an mgcv model and is never parity-compared; these tests
pin the composition contract instead: the GAM stage must equal a standalone
GAM fit exactly, and predictions must compose on the link scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.hybrid import GAMResidualClassifier, GAMResidualRegressor
from nampy.models.linreg import LinRegClassifier, LinRegRegressor

_NEURAL_FIT_KWARGS = {
    "max_epochs": 40,
    "patience": 40,
    "lr": 5e-2,
    "batch_size": 64,
    "logger": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}


def _gaussian_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {"x0": rng.uniform(size=n), "x3": rng.normal(size=n)}
    )
    data["y"] = (
        np.sin(3.0 * data["x0"])
        + 2.0 * data["x3"]
        + rng.normal(scale=0.1, size=n)
    )
    return data


def _regressor():
    return LinRegRegressor(numerical_preprocessing="standardization")


def _fit_gaussian(data, tmp_path):
    hybrid = GAMResidualRegressor("y ~ s(x0, k=6)", _regressor())
    kwargs = dict(_NEURAL_FIT_KWARGS, checkpoint_path=str(tmp_path))
    hybrid.fit(data, neural_features=["x3"], neural_fit_kwargs=kwargs)
    return hybrid


def test_gam_stage_is_identical_to_standalone_fit(tmp_path):
    data = _gaussian_data()
    hybrid = _fit_gaussian(data, tmp_path)

    alone = GAM(
        formula="y ~ s(x0, k=6)",
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="reml",
    )
    alone.fit(data=data)

    np.testing.assert_array_equal(
        hybrid.gam_.fit_result().coef_full, alone.fit_result().coef_full
    )
    np.testing.assert_array_equal(
        hybrid.gam_.predict(data, type="link"), alone.predict(data, type="link")
    )


def test_gaussian_composite_beats_gam_alone(tmp_path):
    data = _gaussian_data()
    hybrid = _fit_gaussian(data, tmp_path)

    from sklearn.metrics import r2_score

    alone = GAM(
        formula="y ~ s(x0, k=6)",
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="reml",
    )
    alone.fit(data=data)

    r2_hybrid = hybrid.score(data, data["y"])
    r2_alone = r2_score(data["y"], alone.predict(data))
    # The x3 signal is invisible to the GAM formula; the correction must
    # recover most of it.
    assert r2_hybrid > r2_alone + 0.5
    assert r2_hybrid > 0.9


def test_link_scale_composition_is_exact(tmp_path):
    data = _gaussian_data()
    hybrid = _fit_gaussian(data, tmp_path)

    eta_gam = np.asarray(hybrid.gam_.predict(data, type="link"))
    eta_nn = (
        hybrid.neural_._predict(data[["x3"]])["output"]
        .squeeze(-1)
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(
        hybrid.predict_link(data), eta_gam + eta_nn, atol=1e-10
    )
    # Gaussian identity link: response == link.
    np.testing.assert_allclose(
        hybrid.predict(data), hybrid.predict_link(data), atol=1e-10
    )


def test_binomial_composition_matches_sigmoid(tmp_path):
    rng = np.random.default_rng(1)
    n = 200
    data = pd.DataFrame({"x0": rng.uniform(size=n), "x3": rng.normal(size=n)})
    eta = np.sin(3.0 * data["x0"]) + 1.5 * data["x3"]
    data["y"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

    hybrid = GAMResidualClassifier(
        "y ~ s(x0, k=6)",
        LinRegClassifier(numerical_preprocessing="standardization"),
    )
    kwargs = dict(_NEURAL_FIT_KWARGS, checkpoint_path=str(tmp_path))
    hybrid.fit(data, neural_features=["x3"], neural_fit_kwargs=kwargs)

    eta_composite = hybrid.predict_link(data)
    proba = hybrid.predict_proba(data)
    np.testing.assert_allclose(
        proba[:, 1], 1.0 / (1.0 + np.exp(-eta_composite)), atol=1e-10
    )
    assert proba.shape == (n, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    assert list(hybrid.classes_) == [0, 1]
    assert hybrid.score(data, data["y"]) > 0.5


def test_predict_components_merges_backends(tmp_path):
    data = _gaussian_data()
    hybrid = _fit_gaussian(data, tmp_path)

    components = hybrid.predict_components(data)
    assert components.backend == "hybrid"
    assert "gam:s(x0, k=6)" in components.terms
    assert "nn:x3" in components.terms
    np.testing.assert_allclose(
        components.link, hybrid.predict_link(data), atol=1e-10
    )


def test_persistence_round_trip(tmp_path):
    data = _gaussian_data()
    hybrid = _fit_gaussian(data, tmp_path)
    expected = hybrid.predict(data)

    path = hybrid.save_model(tmp_path / "hybrid.nampy")
    restored = GAMResidualRegressor.load_model(path)
    np.testing.assert_allclose(restored.predict(data), expected, atol=1e-10)


def test_configuration_guards_raise_at_fit(tmp_path):
    data = _gaussian_data(n=60)
    with pytest.raises(ValueError, match="offset"):
        GAMResidualRegressor("y ~ s(x0) + offset(o)", _regressor()).fit(
            data, neural_features=["x3"]
        )
    with pytest.raises(ValueError, match="supports families"):
        GAMResidualRegressor("y ~ s(x0)", _regressor(), family="gamma").fit(
            data, neural_features=["x3"]
        )
    with pytest.raises(TypeError, match="requires an unfitted"):
        GAMResidualRegressor(
            "y ~ s(x0)",
            LinRegClassifier(numerical_preprocessing="standardization"),
            family="gaussian",
        ).fit(data, neural_features=["x3"])
    with pytest.raises(ValueError, match="supports families"):
        GAMResidualClassifier(
            "y ~ s(x0)",
            LinRegClassifier(numerical_preprocessing="standardization"),
            family="gaussian",
        ).fit(data, neural_features=["x3"])


def test_clone_does_not_mutate_template(tmp_path):
    data = _gaussian_data(n=80)
    template = _regressor()
    hybrid = GAMResidualRegressor("y ~ s(x0, k=5)", template)
    kwargs = dict(_NEURAL_FIT_KWARGS, max_epochs=2, checkpoint_path=str(tmp_path))
    hybrid.fit(data, neural_features=["x3"], neural_fit_kwargs=kwargs)

    assert template.model is None
    assert hybrid.neural_ is not template
    assert hybrid.neural_.model is not None


def test_fit_guards(tmp_path):
    data = _gaussian_data(n=60)
    hybrid = GAMResidualRegressor("y ~ s(x0, k=5)", _regressor())
    with pytest.raises(ValueError, match="at least one column"):
        hybrid.fit(data, neural_features=[])
    with pytest.raises(ValueError, match="not found in data"):
        hybrid.fit(data, neural_features=["missing"])
    with pytest.raises(ValueError, match="not fitted"):
        hybrid.predict(data)
