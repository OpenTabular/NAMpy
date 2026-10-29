"""Public Fisher contracts agree across numerical backends."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from nampy import (
    GAMLSS,
    FisherDiagnostics,
    GAMClassifier,
    GAMRegressor,
    GradientTrainingConfig,
    LinRegRegressor,
    PreprocessingConfig,
)


def frame():
    x = np.linspace(-1, 1, 36).round(3)
    return pd.DataFrame(
        {"x": x, "z": np.cos(np.arange(len(x))).round(3), "y": 0.3 + 0.4 * x}
    )


def test_same_known_scale_gaussian_information_in_both_backends():
    data = frame()
    config = FisherDiagnostics(regression_scale=2, group_by="tensor")
    statistical = GAMRegressor(formula="y~x+z", scale=4, optimize_smoothing=False).fit(
        data
    )
    neural = LinRegRegressor(
        preprocessing=PreprocessingConfig(numerical_preprocessing="none"),
        training=GradientTrainingConfig(
            max_epochs=1,
            trainer_kwargs={
                "logger": False,
                "enable_model_summary": False,
                "enable_progress_bar": False,
            },
        ),
    ).fit(data[["x", "z"]], data.y)
    gam = statistical.fisher_information(data, config=config)
    nn = neural.fisher_information(data[["x", "z"]], config=config, batch_size=5)
    expected = np.r_[1.0, (data[["x", "z"]].to_numpy() ** 2).mean(axis=0)] / 4
    assert_allclose(np.sort(gam.diagonal), np.sort(expected), atol=1e-12)
    # LinReg also has redundant per-feature biases. Compare matching slope
    # and global-intercept coordinates, not whole parameterizations.
    matching = [
        start
        for name, (start, stop) in zip(
            nn.parameter_names, nn.parameter_slices, strict=True
        )
        if name == "model.intercept" or name.endswith(".weight")
    ]
    assert_allclose(np.sort(nn.diagonal[matching]), np.sort(expected), rtol=2e-7)
    assert gam.normalization == nn.normalization == "mean"
    assert gam.n_inputs == nn.n_inputs == len(data)


def test_statistical_constructor_collection_clone_offsets_and_failure_isolation(
    tmp_path,
):
    data = frame()
    y = (np.arange(len(data)) % 3 == 0).astype(int)
    config = FisherDiagnostics(n_inputs=5, random_state=17)
    model = GAMClassifier(
        formula="y~x+z", optimize_smoothing=False, fisher_diagnostics=config
    )
    assert clone(model).fisher_diagnostics == config
    data["y"] = y
    offsets = np.linspace(-0.3, 0.3, len(data))
    model.fit(data, y, offset=offsets)
    assert model.fisher_information_.n_inputs == 5
    assert model.fisher_history_[-1].stage == "final"
    with pytest.raises(ValueError, match="offset"):
        model.fisher_information(data)
    result = model.fisher_information(data, offset=offsets)
    order = np.arange(len(data))[::-1]
    assert_allclose(
        result.diagonal,
        model.fisher_information(data.iloc[order], offset=offsets[order]).diagonal,
    )
    assert not np.allclose(
        result.diagonal, model.fisher_information(data, offset=offsets + 3).diagonal
    )
    restored = GAMClassifier.load_model(model.save_model(tmp_path / "fisher.nampy"))
    assert_allclose(
        restored.fisher_information_.diagonal, model.fisher_information_.diagonal
    )
    previous = model.gam_
    model.fisher_diagnostics = replace(
        config, estimator="negative_hessian", every_n_epochs=1
    )
    with pytest.raises((ValueError, NotImplementedError), match="epoch|Periodic"):
        model.fit(data, y, offset=offsets)
    assert model.gam_ is previous


def test_general_normal_information_and_sampled_estimators_use_public_family_coordinates():
    data = frame()
    data["y"] += 0.3 * np.sin(np.arange(len(data)))
    model = GAMLSS(
        family="normal",
        formula={"mu": "y~x", "sigma": "~1"},
        optimize_smoothing=False,
        fisher_diagnostics=FisherDiagnostics(n_inputs=5),
    ).fit(data)
    assert model.fisher_information_.n_parameters == len(
        model.gam_.fit_result().coef_full
    )
    assert model.fisher_information_.backend == "gam"
    expected = model.fisher_information(data)
    for estimator in ("score_outer_product", "negative_hessian"):
        config = FisherDiagnostics(
            estimator=estimator, n_draws=300, random_state=5, input_sampling="iid"
        )
        result = model.fisher_information(data, config=config)
        assert_allclose(result.diagonal, expected.diagonal, rtol=0.12)
        assert np.isfinite(result.total_standard_error).all()
