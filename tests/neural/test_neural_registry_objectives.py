from __future__ import annotations

import numpy as np
import pytest
import torch

from nampy.models._registered import estimator_family, registered_estimator_class
from nampy.neural.distributions.registry import FAMILY_REGISTRY, resolve_family
from nampy.neural.registry import architectures, get_architecture


def test_builtin_architectures_are_registered_once_with_explicit_capabilities():
    expected = {
        "linreg",
        "nam",
        "sian",
        "snam",
        "gpnam",
        "igann",
        "nbm",
        "nbm_spam",
        "natt",
        "namformer",
        "treenam",
        "ensemble_treenam",
        "nodegam",
        "qnam",
        "spline_nam",
        "spam",
    }
    assert set(architectures()) == expected
    assert get_architecture("nam").supports("distributional")
    assert get_architecture("nam").supports("interactions")
    assert get_architecture("nam").preprocessor_defaults == {
        "numerical_method": "none",
        "categorical_method": "one-hot",
        "scaling": "minmax",
        "dtype": np.float32,
    }
    assert get_architecture("sian").supports("interaction_selection")
    assert get_architecture("sian").estimator_mixin is not None
    assert not get_architecture("linreg").supports("interactions")
    assert get_architecture("gpnam").supports("interactions")
    assert get_architecture("gpnam").supports("fixed_linear_design")
    assert get_architecture("gpnam").preprocessor_defaults == {
        "numerical_method": "none",
        "categorical_method": "one-hot",
        "scaling": None,
    }
    for name in ("nbm", "nbm_spam", "spam"):
        assert get_architecture(name).preprocessor_defaults == {
            "numerical_method": "none",
            "categorical_method": "one-hot",
            "scaling": "minmax",
            "dtype": np.float32,
        }
    assert get_architecture("nodegam").supports("masked_pretraining")
    assert get_architecture("igann").capabilities == {
        "regression",
        "classification",
        "distributional",
        "additive_components",
        "native_training",
    }
    assert get_architecture("igann").preprocessor_defaults == {
        "numerical_method": "none",
        "categorical_method": "int",
        "scaling": None,
    }
    assert get_architecture("igann").objective_defaults == {
        "distributional": {"n_estimators": 100}
    }
    assert get_architecture("qnam").capabilities == {
        "distributional",
        "additive_components",
        "interactions",
    }
    assert get_architecture("spam").supports("local_term_importance")
    assert get_architecture("spline_nam").capabilities == {
        "regression",
        "additive_components",
        "interactions",
    }
    assert get_architecture("spline_nam").preprocessor_defaults == {
        "numerical_method": "minmax",
        "categorical_method": "int",
        "scaling": None,
    }
    assert ".architectures." in get_architecture("nam").module_path


def test_registry_prevents_unsupported_estimator_surfaces():
    with pytest.raises(TypeError, match="does not support"):
        registered_estimator_class(
            "SplineNAMClassifier",
            architecture="spline_nam",
            objective="classification",
            module_name=__name__,
        )


def test_estimator_family_generates_only_declared_public_surfaces():
    nam = estimator_family("nam", module_name=__name__)
    assert nam.regressor.__name__ == "NAMRegressor"
    assert nam.classifier.__name__ == "NAMClassifier"
    assert nam.lss.__name__ == "NAMLSS"

    qnam = estimator_family("qnam", module_name=__name__)
    assert qnam.regressor is None
    assert qnam.classifier is None
    assert qnam.lss.__name__ == "QNAMLSS"


_POSITIVE = np.array([0.2, 0.7, 1.3, 2.0], dtype=np.float32)
_COUNT = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
_REAL = np.array([-1.0, 0.0, 0.5, 2.0], dtype=np.float32)
_FAMILY_CASES = {
    "normal": (_REAL, {}),
    "poisson": (_COUNT, {}),
    "gamma": (_POSITIVE, {}),
    "beta": (np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32), {}),
    "dirichlet": (
        np.array(
            [
                [0.2, 0.3, 0.5],
                [0.1, 0.7, 0.2],
                [0.4, 0.4, 0.2],
                [0.6, 0.1, 0.3],
            ],
            dtype=np.float32,
        ),
        {},
    ),
    "studentt": (_REAL, {}),
    "negativebinom": (_COUNT, {}),
    "inversegamma": (_POSITIVE, {}),
    "categorical": (np.array([0, 1, 2, 1]), {}),
    "quantile": (_REAL, {}),
    "robustnormal": (_REAL, {}),
    "lognormal": (_POSITIVE, {}),
    "weibull": (_POSITIVE, {}),
    "loglogistic": (_POSITIVE, {}),
    "zip": (_COUNT, {}),
    "zinb": (_COUNT, {}),
    "hurdlepoisson": (_COUNT, {}),
    "hurdlenegativebinom": (_COUNT, {}),
    "tweedie": (_COUNT, {"series_max_terms": 20}),
    "ordinal": (np.array([0, 1, 2, 1]), {}),
    "mvnormdiag": (
        np.array(
            [[-1.0, 0.2], [0.0, 0.5], [1.0, -0.3], [0.4, 0.8]],
            dtype=np.float32,
        ),
        {},
    ),
}


@pytest.mark.parametrize("family_name", sorted(FAMILY_REGISTRY))
def test_every_distribution_exposes_unreduced_per_row_loss(family_name):
    target, kwargs = _FAMILY_CASES[family_name]
    family, inferred = resolve_family(family_name).instantiate(target, kwargs)
    predictions = torch.zeros(
        (len(target), family.param_count), dtype=torch.float32, requires_grad=True
    )
    target_tensor = torch.as_tensor(target)

    values = family.compute_loss(predictions, target_tensor, reduction="none")
    reduced = family.compute_loss(predictions, target_tensor)

    assert values.shape == (len(target),)
    assert torch.isfinite(values).all()
    torch.testing.assert_close(reduced, values.mean())
    reduced.backward()
    assert predictions.grad is not None
    if family_name in {"dirichlet", "mvnormdiag"}:
        assert inferred["n_dim"] == target.shape[1]
    if family_name in {"categorical", "ordinal"}:
        assert inferred["num_classes"] == 3
