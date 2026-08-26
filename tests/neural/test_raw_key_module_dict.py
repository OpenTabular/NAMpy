"""Raw dataframe names remain valid PyTorch module-map keys."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn

from nampy.models.qnam import QNAMLSS
from nampy.neural.architectures.components.interactions import (
    create_interaction_networks,
)
from nampy.neural.architectures.components.module_dict import RawKeyModuleDict


def _linear_map(names: list[str]) -> RawKeyModuleDict:
    return RawKeyModuleDict({name: nn.Linear(1, 1) for name in names})


def test_raw_key_module_dict_hides_pytorch_registration_constraints():
    names = ["ordinary", "children", "modules", "cost.amount", "größe_日本"]
    modules = _linear_map(names)

    assert list(modules) == names
    assert list(modules.keys()) == names
    assert [name for name, _module in modules.items()] == names
    assert list(modules.values()) == [modules[name] for name in names]
    assert all(name in modules for name in names)

    state_keys = set(modules.state_dict())
    # Directly registerable keys retain historical state-dict paths.
    assert "ordinary.weight" in state_keys
    assert "größe_日本.weight" in state_keys
    # PyTorch-reserved/dotted names use safe internal paths only.
    assert "children.weight" not in state_keys
    assert "modules.weight" not in state_keys
    assert "cost.amount.weight" not in state_keys


def test_raw_key_module_dict_state_and_pickle_round_trips_are_deterministic():
    names = ["children", "modules", "cost.amount", "größe_日本"]
    original = _linear_map(names)
    with torch.no_grad():
        for index, module in enumerate(original.values(), start=1):
            module.weight.fill_(index)
            module.bias.fill_(-index)

    # Internal names depend only on the raw key, not insertion order.
    rebuilt = _linear_map(list(reversed(names)))
    rebuilt.load_state_dict(original.state_dict(), strict=True)
    restored = pickle.loads(pickle.dumps(original))

    inputs = torch.tensor([[2.0]])
    for name in names:
        torch.testing.assert_close(rebuilt[name](inputs), original[name](inputs))
        torch.testing.assert_close(restored[name](inputs), original[name](inputs))
    assert [name for name, _module in restored.items()] == names


def test_interaction_module_map_exposes_raw_interaction_names_with_dots():
    networks = create_interaction_networks(
        ["children", "cost.amount"],
        2,
        lambda interaction: nn.Linear(len(interaction), 1),
    )

    assert list(networks) == ["children:cost.amount"]
    assert isinstance(networks["children:cost.amount"], nn.Linear)
    assert not any("cost.amount" in key for key in networks.state_dict())


def test_qnamlss_reserved_and_dotted_columns_fit_predict_and_reload(tmp_path):
    x = np.linspace(0.05, 0.95, 24)
    data = pd.DataFrame(
        {
            "children": np.resize(np.array([0, 1, 2]), x.size),
            "modules": x,
            "cost.amount": x**2,
            "größe_日本": np.sin(x),
        }
    )
    target = 500.0 + 100.0 * x + 25.0 * data["children"].to_numpy()
    estimator = QNAMLSS(
        layer_sizes=[4],
        dropout=0.0,
        numerical_preprocessing="minmax",
    )
    estimator.fit(
        data,
        target,
        max_epochs=1,
        batch_size=20,
        shuffle=False,
        checkpoint_path=tmp_path / "checkpoints",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        fast_dev_run=True,
    )

    expected = estimator.predict(data, raw=True)
    artifact = estimator.save_model(tmp_path / "qnam-safe-columns.nampy")
    restored = QNAMLSS.load_model(artifact)
    actual = restored.predict(data, raw=True)

    assert expected.shape == (len(data), 3)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert list(restored.model.model.num_feature_networks) == list(data.columns)
