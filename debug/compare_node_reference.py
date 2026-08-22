"""Compare the shared NODE tree primitive with Qwicen's reference implementation."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstreams" / "node"))

category_encoders = types.ModuleType("category_encoders")
category_encoders.LeaveOneOutEncoder = object
sys.modules.setdefault("category_encoders", category_encoders)
tensorboard_x = types.ModuleType("tensorboardX")
tensorboard_x.SummaryWriter = object
sys.modules.setdefault("tensorboardX", tensorboard_x)

from lib.arch import DenseBlock as ReferenceDenseBlock  # noqa: E402
from lib.nn_utils import sparsemax, sparsemoid  # noqa: E402
from lib.odst import ODST as ReferenceODST  # noqa: E402

from nampy.neural.architectures.components.oblivious_trees import (  # noqa: E402
    ODST,
    ODSTBlock,
)
from nampy.neural.architectures.components.sparse_activations import (  # noqa: E402
    sparsemax as local_sparsemax,
)
from nampy.neural.architectures.components.sparse_activations import (
    sparsemoid as local_sparsemoid,
)


def _ready(module):
    with torch.no_grad():
        module.feature_thresholds.copy_(
            torch.tensor([[0.15, -0.25], [0.45, 0.35], [-0.2, 0.6]])
        )
        module.log_temperatures.fill_(-0.4)
        module._is_initialized_tensor.fill_(1)
    module._is_initialized_bool = True
    return module.eval()


def main() -> None:
    kwargs = {"in_features": 4, "num_trees": 3, "depth": 2, "tree_dim": 2}
    torch.manual_seed(19)
    reference = ReferenceODST(
        **kwargs, choice_function=sparsemax, bin_function=sparsemoid
    )
    torch.manual_seed(19)
    local = ODST(
        **kwargs, choice_function=local_sparsemax, bin_function=local_sparsemoid
    )
    local.load_state_dict(reference.state_dict())
    _ready(reference)
    _ready(local)

    x_reference = torch.tensor(
        [[-0.3, 0.2, 0.7, 1.1], [0.4, -0.8, 0.1, 0.5]], requires_grad=True
    )
    x_local = x_reference.detach().clone().requires_grad_(True)
    reference_output = reference(x_reference)
    local_output = local(x_local)
    torch.testing.assert_close(local_output, reference_output)

    reference_output.square().sum().backward()
    local_output.square().sum().backward()
    torch.testing.assert_close(x_local.grad, x_reference.grad)

    x3_reference = x_reference.detach().view(1, 2, 4)
    x3_local = x_local.detach().view(1, 2, 4)
    torch.testing.assert_close(local(x3_local), reference(x3_reference))

    dense_kwargs = {
        "input_dim": 4,
        "layer_dim": 3,
        "num_layers": 2,
        "tree_dim": 2,
        "max_features": 8,
        "input_dropout": 0.25,
        "flatten_output": False,
        "depth": 2,
        "choice_function": sparsemax,
        "bin_function": sparsemoid,
    }
    torch.manual_seed(23)
    reference_dense = ReferenceDenseBlock(**dense_kwargs)
    torch.manual_seed(23)
    local_dense = ODSTBlock(
        in_features=4,
        num_trees=3,
        num_layers=2,
        num_classes=2,
        max_features=8,
        input_dropout=0.25,
        flatten_output=False,
        add_last_linear=False,
        init_bias=False,
        depth=2,
        choice_function=local_sparsemax,
        bin_function=local_sparsemoid,
    )
    for reference_layer, local_layer in zip(
        reference_dense, local_dense, strict=True
    ):
        local_layer.load_state_dict(reference_layer.state_dict())
        _ready(reference_layer)
        _ready(local_layer)
    reference_dense.eval()
    local_dense.eval()
    dense_reference_output = reference_dense(x_reference.detach())
    dense_local_output = local_dense.run_with_layers(x_local.detach())
    torch.testing.assert_close(dense_local_output, dense_reference_output)
    reference_dense.train()
    local_dense.train()
    torch.manual_seed(31)
    dense_reference_dropout = reference_dense(x_reference.detach())
    torch.manual_seed(31)
    dense_local_dropout = local_dense.run_with_layers(x_local.detach())
    torch.testing.assert_close(dense_local_dropout, dense_reference_dropout)

    torch.manual_seed(19)
    reference_default = _ready(ReferenceODST(**kwargs))
    torch.manual_seed(19)
    local_default = _ready(ODST(**kwargs))
    local_default.load_state_dict(reference_default.state_dict())
    default_difference = (
        local_default(x_local.detach()) - reference_default(x_reference.detach())
    ).abs().max()
    print(
        "NODE ODST reference matched: "
        f"output_shape={tuple(local_output.shape)}, "
        f"max_abs={float((local_output - reference_output).abs().max().detach()):.3g}; "
        "default activation max_abs="
        f"{float(default_difference.detach()):.3g}; "
        "dense_block_max_abs="
        f"{float((dense_local_output - dense_reference_output).abs().max().detach()):.3g}"
    )


if __name__ == "__main__":
    main()
