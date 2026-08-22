"""Compare the local NODE-GAM block with the vendored upstream contract."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstreams" / "nodegam"))
sys.path.insert(0, str(ROOT))

# The vendored reference imports its optional preprocessing dependency from
# package __init__; the architecture comparison does not use that dependency.
category_encoders = types.ModuleType("category_encoders")
category_encoders.LeaveOneOutEncoder = object
sys.modules.setdefault("category_encoders", category_encoders)

from nodegam.arch import GAMBlock as ReferenceGAMBlock  # noqa: E402

from nampy.neural.architectures.components.additive_trees import (  # noqa: E402
    GAMBlock,
)


def check_block(block, x):
    block.eval()
    with torch.no_grad():
        prediction = block(x)
        terms = block.run_with_additive_terms(x)
    assert terms.ndim == 3
    assert terms.shape[0] == x.shape[0]
    assert terms.shape[2] == block.num_classes
    additive = terms.sum(dim=1) + block.bias
    if prediction.ndim == additive.ndim - 1:
        additive = additive.squeeze(-1)
    torch.testing.assert_close(prediction, additive)
    return prediction.shape, terms.shape


def main() -> None:
    torch.manual_seed(123)
    kwargs = {
        "in_features": 2,
        "num_trees": 4,
        "num_layers": 1,
        "num_classes": 1,
        "depth": 2,
        "add_last_linear": True,
    }
    x = torch.tensor([[0.1, 0.7], [0.8, 0.2]])
    reference_shapes = check_block(ReferenceGAMBlock(**kwargs), x)
    local_shapes = check_block(GAMBlock(**kwargs), x)
    assert local_shapes[1] == reference_shapes[1], (local_shapes, reference_shapes)
    print(
        "NODE-GAM reference term contract matched: "
        f"reference={reference_shapes}, local={local_shapes}"
    )


if __name__ == "__main__":
    main()
