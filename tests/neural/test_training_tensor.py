"""Training conversion must own writable storage when NumPy cannot share safely."""

import warnings

import numpy as np
import pytest
import torch

from nampy.neural.training.common import _training_tensor


@pytest.mark.parametrize("shape", [(8,), (8, 2)])
def test_readonly_training_array_has_safe_storage(shape):
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    source.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _training_tensor(source, device="cpu")
    result.add_(1)
    np.testing.assert_array_equal(source, np.arange(np.prod(shape)).reshape(shape))
    assert result.dtype == torch.float32


def test_existing_tensor_and_optional_input_are_preserved():
    source = torch.ones(3, dtype=torch.float64)
    assert _training_tensor(source) is source
    assert _training_tensor(None) is None
