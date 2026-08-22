"""Generic tensor helpers shared by torch components."""

import numpy as np
import torch


def check_numpy(x):
    """Makes sure x is a numpy array. If not, make it as one."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


def process_in_chunks(function, *args, batch_size, out=None, **kwargs):
    """Computes output by applying batch-parallel function to large data tensor in chunks.

    Args:
        function: a function(*[x[indices, ...] for x in args]) -> out[indices, ...].
        args: one or many tensors, each [num_instances, ...].
        batch_size: maximum chunk size processed in one go.
        out: memory buffer for out, defaults to torch.zeros of appropriate size and type.

    Returns:
        out: the outputs of function(data), computed in a memory-efficient (mini-batch) way.
    """
    total_size = args[0].shape[0]
    first_output = function(*[x[0:batch_size] for x in args])
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(
            *output_shape,
            dtype=first_output.dtype,
            device=first_output.device,
            layout=first_output.layout,
            **kwargs,
        )

    out[0:batch_size] = first_output
    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = function(*[x[batch_ix] for x in args])
    return out
