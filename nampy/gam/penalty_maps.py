"""Canonical log-SP rows for compiled penalties (mgcv ``paraPen$L``)."""

import numpy as np


def penalty_log_sp_row(block, n_sp):
    """Return one full-map row; ordinary penalties retain their one-hot row."""
    row = np.zeros(int(n_sp), dtype=np.float64)
    mapping = block.metadata.get("log_sp_map")
    if mapping is None:
        row[int(block.smoothing_index)] = 1.0
    else:
        for index, value in mapping.items():
            row[int(index)] = float(value)
    return row


def penalty_log_sp_map(blocks, n_sp):
    return np.asarray(
        [penalty_log_sp_row(block, n_sp) for block in blocks], dtype=np.float64
    ).reshape(len(blocks), int(n_sp))


def has_mapped_penalties(blocks):
    return any("log_sp_map" in block.metadata for block in blocks)


def remap_penalty_log_sp(block, indices):
    mapping = block.metadata.get("log_sp_map")
    if mapping is not None:
        block.metadata = dict(
            block.metadata,
            log_sp_map={indices[int(index)]: value for index, value in mapping.items()},
        )
