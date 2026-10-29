"""Explicit offsets in raw predictor coordinates, independent of family parameters."""

import torch

from .._offsets import predictor_offset as validate_predictor_offset


def predictor_offset(offset, n_rows, n_predictors):
    """Validate exact predictor width; vectors are valid only for one predictor."""
    if torch.is_tensor(offset):
        offset = offset.detach().cpu().numpy()
    return validate_predictor_offset(offset, n_rows, n_predictors)


def add_predictor_offset(predictions, offset):
    if offset is None:
        return predictions
    value = predictor_offset(offset, len(predictions), predictions.shape[1])
    return predictions + torch.as_tensor(
        value, device=predictions.device, dtype=predictions.dtype
    )
