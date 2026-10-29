"""Explicit offsets in raw predictor coordinates, independent of family parameters."""

import numpy as np
import torch


def predictor_offset(offset, n_rows, n_predictors):
    """Validate exact predictor width; vectors are valid only for one predictor."""
    if offset is None:
        return np.zeros((n_rows, n_predictors))
    if torch.is_tensor(offset):
        offset = offset.detach().cpu().numpy()
    value = np.asarray(offset, dtype=float)
    if value.ndim == 1 and n_predictors == 1:
        value = value[:, None]
    if value.shape != (n_rows, n_predictors) or not np.isfinite(value).all():
        raise ValueError(
            f"offset must be finite with shape ({n_rows}, {n_predictors}); no predictor broadcasting is supported."
        )
    return value


def add_predictor_offset(predictions, offset):
    if offset is None:
        return predictions
    value = predictor_offset(offset, len(predictions), predictions.shape[1])
    return predictions + torch.as_tensor(
        value, device=predictions.device, dtype=predictions.dtype
    )
