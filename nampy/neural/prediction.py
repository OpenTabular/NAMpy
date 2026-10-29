"""Neural prediction batching and architecture-specific feature extraction.

Inputs have already passed the public estimator's fitted feature-schema check.
This module owns Torch execution and never imports estimator facades.
"""

from typing import Any

import torch


def _prediction_batches(preprocessing, X, batch_size, device):
    from numbers import Integral

    size = 1024 if batch_size is None else batch_size
    if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not len(X):
        raise ValueError("Prediction requires at least one row.")
    for start in range(0, len(X), size):
        rows = (
            X.iloc[start : start + size]
            if hasattr(X, "iloc")
            else X[start : start + size]
        )
        cat, num = preprocessing.preprocess_tensors(rows)
        if not cat and not num:
            raise ValueError("Prediction requires at least one transformed feature.")
        yield (
            {key: value.to(device) for key, value in cat.items()},
            {key: value.to(device) for key, value in num.items()},
            len(rows),
        )


def predict_raw(model, preprocessing, X, *, batch_size=None):
    """Batch preprocessing and execution; accumulate detached results on CPU."""
    from .architectures.components.regularization import evaluating
    from .contracts import is_penalty_key

    collected: dict[str, list[torch.Tensor]] = {}
    constants = {}
    device = next(model.parameters()).device
    with evaluating([model]), torch.no_grad():
        for cat, num, n_rows in _prediction_batches(
            preprocessing, X, batch_size, device
        ):
            for key, value in model(num_features=num, cat_features=cat).items():
                value = value.detach().cpu()
                if (
                    key != "intercept"
                    and not is_penalty_key(key)
                    and value.ndim > 0
                    and value.shape[0] == n_rows
                ):
                    collected.setdefault(key, []).append(value)
                elif key not in constants:
                    constants[key] = value
    return {
        **{key: torch.cat(values, dim=0) for key, values in collected.items()},
        **constants,
    }


def basis_transform(architecture, preprocessing, X, *, batch_size=None):
    """Evaluate a fixed design using bounded preprocessing batches."""
    from .architectures.components.regularization import evaluating

    transform = getattr(architecture, "linear_design", None)
    if transform is None:
        raise NotImplementedError(
            f"{type(architecture).__name__} does not expose a fixed basis."
        )
    device = next(architecture.parameters()).device
    with evaluating([architecture]), torch.no_grad():
        blocks = [
            transform(num, cat).detach().cpu()
            for cat, num, _ in _prediction_batches(preprocessing, X, batch_size, device)
        ]
    return torch.cat(blocks).numpy()


def local_term_importance(
    architecture, preprocessing, X, *, target=0, top_k=10, batch_size=None
):
    """Evaluate an architecture-owned local expansion in bounded batches."""
    from .architectures.components.regularization import evaluating

    importance = getattr(architecture, "local_term_importance", None)
    if importance is None:
        raise NotImplementedError(
            f"{type(architecture).__name__} has no local term expansion."
        )
    rows = []
    device = next(architecture.parameters()).device
    with evaluating([architecture]), torch.no_grad():
        for cat, num, _ in _prediction_batches(preprocessing, X, batch_size, device):
            rows.extend(importance(num, cat, target=target, top_k=top_k))
    return rows


def split_output_components(pred_dict):
    """Split architecture output tensors into NumPy terms and intercept."""
    from .contracts import is_penalty_key

    terms: dict[str, Any] = {}
    intercept: float | Any = 0.0
    for key, value in pred_dict.items():
        if key == "output" or is_penalty_key(key):
            continue
        array = value.detach().cpu().numpy()
        if key == "intercept":
            intercept = float(array.item()) if array.size == 1 else array
        else:
            terms[key] = array
    return terms, intercept
