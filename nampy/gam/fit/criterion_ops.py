"""Criterion wrappers forwarding to smoothing-selection subsystem."""

from __future__ import annotations


def criterion_value(model, y, log_sp, method="gcv"):
    from ..selection import criterion_value as _criterion_value

    return _criterion_value(model, y, log_sp, method=method)


def criterion_gradient(model, y, log_sp, method="gcv"):
    from ..selection import criterion_gradient as _criterion_gradient

    return _criterion_gradient(model, y, log_sp, method=method)


def criterion_hessian(model, y, log_sp, method="gcv"):
    from ..selection import criterion_hessian as _criterion_hessian

    return _criterion_hessian(model, y, log_sp, method=method)
