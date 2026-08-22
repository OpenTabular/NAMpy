"""Transient numerical workspace owned by one GAM fit session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_UNSET = object()


@dataclass
class FitWorkspace:
    """Warm starts and evaluation caches that must never cross fit sessions."""

    pirls_coef_start: Any = _UNSET
    pirls_eta_start: Any = _UNSET
    pirls_mu_start: Any = _UNSET
    pirls_eval_start: Any = _UNSET
    pirls_eval_eta_start: Any = _UNSET
    pirls_eval_mu_start: Any = _UNSET
    pirls_lock_start: Any = _UNSET
    pirls_last_coef: Any = _UNSET
    pirls_last_eta: Any = _UNSET
    pirls_last_mu: Any = _UNSET
    pirls_last_inner_trace: Any = _UNSET
    pirls_reml_gamma_state: Any = _UNSET
    pirls_reml_gaussian_state: Any = _UNSET
    pirls_reml_negbin_state: Any = _UNSET
    pirls_reml_derivative_kernel_state: Any = _UNSET
    pirls_disable_theta_efs: Any = _UNSET
    general_family_outer_eval_cache: Any = _UNSET
    penalty_subspace_cache: Any = _UNSET
    shape_gcv_ubre_state: Any = _UNSET
    transformed_gcv_ubre_state: Any = _UNSET

    def get(self, name: str, default: Any = None) -> Any:
        value = getattr(self, name)
        return default if value is _UNSET else value


def _fit_workspace(obj: Any) -> FitWorkspace:
    workspace = getattr(obj, "_ws", None)
    if workspace is None:
        workspace = FitWorkspace()
        obj._ws = workspace
    return workspace


__all__ = ["FitWorkspace", "_fit_workspace"]
