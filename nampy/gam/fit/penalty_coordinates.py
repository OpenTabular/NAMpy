"""Source full-penalty coordinates for analytic ``paraPen$L`` calculations.

mgcv evaluates its fitting/derivative kernels in log full-SP coordinates and
contracts the results with L in gam.outer. This view performs that exact lift;
it does not change coefficient coordinates or construct additional penalties.
"""

from contextlib import contextmanager
from copy import copy
from dataclasses import fields, replace

import numpy as np

from ..model_state import _compiled_model, _fit_workspace, _penalty_blocks_seq
from ..penalty_maps import has_mapped_penalties


class _PenaltyCoordinateModel:
    def __init__(self, model, compiled, sp):
        object.__setattr__(self, "_parent", model)
        result = replace(model.gam_result_, compiled_model=compiled)
        object.__setattr__(self, "gam_result_", result)
        object.__setattr__(self, "smoothing_params", sp)
        object.__setattr__(self, "smoothing_fixed_mask_", np.zeros(len(sp), dtype=bool))
        workspace = copy(_fit_workspace(model))
        workspace.penalty_subspace_cache = None
        object.__setattr__(self, "_ws", workspace)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name in {
            "gam_result_",
            "smoothing_params",
            "smoothing_fixed_mask_",
            "_ws",
            "_parent",
        }:
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)


def uses_penalty_coordinate_map(model):
    return has_mapped_penalties(_penalty_blocks_seq(model))


@contextmanager
def penalty_coordinate_model(model, sp):
    """Expose the same fit at full penalty SPs, retaining source zero offsets."""
    from .selection.reparam import (
        _penalty_log_smoothing_map,
        build_estimate_gam_setup_state,
    )

    compiled = _compiled_model(model)
    blocks = tuple(_penalty_blocks_seq(model))
    setup = build_estimate_gam_setup_state(model)
    _, log_sp, mapping = _penalty_log_smoothing_map(model, setup, sp)
    full_sp = np.exp(log_sp)
    full_blocks = []
    for i, block in enumerate(blocks):
        metadata = dict(block.metadata)
        metadata.pop("log_sp_map", None)
        full_blocks.append(replace(block, smoothing_index=i, metadata=metadata))
    lifted = replace(
        compiled, compiled_penalties=tuple(full_blocks), n_smoothing_params=len(blocks)
    )
    view = _PenaltyCoordinateModel(model, lifted, full_sp)
    try:
        yield view, full_sp, mapping
    finally:
        # Coefficient warm starts belong to the fit, while the full-SP setup
        # cache belongs only to this coordinate view.
        workspace = _fit_workspace(model)
        for field in fields(view._ws):
            if field.name != "penalty_subspace_cache":
                setattr(workspace, field.name, getattr(view._ws, field.name))


def _map_columns(values, mapping):
    if values is None:
        return None
    return list(np.einsum("ia,i...->a...", mapping, np.asarray(values)))


def _map_second(values, mapping):
    if values is None:
        return None
    result = np.einsum("ia,ij...,jb->ab...", mapping, np.asarray(values), mapping)
    return [list(row) for row in result]


def map_ift(state, mapping):
    changes = {}
    for name in ("P_derivs", "dbeta", "deta", "dA", "dXtWX", "dW_obs"):
        if hasattr(state, name):
            changes[name] = _map_columns(getattr(state, name), mapping)
    for name in ("d2beta_mat", "d2A_mat", "d2XtWX_mat", "d2W_obs_mat"):
        if hasattr(state, name):
            changes[name] = _map_second(getattr(state, name), mapping)
    if hasattr(state, "root_blocks"):
        # Signed log-map weights have no square-root representation. Consumers
        # use the contracted derivative matrices and coefficient directions.
        changes["root_blocks"] = None
    return replace(state, **changes)


def map_gdi1(kernel, mapping):
    changes = {"ift": map_ift(kernel.ift, mapping)}
    for first, second in (
        ("D1", "D2"),
        ("P1", "P2"),
        ("bSb1", "bSb2"),
        ("det1", "det2"),
        ("trA1", "trA2"),
        ("K1", "K2"),
    ):
        changes[first] = mapping.T @ getattr(kernel, first)
        changes[second] = mapping.T @ getattr(kernel, second) @ mapping
    changes["dVkk"] = mapping.T @ kernel.dVkk @ mapping
    mapped = replace(kernel, **changes)
    # gam.fit3.post.proc maps the full-SP coefficient/weight directions itself.
    # Keep that source state available without inverting a rectangular L.
    mapped.full_penalty_kernel = kernel
    return mapped


def map_gdi2(kernel, mapping):
    ntheta = int(kernel.ift.ntheta)
    joint = np.zeros((ntheta + mapping.shape[0], ntheta + mapping.shape[1]))
    joint[:ntheta, :ntheta] = np.eye(ntheta)
    joint[ntheta:, ntheta:] = mapping
    changes = {
        "gdi1": map_gdi1(kernel.gdi1, mapping),
        "ift": map_ift(kernel.ift, joint),
    }
    for name in ("Dp1", "D1_full", "K1_full"):
        value = getattr(kernel, name)
        changes[name] = None if value is None else joint.T @ value
    for name in ("Dp2", "D2_full", "K2_full"):
        value = getattr(kernel, name)
        changes[name] = None if value is None else joint.T @ value @ joint
    return replace(kernel, **changes)
