"""Validated controls for the mgcv-aligned fitting stack."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

_KEY_ALIASES = {
    "ncv.threads": "ncv_threads",
    "irls.reg": "irls_reg",
    "mgcv.tol": "mgcv_tol",
    "mgcv.half": "mgcv_half",
    "rank.tol": "rank_tol",
    "idLinksBases": "id_links_bases",
    "scalePenalty": "scale_penalty",
    "efs.lspmax": "efs_lspmax",
    "efs.tol": "efs_tol",
    "keepData": "keep_data",
    "scale.est": "scale_est",
    "edge.correct": "edge_correct",
    # SCAM control spellings.
    "maxHalf": "scam_max_half",
    "devtol.fit": "scam_devtol_fit",
    "steptol.fit": "scam_steptol_fit",
    "print.warn": "scam_print_warn",
    "b.notexp": "scam_b_notexp",
    "threshold.notexp": "scam_threshold_notexp",
    "bfgs": "scam_bfgs",
}


@dataclass(frozen=True)
class GAMControl:
    """Python representation of meaningful ``mgcv::gam.control`` settings."""

    nthreads: int = 1
    ncv_threads: int = 1
    irls_reg: float = 0.0
    epsilon: float = 1e-7
    maxit: int = 200
    mgcv_tol: float = 1e-7
    mgcv_half: int = 15
    trace: bool = False
    rank_tol: float = float(np.sqrt(np.finfo(np.float64).eps))
    nlm: Mapping[str, Any] = field(default_factory=dict)
    optim: Mapping[str, Any] = field(default_factory=dict)
    newton: Mapping[str, Any] = field(default_factory=dict)
    id_links_bases: bool = True
    scale_penalty: bool = True
    efs_lspmax: float = 15.0
    efs_tol: float = 0.1
    keep_data: bool = False
    scale_est: str = "fletcher"
    edge_correct: bool | float = False
    scam_max_half: int = 30
    scam_devtol_fit: float = 1e-7
    scam_steptol_fit: float = 1e-7
    scam_print_warn: bool = False
    scam_b_notexp: float = 1.0
    scam_threshold_notexp: float = 20.0
    scam_bfgs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(round(self.nthreads)) < 1 or int(round(self.ncv_threads)) < 1:
            raise ValueError("nthreads and ncv_threads must be positive integers.")
        object.__setattr__(self, "nthreads", int(round(self.nthreads)))
        object.__setattr__(self, "ncv_threads", int(round(self.ncv_threads)))
        if float(self.irls_reg) < 0.0:
            raise ValueError("irls_reg must be non-negative.")
        if float(self.epsilon) <= 0.0 or int(self.maxit) <= 0:
            raise ValueError("epsilon and maxit must be positive.")
        if float(self.mgcv_tol) <= 0.0 or int(self.mgcv_half) < 0:
            raise ValueError("mgcv_tol must be positive and mgcv_half non-negative.")
        if not 0.0 <= float(self.rank_tol) <= 1.0:
            warnings.warn(
                "silly value supplied for rank_tol: reset to square root of "
                "machine precision.",
                stacklevel=2,
            )
            object.__setattr__(self, "rank_tol", float(np.sqrt(np.finfo(float).eps)))
        if float(self.efs_tol) <= 0.0:
            object.__setattr__(self, "efs_tol", 0.1)
        scale_est = str(self.scale_est).lower()
        if scale_est not in {"fletcher", "pearson", "deviance"}:
            raise ValueError("scale_est must be 'fletcher', 'pearson', or 'deviance'.")
        object.__setattr__(self, "scale_est", scale_est)
        edge = self.edge_correct
        if not isinstance(edge, (bool, np.bool_)) and float(edge) < 0.0:
            raise ValueError("edge_correct must be boolean or non-negative.")
        if int(self.scam_max_half) <= 0:
            raise ValueError("scam maxHalf must be positive.")
        if float(self.scam_devtol_fit) <= 0.0 or float(self.scam_steptol_fit) <= 0.0:
            raise ValueError("SCAM devtol.fit and steptol.fit must be positive.")
        if float(self.scam_b_notexp) <= 0.0 or float(self.scam_threshold_notexp) <= 0.0:
            raise ValueError("SCAM b.notexp and threshold.notexp must be positive.")
        for name in ("nlm", "optim", "newton", "scam_bfgs"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} control must be a mapping.")
            object.__setattr__(self, name, dict(value))

    def with_overrides(self, **kwargs) -> "GAMControl":
        return replace(self, **kwargs)


def gam_control(
    control: GAMControl | Mapping[str, Any] | None = None, **kwargs
) -> GAMControl:
    """Create a validated control object from Python or upstream-style keys."""
    if control is None:
        values: dict[str, Any] = {}
    elif isinstance(control, GAMControl):
        values = {name: getattr(control, name) for name in control.__dataclass_fields__}
    elif isinstance(control, Mapping):
        values = dict(control)
    else:
        raise TypeError("control must be a GAMControl, mapping, or None.")
    values.update(kwargs)
    normalized = {
        _KEY_ALIASES.get(str(key), str(key)): value for key, value in values.items()
    }
    known = set(GAMControl.__dataclass_fields__)
    unknown = sorted(set(normalized) - known)
    if unknown:
        raise TypeError(f"Unknown GAM control field(s): {unknown}")

    epsilon = float(normalized.get("epsilon", 1e-7))
    nlm = dict(normalized.get("nlm", {}) or {})
    nlm_aliases = {
        "check.analyticals": "check_analyticals",
    }
    nlm = {nlm_aliases.get(str(k), str(k)): v for k, v in nlm.items()}
    if nlm.get("ndigit", None) is None or float(nlm["ndigit"]) < 2.0:
        nlm["ndigit"] = max(2, int(np.ceil(-np.log10(epsilon))))
    nlm["ndigit"] = min(int(round(nlm["ndigit"])), 15)
    nlm.setdefault("gradtol", epsilon * 10.0)
    nlm["gradtol"] = abs(float(nlm["gradtol"]))
    nlm.setdefault("stepmax", 2.0)
    if float(nlm["stepmax"]) == 0.0:
        nlm["stepmax"] = 2.0
    nlm["stepmax"] = abs(float(nlm["stepmax"]))
    nlm.setdefault("steptol", 1e-4)
    nlm["steptol"] = abs(float(nlm["steptol"]))
    nlm.setdefault("iterlim", 200)
    nlm["iterlim"] = int(round(abs(float(nlm["iterlim"]))))
    nlm.setdefault("check_analyticals", False)
    normalized["nlm"] = nlm

    optim = dict(normalized.get("optim", {}) or {})
    optim.setdefault("factr", 1e7)
    optim["factr"] = abs(float(optim["factr"]))
    normalized["optim"] = optim

    newton = dict(normalized.get("newton", {}) or {})
    aliases = {
        "conv.tol": "conv_tol",
        "maxNstep": "max_n_step",
        "maxSstep": "max_s_step",
        "maxHalf": "max_half",
        "use.svd": "use_svd",
    }
    newton = {aliases.get(str(k), str(k)): v for k, v in newton.items()}
    newton.setdefault("conv_tol", 1e-6)
    newton.setdefault("max_n_step", 5.0)
    newton.setdefault("max_s_step", 2.0)
    newton.setdefault("max_half", 30)
    newton.setdefault("use_svd", False)
    normalized["newton"] = newton

    scam_bfgs = dict(normalized.get("scam_bfgs", {}) or {})
    bfgs_aliases = {
        "check.analytical": "check_analytical",
        "steptol.bfgs": "steptol_bfgs",
        "gradtol.bfgs": "gradtol_bfgs",
        "maxNstep": "max_n_step",
        "maxHalf": "max_half",
    }
    scam_bfgs = {bfgs_aliases.get(str(k), str(k)): v for k, v in scam_bfgs.items()}
    scam_bfgs.setdefault("check_analytical", False)
    scam_bfgs.setdefault("del", 1e-4)
    scam_bfgs.setdefault("steptol_bfgs", 1e-7)
    scam_bfgs.setdefault("gradtol_bfgs", 1e-6)
    scam_bfgs.setdefault("max_n_step", 5.0)
    scam_bfgs.setdefault("max_half", normalized.get("scam_max_half", 30))
    normalized["scam_bfgs"] = scam_bfgs
    return GAMControl(**normalized)


__all__ = ["GAMControl", "gam_control"]
