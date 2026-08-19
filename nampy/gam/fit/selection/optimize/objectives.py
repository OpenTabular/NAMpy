"""Objective wrappers for smoothing selection (trace + Gaussian REML)."""

from typing import Any

import numpy as np

from ...backends import GENERAL_FAMILY_BACKEND
from ..criteria import (
    criterion_gradient,
    criterion_gradient_ml_reml_gaussian_dynamic_joint,
    criterion_gradient_ml_reml_pirls_gamma_joint,
    criterion_gradient_ml_reml_pirls_gaussian_joint,
    criterion_gradient_ml_reml_pirls_negbin_joint,
    criterion_hessian,
    criterion_hessian_ml_reml_gaussian_dynamic_joint,
    criterion_hessian_ml_reml_pirls_gamma_joint,
    criterion_hessian_ml_reml_pirls_gaussian_joint,
    criterion_hessian_ml_reml_pirls_negbin_joint,
    criterion_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_gaussian_dynamic_profiled,
    criterion_ml_reml_pirls_gamma_joint,
    criterion_ml_reml_pirls_gaussian_joint,
    criterion_ml_reml_pirls_negbin_joint,
    criterion_value,
    resolve_ml_reml_scoring_backend,
)
from ..criteria.gaussian_dyn import _gaussian_dynamic_reml_derivative_terms


class _CriterionObjective:
    def __init__(self, model, y, method, use_gradient):
        self.model = model
        self.y = y
        self.method = method
        self.use_gradient = bool(use_gradient)
        self._last_x = None
        self._last_fun = None
        self._last_grad = None
        self._last_hess = None
        self.n_fun = 0
        self.n_jac = 0
        self.n_hess = 0
        self.capture_trace = True
        self.trace = []
        self._trace_index_by_x = {}

    def _same_x(self, x):
        return self._last_x is not None and np.array_equal(self._last_x, x)

    def _refresh_general_family_score(self, x):
        method = str(self.method).lower()
        if method not in {"ml", "reml", "laml"}:
            return
        try:
            backend = resolve_ml_reml_scoring_backend(self.model, method=method)
        except Exception:
            return
        if backend != GENERAL_FAMILY_BACKEND:
            return
        try:
            val = float(criterion_value(self.model, self.y, x, method=self.method))
        except Exception:
            return

        self._last_fun = val
        if self.capture_trace:
            key = tuple(np.asarray(x, dtype=np.float64).tolist())
            idx = self._trace_index_by_x.get(key, None)
            if idx is not None:
                self.trace[idx]["fun"] = float(val)

    def fun(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if self._same_x(x) and self._last_fun is not None:
            return float(self._last_fun)
        val = float(criterion_value(self.model, self.y, x, method=self.method))
        self.n_fun += 1
        self._last_x = x.copy()
        self._last_fun = val
        self._last_grad = None
        self._last_hess = None
        if self.capture_trace:
            key = tuple(np.asarray(x, dtype=np.float64).tolist())
            idx = self._trace_index_by_x.get(key, None)
            if idx is None:
                idx = len(self.trace)
                self._trace_index_by_x[key] = idx
                self.trace.append(
                    {
                        "x": np.asarray(x, dtype=np.float64).copy(),
                        "fun": float(val),
                        "grad": None,
                        "hess": None,
                        "n_fun": int(self.n_fun),
                        "n_jac": int(self.n_jac),
                        "n_hess": int(self.n_hess),
                    }
                )
            else:
                self.trace[idx]["fun"] = float(val)
                self.trace[idx]["n_fun"] = int(self.n_fun)
        return val

    def jac(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if self._same_x(x) and self._last_grad is not None:
            return self._last_grad.copy()

        if not self._same_x(x) or self._last_fun is None:
            self.fun(x)

        grad = np.asarray(
            criterion_gradient(self.model, self.y, x, method=self.method),
            dtype=np.float64,
        )
        self.n_jac += 1
        self._last_x = x.copy()
        self._last_grad = grad.copy()
        self._last_hess = None
        self._refresh_general_family_score(x)
        if self.capture_trace:
            key = tuple(np.asarray(x, dtype=np.float64).tolist())
            idx = self._trace_index_by_x.get(key, None)
            if idx is None:
                idx = len(self.trace)
                self._trace_index_by_x[key] = idx
                self.trace.append(
                    {
                        "x": np.asarray(x, dtype=np.float64).copy(),
                        "fun": None,
                        "grad": grad.copy(),
                        "hess": None,
                        "n_fun": int(self.n_fun),
                        "n_jac": int(self.n_jac),
                        "n_hess": int(self.n_hess),
                    }
                )
            else:
                self.trace[idx]["grad"] = grad.copy()
                self.trace[idx]["n_jac"] = int(self.n_jac)
        return grad

    def hess(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if self._same_x(x) and self._last_hess is not None:
            return self._last_hess.copy()

        if not self._same_x(x) or self._last_fun is None:
            self.fun(x)

        hess = np.asarray(
            criterion_hessian(self.model, self.y, x, method=self.method),
            dtype=np.float64,
        )
        self.n_hess += 1
        self._last_x = x.copy()
        self._last_hess = hess.copy()
        self._refresh_general_family_score(x)
        if self.capture_trace:
            key = tuple(np.asarray(x, dtype=np.float64).tolist())
            idx = self._trace_index_by_x.get(key, None)
            if idx is None:
                idx = len(self.trace)
                self._trace_index_by_x[key] = idx
                self.trace.append(
                    {
                        "x": np.asarray(x, dtype=np.float64).copy(),
                        "fun": None,
                        "grad": None,
                        "hess": hess.copy(),
                        "n_fun": int(self.n_fun),
                        "n_jac": int(self.n_jac),
                        "n_hess": int(self.n_hess),
                    }
                )
            else:
                self.trace[idx]["hess"] = hess.copy()
                self.trace[idx]["n_hess"] = int(self.n_hess)
        return hess


class _GaussianRemlProfiledObjective:
    """Profiled Gaussian REML/LAML objective (no joint scale parameter)."""

    def __init__(self, model, y, branch_method: str):
        self.model = model
        self.y = y
        self.branch_method = str(branch_method).upper()
        self.method = (
            "REML" if self.branch_method in {"REML", "LAML"} else self.branch_method
        )
        self.n_fun = 0
        self.n_jac = 0
        self.n_hess = 0
        self.accepted_trace: list[dict[str, Any]] = []

    def _raw_fun(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(
            criterion_ml_reml_gaussian_dynamic_profiled(
                self.model,
                self.y,
                x,
                method=self.branch_method,
            )
        )

    def fun(self, x):
        self.n_fun += 1
        return self._raw_fun(x)

    def jac(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_jac += 1
        out = _gaussian_dynamic_reml_derivative_terms(
            self.model,
            self.y,
            x,
            method=self.branch_method,
        )
        if bool(out.get("valid", False)):
            return np.asarray(out["grad"], dtype=np.float64)
        raise NotImplementedError(
            "Gaussian REML/LAML profile objective requires exact derivative terms."
        )

    def hess(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_hess += 1
        out = _gaussian_dynamic_reml_derivative_terms(
            self.model,
            self.y,
            x,
            method=self.branch_method,
        )
        if bool(out.get("valid", False)):
            return np.asarray(out["hess"], dtype=np.float64)
        raise NotImplementedError(
            "Gaussian REML/LAML profile objective requires exact derivative terms."
        )

    def record_iter(self, x, accepted_step_norm: float) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        crit = float(self._raw_fun(x))
        if len(self.accepted_trace) == 0 or not np.array_equal(
            self.accepted_trace[-1]["x"], x
        ):
            self.accepted_trace.append(
                {
                    "x": x.copy(),
                    "fun": crit,
                    "accepted_step_norm": float(accepted_step_norm),
                }
            )


class _GaussianRemlJointObjective:
    """Joint Gaussian REML objective over `(log sp_free..., log sigma^2)`."""

    def __init__(self, model, y, branch_method: str):
        self.model = model
        self.y = y
        self.branch_method = str(branch_method).upper()
        self.method = self.branch_method
        self.uses_joint_log_scale = True
        self.n_fun = 0
        self.n_jac = 0
        self.n_hess = 0
        self.accepted_trace: list[dict[str, Any]] = []

    def _raw_fun(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(
            criterion_ml_reml_gaussian_dynamic_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
                method=self.branch_method,
            )
        )

    def fun(self, x):
        self.n_fun += 1
        return self._raw_fun(x)

    def jac(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_jac += 1
        return np.asarray(
            criterion_gradient_ml_reml_gaussian_dynamic_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
                method=self.branch_method,
            ),
            dtype=np.float64,
        )

    def hess(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_hess += 1
        return np.asarray(
            criterion_hessian_ml_reml_gaussian_dynamic_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
                method=self.branch_method,
            ),
            dtype=np.float64,
        )

    def record_iter(self, x, accepted_step_norm: float) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        crit = float(self._raw_fun(x))
        if len(self.accepted_trace) == 0 or not np.array_equal(
            self.accepted_trace[-1]["x"], x
        ):
            self.accepted_trace.append(
                {
                    "x": x.copy(),
                    "fun": crit,
                    "accepted_step_norm": float(accepted_step_norm),
                }
            )


class _JointGammaPirlsRemlObjective:
    """Joint (log sp, log phi) Gamma PIRLS REML/LAML outer objective."""

    def __init__(self, model, y, branch_method: str):
        self.model = model
        self.y = y
        self.branch_method = str(branch_method).upper()
        self.method = self.branch_method
        self.uses_joint_log_scale = True
        self.n_fun = 0
        self.n_jac = 0
        self.n_hess = 0

    def _raw_fun(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(
            criterion_ml_reml_pirls_gamma_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
                method=self.branch_method,
            )
        )

    def fun(self, x):
        self.n_fun += 1
        return self._raw_fun(x)

    def jac(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_jac += 1
        return np.asarray(
            criterion_gradient_ml_reml_pirls_gamma_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
                method=self.branch_method,
            ),
            dtype=np.float64,
        )

    def hess(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_hess += 1
        return np.asarray(
            criterion_hessian_ml_reml_pirls_gamma_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
                method=self.branch_method,
            ),
            dtype=np.float64,
        )


class _GaussianPirlsRemlJointObjective:
    """Joint Gaussian ``gam.fit3`` objective for noncanonical PIRLS links."""

    def __init__(self, model, y, branch_method: str):
        self.model = model
        self.y = y
        self.branch_method = str(branch_method).upper()
        self.method = self.branch_method
        self.uses_joint_log_scale = True
        self.n_fun = 0
        self.n_jac = 0
        self.n_hess = 0
        self.accepted_trace: list[dict[str, Any]] = []

    def _raw_fun(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(
            criterion_ml_reml_pirls_gaussian_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
                method=self.branch_method,
            )
        )

    def fun(self, x):
        self.n_fun += 1
        return self._raw_fun(x)

    def jac(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_jac += 1
        return np.asarray(
            criterion_gradient_ml_reml_pirls_gaussian_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
                method=self.branch_method,
            ),
            dtype=np.float64,
        )

    def hess(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_hess += 1
        return np.asarray(
            criterion_hessian_ml_reml_pirls_gaussian_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
                method=self.branch_method,
            ),
            dtype=np.float64,
        )

    def record_iter(self, x, accepted_step_norm: float) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        crit = float(self._raw_fun(x))
        if len(self.accepted_trace) == 0 or not np.array_equal(
            self.accepted_trace[-1]["x"], x
        ):
            self.accepted_trace.append(
                {
                    "x": x.copy(),
                    "fun": crit,
                    "accepted_step_norm": float(accepted_step_norm),
                }
            )


class _JointNegbinPirlsRemlObjective:
    """Joint `(log theta, log sp...)` NegBin PIRLS REML/LAML objective."""

    def __init__(self, model, y, branch_method: str):
        self.model = model
        self.y = y
        self.branch_method = str(branch_method).upper()
        self.method = self.branch_method
        self.uses_joint_log_theta = True
        self.joint_log_theta_first = True
        self.n_fun = 0
        self.n_jac = 0
        self.n_hess = 0
        self.accepted_trace: list[dict[str, Any]] = []

    def _split_x(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            raise ValueError("Joint NegBin objective requires log(theta).")
        return np.asarray(x[1:], dtype=np.float64), float(x[0])

    @staticmethod
    def _theta_last_to_theta_first_gradient(grad):
        grad = np.asarray(grad, dtype=np.float64).ravel()
        if grad.size <= 1:
            return grad.copy()
        return np.concatenate([np.asarray([grad[-1]], dtype=np.float64), grad[:-1]])

    @staticmethod
    def _theta_last_to_theta_first_hessian(hess):
        hess = np.asarray(hess, dtype=np.float64)
        if hess.size == 0 or hess.shape[0] <= 1:
            return hess.copy()
        last = int(hess.shape[0] - 1)
        perm = np.concatenate(
            [np.asarray([last], dtype=np.int64), np.arange(last, dtype=np.int64)]
        )
        return np.asarray(hess[np.ix_(perm, perm)], dtype=np.float64)

    def _raw_fun(self, x):
        log_sp, log_theta = self._split_x(x)
        return float(
            criterion_ml_reml_pirls_negbin_joint(
                self.model,
                self.y,
                log_sp,
                log_theta,
                method=self.branch_method,
            )
        )

    def fun(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        val = float(self._raw_fun(x))
        self.n_fun += 1
        if len(self.accepted_trace) == 0 or not np.array_equal(
            self.accepted_trace[-1]["x"], x
        ):
            self.accepted_trace.append({"x": x.copy(), "fun": float(val)})
        return val

    def jac(self, x):
        log_sp, log_theta = self._split_x(x)
        self.n_jac += 1
        grad = np.asarray(
            criterion_gradient_ml_reml_pirls_negbin_joint(
                self.model,
                self.y,
                log_sp,
                log_theta,
                method=self.branch_method,
            ),
            dtype=np.float64,
        )
        return self._theta_last_to_theta_first_gradient(grad)

    def hess(self, x):
        log_sp, log_theta = self._split_x(x)
        self.n_hess += 1
        hess = np.asarray(
            criterion_hessian_ml_reml_pirls_negbin_joint(
                self.model,
                self.y,
                log_sp,
                log_theta,
                method=self.branch_method,
            ),
            dtype=np.float64,
        )
        return self._theta_last_to_theta_first_hessian(hess)

    def record_iter(self, x, accepted_step_norm: float) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        crit = float(self._raw_fun(x))
        self.accepted_trace.append(
            {
                "x": x.copy(),
                "fun": crit,
                "accepted_step_norm": float(accepted_step_norm),
            }
        )
