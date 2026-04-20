"""Objective wrappers for smoothing selection (trace + Gaussian REML)."""

import numpy as np

from ..criteria import (
    _gaussian_dynamic_reml_derivative_terms,
    criterion_gradient,
    criterion_gradient_ml_reml_gaussian_dynamic_joint,
    criterion_gradient_ml_reml_pirls_gamma_joint,
    criterion_gradient_ml_reml_pirls_negbin_joint,
    criterion_gradient_ncv_negbin_joint,
    criterion_hessian,
    criterion_hessian_ml_reml_gaussian_dynamic_joint,
    criterion_hessian_ml_reml_pirls_gamma_joint,
    criterion_hessian_ml_reml_pirls_negbin_joint,
    criterion_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_gaussian_dynamic_profiled,
    criterion_ml_reml_pirls_gamma_joint,
    criterion_ml_reml_pirls_negbin_joint,
    criterion_ncv_negbin_joint,
    criterion_value,
)


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
        self.accepted_trace = []

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
        self.accepted_trace = []

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


class _JointNegbinPirlsRemlObjective:
    """Joint (log sp, log theta) NegBin PIRLS REML/LAML outer objective."""

    def __init__(self, model, y, branch_method: str):
        self.model = model
        self.y = y
        self.branch_method = str(branch_method).upper()
        self.n_fun = 0
        self.n_jac = 0
        self.n_hess = 0
        self.accepted_trace = []

    def _raw_fun(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(
            criterion_ml_reml_pirls_negbin_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
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
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_jac += 1
        return np.asarray(
            criterion_gradient_ml_reml_pirls_negbin_joint(
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
            criterion_hessian_ml_reml_pirls_negbin_joint(
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
        self.accepted_trace.append(
            {
                "x": x.copy(),
                "fun": crit,
                "accepted_step_norm": float(accepted_step_norm),
            }
        )


class _JointNegbinNcvObjective:
    """Joint `(log sp, log theta)` NegBin NCV/QNCV outer objective."""

    def __init__(self, model, y, *, qapprox: bool):
        self.model = model
        self.y = y
        self.qapprox = bool(qapprox)
        self.uses_joint_log_theta = True
        self.n_fun = 0
        self.n_jac = 0
        self.n_hess = 0
        self.accepted_trace = []

    def _raw_fun(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(
            criterion_ncv_negbin_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
                qapprox=self.qapprox,
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
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_jac += 1
        return np.asarray(
            criterion_gradient_ncv_negbin_joint(
                self.model,
                self.y,
                x[:-1],
                float(x[-1]),
                qapprox=self.qapprox,
            ),
            dtype=np.float64,
        )

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
