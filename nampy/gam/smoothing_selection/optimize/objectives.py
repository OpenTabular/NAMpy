"""Objective wrappers for smoothing selection (trace + joint Gaussian REML)."""
import numpy as np

from ..criteria import (
    criterion_gradient,
    criterion_gradient_ml_reml_gaussian_dynamic_joint,
    criterion_gradient_ml_reml_pirls_gamma_joint,
    criterion_hessian,
    criterion_hessian_ml_reml_pirls_gamma_joint,
    criterion_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_gaussian_exact_joint,
    criterion_ml_reml_pirls_gamma_joint,
    criterion_value,
)

try:
    from scipy.optimize import approx_derivative as _approx_derivative
except Exception:  # pragma: no cover
    _approx_derivative = None

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


def _design_has_mrf_smooth(model) -> bool:
    for tb in getattr(model, "term_blocks_", None) or []:
        if str(getattr(tb, "basis_name", "")).lower() == "mrf":
            return True
    return False


class _JointGaussianRemlObjective:
    """Joint (log sp, log sigma^2) Gaussian REML/LAML outer objective (Wood-style)."""

    def __init__(self, model, y, branch_method: str, backend: str):
        self.model = model
        self.y = y
        self.branch_method = str(branch_method).upper()
        self.backend = str(backend)
        self.n_fun = 0
        self.n_jac = 0

    def _raw_fun(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if self.backend == "gaussian_exact":
            return float(
                criterion_ml_reml_gaussian_exact_joint(
                    self.model,
                    self.y,
                    x[:-1],
                    float(x[-1]),
                    method=self.branch_method,
                )
            )
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
        if self.backend == "gaussian_exact":
            if _approx_derivative is None:
                return None
            return np.asarray(
                _approx_derivative(self._raw_fun, x, method="2-point"),
                dtype=np.float64,
            )
        g = criterion_gradient_ml_reml_gaussian_dynamic_joint(
            self.model,
            self.y,
            x[:-1],
            float(x[-1]),
            method=self.branch_method,
        )
        if g is not None:
            return np.asarray(g, dtype=np.float64)
        if _approx_derivative is not None:
            return np.asarray(
                _approx_derivative(self._raw_fun, x, method="2-point"),
                dtype=np.float64,
            )
        return None


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
