"""Objective wrappers for smoothing selection (trace + joint Gaussian REML)."""

import numpy as np

from ..criteria import (
    criterion_gradient,
    criterion_gradient_ml_reml_gaussian_dynamic_joint,
    criterion_gradient_ml_reml_pirls_gamma_joint,
    criterion_hessian,
    criterion_hessian_ml_reml_pirls_gamma_joint,
    criterion_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_pirls_gamma_joint,
    criterion_ml_reml_pirls_negbin_joint,
    criterion_value,
)

try:
    from scipy.optimize import approx_derivative as _approx_derivative
except Exception:  # pragma: no cover
    _approx_derivative = None


def _safe_scalar_fd_gradient(fun, x, *, eps_abs=1e-6, eps_rel=1e-6):
    """Finite-difference gradient that avoids SciPy _numdiff warnings on non-finite probes."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return np.empty((0,), dtype=np.float64)

    f0 = float(fun(x))
    grad = np.zeros_like(x, dtype=np.float64)
    steps = np.maximum(float(eps_abs), float(eps_rel) * (1.0 + np.abs(x)))

    for j, h0 in enumerate(steps):
        h = float(h0)
        g_j = None
        for _ in range(8):
            xp = x.copy()
            xm = x.copy()
            xp[j] += h
            xm[j] -= h

            fp = float(fun(xp))
            fm = float(fun(xm))

            if np.isfinite(fp) and np.isfinite(fm):
                g_j = (fp - fm) / (2.0 * h)
                break
            if np.isfinite(fp) and np.isfinite(f0):
                g_j = (fp - f0) / h
                break
            if np.isfinite(fm) and np.isfinite(f0):
                g_j = (f0 - fm) / h
                break
            h *= 0.5

        grad[j] = 0.0 if g_j is None or not np.isfinite(g_j) else float(g_j)

    return grad


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


class _JointGaussianRemlObjective:
    """Joint (log sp, log sigma^2) Gaussian REML/LAML outer objective (Wood-style)."""

    def __init__(self, model, y, branch_method: str, backend: str):
        self.model = model
        self.y = y
        self.branch_method = str(branch_method).upper()
        self.backend = str(backend)
        self.n_fun = 0
        self.n_jac = 0
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
        if self.backend == "gaussian_exact":
            return np.asarray(_safe_scalar_fd_gradient(self._raw_fun, x), dtype=np.float64)
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
    """Joint (log_sp, log_theta) NegBin PIRLS REML/LAML outer objective.

    Mirrors mgcv's ``gam.fit4`` extended-family outer problem only partially:
    ``log_theta`` is optimized jointly with ``log_sp``, but the optimizer is a
    generic finite-difference L-BFGS-B step rather than mgcv's EFS smoothing
    update loop. This means prediction/theta parity can hold while the chosen
    ``log_sp`` lands at a different point on a shallow REML ridge.

    Gradient is evaluated via numerical finite differences because analytic
    theta derivatives of the NB REML criterion are not yet implemented.
    """

    def __init__(self, model, y, branch_method: str):
        self.model = model
        self.y = y
        self.branch_method = str(branch_method).upper()
        self.method = self.branch_method
        self.n_fun = 0
        self.n_jac = 0
        self.n_hess = 0
        self.trace = []
        self._trace_index_by_x = {}

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
        self.n_fun += 1
        val = self._raw_fun(x)
        key = tuple(np.asarray(x, dtype=np.float64).tolist())
        idx = self._trace_index_by_x.get(key, None)
        if idx is None:
            self._trace_index_by_x[key] = len(self.trace)
            self.trace.append({"x": x.copy(), "fun": float(val), "grad": None})
        else:
            self.trace[idx]["fun"] = float(val)
        return val

    def jac(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_jac += 1
        f0 = self._raw_fun(x)
        if not np.isfinite(f0):
            return np.full_like(x, np.nan)
        grad = np.empty_like(x)
        for i in range(x.size):
            step = max(1e-5, 1e-4 * (1.0 + abs(float(x[i]))))
            xp = x.copy()
            xp[i] += step
            fp = self._raw_fun(xp)
            if np.isfinite(fp):
                grad[i] = (fp - f0) / step
            else:
                grad[i] = np.nan
        key = tuple(np.asarray(x, dtype=np.float64).tolist())
        idx = self._trace_index_by_x.get(key, None)
        if idx is None:
            self._trace_index_by_x[key] = len(self.trace)
            self.trace.append({"x": x.copy(), "fun": float(f0), "grad": grad.copy()})
        else:
            self.trace[idx]["grad"] = grad.copy()
        return grad

    def hess(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_hess += 1
        if _approx_derivative is not None:
            return np.asarray(
                _approx_derivative(self.jac, x, method="2-point"), dtype=np.float64
            )
        n = x.size
        h = np.empty((n, n), dtype=np.float64)
        g0 = self.jac(x)
        for i in range(n):
            step = max(1e-5, 1e-4 * (1.0 + abs(float(x[i]))))
            xp = x.copy()
            xp[i] += step
            gp = self.jac(xp)
            h[:, i] = (gp - g0) / step
        return 0.5 * (h + h.T)
