"""R stats::optim BFGS (src/appl/optim.c::vmmin), unmasked dense path.

Used by qgam's residual SHASH initialization. This is distinct from mgcv's
outer smoothing BFGS and does not replace its dispatch or stopping conventions.
"""

import numpy as np
from scipy.optimize import OptimizeResult


def optim_bfgs(
    function,
    gradient,
    initial,
    *,
    max_iter=100,
    relative_tolerance=np.sqrt(np.finfo(float).eps),
    absolute_tolerance=-np.inf,
):
    parameter = np.asarray(initial, dtype=float).copy()
    n = len(parameter)
    value = float(function(parameter))
    if not np.isfinite(value):
        raise ValueError("Initial R BFGS objective is nonfinite.")
    if max_iter <= 0:
        return OptimizeResult(
            x=parameter, fun=value, success=True, nit=0, nfev=0, njev=0
        )
    g = np.asarray(gradient(parameter))
    function_count = gradient_count = iteration = 1
    last_reset = gradient_count
    B = np.zeros((n, n))
    while True:
        if last_reset == gradient_count:
            B = np.eye(n)
        origin = parameter.copy()
        previous_gradient = g.copy()
        direction = np.zeros(n)
        projection = 0.0
        for i in range(n):
            for j in range(i + 1):
                direction[i] -= B[i, j] * g[j]
            for j in range(i + 1, n):
                direction[i] -= B[j, i] * g[j]
            projection += direction[i] * g[i]
        if projection < 0:
            length = 1.0
            accepted = False
            candidate_value = value
            while True:
                parameter = origin + length * direction
                unchanged = int(np.sum(10.0 + origin == 10.0 + parameter))
                if unchanged < n:
                    candidate_value = float(function(parameter))
                    function_count += 1
                    accepted = (
                        np.isfinite(candidate_value)
                        and candidate_value <= value + projection * length * 1e-4
                    )
                    if not accepted:
                        length *= 0.2
                if unchanged == n or accepted:
                    break
            enough = candidate_value > absolute_tolerance and abs(
                candidate_value - value
            ) > relative_tolerance * (abs(value) + relative_tolerance)
            if not enough:
                unchanged = n
                value = candidate_value
            if unchanged < n:
                value = candidate_value
                g = np.asarray(gradient(parameter))
                gradient_count += 1
                iteration += 1
                step = length * direction
                change = g - previous_gradient
                curvature = sum(step[i] * change[i] for i in range(n))
                if curvature > 0:
                    transformed = np.zeros(n)
                    quadratic = 0.0
                    for i in range(n):
                        for j in range(i + 1):
                            transformed[i] += B[i, j] * change[j]
                        for j in range(i + 1, n):
                            transformed[i] += B[j, i] * change[j]
                        quadratic += transformed[i] * change[i]
                    multiplier = 1 + quadratic / curvature
                    for i in range(n):
                        for j in range(i + 1):
                            B[i, j] += (
                                multiplier * step[i] * step[j]
                                - transformed[i] * step[j]
                                - step[i] * transformed[j]
                            ) / curvature
                else:
                    last_reset = gradient_count
            elif last_reset < gradient_count:
                unchanged = 0
                last_reset = gradient_count
        else:
            unchanged = n if last_reset == gradient_count else 0
            last_reset = gradient_count
        if iteration >= max_iter:
            break
        if gradient_count - last_reset > 2 * n:
            last_reset = gradient_count
        if unchanged == n and last_reset == gradient_count:
            break
    return OptimizeResult(
        x=parameter,
        fun=value,
        success=iteration < max_iter,
        nit=iteration,
        nfev=function_count,
        njev=gradient_count,
        message="converged" if iteration < max_iter else "iteration limit reached",
    )
