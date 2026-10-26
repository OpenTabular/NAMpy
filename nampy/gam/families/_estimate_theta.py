"""Fixed-dispersion ancillary Newton updates, efam.r::estimate.theta."""

import warnings

import numpy as np


def update_theta_efs(family, y, mu, weights, scale=1.0, tol=1e-7):
    """Port the free-vector theta branch; dispersion stays externally fixed."""
    theta = np.asarray(family.getTheta(), float).copy()

    def state(value, deriv=2):
        deviance = np.sum(family.deviance_obs(y, mu, weights, theta=value)) / scale
        saturation = family.ls(y, weights, theta=value, scale=scale)
        objective = float(deviance / 2 - saturation["ls"])
        if not deriv:
            return objective, None, None
        derivatives = family.Dd(y, mu, value, weights, level=2)
        gradient = np.sum(np.asarray(derivatives["Dth"]), axis=0) / (2 * scale)
        gradient = np.atleast_1d(gradient) - np.asarray(saturation["lsth1"])
        second = np.sum(np.asarray(derivatives["Dth2"]), axis=0) / (2 * scale)
        hessian = np.zeros((theta.size, theta.size))
        k = 0
        for i in range(theta.size):
            for j in range(i, theta.size):
                hessian[i, j] = hessian[j, i] = np.atleast_1d(second)[k]
                k += 1
        hessian -= np.asarray(saturation["lsth2"])
        return objective, gradient, hessian

    value, gradient, hessian = state(theta)
    unconverged = np.abs(gradient) > tol * (abs(value) + 1)
    if np.any(unconverged):
        for _ in range(100):
            eigenvalues, vectors = np.linalg.eigh(hessian[np.ix_(unconverged, unconverged)])
            # R eigen returns descending eigenvalues; preserve contraction order.
            eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
            if np.any(eigenvalues <= 0):
                eigenvalues = np.abs(eigenvalues)
                threshold = np.max(eigenvalues) * 1e-5
                eigenvalues[eigenvalues < threshold] = threshold
            step0 = -(vectors @ ((vectors.T @ gradient[unconverged]) / eigenvalues))
            length = np.max(np.abs(step0))
            if length > 4:
                step0 = step0 * 4 / length
            step = np.zeros_like(theta)
            step[unconverged] = step0
            candidate = state(theta + step)
            halvings = 0
            failed = False
            while candidate[0] - value > np.finfo(float).eps**0.75 * abs(value):
                step /= 2
                halvings += 1
                if not np.any(theta != theta + step) or halvings > 25:
                    failed = True
                    break
                candidate = state(theta + step, deriv=0)
            if failed:
                warnings.warn("step failure in theta estimation", RuntimeWarning, stacklevel=2)
                break
            theta += step
            value, gradient, hessian = state(theta) if halvings else candidate
            unconverged = np.abs(gradient) > tol * (abs(value) + 1)
            if not np.any(unconverged):
                break
    family.putTheta(theta)
