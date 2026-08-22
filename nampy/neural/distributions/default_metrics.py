"""Default evaluation metrics owned by distribution-family metadata."""

from __future__ import annotations

import numpy as np
import properscoring as ps
from sklearn.metrics import accuracy_score, mean_squared_error

from .metrics import (
    beta_mean_mse,
    dirichlet_error,
    gamma_deviance,
    inverse_gamma_loss,
    negative_binomial_deviance,
    poisson_deviance,
    student_t_loss,
)


def default_metrics_for(metric_profile, family_instance):
    """
    Provide sensible default metrics for each supported distribution family.

    Metrics use transformed distribution parameters returned by
    ``family_instance(raw_predictions)``. For example, normal and robust-normal
    families return ``[mean, scale]``; count families return their rate or
    mean/dispersion parameters; and categorical families return class
    probabilities.

    Parameters
    ----------
    distribution_family : str
        Family identifier.

    Returns
    -------
    metrics : dict
        Mapping of metric_name -> callable(y_true, transformed_predictions)
    """
    if metric_profile is None:
        # No profile declared for this family: no default metrics.
        return {}
    family = str(metric_profile).lower()

    def _y_1d(y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        return y.reshape(-1) if y.ndim == 1 else y

    def _categorical_labels(y):
        y = np.asarray(y)
        # Accept labels [N], [N,1], or one-hot/probs [N,K]
        if y.ndim == 2 and y.shape[1] == 1:
            return y[:, 0].astype(int)
        if y.ndim == 2:
            return np.argmax(y, axis=1).astype(int)
        return y.reshape(-1).astype(int)

    def _normal_crps(y, pred):
        # pred = [mean, scale]
        y = _y_1d(y).astype(float)
        pred = np.asarray(pred, dtype=float)
        mu = pred[:, 0]
        scale = np.clip(pred[:, 1], 1e-9, None)  # std, not variance
        return float(
            np.mean(
                [
                    ps.crps_gaussian(y[i], mu=mu[i], sig=scale[i])
                    for i in range(len(y))
                ]
            )
        )

    def _normal_mse(y, pred):
        y = _y_1d(y).astype(float)
        pred = np.asarray(pred, dtype=float)
        return float(mean_squared_error(y, pred[:, 0]))

    def _normal_mae(y, pred):
        y = _y_1d(y).astype(float)
        pred = np.asarray(pred, dtype=float)
        return float(np.mean(np.abs(y - pred[:, 0])))

    def _quantile_pinball(y, pred):
        # pred shape [N, Q], uses family.quantiles if available
        y = _y_1d(y).astype(float)
        pred = np.asarray(pred, dtype=float)
        if pred.ndim != 2:
            raise ValueError(
                "Quantile predictions must be 2D (n_samples, n_quantiles)."
            )

        quantiles = getattr(family_instance, "quantiles", None)
        if quantiles is None:
            raise ValueError(
                "Quantile default metric requires `self.family_.quantiles`."
            )
        q = np.asarray(quantiles, dtype=float)
        if pred.shape[1] != len(q):
            raise ValueError(
                f"Predictions have {pred.shape[1]} quantiles but family.quantiles has {len(q)} entries."
            )

        y2 = y[:, None]
        e = y2 - pred
        loss = np.maximum((q[None, :] - 1.0) * e, q[None, :] * e)
        return float(np.mean(np.sum(loss, axis=1)))

    def _quantile_median_mae(y, pred):
        y = _y_1d(y).astype(float)
        pred = np.asarray(pred, dtype=float)
        quantiles = getattr(family_instance, "quantiles", None)
        if quantiles is None:
            # fallback: use center column
            median_pred = pred[:, pred.shape[1] // 2]
            return float(np.mean(np.abs(y - median_pred)))

        q = list(map(float, quantiles))
        if 0.5 in q:
            idx = q.index(0.5)
        else:
            idx = int(np.argmin(np.abs(np.asarray(q) - 0.5)))
        return float(np.mean(np.abs(y - pred[:, idx])))

    default_metrics = {
        "normal": {
            "MSE": _normal_mse,
            "MAE": _normal_mae,
            "CRPS": _normal_crps,
        },
        "robustnormal": {
            "MSE": _normal_mse,
            "MAE": _normal_mae,
            "CRPS": _normal_crps,
        },
        "poisson": {
            # poisson_deviance accepts [rate] or 1D mean/rate
            "Poisson Deviance": poisson_deviance,
        },
        "gamma": {
            # gamma_deviance accepts transformed [shape, rate] directly
            "Gamma Deviance": gamma_deviance,
        },
        "beta": {
            "Beta Mean MSE": beta_mean_mse,
        },
        "dirichlet": {
            "Dirichlet Error": dirichlet_error,
        },
        "studentt": {
            # student_t_loss expects transformed [df, loc, scale]
            "Student-T NLL": student_t_loss,
        },
        "negativebinom": {
            # negative_binomial_deviance accepts transformed [mean,dispersion] directly
            "Negative Binomial Deviance": negative_binomial_deviance,
        },
        "inversegamma": {
            # inverse_gamma_loss expects transformed [shape, rate]
            "Inverse Gamma NLL": inverse_gamma_loss,
        },
        "categorical": {
            "Accuracy": lambda y, p: float(
                accuracy_score(
                    _categorical_labels(y), np.argmax(np.asarray(p), axis=1)
                )
            ),
        },
        "quantile": {
            "Pinball Loss": _quantile_pinball,
            "Median MAE": _quantile_median_mae,
        },
    }

    return default_metrics.get(family, {})


__all__ = ["default_metrics_for"]
