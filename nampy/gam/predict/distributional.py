"""Natural-parameter uncertainty for statistical distributional predictions."""

import numpy as np


def parameter_standard_errors(X, jj, covariance, jacobian, *, eta_se=None):
    """Transport coefficient covariance through diagonal or coupled parameters.

    Predictor blocks may share coefficients. A full Jacobian has shape
    (n, natural_parameters, predictors); diagonal transforms use (n, predictors).
    This helper owns the common transport for extension-family prediction and
    the public statistical natural-parameter uncertainty adapter.
    """
    X, covariance, jacobian = (
        np.asarray(value, dtype=float) for value in (X, covariance, jacobian)
    )
    if jacobian.ndim == 2:
        if jacobian.shape != (len(X), len(jj)):
            raise ValueError(
                "Diagonal parameter Jacobian must have shape (n, predictors)."
            )
        if eta_se is None:
            eta_se = np.column_stack(
                [
                    np.sqrt(
                        np.maximum(
                            np.einsum(
                                "ij,jk,ik->i",
                                X[:, cols],
                                covariance[np.ix_(cols, cols)],
                                X[:, cols],
                            ),
                            0,
                        )
                    )
                    for cols in jj
                ]
            )
        return np.abs(jacobian) * np.asarray(eta_se, dtype=float)
    if (
        jacobian.ndim != 3
        or jacobian.shape[0] != len(X)
        or jacobian.shape[2] != len(jj)
    ):
        raise ValueError(
            "Coupled parameter Jacobian must have shape (n, parameters, predictors)."
        )
    gradient = np.zeros((X.shape[0], jacobian.shape[1], X.shape[1]))
    for k, indices in enumerate(jj):
        gradient[:, :, indices] += jacobian[:, :, k, None] * X[:, None, indices]
    variance = np.einsum("npi,ij,npj->np", gradient, covariance, gradient)
    return np.sqrt(np.maximum(variance, 0))


def distributional_standard_errors(
    model,
    X=None,
    *,
    cov="bayes",
    offset=None,
    block_size=None,
    newdata_guaranteed=False,
    na_action="pass",
    unconditional=False,
):
    """Transport link covariance through the family-owned parameter Jacobian.

    Coupled parameters retain cross-predictor covariance. Operation ordering
    follows the existing general-family prediction transport.
    """
    eta, eta_se = model.predict(
        X,
        return_se=True,
        type="link",
        offset=offset,
        block_size=block_size,
        newdata_guaranteed=newdata_guaranteed,
        na_action=na_action,
        unconditional=unconditional,
    )
    jacobian = model.family.distribution_parameter_jacobian(eta)
    if np.asarray(jacobian).ndim == 3:
        # Coupled natural parameters (multinom probabilities) require the
        # cross-predictor covariance, not independent marginal eta SEs.
        from ..model_state import _predictor_full_indices
        from .general import _general_family_covariance_for_prediction
        from .predictions import select_prediction_covariance_matrix

        P = np.asarray(
            model.predict(
                X,
                type="lpmatrix",
                offset=offset,
                block_size=block_size,
                newdata_guaranteed=newdata_guaranteed,
                na_action=na_action,
            ),
            dtype=float,
        )
        covariance = select_prediction_covariance_matrix(
            model, cov=cov, unconditional=unconditional
        )
        covariance = _general_family_covariance_for_prediction(model, covariance)
        return parameter_standard_errors(
            P, _predictor_full_indices(model), covariance, jacobian
        )
    return np.abs(jacobian) * np.asarray(eta_se, dtype=np.float64)
