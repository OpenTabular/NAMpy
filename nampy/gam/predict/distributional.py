"""Natural-parameter uncertainty for statistical distributional predictions."""

import numpy as np


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
        gradient = np.zeros((P.shape[0], jacobian.shape[1], P.shape[1]))
        for k, indices in enumerate(_predictor_full_indices(model)):
            gradient[:, :, indices] += jacobian[:, :, k, None] * P[:, None, indices]
        variance = np.einsum("npi,ij,npj->np", gradient, covariance, gradient)
        return np.sqrt(np.maximum(variance, 0))
    return np.abs(jacobian) * np.asarray(eta_se, dtype=np.float64)
