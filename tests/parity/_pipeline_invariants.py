"""Compare observable functions/operators and scalar statistics across bases."""

import numpy as np

from tests.mgcv_invariant_policy import stable_column_space_projector


def predictor_evaluation(model, data):
    P = model.lpmatrix(data)
    matrices = []
    for indices in model.gam_result_.compiled_model.predictor_full_indices:
        block = np.zeros_like(P)
        block[:, indices] = P[:, indices]
        matrices.append(block)
    return np.vstack(matrices)


def assert_reference_invariants(model, data, newdata, expected, *, tolerance=2e-7):
    fit = model.fit_result()
    actual = {
        "edf": fit.edf_total,
        "edf1": float(np.sum(model.edf1())),
        "aic": model.aic(),
        "bic": model.bic(),
        "loglik": model.loglik(),
        "likelihood_df": model._loglik_effective_df(),
        "deviance": fit.deviance,
        "scale": fit.scale,
    }
    for name, value in actual.items():
        if name == "edf1" and expected[name] is None:
            continue
        if expected[name] is None:
            assert np.isnan(value), (name, value)
        else:
            np.testing.assert_allclose(
                value, expected[name], atol=tolerance, rtol=tolerance, err_msg=name
            )
    np.testing.assert_allclose(
        model.predict(data, type="link"),
        expected["fitted"],
        atol=tolerance,
        rtol=tolerance,
        err_msg="training link",
    )
    link, se = model.predict(newdata, type="link", return_se=True)
    np.testing.assert_allclose(
        link, expected["eta"], atol=tolerance, rtol=tolerance, err_msg="newdata link"
    )
    np.testing.assert_allclose(
        se,
        expected["se"],
        atol=tolerance,
        rtol=tolerance,
        err_msg="link standard errors",
    )
    np.testing.assert_allclose(
        model.predict(newdata, type="response"),
        expected["response"],
        atol=tolerance,
        rtol=tolerance,
        err_msg="response",
    )
    E = predictor_evaluation(model, newdata)
    np.testing.assert_allclose(
        E @ model.vcov() @ E.T,
        expected["covariance_kernel"],
        atol=tolerance,
        rtol=tolerance,
        err_msg="joint prediction covariance kernel",
    )
    if expected["frequentist_kernel"] is not None:
        np.testing.assert_allclose(
            E @ model.vcov(freq=True) @ E.T,
            expected["frequentist_kernel"],
            atol=tolerance,
            rtol=tolerance,
            err_msg="frequentist prediction kernel",
        )
    if expected["unconditional_kernel"] is not None:
        np.testing.assert_allclose(
            E @ model.vcov(unconditional=True) @ E.T,
            expected["unconditional_kernel"],
            atol=tolerance,
            rtol=tolerance,
            err_msg="smoothing-adjusted prediction kernel",
        )
    if expected["theta"] is not None:
        np.testing.assert_allclose(
            model.family.getTheta(True),
            expected["theta"],
            atol=tolerance,
            rtol=tolerance,
            err_msg="physical nuisance parameters",
        )
    np.testing.assert_allclose(
        stable_column_space_projector(predictor_evaluation(model, data)),
        expected["design_projector"],
        atol=2e-8,
        rtol=2e-8,
        err_msg="identified function space",
    )
    for name, value in expected["residuals"].items():
        if value is not None:
            np.testing.assert_allclose(
                model.residuals(type=name),
                value,
                atol=tolerance,
                rtol=tolerance,
                err_msg=f"{name} residuals",
            )
