import numpy as np

from nampy.utils.distributional_metrics import (
    beta_brier_score,
    dirichlet_error,
    gamma_deviance,
    inverse_gamma_loss,
    negative_binomial_deviance,
    poisson_deviance,
    student_t_loss,
)


def test_distributional_metrics_finite():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.2, 1.9, 3.1])

    assert np.isfinite(poisson_deviance(y_true, y_pred))
    assert np.isfinite(gamma_deviance(y_true, y_pred))

    y_beta = np.array([0.2, 0.5, 0.8])
    assert np.isfinite(beta_brier_score(y_beta, y_beta + 0.01))

    y_dirichlet = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
    y_dirichlet_pred = np.array([[0.21, 0.31, 0.48], [0.1, 0.35, 0.55]])
    assert np.isfinite(dirichlet_error(y_dirichlet, y_dirichlet_pred))

    y_student = np.array([0.5, -0.2, 1.0])
    y_student_pred = np.column_stack([y_student, np.full_like(y_student, 1.0)])
    assert np.isfinite(student_t_loss(y_student, y_student_pred))

    y_nb = np.array([1.0, 2.0, 3.0])
    y_nb_pred = np.array([1.1, 1.9, 3.2])
    assert np.isfinite(negative_binomial_deviance(y_nb, y_nb_pred, alpha=0.5))

    y_inv = np.column_stack([np.full(3, 2.0), np.full(3, 1.0)])
    assert np.isfinite(inverse_gamma_loss(y_true, y_inv))
