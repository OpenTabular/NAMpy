"""Boundary accuracy, derivative algebra and sparse storage regressions."""

import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.stats import poisson

from nampy.gam.extensions import fit_quantile
from nampy.gam.extensions._quantile_qp import quantile_problem
from nampy.gam.families.gamlss._taylor import Taylor
from nampy.gam.families.registry import make_gam_family


def test_nb_boundaries_match_high_precision_and_accurate_r_cases():
    fixture = json.loads(
        (
            Path(__file__).parents[1] / "reference_fixtures/extensions/neighbors.json"
        ).read_text()
    )
    data = fixture["inputs"]["nb_boundary"]
    mu, y, alpha = (np.asarray(data[name]) for name in ["mu", "y", "alpha"])
    family = make_gam_family("nb_lss")
    likelihood = family._log_density(
        y, Taylor(np.log(mu), 2, 4, 0), Taylor(np.log(alpha), 2, 4, 1)
    )
    expected = np.asarray(fixture["upstream"]["nb_high_precision"]["log_density"])
    assert_allclose(likelihood.value, expected, atol=3e-8, rtol=1e-10)
    assert_allclose(
        family.logpdf_from_parameters(y, np.column_stack([mu, alpha])),
        expected,
        atol=3e-8,
        rtol=1e-10,
    )
    r_values = np.asarray(fixture["upstream"]["nb_boundary"]["exact"])
    # R 4.6's dnbinom loses about 5e-5 in these three extreme tail cases.
    # Keep their high-precision checks strict, rather than weakening every
    # comparison or reproducing the upstream cancellation error.
    r_roundoff = (mu == 10000) & (alpha == 1e-12) & np.isin(y, [1, 5, 10])
    assert_allclose(
        likelihood.value[~r_roundoff], r_values[~r_roundoff], atol=3e-8, rtol=1e-10
    )
    assert np.max(np.abs(r_values[r_roundoff] - expected[r_roundoff])) < 6e-5
    for degree in range(1, 5):
        assert np.isfinite(likelihood.packed_derivatives(degree)).all()


@pytest.mark.parametrize("alpha", [1e-8, 1e-12, 1e-16])
def test_negative_binomial_poisson_limit_value_and_all_derivatives(alpha):
    family = make_gam_family("nb_lss")
    y = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    mu = np.full(len(y), 2.0)
    jet = family._log_density(
        y, Taylor(np.log(mu), 2, 4, 0), Taylor(np.full(len(y), np.log(alpha)), 2, 4, 1)
    )
    assert_allclose(
        jet.value, poisson.logpmf(y, mu), atol=3e-6 if alpha == 1e-8 else 3e-10
    )
    assert_allclose(
        family.logpdf_from_parameters(y, np.column_stack([mu, np.full(len(y), alpha)])),
        jet.value,
    )
    assert_allclose(jet.packed_derivatives(1)[:, 0], y - mu, atol=2e-6)
    for degree in range(1, 5):
        assert np.isfinite(jet.packed_derivatives(degree)).all()


@pytest.mark.parametrize(
    "log_alpha,count,mean",
    [
        (-36.0, 5, 2),
        (-16.0, 5, 2),
        (-5.0, 5, 2),
        (np.log(0.01) - 1e-7, 5, 2),
        (np.log(0.01) + 1e-7, 5, 2),
        (-5.2, 10000, 2),
        (-5.2, 5, 22000),
        (np.log(0.004), 60, 2),
        (np.log(0.004), 65, 2),
    ],
)
def test_nb_fourth_order_derivatives_against_high_precision(log_alpha, count, mean):
    mp = pytest.importorskip("mpmath")
    family = make_gam_family("nb_lss")
    eta = [np.log(mean), log_alpha]
    jet = family._log_density(
        np.array([float(count)]),
        Taylor(np.array([eta[0]]), 2, 4, 0),
        Taylor(np.array([eta[1]]), 2, 4, 1),
    )
    with mp.workdps(90):

        def density(a, b):
            mu, r = mp.exp(a), mp.exp(-b)
            return (
                mp.loggamma(r + count)
                - mp.loggamma(r)
                - mp.loggamma(count + 1)
                + r * (mp.log(r) - mp.log(r + mu))
                + count * (a - mp.log(r + mu))
            )

        for degree in range(1, 5):
            expected = []
            for term in combinations_with_replacement(range(2), degree):
                counts = (term.count(0), term.count(1))
                expected.append(
                    float(mp.diff(density, tuple(map(mp.mpf, eta)), counts))
                )
            assert_allclose(
                jet.packed_derivatives(degree)[0], expected, atol=3e-10, rtol=2e-8
            )


def test_taylor_algebra_identity_and_incompatible_spaces():
    x = Taylor(np.array([-0.2, 0, 0.2]), 3, 4, 0)
    restored = x.exp().log()
    for degree in range(1, 5):
        assert_allclose(
            restored.packed_derivatives(degree),
            x.packed_derivatives(degree),
            atol=2e-13,
        )
    y, z = Taylor(np.ones(3), 3, 4, 1), Taylor(np.ones(3), 3, 4, 2)
    polynomial = x * x * y * z
    assert polynomial.packed_derivatives(4)[0, -1] == 0
    idx = list(combinations_with_replacement(range(3), 4)).index((0, 0, 1, 2))
    assert_allclose(polynomial.packed_derivatives(4)[:, idx], 2)
    with pytest.raises(ValueError):
        x + Taylor(np.ones(3), 2, 4)
    with pytest.raises(ValueError):
        x * Taylor(np.ones(3), 3, 2)


def test_exact_quantile_sparse_problem_has_linear_observation_storage():
    sizes = []
    for n in [1000, 10000]:
        x = np.linspace(-1, 1, n)
        X = np.column_stack([np.ones(n), x])
        P, _, A, _ = quantile_problem(X, x, np.eye(2), np.ones(n), 0.7)
        assert sparse.isspmatrix_csc(A) and sparse.isspmatrix_csc(P)
        assert A.shape == (2 * n, n + 2)
        assert A.nnz <= 6 * n
        sizes.append(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)
    assert sizes[1] <= 10 * sizes[0] + 100
    # End-to-end exact fitting at the size that formerly needed a 1.6GB A.
    fit = fit_quantile(X, 0.3 + 0.7 * x, penalty=np.diag([0.0, 0.2]), level=0.7)
    assert fit.converged
    assert_allclose(fit.fitted, 0.3 + 0.7 * x, atol=1e-7)
    assert fit.optimality < 1e-6
