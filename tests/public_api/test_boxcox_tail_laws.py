"""BCT/BCPE normalized probabilities, upstream semantics and support limits."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, special, stats

from nampy.distributions import AtomMixtureLaw, BCCGLaw, BCPELaw, BCTLaw, DistributionPrediction
from nampy.diagnostics import prepare_centiles, prepare_worm_plot

FIXTURES = Path(__file__).parents[1] / "reference_fixtures" / "boxcox_tail"


def distribution(law, mu=1.0, sigma=0.6, nu=0.7, tau=3.0):
    parameters = np.broadcast_arrays(*[np.atleast_1d(v) for v in [mu, sigma, nu, tau]])
    return DistributionPrediction(
        law(), dict(zip(law.spec.parameter_names, parameters, strict=True))
    )


@pytest.mark.parametrize("law", [BCTLaw, BCPELaw])
def test_saved_gamlss_probability_references(law):
    frame = pd.read_csv(FIXTURES / f"{law.spec.name}.csv")
    dist = distribution(law, frame.mu, frame.sigma, frame.nu, frame.tau)
    assert_allclose(dist.log_prob(frame.y), frame.log_prob, atol=4e-12, rtol=3e-12)
    assert_allclose(dist.cdf(frame.y), frame.cdf, atol=4e-14, rtol=2e-12)
    assert_allclose(dist.survival(frame.y), frame.survival, atol=4e-14, rtol=2e-12)
    assert_allclose(dist.log_cdf(frame.y), frame.log_cdf, atol=4e-12, rtol=2e-12)
    assert_allclose(
        dist.log_survival(frame.y), frame.log_survival, atol=4e-12, rtol=2e-12
    )
    for row, p in enumerate(frame.probability):
        assert_allclose(
            dist.quantile(p)[row, 0], frame["quantile"][row], rtol=5e-12, atol=5e-12
        )


@pytest.mark.parametrize("law", [BCTLaw, BCPELaw])
@pytest.mark.parametrize("nu", [-2.0, -0.3, 0.0, 0.3, 2.0])
def test_density_normalizes_with_jacobian_and_positive_support_normalizer(law, nu):
    tau = 5.0 if law is BCTLaw else 1.3
    dist = distribution(law, mu=2.0, sigma=1.4, nu=nu, tau=tau)

    def integrand(log_y):
        return np.exp(dist.log_prob(np.exp(log_y))[0] + log_y)

    value, error = integrate.quad(
        integrand, -700, 700, points=[np.log(2)], epsabs=2e-10, epsrel=2e-10, limit=300
    )
    assert error < 2e-8
    assert_allclose(value, 1, atol=3e-9)


@pytest.mark.parametrize("law", [BCTLaw, BCPELaw])
@pytest.mark.parametrize("nu", [-2.0, -0.3, 0.0, 0.3, 2.0])
def test_cdf_quantile_inverse_and_support_endpoints(law, nu):
    dist = distribution(
        law, mu=1.2, sigma=1.1, nu=nu, tau=5.0 if law is BCTLaw else 1.3
    )
    p = np.array([1e-4 if nu == 0 else 1e-20, 1e-3, 0.01, 0.3, 0.5, 0.9, 1 - 1e-10])
    if nu == 0:
        p[-1] = 1 - 1e-4
    y = dist.quantile(p)[0]
    repeated = distribution(
        law,
        mu=np.full(len(p), 1.2),
        sigma=1.1,
        nu=nu,
        tau=5.0 if law is BCTLaw else 1.3,
    )
    assert np.all(np.diff(y) >= 0)
    assert_allclose(
        repeated.log_cdf(y)[p <= 0.5], np.log(p[p <= 0.5]), atol=3e-10, rtol=2e-12
    )
    assert_allclose(
        repeated.log_survival(y)[p > 0.5], np.log1p(-p[p > 0.5]), atol=3e-10, rtol=2e-12
    )
    assert_array_equal(dist.quantile([0, 1]), [[0, np.inf]])
    assert dist.quantile([]).shape == (1, 0)
    for y, cdf in [(-np.inf, 0), (-1, 0), (0, 0), (np.inf, 1)]:
        assert_array_equal(dist.cdf(y), [cdf])
        assert_array_equal(dist.cdf_left(y), [cdf])
        assert_array_equal(dist.survival(y), [1 - cdf])
        assert dist.log_prob(y)[0] == -np.inf
    assert np.isnan(dist.log_prob(np.nan)[0])


@pytest.mark.parametrize("nu", [-3.0, -0.3, 0.0, 0.3, 2.0])
def test_bcpe_tau_two_is_the_existing_bccg_law(nu):
    y = np.array([0.001, 0.2, 1.0, 2.0, 10.0, 50.0])
    parameters = {
        "mu": np.full(len(y), 1.2),
        "sigma": np.full(len(y), 0.7),
        "nu": np.full(len(y), nu),
    }
    reference = DistributionPrediction(BCCGLaw(), parameters)
    dist = DistributionPrediction(
        BCPELaw(), {**parameters, "tau": np.full(len(y), 2.0)}
    )
    for operation in [
        "log_prob",
        "cdf",
        "cdf_left",
        "log_cdf",
        "survival",
        "log_survival",
    ]:
        assert_allclose(
            getattr(dist, operation)(y),
            getattr(reference, operation)(y),
            atol=4e-11,
            rtol=3e-12,
        )
    assert_allclose(
        dist.quantile([0.001, 0.2, 0.5, 0.99]),
        reference.quantile([0.001, 0.2, 0.5, 0.99]),
        atol=3e-11,
        rtol=4e-12,
    )


@pytest.mark.parametrize("law,tau", [(BCTLaw, 5.0), (BCPELaw, 1.3)])
@pytest.mark.parametrize("nu", [-1e-12, 0.0, 1e-12, 1e-300])
def test_zero_power_log_latent_law_and_removable_limit(law, tau, nu):
    y = np.array([0.2, 0.8, 1.2, 4.0])
    mu, sigma = 1.2, 0.6
    z = np.log(y / mu) / sigma
    latent = (
        stats.t(tau)
        if law is BCTLaw
        else stats.gennorm(
            tau,
            scale=np.exp(0.5 * (special.gammaln(1 / tau) - special.gammaln(3 / tau))),
        )
    )
    dist = distribution(law, mu=np.full(len(y), mu), sigma=sigma, nu=nu, tau=tau)
    assert_allclose(
        dist.log_prob(y), latent.logpdf(z) - np.log(sigma * y), atol=2e-11, rtol=2e-11
    )
    assert_allclose(dist.cdf(y), latent.cdf(z), atol=2e-12)
    p = np.array([0.01, 0.5, 0.99])
    assert_allclose(
        dist.quantile(p),
        np.broadcast_to(mu * np.exp(sigma * latent.ppf(p)), (len(y), len(p))),
        atol=3e-10,
        rtol=2e-11,
    )


@pytest.mark.parametrize("law", [BCTLaw, BCPELaw])
@pytest.mark.parametrize(
    "nu,y", [(1.0, 1e-200), (-1.0, 1e200), (2.0, 1e-100), (-2.0, 1e100)]
)
def test_boundary_log_tails_against_high_precision_latent_interval(law, nu, y):
    mp = pytest.importorskip("mpmath")
    tau, sigma = (4.0 if law is BCTLaw else 1.5), 1.1
    with mp.workdps(260):
        ss, nn, tt, yy = [mp.mpf(str(v)) for v in [sigma, nu, tau, y]]
        a = 1 / (ss * abs(nn))
        delta = yy**nn / (ss * abs(nn))
        if law is BCTLaw:

            def pdf(z):
                return (
                    mp.gamma((tt + 1) / 2)
                    / (mp.sqrt(tt * mp.pi) * mp.gamma(tt / 2))
                    * (1 + z * z / tt) ** (-(tt + 1) / 2)
                )

            c = (
                1
                - mp.betainc(
                    tt / 2, mp.mpf(".5"), 0, tt / (tt + a * a), regularized=True
                )
                / 2
            )
        else:
            alpha = mp.sqrt(mp.gamma(1 / tt) / mp.gamma(3 / tt))

            def pdf(z):
                return (
                    tt
                    / (2 * alpha * mp.gamma(1 / tt))
                    * mp.exp(-((abs(z) / alpha) ** tt))
                )

            c = (
                1
                - mp.gammainc(1 / tt, (a / alpha) ** tt, mp.inf) / mp.gamma(1 / tt) / 2
            )
        mass = delta * mp.quad(lambda s: pdf(-a + delta * s), [0, 1])
        expected = float(mp.log(mass) - mp.log(c))
    dist = distribution(law, sigma=sigma, nu=nu, tau=tau)
    actual = dist.log_cdf(y) if nu > 0 else dist.log_survival(y)
    assert_allclose(actual, expected, atol=3e-12, rtol=3e-14)


def test_bct_preserves_student_density_above_upstream_gaussian_switch():
    frame = pd.read_csv(FIXTURES / "bct_density_switch.csv").iloc[0]
    assert frame.log_prob == frame.gaussian_log_prob
    dist = distribution(BCTLaw, nu=0.0, sigma=1.0, tau=frame.tau)
    exact = stats.t.logpdf(100, frame.tau) - 100
    assert_allclose(dist.log_prob(frame.y), exact, atol=3e-10)
    assert abs(exact - frame.log_prob) > 1
    assert np.isfinite(dist.log_survival(frame.y)).all()


def test_bct_log_density_handles_an_unrepresentable_transformed_latent_value():
    y, tau, nu = 1e300, 5.0, 2.0
    lr = np.log(y)
    logz = nu * lr + np.log1p(-np.exp(-nu * lr)) - np.log(nu)
    expected = (
        (nu - 1) * lr
        - 0.5 * np.log(tau)
        - special.betaln(tau / 2, 0.5)
        - (tau + 1) * (logz - 0.5 * np.log(tau))
        - stats.t.logcdf(1 / nu, tau)
    )
    assert_allclose(
        distribution(BCTLaw, sigma=1.0, nu=nu, tau=tau).log_prob(y),
        expected,
        atol=3e-12,
    )
    assert np.isfinite(
        distribution(BCTLaw, sigma=1.0, nu=nu, tau=tau).log_survival(y)
    ).all()


def test_bcpe_log_survival_is_not_the_log_of_an_underflowed_probability():
    dist = distribution(BCPELaw, nu=0.0, sigma=1.0, tau=2.0)
    assert dist.survival(np.exp(50))[0] == 0
    assert_allclose(dist.log_survival(np.exp(50)), stats.norm.logsf(50), atol=3e-12)


@pytest.mark.parametrize("law", [BCTLaw, BCPELaw])
def test_sampling_missing_rows_generic_diagnostics_and_capabilities(law):
    dist = distribution(
        law,
        mu=[1.0, 2.0, 3.0, np.nan],
        sigma=[0.3, 0.2, 0.4, np.nan],
        nu=[0.0, 1.0, -2.0, np.nan],
        tau=[5.0, 2.0, 3.0, np.nan],
    )
    draws = dist.sample(1500, random_state=71)
    assert draws.shape == (1500, 4)
    assert_array_equal(draws, dist.sample(1500, random_state=71))
    assert np.isnan(draws[:, -1]).all() and np.all(draws[:, :3] > 0)
    medians = dist.quantile(0.5)[:, 0]
    assert_allclose(np.mean(draws[:, :3] <= medians[:3], axis=0), 0.5, atol=0.04)
    assert np.isnan(dist.cdf(1)[-1])
    residual = dist.quantile_residuals(medians)
    assert_allclose(residual.residuals[:3], 0, atol=2e-11)
    assert len(prepare_worm_plot(residual).panels) == 1
    assert prepare_centiles(dist, [0, 1, 2, 3]).quantiles.shape == (4, 3)
    assert {"mean", "variance"} <= dist.spec.operations
    assert dist.sample(0, random_state=0).shape == (0, 4)
    mixed = AtomMixtureLaw(law(), (0.0,), ("p0",))
    assert {"cdf", "mean", "variance"} <= mixed.spec.operations


@pytest.mark.parametrize("law", [BCTLaw, BCPELaw])
@pytest.mark.parametrize("factor", [1e-200, 1e200])
def test_response_units_keep_probabilities_and_jacobian(law, factor):
    mu, nu, y = (
        np.array([1.0, 2.0, 3.0]),
        np.array([-1.0, 0.0, 2.0]),
        np.array([0.2, 2.1, 7.0]),
    )
    base = distribution(law, mu=mu, nu=nu)
    scaled = distribution(law, mu=mu * factor, nu=nu)
    assert_allclose(
        scaled.log_prob(y * factor), base.log_prob(y) - np.log(factor), atol=4e-12
    )
    assert_allclose(scaled.log_cdf(y * factor), base.log_cdf(y), atol=4e-12)
    assert_allclose(
        scaled.quantile([0.1, 0.5, 0.9]) / factor,
        base.quantile([0.1, 0.5, 0.9]),
        rtol=4e-12,
    )


@pytest.mark.parametrize("law", [BCTLaw, BCPELaw])
@pytest.mark.parametrize(
    "name,value",
    [("mu", 0), ("sigma", -1), ("tau", 0), ("tau", np.inf), ("nu", np.inf)],
)
def test_invalid_parameters_are_not_clipped(law, name, value):
    with pytest.raises(ValueError):
        distribution(law, **{name: value})


def test_statistical_fitting_registration_and_mgcv_bcg_remain_distinct():
    from nampy.gam.families.registry import make_gam_family

    for name in ["bct", "bcpe"]:
        with pytest.raises(ValueError, match="Unknown GAM family"):
            make_gam_family(name)
        assert make_gam_family(name + "_lss").name == name + "_lss"
    assert make_gam_family("bcg").name == "bcg"


@pytest.mark.parametrize("nu,p", [(10.0, 1 - 1e-12), (-10.0, 1e-200)])
def test_bct_small_df_inverse_keeps_unrepresentable_latent_quantiles(nu, p):
    dist = distribution(BCTLaw, nu=nu, sigma=0.6, tau=0.1)
    value = dist.quantile(p)[0, 0]
    assert 0 < value < np.inf
    actual = dist.log_cdf(value) if p < 0.5 else dist.log_survival(value)
    expected = np.log(p) if p < 0.5 else np.log1p(-p)
    assert_allclose(actual, expected, atol=3e-10)


def test_bcpe_large_shape_keeps_finite_latent_quantiles_when_gamma_quantile_underflows():
    mp = pytest.importorskip("mpmath")
    tau = 1000.0
    source = pd.read_csv(FIXTURES / "bcpe_gamma_underflow.csv").iloc[0]
    assert source.cdf == 0.5 and source["quantile"] == 1.0
    # nu=0 exposes the latent PE directly on log-response scale.
    y = np.exp(0.02)
    dist = distribution(BCPELaw, nu=0.0, sigma=1.0, tau=tau)
    with mp.workdps(80):
        t = mp.mpf(tau)
        alpha = mp.sqrt(mp.gamma(1 / t) / mp.gamma(3 / t))
        x = (mp.mpf(".02") / alpha) ** t
        expected = float((1 + mp.gammainc(1 / t, 0, x) / mp.gamma(1 / t)) / 2)
    assert_allclose(dist.cdf(y), expected, atol=3e-14)
    p = [0.49, 0.5, 0.51]
    values = dist.quantile(p)[0]
    repeated = distribution(BCPELaw, mu=np.ones(3), nu=0.0, sigma=1.0, tau=tau)
    assert np.all(np.diff(values) > 0)
    assert_allclose(repeated.cdf(values), p, atol=3e-13)


@pytest.mark.parametrize("law", [BCTLaw, BCPELaw])
def test_recorded_upstream_support_and_boundary_tail_quirks_are_not_copied(law):
    frame = pd.read_csv(FIXTURES / f"{law.spec.name}_quirks.csv")
    assert_array_equal(frame.log_prob[:2], [0, 0])
    if law is BCTLaw:
        assert_array_equal(frame.log_cdf[:2], [0, 0])
    else:
        assert np.isnan(frame.log_cdf[0]) and np.isneginf(frame.log_cdf[1])
        assert frame.log_survival[0] > 0  # Upstream reports survival > 1.
    assert np.isneginf(frame.log_cdf[2]) and np.isneginf(frame.log_survival[3])
    dist = distribution(law, frame.mu, frame.sigma, frame.nu, frame.tau)
    assert np.isneginf(dist.log_prob(frame.y)[:2]).all()
    assert np.isneginf(dist.log_cdf(frame.y)[:2]).all()
    assert np.isfinite(dist.log_cdf(frame.y)[2])
    assert np.isfinite(dist.log_survival(frame.y)[3])


def test_bcpe_tau_one_is_log_laplace_with_unit_latent_variance():
    y = np.exp(np.array([-3.0, -0.3, 0.0, 0.3, 3.0]))
    dist = distribution(BCPELaw, mu=np.ones(len(y)), sigma=1.0, nu=0.0, tau=1.0)
    reference = stats.laplace(scale=1 / np.sqrt(2))
    assert_allclose(
        dist.log_prob(y), reference.logpdf(np.log(y)) - np.log(y), atol=2e-14
    )
    assert_allclose(dist.cdf(y), reference.cdf(np.log(y)), atol=2e-15)


def test_student_far_tail_log_survival_matches_independent_scaled_integral():
    mp = pytest.importorskip("mpmath")
    df = 2e6
    with mp.workdps(80):
        d = mp.mpf(df)
        constant = mp.loggamma((d + 1) / 2) - mp.loggamma(d / 2) - mp.log(d * mp.pi) / 2

        def density(z):
            return constant - (d + 1) / 2 * mp.log1p(z * z / d)

        peak = density(100)
        mass = mp.quad(lambda v: mp.exp(density(100 + v / 100) - peak), [0, mp.inf])
        expected = float(peak - mp.log(100) + mp.log(mass))
    actual = distribution(BCTLaw, sigma=1.0, nu=0.0, tau=df).log_survival(np.exp(100))
    assert_allclose(actual, expected, atol=3e-10, rtol=2e-13)


@pytest.mark.parametrize("law", [BCTLaw, BCPELaw])
def test_probability_laws_compose_with_the_generic_atom_interface(law):
    base = distribution(law)
    mixed = DistributionPrediction(
        AtomMixtureLaw(law(), (0.0,), ("p0",)), {**dict(base.parameters), "p0": [0.2]}
    )
    assert_allclose(mixed.cdf(1), 0.2 + 0.8 * base.cdf(1), atol=3e-15)
    assert_allclose(mixed.log_prob(1), np.log(0.8) + base.log_prob(1), atol=3e-15)
    assert_array_equal(mixed.cdf_left(0), [0])
    assert_array_equal(mixed.cdf(0), [0.2])
    assert_allclose(mixed.quantile(0.5), base.quantile(0.375), rtol=2e-12)


def test_failed_boundary_integration_is_explicit_not_a_substitute_law(monkeypatch):
    from nampy.distributions import _boxcox_latent

    monkeypatch.setattr(_boxcox_latent, "quad", lambda *args, **kwargs: (1.0, 1.0))
    with pytest.raises(NotImplementedError, match="certify accuracy"):
        distribution(BCTLaw, nu=1.0).log_cdf(1e-200)


def test_unrepresentable_bcpe_latent_scale_is_explicit():
    with pytest.raises(NotImplementedError, match="scale.*range"):
        distribution(BCPELaw, tau=0.001)
