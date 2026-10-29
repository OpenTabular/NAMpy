"""Mixed CDFs, conditional quantiles and reusable capability boundaries."""

import pickle
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate, stats

from nampy.distributions import AtomMixtureLaw, BetaLaw, DistributionPrediction, GammaLaw, NegativeBinomialLaw, NormalLaw, StudentTLaw
from nampy.diagnostics import prepare_centiles


def test_interior_and_exterior_atoms_normalization_jumps_and_quantile_inversion():
    law = AtomMixtureLaw(NormalLaw(), (-2.0, 1.0), ("p0", "p1"))
    dist = DistributionPrediction(
        law,
        {"loc": [0.0, 3.0], "scale": [1.0, 0.5], "p0": [0.2, 0.1], "p1": [0.3, 0.25]},
    )
    for atom, name in zip(law.atoms, law.mass_names, strict=True):
        assert_allclose(
            dist.cdf(atom) - dist.cdf_left(atom), dist.parameters[name], atol=1e-15
        )
        assert_allclose(np.exp(dist.log_prob(atom)), dist.parameters[name])
    assert_allclose(
        dist.cdf(0), [0.2 + 0.5 * 0.5, 0.1 + 0.65 * stats.norm.cdf(0, 3, 0.5)]
    )
    assert_allclose(dist.cdf(2) + dist.survival(2), 1)
    assert_allclose(np.exp(dist.log_cdf(2)), dist.cdf(2))
    assert_allclose(np.exp(dist.log_survival(2)), dist.survival(2))
    probabilities = np.linspace(0.001, 0.999, 101)
    quantiles = dist.quantile(probabilities)
    for j, p in enumerate(probabilities):
        assert np.all(dist.cdf(quantiles[:, j]) >= p - 1e-14)
        assert np.all(dist.cdf_left(quantiles[:, j]) <= p + 1e-14)
    assert np.isneginf(dist.quantile(0)).all()
    assert np.isposinf(dist.quantile(1)).all()
    # Integrate the continuous component away from its atoms.
    one = DistributionPrediction(
        law, {name: value[:1] for name, value in dist.parameters.items()}
    )
    mass = sum(
        integrate.quad(lambda y: np.exp(one.log_prob(y)[0]), lo, hi)[0]
        for lo, hi in [(-np.inf, -2), (-2, 1), (1, np.inf)]
    )
    assert_allclose(mass + 0.2 + 0.3, 1, atol=2e-11)


def test_beta_with_two_boundary_atoms_moments_samples_and_immutable_snapshots():
    law = AtomMixtureLaw(BetaLaw(), (0.0, 1.0), ("pi0", "pi1"), name="inflated_beta")
    params = {
        "alpha": [2.0, 3.0],
        "beta": [4.0, 2.0],
        "pi0": [0.2, 0],
        "pi1": [0.3, 1.0],
    }
    dist = DistributionPrediction(law, params)
    assert_allclose(dist.mean(), [0.5 / 3 + 0.3, 1])
    expected_var = (
        0.5 * (2 * 4 / (6**2 * 7) + (1 / 3 - dist.mean()[0]) ** 2)
        + 0.2 * dist.mean()[0] ** 2
        + 0.3 * (1 - dist.mean()[0]) ** 2
    )
    assert_allclose(dist.variance(), [expected_var, 0])
    draws = dist.sample(n_draws=30000, random_state=19)
    assert_allclose(draws.mean(axis=0), dist.mean(), atol=0.006)
    assert_allclose(draws.var(axis=0), dist.variance(), atol=0.006)
    assert_allclose((draws == 0).mean(axis=0), [0.2, 0], atol=0.006)
    assert_allclose((draws == 1).mean(axis=0), [0.3, 1], atol=0.006)
    assert_allclose(dist.quantile([0, 0.1, 0.8, 1])[1], 1)
    assert_allclose(dist.quantile([0, 1])[0], [0, 1])
    params["pi0"][0] = 0.9
    assert dist.parameters["pi0"][0] == 0.2
    restored = pickle.loads(pickle.dumps(dist))
    assert_allclose(
        restored.quantile([0, 0.2, 0.5, 1]), dist.quantile([0, 0.2, 0.5, 1])
    )
    assert not restored.parameters["pi0"].flags.writeable


def test_zero_gamma_tails_and_pure_component_limits():
    law = AtomMixtureLaw(GammaLaw(), (0.0,), ("pi0",))
    dist = DistributionPrediction(
        law, {"shape": [2.0, 2.0, 2.0], "rate": [1.0, 1.0, 1.0], "pi0": [0.3, 0.0, 1.0]}
    )
    assert_allclose(dist.log_prob(0), [np.log(0.3), -np.inf, 0])
    assert_allclose(dist.cdf_left(0), 0)
    assert_allclose(dist.cdf(0), [0.3, 0, 1])
    assert_allclose(dist.mean(), [1.4, 2, 0])
    assert_allclose(dist.variance(), [0.7 * 2 + 0.3 * 0.7 * 4, 2, 0])
    assert_allclose(dist.quantile([0, 0.3, 0.9, 1])[[0, 1], 0], 0)
    assert_allclose(dist.quantile([0, 0.3, 0.9, 1])[2], 0)
    assert_allclose(
        dist.log_survival([1000, 1000, 1])[:2],
        stats.gamma.logsf(1000, 2) + np.log([0.7, 1]),
        equal_nan=True,
    )
    # Use a normal base to assert a finite tail beyond ordinary CDF/SF range.
    normal = DistributionPrediction(
        AtomMixtureLaw(NormalLaw(), (0.0,), ("p",)),
        {"loc": [0.0], "scale": [1.0], "p": [0.4]},
    )
    assert_allclose(normal.log_survival(40), stats.norm.logsf(40) + np.log(0.6))
    assert_allclose(normal.log_cdf(-40), stats.norm.logcdf(-40) + np.log(0.6))


def test_mixed_residuals_and_centiles_require_no_family_specific_dispatch():
    dist = DistributionPrediction(
        AtomMixtureLaw(GammaLaw(), (0.0,), ("pi0",)),
        {"shape": [2.0, 3.0, 4.0], "rate": [1.0, 2.0, 3.0], "pi0": [0.2, 0.3, 0.4]},
    )
    y = [0.0, 1.0, 0.0]
    residual = dist.quantile_residuals(y, uniforms=[0.25, 0.5, 0.75])
    pit = dist.cdf_left(y) + np.array([0.25, 0.5, 0.75]) * (
        dist.cdf(y) - dist.cdf_left(y)
    )
    assert_allclose(residual.residuals, stats.norm.ppf(pit))
    assert_allclose(
        prepare_centiles(dist, [0, 1, 2]).quantiles, dist.quantile([0.05, 0.5, 0.95])
    )


def test_missing_rows_and_operation_capabilities():
    class CdfOnly:
        spec = replace(NormalLaw.spec, operations=frozenset({"cdf"}))

        def validate_parameters(self, parameters, *, context):
            NormalLaw().validate_parameters(parameters, context=context)

        def evaluate(self, operation, parameters, **kwargs):
            return NormalLaw().evaluate(operation, parameters, **kwargs)

    law = AtomMixtureLaw(CdfOnly(), (0.0,), ("p",))
    assert law.spec.operations == {"cdf", "cdf_left"}
    dist = DistributionPrediction(
        law, {"loc": [np.nan, 0], "scale": [np.nan, 1], "p": [np.nan, 0.3]}
    )
    assert_allclose(dist.cdf(0), [np.nan, 0.65])
    with pytest.raises(NotImplementedError, match="quantile"):
        dist.quantile(0.5)
    for base in [NegativeBinomialLaw(), law]:
        with pytest.raises(NotImplementedError, match="continuous"):
            AtomMixtureLaw(base, (0.0,), ("p",))


def test_pure_atom_does_not_inherit_undefined_base_moments():
    law = AtomMixtureLaw(StudentTLaw(), (2.0,), ("p",))
    dist = DistributionPrediction(
        law, {"df": [0.5], "loc": [0.0], "scale": [1.0], "p": [1.0]}
    )
    assert_allclose(dist.mean(), 2)
    assert_allclose(dist.variance(), 0)


def test_explicit_continuous_weight_preserves_tiny_density_and_log_tail():
    law = AtomMixtureLaw(NormalLaw(), (0.0,), ("p",), continuous_mass_name="pc")
    dist = DistributionPrediction(
        law, {"loc": [0.0], "scale": [1.0], "p": [1.0], "pc": [1e-24]}
    )
    assert_allclose(dist.log_prob(1), np.log(1e-24) + stats.norm.logpdf(1))
    assert_allclose(dist.log_survival(0), np.log(1e-24) + stats.norm.logsf(0))
    assert np.isposinf(dist.quantile(1)).all()
    ordinary = DistributionPrediction(
        AtomMixtureLaw(NormalLaw(), (0.0,), ("p",)),
        {"loc": [0.0], "scale": [1.0], "p": [0.3]},
    )
    explicit = DistributionPrediction(
        law, {"loc": [0.0], "scale": [1.0], "p": [0.3], "pc": [0.7]}
    )
    assert_allclose(
        explicit.quantile([0, 0.2, 0.5, 0.9, 1]),
        ordinary.quantile([0, 0.2, 0.5, 0.9, 1]),
    )
    assert_allclose(explicit.mean(), ordinary.mean())
    assert_allclose(explicit.variance(), ordinary.variance())


@pytest.mark.parametrize("pc", [-0.1, 0.5, 1.1])
def test_explicit_continuous_weights_reject_invalid_or_unnormalized_inputs(pc):
    law = AtomMixtureLaw(NormalLaw(), (0.0,), ("p",), continuous_mass_name="pc")
    with pytest.raises(ValueError, match="probabilities"):
        DistributionPrediction(
            law, {"loc": [0.0], "scale": [1.0], "p": [0.3], "pc": [pc]}
        )


def test_base_context_is_aligned_in_missing_rows_and_rowwise_quantiles():
    class ShiftedNormal:
        spec = replace(NormalLaw.spec, context_names=("shift",))

        def validate_parameters(self, parameters, *, context):
            NormalLaw().validate_parameters(parameters, context={})
            assert np.all(np.isfinite(context["shift"]))

        def evaluate(self, operation, parameters, *, context, **kwargs):
            shifted = {**parameters, "loc": parameters["loc"] + context["shift"]}
            return NormalLaw().evaluate(operation, shifted, context={}, **kwargs)

    dist = DistributionPrediction(
        AtomMixtureLaw(ShiftedNormal(), (0.0,), ("p",)),
        {
            "loc": [0.0, np.nan, 1.0],
            "scale": [1.0, np.nan, 0.5],
            "p": [0.2, np.nan, 0.3],
        },
        {"shift": [-2.0, np.nan, 3.0]},
    )
    q = dist.quantile(0.995)[:, 0]
    # At this upper centile both atoms are below the query.
    assert_allclose(
        q,
        [
            stats.norm.ppf((0.995 - 0.2) / 0.8, -2, 1),
            np.nan,
            stats.norm.ppf((0.995 - 0.3) / 0.7, 4, 0.5),
        ],
    )
    assert_allclose(dist.cdf(q), [0.995, np.nan, 0.995])


@pytest.mark.parametrize(
    "atoms,names",
    [
        ((1.0, 0.0), ("p", "q")),
        ((0.0, 0.0), ("p", "q")),
        ((np.inf,), ("p",)),
        ((0.0,), ("scale",)),
        ((0.0,), ()),
    ],
)
def test_invalid_composition_configuration(atoms, names):
    with pytest.raises(ValueError):
        AtomMixtureLaw(NormalLaw(), atoms, names)


@pytest.mark.parametrize("mass", [-0.1, 1.1])
def test_invalid_atom_probabilities(mass):
    with pytest.raises(ValueError, match="probabilities"):
        DistributionPrediction(
            AtomMixtureLaw(NormalLaw(), (0.0,), ("p",)),
            {"loc": [0.0], "scale": [1.0], "p": [mass]},
        )
