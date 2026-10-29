"""Expected and sampled likelihood Fisher diagonals in GAM coordinates.

GLM score/Hessian expressions use the ordinary exponential-dispersion family
identity; general-family sampled derivatives reuse the existing analytic
``family.ll`` kernels. Neither smoothing penalties nor coefficient covariance
are information. No parameter-by-parameter matrix is constructed here.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import polygamma

from ..inspection.fisher import aggregate_fisher
from .data import combine_offsets
from .families import binomial, gamma, gaussian, poisson
from .families.gamlss.distributional import (
    BetaLocationPrecision,
    NegativeBinomialLocationDispersion,
    StudentLocationScale,
)
from .families.gamlss.gammals import GammalsFamily
from .families.gamlss.gaulss import GaulssFamily
from .families.gamlss.positive import LogNormalLocationScale, WeibullLocationShape
from .families.inverse_gaussian import InverseGaussianFamily
from .families.negbin import NegativeBinomialLogFamily
from .fit.capabilities import observation_transform
from .model_state import (
    _fit_scale,
    _predictor_full_indices,
    _term_blocks_seq,
    _term_full_coefficient_indices,
)
from .predict.distribution_adapter import distribution_prediction

_GLM_TYPES = (
    gaussian.GaussianIdentityFamily,
    gaussian.GaussianLogFamily,
    gaussian.GaussianInverseFamily,
    poisson.PoissonLogFamily,
    poisson.PoissonIdentityFamily,
    poisson.PoissonSqrtFamily,
    binomial.BinomialLogitFamily,
    binomial.BinomialProbitFamily,
    binomial.BinomialCloglogFamily,
    binomial.BinomialCauchitFamily,
    binomial.BinomialLogFamily,
    gamma.GammaIdentityFamily,
    gamma.GammaLogFamily,
    gamma.GammaInverseFamily,
    InverseGaussianFamily,
    NegativeBinomialLogFamily,
)
_GENERAL_TYPES = (
    GaulssFamily,
    GammalsFamily,
    LogNormalLocationScale,
    WeibullLocationShape,
    BetaLocationPrecision,
    StudentLocationScale,
    NegativeBinomialLocationDispersion,
)


def validate_fisher_family(family, config):
    """Eligibility is an explicit likelihood contract, never GLM inheritance."""
    if config.every_n_epochs is not None:
        raise NotImplementedError(
            "Periodic GAM Fisher diagnostics have no iteration contract; use final diagnostics."
        )
    if getattr(family, "is_quasi", False):
        raise NotImplementedError(
            "Quasi-likelihood families do not define model Fisher information."
        )
    if type(family) not in _GLM_TYPES + _GENERAL_TYPES:
        raise NotImplementedError(
            f"Fisher likelihood derivatives are not declared for {type(family).__name__}."
        )
    if (
        config.estimator == "exact"
        and type(family) is NegativeBinomialLocationDispersion
    ):
        raise NotImplementedError(
            "Exact joint negative-binomial information is not declared; use the sampled score or Hessian estimator."
        )


def _groups(model, width, group_by):
    if group_by in {"tensor", "layer"}:
        return (("coefficients", np.arange(width)),)
    # Shared coefficients belong to one shared group, never twice to two terms.
    owners: list[set[str]] = [set() for _ in range(width)]
    for term in _term_blocks_seq(model):
        for index in _term_full_coefficient_indices(model, term):
            owners[index].add(str(term.label))
    mapping: dict[str, list[int]] = {}
    for index, names in enumerate(owners):
        label = " + ".join(sorted(names)) if names else "__global__"
        mapping.setdefault(label, []).append(index)
    return tuple((name, np.asarray(indices)) for name, indices in mapping.items())


def _take(value, indices):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [_take(part, indices) for part in value]
    return value.iloc[indices] if hasattr(value, "iloc") else np.asarray(value)[indices]


def _reference(model, X, offset, config, *, probe=False):
    """Select before building prediction designs; preserve training offsets."""
    if offset is None and getattr(model, "_fisher_offset_required_", False):
        raise ValueError(
            "Fisher evaluation requires explicit row-aligned offsets for this fitted model."
        )
    count = int(model.n_samples_) if X is None else len(X)
    if count < 1:
        raise ValueError("Fisher evaluation requires at least one input row.")
    offset = model._coerce_api_offset(offset, count)
    if X is None and offset is not None:
        offset = combine_offsets(model.offset_predict_default_, offset)
    indices = (
        np.sort(
            np.random.default_rng(config.random_state).choice(
                count, min(count, config.n_inputs), replace=False
            )
        )
        if probe
        else np.arange(count)
    )
    selected = len(indices)
    transform = observation_transform(model)
    if not transform.is_identity:
        if (
            X is not None
            or selected != count
            or config.estimator != "exact"
            or config.input_sampling != "fixed"
        ):
            raise NotImplementedError(
                "Correlated Fisher requires the entire fitted sequence, exact information and fixed inputs."
            )
        design = transform.apply(np.asarray(model.lpmatrix(None), dtype=float))
        eta = np.asarray(model.predict(None, type="link", offset=offset), dtype=float)
        return design, eta.reshape(count, -1), indices, count
    if X is None:
        # Link output is only n-by-predictors, not n-by-coefficients. Asking
        # predict(None) preserves formula/default offset semantics exactly.
        eta = np.asarray(
            model.predict(None, type="link", offset=offset), dtype=float
        ).reshape(count, -1)[indices]
        source = model.vis_training_data_
        reference = (
            source.iloc[indices]
            if hasattr(source, "iloc")
            else np.asarray(source)[indices]
        )
        design = model.lpmatrix(reference)
    else:
        reference = X.iloc[indices] if hasattr(X, "iloc") else np.asarray(X)[indices]
        design = model.lpmatrix(reference)
        eta = np.asarray(
            model.predict(reference, type="link", offset=_take(offset, indices)),
            dtype=float,
        ).reshape(selected, -1)
    design = (
        design.toarray() if sparse.issparse(design) else np.asarray(design, dtype=float)
    )
    if not np.isfinite(design).all() or not np.isfinite(eta).all():
        raise ValueError("Fisher inputs must produce finite predictor/design rows.")
    return design, eta, indices, count


def _natural_information(family, parameters):
    """Small expected information matrices in the family's natural columns."""
    a, b = parameters.T
    information = np.zeros((len(parameters), 2, 2))
    if type(family) in {GaulssFamily, LogNormalLocationScale}:
        information[:, 0, 0] = 1 / b**2
        information[:, 1, 1] = 2 / b**2
    elif type(family) is StudentLocationScale:
        nu = family.df
        information[:, 0, 0] = (nu + 1) / ((nu + 3) * b**2)
        information[:, 1, 1] = 2 * nu / ((nu + 3) * b**2)
    elif type(family) is GammalsFamily:
        shape = 1 / b**2
        information[:, 0, 0] = shape / a**2
        information[:, 1, 1] = 4 * shape**2 * (polygamma(1, shape) - 1 / shape) / b**2
    elif type(family) is WeibullLocationShape:
        information[:, 0, 0] = b**2 / a**2
        information[:, 0, 1] = information[:, 1, 0] = (np.euler_gamma - 1) / a
        information[:, 1, 1] = (np.pi**2 / 6 + (1 - np.euler_gamma) ** 2) / b**2
    elif type(family) is BetaLocationPrecision:
        ta, tb, total = polygamma(1, a * b), polygamma(1, (1 - a) * b), polygamma(1, b)
        information[:, 0, 0] = b**2 * (ta + tb)
        information[:, 0, 1] = information[:, 1, 0] = b * (a * ta - (1 - a) * tb)
        information[:, 1, 1] = a**2 * ta + (1 - a) ** 2 * tb - total
    else:
        raise NotImplementedError(
            f"Exact information is not declared for {family.name}."
        )
    return information


def _general_information(family, eta):
    parameters = family.distribution_parameters_from_eta(eta)
    natural = _natural_information(family, parameters)
    jacobian = np.asarray(family.distribution_parameter_jacobian(eta))
    if jacobian.ndim == 2:
        return natural * jacobian[:, :, None] * jacobian[:, None, :]
    return np.einsum("nai,nab,nbj->nij", jacobian, natural, jacobian)


def _general_derivatives(family, eta, value):
    # One coefficient per predictor makes the existing exact family kernel's
    # coefficient derivatives its predictor derivatives. Only this tiny
    # output-sized Hessian is assembled; no fitted coefficient Hessian exists.
    width = len(eta)
    result = family.ll(
        np.asarray([value]),
        np.ones((1, width)),
        [np.asarray([i]) for i in range(width)],
        eta,
        np.ones(1),
        deriv=1,
    )
    return np.asarray(result["lb"]), np.asarray(result["lbb"])


def _glm_state(family, eta, scale):
    mu = np.asarray(family.inverse_link(eta), dtype=float)
    first = np.asarray(family.mu_eta(eta), dtype=float)
    eps = np.finfo(float).eps
    link = family.link_name
    clipped = (
        np.any(np.abs(eta) > 30)
        if link == "logit"
        else np.any((mu <= eps) | (mu >= 1 - eps) | (first <= eps))
        if link in {"probit", "cloglog", "cauchit"}
        else np.any(mu <= eps)
        if link == "log"
        else False
    )
    if clipped:
        raise ValueError(
            "Fisher is undefined for solver-clipped link derivatives; use interior predictors."
        )
    if (
        not np.isfinite(mu).all()
        or not np.isfinite(first).all()
        or (family.name == "binomial" and np.any((mu <= 0) | (mu >= 1)))
        or (family.name not in {"gaussian", "binomial"} and np.any(mu <= 0))
    ):
        raise ValueError(
            "Fisher requires finite predictors inside the likelihood domain."
        )
    variance = np.broadcast_to(family.variance(mu), mu.shape)
    second = -np.asarray(family.d2link(mu)) * first**3
    expected = first**2 / (scale * variance)
    return mu, first, second, variance, expected


def _glm_sample(family, mu, scale, trials, rng):
    if family.name == "gaussian":
        return rng.normal(mu, np.sqrt(scale))
    if family.name == "binomial":
        return rng.binomial(trials, mu) / trials
    if family.name == "poisson":
        return rng.poisson(mu)
    if family.name == "gamma":
        return rng.gamma(1 / scale, mu * scale)
    if family.name == "inverse_gaussian":
        return rng.wald(mu, 1 / scale)
    if family.name == "negbin":
        return rng.negative_binomial(family.theta, family.theta / (family.theta + mu))
    raise NotImplementedError(f"No declared sampling likelihood for {family.name}.")


def _glm_draw(family, state, scale, trials, rng):
    mu, first, second, variance, expected = state
    value = _glm_sample(family, mu, scale, trials, rng)
    score = trials * (value - mu) * first / (scale * variance)
    hessian = trials * (
        -expected
        + (value - mu)
        / scale
        * (second / variance - first**2 * family.dvar(mu) / variance**2)
    )
    return np.asarray([score]), np.asarray([[hessian]])


def _row_moments(draws, config, groups):
    mean = m2 = group_mean = group_m2 = None
    for count, value in enumerate(draws, 1):
        traces = np.asarray([value[index].sum() for _, index in groups])
        if count == 1:
            mean, m2 = value.copy(), np.zeros_like(value)
            group_mean, group_m2 = traces.copy(), np.zeros_like(traces)
        else:
            delta = value - mean
            mean += delta / count
            m2 += delta * (value - mean)
            delta = traces - group_mean
            group_mean += delta / count
            group_m2 += delta * (traces - group_mean)
    return mean, m2 / (config.n_draws - 1), group_m2 / (config.n_draws - 1)


def statistical_fisher_information(
    model,
    X=None,
    *,
    config,
    offset=None,
    context=None,
    return_diagonal=True,
    probe=False,
):
    """Mean likelihood information in public GAM coefficient coordinates."""
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")
    family = model.family
    validate_fisher_family(family, config)
    design, eta, indices, available = _reference(model, X, offset, config, probe=probe)
    if (
        probe
        and config.input_sampling == "without_replacement"
        and config.population_size is not None
        and config.population_size != available
    ):
        raise ValueError("Fit-time Fisher population_size must match the selected split.")
    n_inputs, width = design.shape
    groups = _groups(model, width, config.group_by)
    prior = getattr(model, "prior_weights_", None)
    declared_trials = getattr(family, "_trial_counts", None)
    unit_weights = np.ones(available) if declared_trials is None else declared_trials
    if X is None and prior is not None and np.any(np.asarray(prior) != unit_weights):
        raise NotImplementedError(
            "Fisher case/frequency weights need an explicit likelihood-unit contract; use unit-weight fits."
        )
    general = type(family) in _GENERAL_TYPES
    scale = float(_fit_scale(model)) if not general else None
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError("Fisher requires a positive finite fitted likelihood scale.")
    if context:
        if general or family.name != "binomial" or set(context) != {"trials"}:
            raise ValueError(
                "This Fisher family does not accept that observation context."
            )
    trials = np.ones(n_inputs)
    if not general and family.name == "binomial":
        declared = (
            context.get("trials")
            if context
            else getattr(family, "_trial_counts", None)
            if X is None
            else None
        )
        if declared is not None:
            declared = np.asarray(declared, dtype=float)
            if (
                declared.shape != (available,)
                or not np.isfinite(declared).all()
                or np.any(declared < 1)
                or np.any(declared != np.floor(declared))
            ):
                raise ValueError(
                    "Binomial Fisher trials must be positive row-aligned integers."
                )
            trials = declared[indices]
        elif X is not None and getattr(family, "_trial_counts", None) is not None:
            raise ValueError(
                "Grouped-binomial Fisher requires explicit new-row trials context."
            )
    blocks = _predictor_full_indices(model) if general else (np.arange(width),)
    if len(blocks) != eta.shape[1]:
        raise NotImplementedError(
            "Fisher predictor/design mapping is not available for this model."
        )
    rng = np.random.default_rng(config.random_state)
    expected = (
        _general_information(family, eta)
        if general and config.estimator == "exact"
        else None
    )

    def rows():
        for row in range(n_inputs):
            jacobian = np.zeros((len(blocks), width))
            for k, columns in enumerate(blocks):
                jacobian[k, columns] = design[row, columns]
            if config.estimator == "exact":
                if general:
                    assert expected is not None
                    metric = expected[row]
                else:
                    metric = np.asarray(
                        [[_glm_state(family, eta[row, 0], scale)[-1] * trials[row]]]
                    )
                value = np.einsum("ki,kl,li->i", jacobian, metric, jacobian)
                yield value, None, None
                continue
            samples = state = None
            if general:
                parameters = family.distribution_parameters_from_eta(eta[row : row + 1])
                distribution = distribution_prediction(family, parameters)
                samples = distribution.sample(config.n_draws, random_state=rng)[:, 0]
            else:
                state = _glm_state(family, eta[row, 0], scale)

            def draws(row=row, samples=samples, state=state, jacobian=jacobian):
                for draw in range(config.n_draws):
                    score, hessian = (
                        _general_derivatives(family, eta[row], samples[draw])
                        if general
                        else _glm_draw(family, state, scale, trials[row], rng)
                    )
                    yield (
                        (score @ jacobian) ** 2
                        if config.estimator == "score_outer_product"
                        else -np.einsum("ki,kl,li->i", jacobian, hessian, jacobian)
                    )

            yield _row_moments(draws(), config, groups)

    held_fixed = (
        ("fitted_scale",)
        if family.name in {"gaussian", "gamma", "inverse_gaussian"}
        else ("theta",)
        if family.name == "negbin"
        else ("df",)
        if type(family) is StudentLocationScale
        else ()
    )
    return aggregate_fisher(
        rows(),
        config=config,
        parameter_names=("coefficients",),
        parameter_shapes=((width,),),
        parameter_slices=((0, width),),
        groups=groups,
        backend="gam",
        estimator=config.estimator,
        n_inputs=n_inputs,
        grouping=config.group_by,
        split="train" if X is None else None,
        population_size=config.population_size or available
        if config.input_sampling == "without_replacement"
        else None,
        return_diagonal=return_diagonal,
        held_fixed=held_fixed,
    )
