from .base import BaseFamily
from .exponential import (
    GaussianIdentityFamily,
    BinomialLogitFamily,
    PoissonLogFamily,
    GammaLogFamily,
    NegativeBinomialLogFamily,
)


def make_gam_family(family):
    if isinstance(family, BaseFamily):
        return family

    if family is None:
        return GaussianIdentityFamily()

    if isinstance(family, dict):
        name = str(family.get("name", "")).lower()
        if name in {"negbin", "negativebinomial", "negative_binomial"}:
            return NegativeBinomialLogFamily(
                theta=family.get("theta", 1.0),
                estimate_theta=bool(family.get("estimate_theta", False)),
            )
        family = name

    if isinstance(family, tuple) and len(family) == 2:
        key = str(family[0]).lower()
        spec = family[1]
        if key in {"negbin", "negativebinomial", "negative_binomial"}:
            if isinstance(spec, dict):
                return NegativeBinomialLogFamily(
                    theta=spec.get("theta", 1.0),
                    estimate_theta=bool(spec.get("estimate_theta", False)),
                )
            return NegativeBinomialLogFamily(theta=spec)
        family = key

    key = str(family).lower()
    if key in {"gaussian", "normal"}:
        return GaussianIdentityFamily()
    if key in {"binomial", "bernoulli", "logistic"}:
        return BinomialLogitFamily()
    if key in {"poisson"}:
        return PoissonLogFamily()
    if key in {"gamma"}:
        return GammaLogFamily()
    if key in {"negbin", "negativebinomial", "negative_binomial"}:
        return NegativeBinomialLogFamily(theta=1.0)

    raise ValueError(
        f"Unknown GAM family {family!r}. "
        "Valid options: gaussian, binomial, poisson, gamma, negbin."
    )