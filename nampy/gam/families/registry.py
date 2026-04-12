from .exponential import (
    BinomialCloglogFamily,
    BinomialLogitFamily,
    BinomialProbitFamily,
    GammaIdentityFamily,
    GammaInverseFamily,
    GammaLogFamily,
    GaussianIdentityFamily,
    NegativeBinomialLogFamily,
    PoissonLogFamily,
)
from .family_base import BaseFamily
from .gamlss import gammals, gaulss, gevlss, shashlss, ziplss

_BINOMIAL_LINK_MAP = {
    "logit": BinomialLogitFamily,
    "probit": BinomialProbitFamily,
    "cloglog": BinomialCloglogFamily,
}

_GAMMA_LINK_MAP = {
    "identity": GammaIdentityFamily,
    "log": GammaLogFamily,
    "inverse": GammaInverseFamily,
}


def make_gam_family(family):
    if isinstance(family, BaseFamily):
        return family

    if family is None:
        return GaussianIdentityFamily()

    if isinstance(family, dict):
        name = str(family.get("name", "")).lower()
        link = str(family.get("link", "")).lower() or None
        if name in {"negbin", "negativebinomial", "negative_binomial"}:
            return NegativeBinomialLogFamily(
                theta=family.get("theta", 1.0),
                estimate_theta=bool(family.get("estimate_theta", False)),
            )
        if name in {"binomial", "bernoulli", "logistic"}:
            cls = _BINOMIAL_LINK_MAP.get(link or "logit", BinomialLogitFamily)
            return cls()
        if name in {"gamma"}:
            resolved = (link or "log").lower()
            try:
                cls = _GAMMA_LINK_MAP[resolved]
            except KeyError:
                raise ValueError(
                    f"Unknown gamma link {link!r}. "
                    f"Supported: {', '.join(sorted(_GAMMA_LINK_MAP))}."
                ) from None
            return cls()
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
    if key in {"gaulss"}:
        return gaulss()
    if key in {"gammals"}:
        return gammals()
    if key in {"ziplss"}:
        return ziplss()
    if key in {"gevlss"}:
        return gevlss()
    if key in {"shashlss", "shash"}:
        return shashlss()

    raise ValueError(
        f"Unknown GAM family {family!r}. "
        "Valid options: gaussian, binomial, poisson, gamma, negbin, gaulss, gammals, ziplss, gevlss, shashlss."
    )
