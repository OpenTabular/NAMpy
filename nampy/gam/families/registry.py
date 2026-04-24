from .binomial import (
    BinomialCauchitFamily,
    BinomialCloglogFamily,
    BinomialLogFamily,
    BinomialLogitFamily,
    BinomialProbitFamily,
)
from .family_base import BaseFamily
from .gamlss.gammals import gammals
from .gamlss.gaulss import gaulss
from .gamlss.gevlss import gevlss
from .gamlss.shashlss import shashlss
from .gamlss.ziplss import ziplss
from .gamma import GammaIdentityFamily, GammaInverseFamily, GammaLogFamily
from .gaussian import GaussianIdentityFamily, GaussianInverseFamily, GaussianLogFamily
from .negbin import NegativeBinomialLogFamily
from .poisson import PoissonIdentityFamily, PoissonLogFamily, PoissonSqrtFamily

_BINOMIAL_LINK_MAP = {
    "logit": BinomialLogitFamily,
    "probit": BinomialProbitFamily,
    "cloglog": BinomialCloglogFamily,
    "cauchit": BinomialCauchitFamily,
    "log": BinomialLogFamily,
}

_GAUSSIAN_LINK_MAP = {
    "identity": GaussianIdentityFamily,
    "log": GaussianLogFamily,
    "inverse": GaussianInverseFamily,
}

_POISSON_LINK_MAP = {
    "log": PoissonLogFamily,
    "identity": PoissonIdentityFamily,
    "sqrt": PoissonSqrtFamily,
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
        if name in {"nb"}:
            return NegativeBinomialLogFamily(
                theta=family.get("theta", 1.0),
                estimate_theta=bool(family.get("estimate_theta", True)),
                link=link or "log",
            )
        if name in {"negbin", "negativebinomial", "negative_binomial"}:
            if "theta" not in family:
                raise ValueError(
                    "mgcv::negbin requires explicit theta. Use {'name': 'nb'} "
                    "for mgcv::nb theta estimation."
                )
            return NegativeBinomialLogFamily(
                theta=family.get("theta", 1.0),
                estimate_theta=bool(family.get("estimate_theta", False)),
                link=link or "log",
            )
        if name in {"binomial", "bernoulli", "logistic"}:
            resolved = link or "logit"
            try:
                cls = _BINOMIAL_LINK_MAP[resolved]
            except KeyError:
                raise ValueError(
                    f"Unknown binomial link {link!r}. "
                    f"Supported: {', '.join(sorted(_BINOMIAL_LINK_MAP))}."
                ) from None
            return cls()
        if name in {"gaussian", "normal"}:
            resolved = link or "identity"
            try:
                cls = _GAUSSIAN_LINK_MAP[resolved]
            except KeyError:
                raise ValueError(
                    f"Unknown gaussian link {link!r}. "
                    f"Supported: {', '.join(sorted(_GAUSSIAN_LINK_MAP))}."
                ) from None
            return cls()
        if name in {"poisson"}:
            resolved = link or "log"
            try:
                cls = _POISSON_LINK_MAP[resolved]
            except KeyError:
                raise ValueError(
                    f"Unknown poisson link {link!r}. "
                    f"Supported: {', '.join(sorted(_POISSON_LINK_MAP))}."
                ) from None
            return cls()
        if name in {"gamma"}:
            resolved = (link or "inverse").lower()
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
        if key in {"gaussian", "normal"}:
            link_spec = (
                str(spec.get("link", "identity")).lower()
                if isinstance(spec, dict)
                else str(spec).lower()
            )
            try:
                return _GAUSSIAN_LINK_MAP[link_spec]()
            except KeyError:
                raise ValueError(
                    f"Unknown gaussian link {link_spec!r}. "
                    f"Supported: {', '.join(sorted(_GAUSSIAN_LINK_MAP))}."
                ) from None
        if key in {"binomial", "bernoulli", "logistic"}:
            link_spec = (
                str(spec.get("link", "logit")).lower()
                if isinstance(spec, dict)
                else str(spec).lower()
            )
            try:
                return _BINOMIAL_LINK_MAP[link_spec]()
            except KeyError:
                raise ValueError(
                    f"Unknown binomial link {link_spec!r}. "
                    f"Supported: {', '.join(sorted(_BINOMIAL_LINK_MAP))}."
                ) from None
        if key in {"poisson"}:
            link_spec = (
                str(spec.get("link", "log")).lower()
                if isinstance(spec, dict)
                else str(spec).lower()
            )
            try:
                return _POISSON_LINK_MAP[link_spec]()
            except KeyError:
                raise ValueError(
                    f"Unknown poisson link {link_spec!r}. "
                    f"Supported: {', '.join(sorted(_POISSON_LINK_MAP))}."
                ) from None
        if key in {"gamma"}:
            link_spec = (
                str(spec.get("link", "inverse")).lower()
                if isinstance(spec, dict)
                else str(spec).lower()
            )
            try:
                return _GAMMA_LINK_MAP[link_spec]()
            except KeyError:
                raise ValueError(
                    f"Unknown gamma link {link_spec!r}. "
                    f"Supported: {', '.join(sorted(_GAMMA_LINK_MAP))}."
                ) from None
        if key in {"nb"}:
            if isinstance(spec, dict):
                return NegativeBinomialLogFamily(
                    theta=spec.get("theta", 1.0),
                    estimate_theta=bool(spec.get("estimate_theta", True)),
                    link=str(spec.get("link", "log")).lower(),
                )
            return NegativeBinomialLogFamily(theta=spec, estimate_theta=True)
        if key in {"negbin", "negativebinomial", "negative_binomial"}:
            if isinstance(spec, dict):
                if "theta" not in spec:
                    raise ValueError("mgcv::negbin tuple/dict specs require theta.")
                return NegativeBinomialLogFamily(
                    theta=spec.get("theta", 1.0),
                    estimate_theta=bool(spec.get("estimate_theta", False)),
                    link=str(spec.get("link", "log")).lower(),
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
        return GammaInverseFamily()
    if key in {"negbin", "negativebinomial", "negative_binomial"}:
        raise ValueError("mgcv::negbin requires explicit theta.")
    if key in {"nb"}:
        return NegativeBinomialLogFamily(theta=1.0, estimate_theta=True)
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
        "Valid options: gaussian, binomial, poisson, gamma, nb, negbin, gaulss, gammals, ziplss, gevlss, shashlss."
    )
