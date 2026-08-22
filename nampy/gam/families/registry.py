import copy

import numpy as np

from .betar import BetaRegressionFamily
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
from .gamma import GammaIdentityFamily, GammaInverseFamily, GammaLogFamily
from .gaussian import GaussianIdentityFamily, GaussianInverseFamily, GaussianLogFamily
from .negbin import NegativeBinomialLogFamily
from .ocat import OrderedCategoricalFamily
from .poisson import PoissonIdentityFamily, PoissonLogFamily, PoissonSqrtFamily
from .tweedie import TweedieTwFamily

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


def clone_gam_family(family: BaseFamily) -> BaseFamily:
    """Return an independent family instance for one model or fit session.

    Extended and general mgcv families carry mutable working parameters (for
    example ``theta`` and ordered-category cut points).  Sharing one Python
    family object across models would therefore also share fitted state, unlike
    R's copy-on-modify family lists.
    """
    if not isinstance(family, BaseFamily):
        raise TypeError(
            f"Expected a BaseFamily instance, got {type(family).__name__}."
        )
    try:
        cloned = copy.deepcopy(family)
    except Exception as exc:  # pragma: no cover - custom family diagnostic
        raise TypeError(
            f"Family {family.__class__.__name__} cannot be cloned for an "
            "independent GAM fit."
        ) from exc
    if cloned is family:  # defensive against a custom ``__deepcopy__``
        raise TypeError(
            f"Family {family.__class__.__name__} returned itself from deepcopy; "
            "families used by GAM must support independent fit state."
        )
    return cloned


def make_gam_family(family):
    if isinstance(family, BaseFamily):
        return clone_gam_family(family)

    if family is None:
        return GaussianIdentityFamily()

    if isinstance(family, dict):
        name = str(family.get("name", "")).lower()
        link = str(family.get("link", "")).lower() or None
        if name in {"nb"}:
            theta = family.get("theta", 1.0)
            estimate_theta = bool(
                family.get(
                    "estimate_theta",
                    "theta" not in family or float(theta) <= 0.0,
                )
            )
            if float(theta) <= 0.0:
                theta = abs(float(theta)) if float(theta) < 0.0 else 1.0
            return NegativeBinomialLogFamily(
                theta=theta,
                estimate_theta=estimate_theta,
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
        if name in {"tw", "tweedie"}:
            return TweedieTwFamily(
                theta=family.get("theta", None),
                link=link or "log",
                a=float(family.get("a", 1.01)),
                b=float(family.get("b", 1.99)),
            )
        if name in {"betar", "beta", "beta_regression"}:
            return BetaRegressionFamily(
                theta=family.get("theta", None),
                link=link or "logit",
                eps=float(family.get("eps", np.finfo(np.float64).eps * 100.0)),
            )
        if name in {"ocat", "ordered_categorical", "ordered"}:
            return OrderedCategoricalFamily(
                theta=family.get("theta", None),
                R=family.get("R", None),
                link=link or "identity",
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
                theta = spec.get("theta", 1.0)
                estimate_theta = bool(
                    spec.get(
                        "estimate_theta",
                        "theta" not in spec or float(theta) <= 0.0,
                    )
                )
                if float(theta) <= 0.0:
                    theta = abs(float(theta)) if float(theta) < 0.0 else 1.0
                return NegativeBinomialLogFamily(
                    theta=theta,
                    estimate_theta=estimate_theta,
                    link=str(spec.get("link", "log")).lower(),
                )
            theta = float(spec)
            return NegativeBinomialLogFamily(
                theta=abs(theta) if theta < 0.0 else (1.0 if theta == 0.0 else theta),
                estimate_theta=theta <= 0.0,
            )
        if key in {"tw", "tweedie"}:
            if isinstance(spec, dict):
                return TweedieTwFamily(
                    theta=spec.get("theta", None),
                    link=str(spec.get("link", "log")).lower(),
                    a=float(spec.get("a", 1.01)),
                    b=float(spec.get("b", 1.99)),
                )
            return TweedieTwFamily(theta=float(spec))
        if key in {"betar", "beta", "beta_regression"}:
            if isinstance(spec, dict):
                return BetaRegressionFamily(
                    theta=spec.get("theta", None),
                    link=str(spec.get("link", "logit")).lower(),
                    eps=float(spec.get("eps", np.finfo(np.float64).eps * 100.0)),
                )
            return BetaRegressionFamily(theta=spec)
        if key in {"ocat", "ordered_categorical", "ordered"}:
            if isinstance(spec, dict):
                return OrderedCategoricalFamily(
                    theta=spec.get("theta", None),
                    R=spec.get("R", None),
                    link=str(spec.get("link", "identity")).lower(),
                )
            if np.isscalar(spec):
                return OrderedCategoricalFamily(R=int(spec))
            return OrderedCategoricalFamily(theta=spec)
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
    if key in {"tw", "tweedie"}:
        return TweedieTwFamily()
    if key in {"betar", "beta", "beta_regression"}:
        return BetaRegressionFamily()
    if key in {"gaulss"}:
        return gaulss()
    if key in {"gammals"}:
        return gammals()
    raise ValueError(
        f"Unknown GAM family {family!r}. "
        "Valid options: gaussian, binomial, poisson, gamma, nb, negbin, tw, "
        "betar, gaulss, gammals."
    )
