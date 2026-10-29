import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, cast

import numpy as np

from ...distributions import (
    BCCGLaw,
    BCPELaw,
    BCTLaw,
    BetaBinomialLaw,
    BetaLaw,
    GammaLaw,
    LogNormalLaw,
    NegativeBinomialLaw,
    NormalLaw,
    StudentTLaw,
    WeibullLaw,
)
from ...distributions.mixtures import INFLATED_BETA_LAW, ZERO_ADJUSTED_GAMMA_LAW
from ..predict.distribution_adapter import (
    beta_prediction,
    gamma_prediction,
    inflated_beta_adapter,
    observation_prediction,
    student_prediction,
    zero_adjusted_gamma_adapter,
)
from .bcg import CensoredBoxCoxGaussianFamily
from .betar import BetaRegressionFamily
from .binomial import (
    BinomialCauchitFamily,
    BinomialCloglogFamily,
    BinomialLogFamily,
    BinomialLogitFamily,
    BinomialProbitFamily,
)
from .clog import CensoredLogisticFamily
from .cnorm import CensoredNormalFamily
from .cpois import CensoredPoissonFamily
from .family_base import BaseFamily
from .gamlss.bccg import BCCGLikelihood
from .gamlss.betabinomial import BetaBinomialLocationPrecision
from .gamlss.boxcox_tail import BCPELikelihood, BCTLikelihood
from .gamlss.distributional import (
    BetaLocationPrecision,
    NegativeBinomialLocationDispersion,
    StudentLocationScale,
)
from .gamlss.gammals import GammalsFamily, gammals
from .gamlss.gaulss import GaulssFamily, gaulss
from .gamlss.gevlss import GevlssFamily
from .gamlss.gumbls import GumblsFamily
from .gamlss.inflated_beta import InflatedBeta
from .gamlss.multinom import MultinomFamily
from .gamlss.observations import (
    LogNormalObservations,
    NormalObservations,
    WeibullObservations,
)
from .gamlss.positive import LogNormalLocationScale, WeibullLocationShape
from .gamlss.semicontinuous import ZeroAdjustedGamma
from .gamlss.shash import ShashFamily
from .gamlss.twlss import TwlssFamily
from .gamlss.ziplss import ZiplssFamily
from .gamma import GammaIdentityFamily, GammaInverseFamily, GammaLogFamily
from .gaussian import GaussianIdentityFamily, GaussianInverseFamily, GaussianLogFamily
from .inverse_gaussian import InverseGaussianFamily
from .negbin import NegativeBinomialLogFamily
from .ocat import OrderedCategoricalFamily
from .poisson import PoissonIdentityFamily, PoissonLogFamily, PoissonSqrtFamily
from .quasi import QuasiBinomialFamily, QuasiFamily, QuasiPoissonFamily
from .scat import ScaledTFamily
from .tweedie import TweedieTwFamily
from .zip import ZeroInflatedPoissonFamily

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


@dataclass(frozen=True)
class GeneralFamilySpec:
    """Runtime constructor contract also used to discover reference pages."""

    name: str
    factory: Callable[..., BaseFamily]
    options: frozenset[str]
    reference_group: str
    family_type: type
    law_factory: Callable | None = None
    prediction_adapter: Callable | None = None
    public_name: str | None = None
    prediction_context_names: tuple[str, ...] = ()

    @property
    def supports_distribution_prediction(self):
        return self.law_factory is not None or self.prediction_adapter is not None


GENERAL_FAMILIES = MappingProxyType(
    {
        spec.name: spec
        for spec in (
            *(
                GeneralFamilySpec(
                    name + "_lss", family, frozenset(),
                    getattr(family, "reference_group", "mgcv"),
                    family_type=family, law_factory=law, public_name=name,
                )
                for name, family, law in (
                    ("bct", BCTLikelihood, BCTLaw),
                    ("bcpe", BCPELikelihood, BCPELaw),
                )
            ),
            *(
                GeneralFamilySpec(
                    name + "_lss", family, frozenset({"lower", "upper"}),
                    getattr(family, "reference_group", "mgcv"),
                    family_type=family, law_factory=lambda family=family: family().law,
                    prediction_adapter=observation_prediction, public_name=name,
                )
                for name, family in (
                    ("lognormal_observation", LogNormalObservations),
                    ("weibull_observation", WeibullObservations),
                )
            ),
            GeneralFamilySpec(
                "normal_observation_lss",
                NormalObservations,
                frozenset({"lower", "upper"}),
                getattr(NormalObservations, "reference_group", "mgcv"),
                family_type=NormalObservations,
                law_factory=NormalLaw,
                prediction_adapter=observation_prediction,
                public_name="normal_observation",
            ),
            GeneralFamilySpec(
                "bccg_lss",
                BCCGLikelihood,
                frozenset(set()),
                getattr(BCCGLikelihood, "reference_group", "mgcv"),
                family_type=BCCGLikelihood,
                law_factory=BCCGLaw,
                public_name="bccg",
            ),
            GeneralFamilySpec(
                "beinf_lss",
                InflatedBeta,
                frozenset(set()),
                getattr(InflatedBeta, "reference_group", "mgcv"),
                family_type=InflatedBeta,
                law_factory=lambda: INFLATED_BETA_LAW,
                prediction_adapter=inflated_beta_adapter,
                public_name="beinf",
            ),
            GeneralFamilySpec(
                "zaga_lss",
                ZeroAdjustedGamma,
                frozenset(set()),
                getattr(ZeroAdjustedGamma, "reference_group", "mgcv"),
                family_type=ZeroAdjustedGamma,
                law_factory=lambda: ZERO_ADJUSTED_GAMMA_LAW,
                prediction_adapter=zero_adjusted_gamma_adapter,
                public_name="zaga",
            ),
            GeneralFamilySpec(
                "betabinomial_lss",
                BetaBinomialLocationPrecision,
                frozenset(set()),
                getattr(BetaBinomialLocationPrecision, "reference_group", "mgcv"),
                family_type=BetaBinomialLocationPrecision,
                law_factory=BetaBinomialLaw,
                public_name="betabinomial",
                prediction_context_names=("trials",),
            ),
            GeneralFamilySpec(
                "student_lss",
                StudentLocationScale,
                frozenset({"df"}),
                getattr(StudentLocationScale, "reference_group", "mgcv"),
                family_type=StudentLocationScale,
                law_factory=StudentTLaw,
                prediction_adapter=student_prediction,
                public_name="studentt",
            ),
            GeneralFamilySpec(
                "beta_lss",
                BetaLocationPrecision,
                frozenset(set()),
                getattr(BetaLocationPrecision, "reference_group", "mgcv"),
                family_type=BetaLocationPrecision,
                law_factory=BetaLaw,
                prediction_adapter=beta_prediction,
                public_name="beta",
            ),
            GeneralFamilySpec(
                "nb_lss",
                NegativeBinomialLocationDispersion,
                frozenset(set()),
                getattr(NegativeBinomialLocationDispersion, "reference_group", "mgcv"),
                family_type=NegativeBinomialLocationDispersion,
                law_factory=NegativeBinomialLaw,
                public_name="negativebinom",
            ),
            GeneralFamilySpec(
                "lognormal_lss",
                LogNormalLocationScale,
                frozenset(set()),
                getattr(LogNormalLocationScale, "reference_group", "mgcv"),
                family_type=LogNormalLocationScale,
                law_factory=LogNormalLaw,
                public_name="lognormal",
            ),
            GeneralFamilySpec(
                "weibull_lss",
                WeibullLocationShape,
                frozenset(set()),
                getattr(WeibullLocationShape, "reference_group", "mgcv"),
                family_type=WeibullLocationShape,
                law_factory=WeibullLaw,
                public_name="weibull",
            ),
            GeneralFamilySpec(
                "gaulss",
                gaulss,
                frozenset({"link", "b"}),
                getattr(gaulss, "reference_group", "mgcv"),
                family_type=GaulssFamily,
                law_factory=NormalLaw,
                public_name="normal",
            ),
            GeneralFamilySpec(
                "gammals",
                gammals,
                frozenset({"link", "b"}),
                getattr(gammals, "reference_group", "mgcv"),
                family_type=GammalsFamily,
                law_factory=GammaLaw,
                prediction_adapter=gamma_prediction,
                public_name="gamma",
            ),
            GeneralFamilySpec(
                "gumbls",
                GumblsFamily,
                frozenset({"link", "b"}),
                getattr(GumblsFamily, "reference_group", "mgcv"),
                family_type=GumblsFamily,
            ),
            GeneralFamilySpec(
                "ziplss",
                ZiplssFamily,
                frozenset({"link"}),
                getattr(ZiplssFamily, "reference_group", "mgcv"),
                family_type=ZiplssFamily,
            ),
            GeneralFamilySpec(
                "multinom",
                MultinomFamily,
                frozenset({"K"}),
                getattr(MultinomFamily, "reference_group", "mgcv"),
                family_type=MultinomFamily,
            ),
            GeneralFamilySpec(
                "gevlss",
                GevlssFamily,
                frozenset({"link"}),
                getattr(GevlssFamily, "reference_group", "mgcv"),
                family_type=GevlssFamily,
            ),
            GeneralFamilySpec(
                "shash",
                ShashFamily,
                frozenset({"link", "b", "phi_pen", "phiPen"}),
                getattr(ShashFamily, "reference_group", "mgcv"),
                family_type=ShashFamily,
            ),
            GeneralFamilySpec(
                "twlss",
                TwlssFamily,
                frozenset({"link", "a", "b"}),
                getattr(TwlssFamily, "reference_group", "mgcv"),
                family_type=TwlssFamily,
            ),
        )
    }
)


def family_spec_for_instance(family):
    """Exact types only: custom subclasses do not inherit probability claims."""
    return next(
        (
            spec
            for spec in GENERAL_FAMILIES.values()
            if type(family) is spec.family_type
        ),
        None,
    )


def make_distributional_family(family):
    """Resolve ordinary distribution names for GAMLSS, keeping mgcv names intact."""
    if isinstance(family, str):
        key = family.lower()
        if key == "gaussian":
            key = "normal"
        for spec in GENERAL_FAMILIES.values():
            if spec.public_name == key:
                return spec.factory()
    elif isinstance(family, dict):
        key = str(family.get("name", "")).lower()
        if key == "gaussian":
            key = "normal"
        for spec in GENERAL_FAMILIES.values():
            if spec.public_name == key:
                return _make_general_family(
                    spec.name, {k: v for k, v in family.items() if k != "name"}
                )
    return make_gam_family(family)


def _make_general_family(name, options):
    spec = GENERAL_FAMILIES[name]
    unknown = set(options) - spec.options
    if unknown:
        raise ValueError(f"Unsupported {name} options: {', '.join(sorted(unknown))}.")
    return spec.factory(**options)


def clone_gam_family(family: BaseFamily) -> BaseFamily:
    """Return an independent family instance for one model or fit session.

    Extended and general mgcv families carry mutable working parameters (for
    example ``theta`` and ordered-category cut points).  Sharing one Python
    family object across models would therefore also share fitted state, unlike
    R's copy-on-modify family lists.
    """
    if not isinstance(family, BaseFamily):
        raise TypeError(f"Expected a BaseFamily instance, got {type(family).__name__}.")
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


def _make_quasi_family(name, options):
    classes = {
        "quasi": QuasiFamily,
        "quasipoisson": QuasiPoissonFamily,
        "quasibinomial": QuasiBinomialFamily,
    }
    allowed = {"link", "variance", "power"} if name == "quasi" else {"link"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unsupported {name} options: {', '.join(sorted(unknown))}.")
    return classes[name](**options)


def make_gam_family(family):
    cls: type[BaseFamily]
    if isinstance(family, BaseFamily):
        return clone_gam_family(family)

    if family is None:
        return GaussianIdentityFamily()

    if isinstance(family, dict):
        name = str(family.get("name", "")).lower()
        if name in GENERAL_FAMILIES:
            return _make_general_family(
                name, {k: v for k, v in family.items() if k != "name"}
            )
        if name == "cpois":
            options = {k: v for k, v in family.items() if k != "name"}
            unknown = set(options) - {"link"}
            if unknown:
                raise ValueError(
                    f"Unsupported cpois options: {', '.join(sorted(unknown))}."
                )
            return CensoredPoissonFamily(**options)
        if name in {"cnorm", "clog", "bcg"}:
            options = {k: v for k, v in family.items() if k != "name"}
            unknown = set(options) - {"theta", "link"}
            if unknown:
                raise ValueError(
                    f"Unsupported {name} options: {', '.join(sorted(unknown))}."
                )
            return {
                "cnorm": CensoredNormalFamily,
                "clog": CensoredLogisticFamily,
                "bcg": CensoredBoxCoxGaussianFamily,
            }[name](**options)
        if name == "zip":
            options = {k: v for k, v in family.items() if k != "name"}
            unknown = set(options) - {"theta", "link", "b"}
            if unknown:
                raise ValueError(
                    f"Unsupported ziP options: {', '.join(sorted(unknown))}."
                )
            return ZeroInflatedPoissonFamily(**options)
        if name == "scat":
            options = {k: v for k, v in family.items() if k != "name"}
            unknown = set(options) - {"theta", "link", "min_df"}
            if unknown:
                raise ValueError(
                    f"Unsupported scat options: {', '.join(sorted(unknown))}."
                )
            return ScaledTFamily(**options)
        if name in {"quasi", "quasipoisson", "quasibinomial"}:
            return _make_quasi_family(
                name, {k: v for k, v in family.items() if k != "name"}
            )
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
        if name in {"inverse_gaussian", "inverse.gaussian", "inversegaussian"}:
            return InverseGaussianFamily(link=link or "1/mu^2")
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
        if key in GENERAL_FAMILIES:
            if not isinstance(spec, dict):
                if key == "multinom":
                    spec = {"K": spec}
                else:
                    raise ValueError(f"{key} tuple options must be a dictionary.")
            return _make_general_family(key, spec)
        if key == "cpois":
            return make_gam_family(
                {"name": key, **(spec if isinstance(spec, dict) else {"link": spec})}
            )
        if key in {"scat", "zip", "cnorm", "clog", "bcg"}:
            options = spec if isinstance(spec, dict) else {"theta": spec}
            return make_gam_family({"name": key, **options})
        if key in {"quasi", "quasipoisson", "quasibinomial"}:
            return _make_quasi_family(
                key, spec if isinstance(spec, dict) else {"link": spec}
            )
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
        if key in {"inverse_gaussian", "inverse.gaussian", "inversegaussian"}:
            link_spec = spec.get("link", "1/mu^2") if isinstance(spec, dict) else spec
            return InverseGaussianFamily(link=link_spec)
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
                return OrderedCategoricalFamily(R=int(cast(Any, spec)))
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
    if key == "bcg":
        return CensoredBoxCoxGaussianFamily()
    if key in GENERAL_FAMILIES:
        return _make_general_family(key, {})
    if key == "cpois":
        return CensoredPoissonFamily()
    if key == "cnorm":
        return CensoredNormalFamily()
    if key == "clog":
        return CensoredLogisticFamily()
    if key == "scat":
        return ScaledTFamily()
    if key == "zip":
        return ZeroInflatedPoissonFamily()
    if key in {"gaussian", "normal"}:
        return GaussianIdentityFamily()
    if key in {"binomial", "bernoulli", "logistic"}:
        return BinomialLogitFamily()
    if key in {"quasi", "quasipoisson", "quasibinomial"}:
        return _make_quasi_family(key, {})
    if key in {"poisson"}:
        return PoissonLogFamily()
    if key in {"inverse_gaussian", "inverse.gaussian", "inversegaussian"}:
        return InverseGaussianFamily()
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
        "Valid options: gaussian, binomial, poisson, gamma, inverse_gaussian, nb, negbin, tw, "
        "betar, ocat, scat, ziP, cnorm, clog, gaulss, gammals, quasi, quasipoisson, quasibinomial."
    )
