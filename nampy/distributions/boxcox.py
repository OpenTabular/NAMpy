"""BCT and BCPE probability laws, independent of both fitting backends.

References
----------
- Rigby, R. A., and Stasinopoulos, D. M. (2004). Smooth centile curves for skew and
  kurtotic data modelled using the Box-Cox power exponential distribution.
  https://doi.org/10.1002/sim.1861
- Rigby, R. A., and Stasinopoulos, D. M. (2006). Using the Box-Cox t distribution in
  GAMLSS to model skewness and kurtosis.
  https://doi.org/10.1191/1471082X06st122oa
"""

from dataclasses import dataclass

from ._boxcox import _OPERATIONS, SymmetricBoxCoxLaw
from ._boxcox_latent import PowerExponentialLatent, StudentLatent
from .contracts import DistributionSpec


@dataclass(frozen=True)
class BCTLaw(SymmetricBoxCoxLaw):
    """Normalized Box-Cox t: mu>0, sigma>0, real nu and Student-t df tau>0.

    z=expm1(nu*log(y/mu))/(nu*sigma), with its log limit at nu=0.
    The latent Student-t is truncated to positive-response support. Sigma is
    its scale, not its SD; mu is not generally the mean or exact median.
    The exact Student-t law is retained above the upstream density-only
    Gaussian-switch threshold. Moments have explicit existence and numerical integration gates. The
    bct fitting family uses four predictors; GAM fitting requires fixed smoothing.
    """

    spec = DistributionSpec(
        "bct",
        ("mu", "sigma", "nu", "tau"),
        _OPERATIONS | {"mean", "variance"},
        "positive_truncated_box_cox_student_t_v1",
        "continuous",
    )
    latent = StudentLatent()


@dataclass(frozen=True)
class BCPELaw(SymmetricBoxCoxLaw):
    """Normalized Box-Cox power exponential, with mu,sigma,tau>0 and real nu.

    The symmetric latent PE has variance one: generalized-normal shape tau
    and scale sqrt(Gamma(1/tau)/Gamma(3/tau)). Tau=2 is exactly BCCG;
    at nu=0 the response is log-PE. Sigma scales the latent variable before
    truncation; mu is not generally a response mean or exact median.
    Moments have explicit existence and numerical integration gates. The bcpe
    fitting family uses four predictors; GAM fitting requires fixed smoothing.
    Untruncated-family aliases are unsupported.
    """

    spec = DistributionSpec(
        "bcpe",
        ("mu", "sigma", "nu", "tau"),
        _OPERATIONS | {"mean", "variance"},
        "positive_truncated_box_cox_variance_one_power_exponential_v1",
        "continuous",
    )
    latent = PowerExponentialLatent()
