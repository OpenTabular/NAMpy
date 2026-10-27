"""Single inventory for statistical exports and reference-page discovery."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class StatisticalEstimator:
    name: str
    module: str
    group: str


ESTIMATORS = MappingProxyType(
    {
        name: StatisticalEstimator(name, module, group)
        for name, module, group in (
            ("GAMRegressor", ".gam", "likelihood"),
        ("GAMClassifier", ".gam", "likelihood"),
            ("GAMLSS", ".gamlss", "likelihood"),
            ("ExpectileGAM", ".asymmetric_gam", "asymmetric"),
            ("QuantileGAM", ".asymmetric_gam", "asymmetric"),
        )
    }
)
