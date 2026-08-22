"""Public estimator family generated from the NBM-SPAM declaration."""

from ._registered import estimator_family

_family = estimator_family("nbm_spam", module_name=__name__)
NBMSPAMRegressor = _family.regressor
NBMSPAMClassifier = _family.classifier
NBMSPAMLSS = _family.lss

__all__ = ["NBMSPAMRegressor", "NBMSPAMClassifier", "NBMSPAMLSS"]
