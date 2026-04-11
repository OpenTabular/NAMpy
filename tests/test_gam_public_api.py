import nampy
import nampy.models
from nampy.gam import GAM


def test_gam_public_api_exports_gam_directly():
    assert nampy.GAM is GAM


def test_removed_gam_wrapper_classes_not_exported():
    assert not hasattr(nampy, "GAMRegressor")
    assert not hasattr(nampy.models, "GAMRegressor")
    assert not hasattr(nampy.models, "GAMClassifier")
