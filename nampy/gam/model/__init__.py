# gam/model/__init__.py
"""GAM model mixins and user-facing model class."""

from .api import GAM
from .gam_data import _GAMDataMixin
from .gam_solve import _GAMSolveMixin
from .gam_specs import _GAMSpecsMixin

__all__ = ["GAM", "_GAMDataMixin", "_GAMSpecsMixin", "_GAMSolveMixin"]
