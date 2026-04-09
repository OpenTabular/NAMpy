# gam/model/__init__.py
"""Mixin classes that implement GAM basemodel behaviour."""
from .gam_data import _GAMDataMixin
from .gam_solve import _GAMSolveMixin
from .gam_specs import _GAMSpecsMixin

__all__ = ["_GAMDataMixin", "_GAMSpecsMixin", "_GAMSolveMixin"]
