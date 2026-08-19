"""Hybrid GAM + neural backends.

The only package allowed to import both the mgcv-parity GAM backend and the
Torch neural backend. Everything here is a documented NON-mgcv model: the GAM
stage is exact mgcv parity, but any composed or jointly trained result has no
mgcv counterpart and must never enter the parity suites.
"""

from .compiled_terms import CompiledGAMTerms, CompiledGAMTermsModule
from .joint import GAMNetClassifier, GAMNetRegressor
from .net import GAMNet
from .residual import GAMResidualClassifier, GAMResidualRegressor

__all__ = [
    "GAMResidualRegressor",
    "GAMResidualClassifier",
    "GAMNetRegressor",
    "GAMNetClassifier",
    "GAMNet",
    "CompiledGAMTerms",
    "CompiledGAMTermsModule",
]
