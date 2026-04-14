"""Compilation boundary between declarative specs and numerical solving."""

from .structures import (
    CompiledModel,
    CompiledPenalty,
    CompiledPredictor,
    CompiledTerm,
    PenaltySpec,
)


def compile_model(*args, **kwargs):
    from .compile_model import compile_model as _compile_model

    return _compile_model(*args, **kwargs)


def compile_predictors(*args, **kwargs):
    from .compile_predictors import compile_predictors as _compile_predictors

    return _compile_predictors(*args, **kwargs)


def construct_smooth(*args, **kwargs):
    from .construct import construct_smooth as _construct_smooth

    return _construct_smooth(*args, **kwargs)


def build_term_matrix(*args, **kwargs):
    from .construct import build_term_matrix as _build_term_matrix

    return _build_term_matrix(*args, **kwargs)


def instantiate_term(*args, **kwargs):
    from .factory import instantiate_term as _instantiate_term

    return _instantiate_term(*args, **kwargs)


def instantiate_predictor_terms(*args, **kwargs):
    from .factory import instantiate_predictor_terms as _instantiate_predictor_terms

    return _instantiate_predictor_terms(*args, **kwargs)


def attach_shared_basis_metadata(*args, **kwargs):
    from .linked_basis import (
        attach_shared_basis_metadata as _attach_shared_basis_metadata,
    )

    return _attach_shared_basis_metadata(*args, **kwargs)


def __getattr__(name):
    if name == "ConstructedSmooth":
        from .construct import ConstructedSmooth

        return ConstructedSmooth
    raise AttributeError(name)


__all__ = [
    "PenaltySpec",
    "CompiledPenalty",
    "CompiledPredictor",
    "CompiledTerm",
    "CompiledModel",
    "compile_model",
    "compile_predictors",
    "ConstructedSmooth",
    "construct_smooth",
    "build_term_matrix",
    "instantiate_term",
    "instantiate_predictor_terms",
    "attach_shared_basis_metadata",
]
