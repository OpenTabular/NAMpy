"""Shared torch building blocks for the neural architectures.

Reusable, composable components: base module classes, MLPs, embeddings,
normalization, transformer layers, interaction scaffolding, spline and tree
layers, sparse activations, and the NODE-style oblivious-tree blocks.
Model architectures themselves live one level up in ``nampy.neural.architectures``.
"""

from .additive_trees import (
    GAM_ODST,
    GAMAdditiveMixin,
    GAMAttBlock,
    GAMAttODST,
    GAMBlock,
)
from .base_model import BaseModel, ModuleWithInit
from .block_masked_additive import BlockMaskedAdditiveNetwork, MaskedLinear
from .concept_bases import ConceptNNBasesNary
from .embeddings import EmbeddingLayer, OneHotEncoding
from .interactions import (
    apply_feature_dropout,
    create_interaction_networks,
    interaction_forward,
    sum_feature_dims,
)
from .mlp import MLP, make_activation, make_norm, resolve_norm_name
from .nam import CenteredReLU, ExU, NAMFeatureNN
from .normalization import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    LearnableLayerScaling,
    RMSNorm,
)
from .oblivious_trees import ODST, ODSTBlock
from .regularization import (
    evaluating,
    mean_squared_term_outputs,
    normalized_parameter_l2,
)
from .sparse_activations import (
    EM15Temp,
    entmax15,
    entmoid15,
    sparsemax,
    sparsemoid,
)
from .splines import CubicSplineLayer
from .tensor_utils import check_numpy, process_in_chunks
from .term_extraction import (
    aggregate_term_values,
    build_terms_frame,
    center_main_effects,
    convert_onehot_vector_to_integers,
    purify_interactions,
    terms_from_feature_selectors,
)
from .transformer import GLU, CustomTransformerEncoderLayer
from .trees import NeuralDecisionTree

__all__ = [
    "BaseModel",
    "ModuleWithInit",
    "BlockMaskedAdditiveNetwork",
    "MaskedLinear",
    "MLP",
    "make_activation",
    "make_norm",
    "resolve_norm_name",
    "EmbeddingLayer",
    "OneHotEncoding",
    "CustomTransformerEncoderLayer",
    "GLU",
    "RMSNorm",
    "LayerNorm",
    "BatchNorm",
    "InstanceNorm",
    "GroupNorm",
    "LearnableLayerScaling",
    "CubicSplineLayer",
    "NeuralDecisionTree",
    "ConceptNNBasesNary",
    "apply_feature_dropout",
    "create_interaction_networks",
    "interaction_forward",
    "sum_feature_dims",
    "entmax15",
    "entmoid15",
    "sparsemax",
    "sparsemoid",
    "EM15Temp",
    "ODST",
    "ODSTBlock",
    "GAM_ODST",
    "GAMAttODST",
    "GAMBlock",
    "GAMAttBlock",
    "GAMAdditiveMixin",
    "aggregate_term_values",
    "build_terms_frame",
    "center_main_effects",
    "convert_onehot_vector_to_integers",
    "purify_interactions",
    "terms_from_feature_selectors",
    "check_numpy",
    "process_in_chunks",
    "CenteredReLU",
    "ExU",
    "NAMFeatureNN",
    "mean_squared_term_outputs",
    "normalized_parameter_l2",
    "evaluating",
]
