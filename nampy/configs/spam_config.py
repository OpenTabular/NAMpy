from dataclasses import dataclass, field
from typing import List, Union

import torch.nn as nn


@dataclass
class DefaultSPAMConfig:
    """
    Configuration dataclass for the Scalable Polynomial Additive Model (SPAM).

    SPAM learns inherently-interpretable classifiers by leveraging low-rank tensor
    decompositions of polynomials. Two variants are supported:
      - SPAM-LINEAR: uses geometric rescaling x̃_l = sign(x)·|x|^(1/l) per interaction order.
      - SPAM-NEURAL: replaces the rescaled features with per-feature MLP sub-networks,
        i.e. F_l(x) = [f_l1(x1), ..., f_ld(xd)], identical in spirit to NAMs.

    The model computes:
        P(x) = b + <u1, F1(x)> + Σ_{i=1}^{r2} λ2i·<u2i, F2(x)>²
                                + Σ_{i=1}^{r3} λ3i·<u3i, F3(x)>³ + ...

    Parameters
    ----------
    # --- Training ---
    lr : float
        Learning rate for the optimizer.
    lr_patience : int
        Number of epochs with no improvement after which LR is reduced.
    weight_decay : float
        L2 weight decay / regularisation applied by the optimizer.
    lr_factor : float
        Factor by which the learning rate is reduced on plateau.

    # --- Polynomial structure ---
    degree : int
        Maximum polynomial degree k (i.e. highest order of feature interaction).
        degree=1 is equivalent to a linear model; degree=2 adds pairwise interactions.
        Interpretability degrades beyond degree=2 per the paper.
    rank : Union[int, List[int]]
        Rank(s) of the tensor decomposition.
        - If an int, the same rank is used for every degree l = 2 … k.
          The order-1 term always uses rank 1 (a single weight vector u1).
        - If a list, must have exactly (degree - 1) elements, one per degree
          l = 2 … k, e.g. [100, 50] for degree=3.
        A larger rank increases model capacity but also model complexity.
        The paper shows performance plateaus after a moderate cumulative rank.

    # --- Variant ---
    use_neural : bool
        If False (default): SPAM-LINEAR — uses geometric rescaling x̃_l = sign(x)·|x|^(1/l).
        If True: SPAM-NEURAL — each degree l gets its own per-feature MLP sub-network
        F_l(x) = [f_l1(x1), ..., f_ld(xd)], identical to a NAM applied per order.
        SPAM-NEURAL generally outperforms SPAM-LINEAR (see Table 1 in the paper)
        but is more expensive and harder to scale to very high dimensions.

    # --- SPAM-NEURAL sub-network architecture ---
    layer_sizes : List[int]
        Hidden layer widths for each per-feature MLP (SPAM-NEURAL only).
        These sub-networks map a scalar feature value to a scalar transformed value.
        Default is [64, 64], matching the NAM sub-network used in the paper.
    activation : nn.Module
        Activation function for the MLP sub-networks (SPAM-NEURAL only).

    # --- Regularisation ---
    dropout : float
        Dropout probability applied to the singular values λ during each forward pass.
        This is "basis dropout" from Section 3.2 of the paper: setting λ to zero at
        random forces the model to learn robust, non-overlapping basis directions.
    feature_dropout : float
        Dropout probability applied across feature contributions before summation.
        Mirrors the feature_dropout used in NAM.
    l1_lambda : float
        Coefficient for an L1 sparsity penalty on the basis vectors U = {u_li}.
        Corresponds to R(θ) = ‖U‖₁ in Equation 2 of the paper. When non-zero,
        only a fraction of pairwise interactions remain active (the paper shows
        ~6% suffice for competitive performance on CUB-200).

    # --- Multi-class ---
    shared_bases : bool
        If True (default) and num_classes > 1, the basis vectors u_li are shared
        across all classes while the singular values λ are learned per class.
        This reduces parameters from O(2drC) to O((d+r)C + rd) and prevents
        overfitting for large C (see Section 3.2 of the paper).
        If False, independent parameters are learned per class.

    # --- Intercept ---
    intercept : bool
        Whether to learn a bias / intercept term b.
    """

    lr: float = 1e-3
    lr_patience: int = 10
    weight_decay: float = 1e-5
    lr_factor: float = 0.1
    degree: int = 2
    rank: Union[int, List[int]] = 100
    use_neural: bool = False
    layer_sizes: List[int] = field(default_factory=lambda: [64, 64])
    activation: nn.Module = field(default_factory=nn.GELU)
    dropout: float = 0.0
    feature_dropout: float = 0.0
    l1_lambda: float = 0.0
    shared_bases: bool = True
    intercept: bool = True