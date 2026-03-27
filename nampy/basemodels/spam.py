from itertools import combinations
from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..configs.spam_config import DefaultSPAMConfig
from .basemodel import BaseModel


class SPAM(BaseModel):
    """
    Scalable Polynomial Additive Model (SPAM).

    Implements both SPAM-LINEAR and SPAM-NEURAL variants from the NeurIPS 2022 paper.
    Inherits from NAMpy's BaseModel (PyTorch Lightning) for full compatibility with
    the NAMpy training API (fit, predict, predict_proba, LSS, etc.).

    Parameters
    ----------
    cat_feature_info : dict
        Mapping from categorical feature name → {"dimension": int, ...}.
        Dimension is the size of the encoded representation (e.g. 1 for ordinal,
        n for one-hot).
    num_feature_info : dict
        Mapping from numerical feature name → {"dimension": int, ...}.
        Dimension is typically 1 for scalar features.
    num_classes : int, optional
        Number of output classes. 1 for regression / binary classification,
        >1 for multi-class. Default is 1.
    config : DefaultSPAMConfig, optional
        Hyperparameter configuration. See DefaultSPAMConfig for full documentation.
    **kwargs
        Passed through to BaseModel (e.g. optimizer overrides).

    Attributes
    ----------
    d : int
        Total input dimensionality (sum of all feature dimensions).
    feature_names : list[str]
        Ordered list of all feature names (numerical first, then categorical).
        Used to map parameter indices back to interpretable feature names.
    feature_dims : list[int]
        Dimension of each feature, parallel to feature_names.
    feature_offsets : list[int]
        Cumulative offsets into the concatenated feature vector, used when
        extracting per-feature slices for the output dict and for SPAM-NEURAL.
    ranks : list[int]
        Resolved rank per degree, indexed as ranks[0] = r2, ranks[1] = r3, ...
        (order-1 always uses a single weight vector, so no rank needed there).
    order1_weights : nn.Linear
        Linear layer implementing the order-1 term: <u1, x> → shape (d, num_classes).
        For SPAM-NEURAL this operates on F1(x); for SPAM-LINEAR on x directly.
    basis_vectors : nn.ParameterList
        One entry per degree l = 2 … k.
        Each entry: nn.Parameter of shape (r_l, d), the basis matrix U^(l).
        Shared across classes when shared_bases=True.
    singular_values : nn.ParameterList
        One entry per degree l = 2 … k.
        Shape (num_classes, r_l) when shared_bases=True and num_classes > 1,
        else (r_l,). These are the λ^(l) eigenvalues.
    neural_transforms : nn.ModuleList or None
        Only present when use_neural=True. One nn.ModuleDict per degree l = 2 … k,
        where each ModuleDict maps feature_name → per-feature MLP.
        The order-1 MLP sub-networks are stored in order1_neural (nn.ModuleDict).
    order1_neural : nn.ModuleDict or None
        Per-feature MLP sub-networks for the order-1 term (SPAM-NEURAL only).
        Maps feature_name → nn.Sequential that transforms a scalar input.
    feature_dropout : nn.Dropout
        Applied to the concatenated per-feature contribution vector.
    lambda_dropout : nn.Dropout
        Applied to the singular values λ at each forward pass (basis dropout
        from Section 3.2 of the paper).
    intercept : nn.Parameter or None
        Learnable bias term b of shape (num_classes,), or None if disabled.
    """

    def __init__(
        self,
        cat_feature_info: dict,
        num_feature_info: dict,
        num_classes: int = 1,
        config: DefaultSPAMConfig = DefaultSPAMConfig(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)

        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.num_classes = num_classes

        degree = self.hparams.get("degree", config.degree)
        use_neural = self.hparams.get("use_neural", config.use_neural)
        shared_bases = self.hparams.get("shared_bases", config.shared_bases)
        l1_lambda = self.hparams.get("l1_lambda", config.l1_lambda)

        self.degree = degree
        self.use_neural = use_neural
        self.shared_bases = shared_bases
        self.l1_lambda = l1_lambda

        raw_rank = self.hparams.get("rank", config.rank)
        self.ranks: List[int] = self._resolve_ranks(raw_rank, degree)

        self.feature_names: List[str] = (
            list(num_feature_info.keys()) + list(cat_feature_info.keys())
        )
        self.feature_dims: List[int] = [
            info["dimension"] for info in num_feature_info.values()
        ] + [
            info["dimension"] for info in cat_feature_info.values()
        ]
        self.feature_offsets: List[int] = []
        offset = 0
        for dim in self.feature_dims:
            self.feature_offsets.append(offset)
            offset += dim
        self.d: int = offset  

        self.n_features: int  = len(self.feature_names)
        self.d_effective: int = self.n_features if use_neural else self.d


        if use_neural:
            self.effective_offsets: List[int] = list(range(self.n_features))
            self.effective_dims: List[int]    = [1] * self.n_features
        else:
            self.effective_offsets = self.feature_offsets
            self.effective_dims    = self.feature_dims

        # Order-1
        self.order1_weights = nn.Linear(self.d_effective, num_classes, bias=False)

        self.order1_neural: Optional[nn.ModuleDict] = None
        if use_neural:
            self.order1_neural = nn.ModuleDict({
                name: self._build_feature_mlp(dim, config)
                for name, dim in zip(self.feature_names, self.feature_dims)
            })

        self.basis_vectors = nn.ParameterList()
        self.singular_values = nn.ParameterList()

        for r_l in self.ranks:  # one entry per degree l = 2 … k
            U = nn.Parameter(torch.randn(r_l, self.d_effective) * 0.01)
            self.basis_vectors.append(U)

            
            if shared_bases and num_classes > 1:
                lam = nn.Parameter(torch.randn(num_classes, r_l) * 0.01)
            else:
                lam = nn.Parameter(torch.randn(r_l) * 0.01)
            self.singular_values.append(lam)

        # SPAM-NEURAL
        self.neural_transforms: Optional[nn.ModuleList] = None
        if use_neural and degree >= 2:
            self.neural_transforms = nn.ModuleList()
            for _ in self.ranks:  # one ModuleDict per degree l = 2 … k
                degree_nets = nn.ModuleDict({
                    name: self._build_feature_mlp(dim, config)
                    for name, dim in zip(self.feature_names, self.feature_dims)
                })
                self.neural_transforms.append(degree_nets)

        self.feature_dropout = nn.Dropout(
            self.hparams.get("feature_dropout", config.feature_dropout)
        )
        self.lambda_dropout = nn.Dropout(
            self.hparams.get("dropout", config.dropout)
        )

        if self.hparams.get("intercept", config.intercept):
            self.intercept = nn.Parameter(torch.zeros(num_classes))
        else:
            self.intercept = None


    @staticmethod
    def _resolve_ranks(rank: Union[int, List[int]], degree: int) -> List[int]:
        """
        Convert the user-supplied rank specification into a list of length
        (degree - 1), one rank per degree l = 2 … k.

        Parameters
        ----------
        rank : int or list[int]
            If int, broadcast to all higher-order degrees.
            If list, must have exactly (degree - 1) elements.
        degree : int
            Maximum polynomial degree k.

        Returns
        -------
        list[int]
            Ranks indexed as [r2, r3, ..., rk].  Empty list when degree < 2.

        Raises
        ------
        ValueError
            If a list rank is provided but its length != degree - 1.
        """
        n_higher = max(0, degree - 1) 
        if n_higher == 0:
            return []

        if isinstance(rank, int):
            return [rank] * n_higher

        if isinstance(rank, (list, tuple)):
            if len(rank) != n_higher:
                raise ValueError(
                    f"When rank is a list it must have exactly (degree - 1) = {n_higher} "
                    f"elements (one per degree l = 2 … {degree}), "
                    f"but got {len(rank)} elements."
                )
            return list(rank)

        raise TypeError(
            f"rank must be an int or a list of ints, got {type(rank).__name__}."
        )

   
    def _build_feature_mlp(
        self,
        input_dim: int,
        config: DefaultSPAMConfig,
    ) -> nn.Sequential:
        """
        Build a per-feature MLP sub-network for SPAM-NEURAL.

        The network maps a single feature of dimension `input_dim` to a scalar
        output of dimension 1. This matches the NAM sub-network convention in
        the paper (Section 3.2 and Appendix C).

        Architecture:
            Linear(input_dim → layer_sizes[0])
            Activation
            [Dropout]
            Linear(layer_sizes[i-1] → layer_sizes[i])
            Activation
            [Dropout]   ×  len(layer_sizes)-1
            Linear(layer_sizes[-1] → 1)

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input feature (typically 1 for scalar features).
        config : DefaultSPAMConfig
            Provides layer_sizes, activation, and dropout.

        Returns
        -------
        nn.Sequential
            The per-feature MLP sub-network.
        """
        layer_sizes = self.hparams.get("layer_sizes", config.layer_sizes)
        activation_cls = self.hparams.get("activation", config.activation)
        dropout_p = self.hparams.get("dropout", config.dropout)

        layers = nn.Sequential()

        layers.add_module("input", nn.Linear(input_dim, layer_sizes[0]))
        layers.add_module("act_0", activation_cls() if isinstance(activation_cls, type) else activation_cls)
        if dropout_p > 0.0:
            layers.add_module("drop_0", nn.Dropout(dropout_p))

        for i in range(1, len(layer_sizes)):
            layers.add_module(f"linear_{i}", nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            layers.add_module(
                f"act_{i}",
                activation_cls() if isinstance(activation_cls, type) else activation_cls,
            )
            if dropout_p > 0.0:
                layers.add_module(f"drop_{i}", nn.Dropout(dropout_p))

        # 1 output per feature
        layers.add_module(
            f"output",
            nn.Linear(layer_sizes[-1], 1),
        )
        return layers


    @staticmethod
    def _geometric_rescale(x: torch.Tensor, l: int) -> torch.Tensor:
        """
        Apply the geometric rescaling from Section 3.2 of the paper:

            x̃_l = sign(x) · |x|^(1/l)

        This ensures that higher-order products stay on the same scale as the
        original features. Without it, a pairwise product x_i·x_j for unit-bounded
        features would shrink the value (e.g. 0.5·0.6 = 0.3), whereas the
        geometric mean √(x_i·x_j) = √(0.5·0.6) ≈ 0.55 preserves the scale.

        It also tightens the variance of higher-order terms: for uncorrelated
        features with V(x_i) = V(x_j) = σ², the product variance grows as
        V(x_i·x_j) ≈ σ⁴, whereas the rescaled version has V(√x_i·√x_j) ≈ σ²,
        keeping all interaction terms on the same gradient scale.

        For l=1 the identity x̃_1 = x is returned (1/1 = 1).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, d).
        l : int
            Interaction order (degree). Must be >= 1.

        Returns
        -------
        torch.Tensor
            Rescaled tensor of shape (batch, d), same dtype as x.
        """
        if l == 1:
            return x
        return torch.sign(x) * torch.abs(x).pow(1.0 / l)

 
    def _apply_neural_transform(
        self,
        num_features: dict,
        cat_features: dict,
        degree_nets: nn.ModuleDict,
    ) -> torch.Tensor:
        """
        Apply per-feature MLP sub-networks for SPAM-NEURAL and assemble the
        transformed feature tensor F_l(x).

        For each feature the sub-network maps its raw value(s) to a single
        scalar, then all scalars are concatenated into a (batch, d) tensor,
        analogous to how a NAM assembles feature contributions.

        Each sub-network in degree_nets maps:
            feature tensor (batch, feature_dim) → (batch, 1)

        The outputs are concatenated in the same order as self.feature_names
        (numerical first, then categorical), yielding F_l(x) of shape (batch, d).

        Parameters
        ----------
        num_features : dict
            Numerical feature tensors, name → (batch, dim).
        cat_features : dict
            Categorical feature tensors, name → (batch, dim).
        degree_nets : nn.ModuleDict
            Per-feature MLP sub-networks for this degree, name → nn.Sequential.

        Returns
        -------
        torch.Tensor
            Transformed feature matrix F_l(x) of shape (batch, d).
            Each column j corresponds to f_lj(x_j), the scalar output of
            the j-th feature's sub-network.
        """
        outputs = []
        all_features = {**num_features, **cat_features}
        for name in self.feature_names:
            feat = all_features[name].float()
            outputs.append(degree_nets[name](feat))       
        return torch.cat(outputs, dim=1)


    def _compute_poly_contribution(
        self,
        z: torch.Tensor,
        degree_idx: int,
    ) -> torch.Tensor:
        """
        Compute the contribution of a single polynomial order l = degree_idx + 2.

        Implements Equation (1) from the paper (one term in the sum):

            Σ_{i=1}^{r_l}  λ_li · <u_li, z>^l

        Where:
          - z   is F_l(x) — either geometrically rescaled (LINEAR) or neural (NEURAL)
          - U^(l) = basis_vectors[degree_idx], shape (r_l, d)
          - λ^(l) = singular_values[degree_idx], shape (r_l,) or (num_classes, r_l)

        The inner product <u_li, z> for a batch is computed as:

            projections = z @ U^(l).T        # (batch, r_l)

        Then raised element-wise to the l-th power:

            projections^l                    # (batch, r_l)

        Basis dropout is applied to λ (not to projections), consistent with
        Section 3.2: "we apply dropout to λ to ensure the network learns
        robust basis directions."

        For multi-class with shared_bases:
          - λ has shape (num_classes, r_l)
          - output = projections^l @ λ.T     # (batch, num_classes)

        For single-class or shared_bases=False:
          - λ has shape (r_l,)
          - output = projections^l · λ summed over r_l  # (batch, 1)

        Parameters
        ----------
        z : torch.Tensor
            Transformed feature matrix of shape (batch, d).
            For LINEAR: z = x̃_l.  For NEURAL: z = F_l(x).
        degree_idx : int
            Index into self.basis_vectors / self.singular_values.
            Corresponds to degree l = degree_idx + 2.

        Returns
        -------
        torch.Tensor
            Contribution tensor of shape (batch, num_classes).
        """
        l = degree_idx + 2                   

        U   = self.basis_vectors[degree_idx]      
        lam = self.singular_values[degree_idx]   

        projections = z @ U.t()               

        projections = projections.pow(l)            

        lam_dropped = self.lambda_dropout(lam)        

        if lam_dropped.dim() == 2:
            contribution = projections @ lam_dropped.t()
        else:
            contribution = (projections * lam_dropped).sum(dim=1, keepdim=True)

        return contribution                           


    def _compute_unary_contributions(
        self,
        z1: torch.Tensor,
    ) -> dict:
        """
        Compute the per-feature contribution of the order-1 (linear) term.

        The order-1 term is <u1, F1(x)>, a standard linear map from d → num_classes.
        To decompose this into individual feature contributions (needed for the
        interpretability output dict), we split the weight matrix by feature offsets
        and apply each slice to the corresponding feature block in z1.

        For feature j with offset o_j and dimension d_j:
            contrib_j = z1[:, o_j : o_j + d_j] @ W1[:, o_j : o_j + d_j].T
            shape: (batch, num_classes)

        Parameters
        ----------
        z1 : torch.Tensor
            The order-1 input, shape (batch, d). For LINEAR this is x; for
            NEURAL this is the stacked output of order1_neural sub-networks.

        Returns
        -------
        dict
            Mapping feature_name → (batch, num_classes) unary contribution tensor.
        """
        W1 = self.order1_weights.weight          

        unary = {}
        for name, offset, dim in zip(
            self.feature_names, self.effective_offsets, self.effective_dims
        ):
            z_feat = z1[:, offset : offset + dim]      
            W_feat = W1[:, offset : offset + dim]       
            unary[name] = z_feat @ W_feat.t()

        return unary


    def _compute_pairwise_contributions(
        self,
        z2: torch.Tensor,
    ) -> dict:
        """
        Compute the interpretable pairwise contribution for every feature pair
        (i, j) from the degree-2 polynomial term.

        From Table 5 of the paper, the pairwise importance for features x_i, x_j is:

            (Σ_{k=1}^{r2} λ2k · u2k_i · u2k_j) · z2_i · z2_j

        Where z2_i = x̃2_i = sign(x_i)·√|x_i| for LINEAR (or f_2i(x_i) for NEURAL).

        This tells us exactly how much features i and j jointly contribute to the
        prediction, which is the primary interpretability output used in the human
        subject evaluation (Section 5, Table 5).

        We compute all n_features*(n_features-1)/2 pairs efficiently:
          1. For each basis vector k, compute the outer product of its loadings:
                outer_k[i,j] = λ2k · u2k_i · u2k_j
          2. Sum over k to get the effective pairwise weight matrix W_pair[i,j].
          3. For each pair (i,j): contribution = W_pair[i,j] · z2_i · z2_j

        This is O(r2·d²) per batch, which is efficient for moderate d.

        Parameters
        ----------
        z2 : torch.Tensor
            Degree-2 input, shape (batch, d). For LINEAR: x̃_2; for NEURAL: F_2(x).

        Returns
        -------
        dict
            Mapping "feat_i:feat_j" → (batch, num_classes) pairwise contribution.
            Keys use colon separator, consistent with NAM interaction convention.
            Only populated when degree >= 2.
        """
        U2  = self.basis_vectors[0]          
        lam = self.singular_values[0]         

        lam_dropped = self.lambda_dropout(lam)

        pairwise = {}

        for i, j in combinations(range(len(self.feature_names)), 2):
            name_i = self.feature_names[i]
            name_j = self.feature_names[j]
            key = f"{name_i}:{name_j}"

            offset_i, dim_i = self.effective_offsets[i], self.effective_dims[i]
            offset_j, dim_j = self.effective_offsets[j], self.effective_dims[j]


            u_i = U2[:, offset_i : offset_i + dim_i].mean(dim=1)  
            u_j = U2[:, offset_j : offset_j + dim_j].mean(dim=1)  


            if lam_dropped.dim() == 2:

                element_weights = lam_dropped * u_i.unsqueeze(0) * u_j.unsqueeze(0)

                W_pair = element_weights.sum(dim=1)        
            else:
 
                W_pair = (lam_dropped * u_i * u_j).sum()   


            z2_i = z2[:, offset_i : offset_i + dim_i].mean(dim=1, keepdim=True)  # (batch, 1)
            z2_j = z2[:, offset_j : offset_j + dim_j].mean(dim=1, keepdim=True)  # (batch, 1)

 
            interaction = z2_i * z2_j                      

            if lam_dropped.dim() == 2:
                pairwise[key] = interaction * W_pair.unsqueeze(0)
            else:
                contrib = interaction * W_pair            
                pairwise[key] = contrib.expand(-1, self.num_classes)

        return pairwise


    def _compute_l1_penalty(self) -> torch.Tensor:
        """
        Compute the L1 sparsity penalty on all basis vectors U^(l).

        Implements R(θ) = ‖U‖₁ from Equation (2) of the paper, scaled by
        self.l1_lambda. This penalty drives individual basis vector entries
        toward zero, producing sparse feature interactions where only a
        fraction of pairwise terms remain active.

        The paper (Figure 1C) shows that on CUB-200 only ~6% of pairwise
        interactions need to be active for competitive accuracy.

        Note: this penalty is intended to be added to the task loss in the
        training step. BaseModel's training_step will call forward() and then
        add this via the loss hook — the integration point is in Part 3
        (the forward / loss methods).

        Returns
        -------
        torch.Tensor
            Scalar L1 penalty: l1_lambda · Σ_l ‖U^(l)‖₁.
            Returns zero (no grad) when l1_lambda == 0 or degree < 2.
        """
        if self.l1_lambda == 0.0 or len(self.basis_vectors) == 0:
            return torch.tensor(0.0, device=self.basis_vectors[0].device
                                if len(self.basis_vectors) > 0 else "cpu")

        penalty = sum(U.abs().sum() for U in self.basis_vectors)
        return self.l1_lambda * penalty


    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of the SPAM model.

        Computes the full polynomial prediction and decomposes it into
        interpretable per-feature and pairwise contributions.

        The computation follows Equation (3) / (1) of the paper:

            P(x) = b
                 + <u1, F1(x)>                          order-1
                 + Σ_i λ2i · <u2i, F2(x)>²             order-2
                 + Σ_i λ3i · <u3i, F3(x)>³             order-3
                 + ...

        Where F_l(x) is either:
          - x̃_l = sign(x)·|x|^(1/l)      [SPAM-LINEAR, use_neural=False]
          - stacked per-feature MLP scalars [SPAM-NEURAL, use_neural=True]

        Steps
        -----
        1.  Concatenate all raw features → x  (batch, d_raw).
        2.  [SPAM-NEURAL] Apply per-degree MLP transforms to get F_l tensors.
            [SPAM-LINEAR] Apply geometric rescaling to get x̃_l tensors.
        3.  Order-1 term: self.order1_weights(z1) → (batch, num_classes).
            Store per-feature unary contributions for the output dict.
        4.  For l = 2 … k: compute _compute_poly_contribution → sum into output.
        5.  Apply feature_dropout across the stacked per-term contributions
            before the final sum (mirrors NAM's feature-level dropout).
        6.  Add intercept b.
        7.  If degree >= 2: compute pairwise interpretability contributions.
        8.  Return result dict.

        Parameters
        ----------
        num_features : dict
            Numerical feature tensors, name → (batch, dim).
        cat_features : dict
            Categorical feature tensors, name → (batch, dim).

        Returns
        -------
        dict with keys:
            "output"        : (batch, num_classes) — final prediction logit/value.
            "intercept"     : (num_classes,) parameter, only if intercept=True.
            <feature_name>  : (batch, num_classes) unary contribution per feature.
            "<f_i>:<f_j>"   : (batch, num_classes) pairwise contribution per pair,
                              only populated when degree >= 2.
        """
        all_raw: dict = {**num_features, **cat_features}
        x = torch.cat(
            [all_raw[name].float() for name in self.feature_names],
            dim=1,
        ) 

        if self.use_neural:
            z1 = self._apply_neural_transform(
                num_features, cat_features, self.order1_neural
            )  
        else:
            z1 = x  

        order1_out = self.order1_weights(z1)

        unary_contribs: dict = self._compute_unary_contributions(z1)

        term_list = [order1_out]

        z2_cache: Optional[torch.Tensor] = None

        for degree_idx, _ in enumerate(self.ranks):
            l = degree_idx + 2  

            if self.use_neural:
                # Apply degree-specific per-feature MLP transforms
                z_l = self._apply_neural_transform(
                    num_features, cat_features,
                    self.neural_transforms[degree_idx],
                )  
            else:
                z_l = self._geometric_rescale(x, l)  

            if degree_idx == 0:
                z2_cache = z_l

            contribution = self._compute_poly_contribution(z_l, degree_idx)
            term_list.append(contribution)  

        stacked = torch.stack(term_list, dim=1)    

        stacked = self.feature_dropout(stacked)    

        output = stacked.sum(dim=1)                

        if self.intercept is not None:
            output = output + self.intercept      

        pairwise_contribs: dict = {}
        if self.degree >= 2 and z2_cache is not None:
            pairwise_contribs = self._compute_pairwise_contributions(z2_cache)

        result: dict = {"output": output}

        result.update(unary_contribs)

        result.update(pairwise_contribs)

        if self.intercept is not None:
            result["intercept"] = self.intercept

        return result

    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        task_loss_fn=None,
    ) -> torch.Tensor:
        """
        Compute the total training loss: task loss + L1 sparsity penalty.

        This method is designed to be called from BaseModel's training_step.
        It adds the L1 penalty on the basis vectors (Equation 2 of the paper)
        on top of the standard regression / classification loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Model output, shape (batch, num_classes).
        targets : torch.Tensor
            Ground-truth labels/values, shape (batch,) or (batch, num_classes).
        task_loss_fn : callable, optional
            Loss function with signature loss_fn(predictions, targets) → scalar.
            If None, defaults to MSELoss for regression (num_classes == 1) or
            CrossEntropyLoss for classification (num_classes > 1).

        Returns
        -------
        torch.Tensor
            Scalar total loss: task_loss + l1_penalty.
        """
        if task_loss_fn is None:
            if self.num_classes == 1:
                task_loss_fn = nn.MSELoss()
            else:
                task_loss_fn = nn.CrossEntropyLoss()

        if self.num_classes > 1 and predictions.shape[-1] > 1:
            task_loss = task_loss_fn(predictions, targets.long())
        else:
            pred_squeezed = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
            task_loss = task_loss_fn(pred_squeezed, targets.float())

        return task_loss + self._compute_l1_penalty()


    def get_feature_importances(
        self,
        num_features: dict,
        cat_features: dict,
    ) -> dict:
        """
        Return mean absolute contributions per feature and per pair over a batch.

        Runs a forward pass in eval mode and aggregates the per-sample
        contribution tensors into a single scalar importance score per feature
        (or pair), averaged over the batch and over output classes.

        This implements the feature importance definition from Table 5 of the
        paper: importance of feature i is |u1i · xi| for unary terms and
        |Σk λ2k·u2ki·u2kj · √(xi·xj)| for pairwise terms.

        Parameters
        ----------
        num_features : dict
            Numerical feature tensors, name → (batch, dim).
        cat_features : dict
            Categorical feature tensors, name → (batch, dim).

        Returns
        -------
        dict
            Mapping feature_name (or "feat_i:feat_j") → float scalar,
            representing the mean absolute contribution over the batch.
            Sorted in descending order of importance.
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(num_features, cat_features)

        importances = {}
        skip = {"output", "intercept"}
        for key, tensor in result.items():
            if key in skip:
                continue
            importances[key] = tensor.abs().mean().item()

        return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))