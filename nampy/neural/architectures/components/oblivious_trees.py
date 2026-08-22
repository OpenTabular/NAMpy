"""Oblivious decision tree ensembles (NODE), adapted from https://github.com/Qwicen/node."""

from numbers import Integral
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import ModuleWithInit
from .sparse_activations import entmax15, entmoid15
from .tensor_utils import check_numpy

# Set the min logits to -20
MIN_LOGITS = -20


class ODST(ModuleWithInit):
    def __init__(
        self,
        in_features,
        num_trees,
        depth=6,
        tree_dim=1,
        flatten_output=True,
        choice_function=entmax15,
        bin_function=entmoid15,
        initialize_response_=nn.init.normal_,
        initialize_selection_logits_=nn.init.uniform_,
        threshold_init_beta=1.0,
        threshold_init_cutoff=1.0,
        colsample_bytree=1.0,
        **kwargs,
    ):
        super().__init__()
        self.in_features, self.depth, self.num_trees, self.tree_dim = (
            in_features,
            depth,
            num_trees,
            tree_dim,
        )
        self.flatten_output = bool(flatten_output)
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = (
            threshold_init_beta,
            threshold_init_cutoff,
        )
        self.colsample_bytree = colsample_bytree

        self.response = nn.Parameter(
            torch.zeros([num_trees, tree_dim, 2**depth]), requires_grad=True
        )
        initialize_response_(self.response)

        self.num_sample_feats = in_features
        if self.colsample_bytree < 1.0:
            self.num_sample_feats = int(np.ceil(in_features * self.colsample_bytree))

        # Do the subsampling
        if self.num_sample_feats < in_features:
            self.colsample = nn.Parameter(
                torch.zeros([in_features, num_trees, 1]), requires_grad=False
            )
            for nt in range(num_trees):
                rand_idx = torch.randperm(in_features)[: self.num_sample_feats]
                self.colsample[rand_idx, nt, 0] = 1.0

        # Only when num_sample_feats > 1, we initialize this logit
        if self.num_sample_feats > 1 or self.colsample_bytree == 1.0:
            self.feature_selection_logits = nn.Parameter(
                torch.zeros([in_features, num_trees, depth]), requires_grad=True
            )
            initialize_selection_logits_(self.feature_selection_logits)

        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )  # nan values will be initialized on first batch (data-aware init)

        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2**self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(
                torch.float32
            )
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
            # ^-- [depth, 2 ** depth, 2]

    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(
                *input.shape[:-1], -1
            )
        # new input shape: [batch_size, in_features]

        feature_values = self.get_feature_selection_values(input)
        # ^--[batch_size, num_trees, depth]

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(
            -self.log_temperatures
        )

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_trees, depth, 2]

        bins = self.bin_function(threshold_logits)
        # ^--[batch_size, num_trees, depth, 2], approximately binary

        bin_matches = torch.einsum("btds,dcs->btdc", bins, self.bin_codes_1hot)
        # ^--[batch_size, num_trees, depth, 2 ** depth]

        response_weights = torch.prod(bin_matches, dim=-2)
        # ^-- [batch_size, num_trees, 2 ** depth]

        response = torch.einsum("bnd,ncd->bnc", response_weights, self.response)
        # ^-- [batch_size, num_trees, tree_dim]

        return response.flatten(1, 2) if self.flatten_output else response
        # ^-- [batch_size, num_trees * tree_dim]

    def initialize(self, input, eps=1e-6):
        # data-aware initializer
        assert len(input.shape) == 2
        if input.shape[0] < 1000:
            warn(
                "Data-aware initialization is performed on less than 1000 data points. This may cause instability. To avoid potential problems, run this model on a data batch with at least 1000 data samples. You can do so manually before training. Use with torch.no_grad() for memory efficiency.",
                stacklevel=2,
            )

        with torch.no_grad():
            feature_values = self.get_feature_selection_values(input)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(
                self.threshold_init_beta,
                self.threshold_init_beta,
                size=[self.num_trees, self.depth],
            )
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(
                    map(
                        np.percentile,
                        check_numpy(feature_values.flatten(1, 2).t()),
                        percentiles_q.flatten(),
                    )
                ),
                dtype=feature_values.dtype,
                device=feature_values.device,
            ).view(self.num_trees, self.depth)

            # init temperatures: make sure enough data points are in the linear region of
            # sparse-sigmoid
            temperatures = np.percentile(
                check_numpy(abs(feature_values - self.feature_thresholds)),
                q=100 * min(1.0, self.threshold_init_cutoff),
                axis=0,
            )

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(
                torch.as_tensor(temperatures) + eps
            )

    def get_feature_selection_values(self, input):
        """Get the selected features of each tree.

        Args:
            input: Input data of shape [batch_size, in_features].

        Returns:
            feature_values: The feature input to trees in a batch with shape as
                [batch_size, num_trees, tree_depth].
        """
        feature_selectors = self.get_feature_selectors()
        # ^--[in_features, num_trees, depth]

        feature_values = torch.einsum("bi,ind->bnd", input, feature_selectors)
        # ^--[batch_size, num_trees, depth]

        return feature_values

    def get_feature_selectors(self):
        """Get the feature selectors of each tree of each depth.

        Returns:
            feature_selectors: Tensor of shape [in_features, num_trees, tree_depth]. The values of
                first dimension sum to 1.
        """
        if self.colsample_bytree < 1.0 and self.num_sample_feats == 1:
            return self.colsample.data

        fsl = self.feature_selection_logits
        if self.colsample_bytree < 1.0:
            fsl = self.colsample * fsl + (1.0 - self.colsample) * MIN_LOGITS
        feature_selectors = self.choice_function(fsl, dim=0)
        return feature_selectors

    def __repr__(self):
        return "{}(in_features={}, num_trees={}, depth={}, tree_dim={}, flatten_output={})".format(
            self.__class__.__name__,
            self.in_features,
            self.num_trees,
            self.depth,
            self.tree_dim,
            self.flatten_output,
        )


class ODSTBlock(nn.Sequential):
    """Original NODE model adapted from https://github.com/Qwicen/node."""

    def __init__(
        self,
        in_features,
        num_trees,
        num_layers,
        num_classes=1,
        addi_tree_dim=0,
        output_dropout=0.0,
        init_bias=True,
        add_last_linear=True,
        last_dropout=0.0,
        l2_lambda=0.0,
        max_features=None,
        input_dropout=0.0,
        flatten_output=True,
        **kwargs,
    ):
        """Neural Oblivious Decision Ensembles (NODE).

        Args:
            in_features: The input dimension of dataset.
            num_trees: How many ODST trees in a layer.
            num_layers: How many layers of trees.
            num_classes: How many classes to predict. It's the output dim.
            addi_tree_dim: Additional dimension for the outputs of each tree. If the value x > 0,
                each tree outputs a (1 + x) dimension of vector.
            output_dropout: The dropout rate on the output of each tree.
            init_bias: If set to True, it adds a trainable bias to the output of the model.
            add_last_linear: If set to True, add a last linear layer to sum outputs of all trees.
            last_dropout: If add_last_layer is True, then it adds a dropout on the weight og last
                linear year.
            l2_lambda: Add a l2 penalty on the outputs of trees.
            max_features: Maximum dense-layer input width. Original features
                are retained and the newest tree outputs fill the remainder.
            input_dropout: Dropout applied independently before every tree layer.
            flatten_output: Whether ``run_with_layers`` flattens tree/output axes.
            kwargs: The kwargs for initializing odst trees.
        """
        if max_features is not None:
            if not isinstance(max_features, Integral):
                raise TypeError("max_features must be an integer or None.")
            if max_features < in_features:
                raise ValueError(
                    "max_features cannot be smaller than the original input width."
                )
            max_features = int(max_features)
        if not 0 <= input_dropout <= 1:
            raise ValueError("input_dropout must lie between 0 and 1.")

        layers = self.create_layers(
            in_features,
            num_trees,
            num_layers,
            tree_dim=num_classes + addi_tree_dim,
            max_features=max_features,
            **kwargs,
        )
        super().__init__(*layers)
        self.num_layers, self.num_trees, self.num_classes, self.addi_tree_dim = (
            num_layers,
            num_trees,
            num_classes,
            addi_tree_dim,
        )
        self.output_dropout = output_dropout
        self.init_bias = init_bias
        self.add_last_linear = add_last_linear
        self.last_dropout = last_dropout
        self.l2_lambda = l2_lambda
        self.max_features = max_features
        self.input_dropout = float(input_dropout)
        self.flatten_output = bool(flatten_output)

        val = (
            torch.tensor(0.0)
            if num_classes == 1
            else torch.full([num_classes], 0.0, dtype=torch.float32)
        )
        self.bias = nn.Parameter(val, requires_grad=init_bias)

        self.last_w = None
        if add_last_linear or addi_tree_dim < 0:
            # Happens when more outputs than intermediate tree dim
            self.last_w = nn.Parameter(
                torch.empty(
                    num_layers * num_trees * (num_classes + addi_tree_dim), num_classes
                )
            )
            nn.init.xavier_uniform_(self.last_w)

        # Record which params need gradient
        self.named_params_requires_grad = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.named_params_requires_grad.add(name)

    def create_layers(
        self, in_features, num_trees, num_layers, tree_dim, max_features=None, **kwargs
    ):
        """Create layers of oblivious trees.

        Args:
            in_features: The dim of input features.
            num_trees: The number of trees in a layer.
            num_layers: The number of layers.
            tree_dim: The output dimension of each tree.
            kwargs: The kwargs for initializing odst trees.
        """
        layers = []
        for _i in range(num_layers):
            oddt = ODST(
                in_features,
                num_trees,
                tree_dim=tree_dim,
                flatten_output=True,
                **kwargs,
            )
            in_features = min(
                in_features + num_trees * tree_dim,
                max_features if max_features is not None else float("inf"),
            )
            layers.append(oddt)
        return layers

    def forward(self, x, return_outputs_penalty=False, feature_masks=None):
        """Model prediction.

        Args:
            x: The input features.
            return_outputs_penalty: If True, it returns the output l2 penalty.
            feature_masks: Only used in the pretraining. If passed, the outputs of trees belonging
                to masked features (masks==1) is zeroed. This is like dropping out features directly.
        """
        outputs = self.run_with_layers(x)

        num_output_trees = self.num_layers * self.num_trees
        outputs = outputs.view(
            *outputs.shape[:-1], num_output_trees, self.num_classes + self.addi_tree_dim
        )

        # During pretraining, we mask the outputs of trees
        if feature_masks is not None:
            assert not self[0].ga2m, "Not supported for ga2m for now!"
            with torch.no_grad():
                tmp = torch.cat(
                    [layer.get_feature_selectors() for layer in self], dim=1
                )
                # ^-- [in_features, layers * num_trees, 1]
                op_masks = torch.einsum("bi,ied->bed", feature_masks, tmp)
            outputs = outputs * (1.0 - op_masks)

        # We can do weighted sum instead of just simple averaging
        if self.last_w is not None:
            last_w = self.last_w
            if self.training and self.last_dropout > 0.0:
                last_w = F.dropout(last_w, self.last_dropout)
            result = torch.einsum(
                "bd,dc->bc", outputs.reshape(outputs.shape[0], -1), last_w
            )
        else:
            outputs = outputs[..., : self.num_classes]
            # ^--[batch_size, num_trees, num_classes]
            result = outputs.mean(dim=-2)

        result += self.bias

        if return_outputs_penalty:
            # Average over batch, num_outputs_units
            output_penalty = self.calculate_l2_penalty(outputs)
            return result, output_penalty
        return result

    def calculate_l2_penalty(self, outputs):
        """Calculate l2 penalty."""
        return self.l2_lambda * (outputs**2).mean()

    def run_with_layers(self, x):
        initial_features = x.shape[-1]

        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = (
                    min(self.max_features, layer_inp.shape[-1]) - initial_features
                )
                if tail_features != 0:
                    layer_inp = torch.cat(
                        [
                            layer_inp[..., :initial_features],
                            layer_inp[..., -tail_features:],
                        ],
                        dim=-1,
                    )
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            if self.training and self.output_dropout:
                h = F.dropout(h, self.output_dropout)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(
                *outputs.shape[:-1],
                self.num_layers * self.num_trees,
                self.num_classes + self.addi_tree_dim,
            )
        return outputs

    def set_bias(self, y_train):
        """Set the bias term for GAM output as logodds of y.

        It's unnecessary to run since we can just use a learnable bias.
        """

        y_cls, counts = np.unique(y_train, return_counts=True)
        bias = np.log(counts / np.sum(counts))
        if len(bias) == 2:
            bias = bias[1] - bias[0]

        self.bias.data = torch.tensor(bias, dtype=torch.float32)

    def freeze_all_but_lastw(self):
        for name, param in self.named_parameters():
            if param.requires_grad and "last_w" not in name:
                param.requires_grad = False

    def unfreeze(self):
        for name, param in self.named_parameters():
            if name in self.named_params_requires_grad:
                param.requires_grad = True

    def get_num_trees_assigned_to_each_feature(self):
        """Get the number of trees assigned to each feature per layer.

        It's helpful for logging. Just to see how many trees focus on some features.

        Returns:
            Counts of trees with shape of [num_layers, num_input_features (in_features)].
        """
        if type(self) is ODSTBlock:
            return None

        num_trees = [layer.get_num_trees_assigned_to_each_feature() for layer in self]
        counts = torch.stack(num_trees)
        return counts
