"""Implementation of NODE-GAM layer."""

from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from nampy.arch_utils.nn_utils import entmax15, entmoid15, check_numpy, ModuleWithInit, process_in_chunks, EM15Temp

# Set the min logits to -20
MIN_LOGITS = -20


class ODST(ModuleWithInit):
    def __init__(self, in_features, num_trees, depth=6, tree_dim=1, choice_function=entmax15,
                 bin_function=entmoid15, initialize_response_=nn.init.normal_,
                 initialize_selection_logits_=nn.init.uniform_, threshold_init_beta=1.0,
                 threshold_init_cutoff=1.0, colsample_bytree=1., **kwargs):
        super().__init__()
        self.in_features, self.depth, self.num_trees, self.tree_dim = in_features, depth, num_trees, tree_dim
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = threshold_init_beta, threshold_init_cutoff
        self.colsample_bytree = colsample_bytree

        self.response = nn.Parameter(torch.zeros([num_trees, tree_dim, 2 ** depth]),
                                     requires_grad=True)
        initialize_response_(self.response)

        self.num_sample_feats = in_features
        if self.colsample_bytree < 1.:
            self.num_sample_feats = int(np.ceil(in_features * self.colsample_bytree))

        # Do the subsampling
        if self.num_sample_feats < in_features:
            self.colsample = nn.Parameter(
                torch.zeros([in_features, num_trees, 1]), requires_grad=False
            )
            for nt in range(num_trees):
                rand_idx = torch.randperm(in_features)[:self.num_sample_feats]
                self.colsample[rand_idx, nt, 0] = 1.

        # Only when num_sample_feats > 1, we initialize this logit
        if self.num_sample_feats > 1 or self.colsample_bytree == 1.:
            self.feature_selection_logits = nn.Parameter(
                torch.zeros([in_features, num_trees, depth]), requires_grad=True
            )
            initialize_selection_logits_(self.feature_selection_logits)

        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )  # nan values will be initialized on first batch (data-aware init)

        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2 ** self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
            # ^-- [depth, 2 ** depth, 2]

    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)
        # new input shape: [batch_size, in_features]

        feature_values = self.get_feature_selection_values(input)
        # ^--[batch_size, num_trees, depth]

        threshold_logits = (feature_values - self.feature_thresholds) \
                           * torch.exp(-self.log_temperatures)

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_trees, depth, 2]

        bins = self.bin_function(threshold_logits)
        # ^--[batch_size, num_trees, depth, 2], approximately binary

        bin_matches = torch.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
        # ^--[batch_size, num_trees, depth, 2 ** depth]
        
        response_weights = torch.prod(bin_matches, dim=-2)
        # ^-- [batch_size, num_trees, 2 ** depth]

        response = torch.einsum('bnd,ncd->bnc', response_weights, self.response)
        # ^-- [batch_size, num_trees, tree_dim]

        return response.flatten(1, 2)
        # ^-- [batch_size, num_trees * tree_dim]

    def initialize(self, input, eps=1e-6):
        # data-aware initializer
        assert len(input.shape) == 2
        if input.shape[0] < 1000:
            warn("Data-aware initialization is performed on less than 1000 data points. This may cause instability. To avoid potential problems, run this model on a data batch with at least 1000 data samples. You can do so manually before training. Use with torch.no_grad() for memory efficiency.")

        with torch.no_grad():
            feature_values = self.get_feature_selection_values(input)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(self.threshold_init_beta, self.threshold_init_beta,
                                                 size=[self.num_trees, self.depth])
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(map(np.percentile, check_numpy(feature_values.flatten(1, 2).t()),
                         percentiles_q.flatten())),
                dtype=feature_values.dtype, device=feature_values.device
            ).view(self.num_trees, self.depth)

            # init temperatures: make sure enough data points are in the linear region of
            # sparse-sigmoid
            temperatures = np.percentile(check_numpy(abs(feature_values - self.feature_thresholds)),
                                         q=100 * min(1.0, self.threshold_init_cutoff), axis=0)

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

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

        feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
        # ^--[batch_size, num_trees, depth]

        return feature_values

    def get_feature_selectors(self):
        """Get the feature selectors of each tree of each depth.

        Returns:
            feature_selectors: Tensor of shape [in_features, num_trees, tree_depth]. The values of
                first dimension sum to 1.
        """
        if self.colsample_bytree < 1. and self.num_sample_feats == 1:
            return self.colsample.data

        fsl = self.feature_selection_logits
        if self.colsample_bytree < 1.:
            fsl = self.colsample * fsl + (1. - self.colsample) * MIN_LOGITS
        feature_selectors = self.choice_function(fsl, dim=0)
        return feature_selectors

    def __repr__(self):
        return "{}(in_features={}, num_trees={}, depth={}, tree_dim={})".format(
            self.__class__.__name__, self.in_features,
            self.num_trees, self.depth, self.tree_dim
        )


class GAM_ODST(ODST):
    def __init__(self, in_features, num_trees, tree_dim=1, depth=6, choice_function=entmax15,
                 bin_function=entmoid15, initialize_response_=nn.init.normal_,
                 initialize_selection_logits_=nn.init.uniform_, colsample_bytree=1.,
                 selectors_detach=True, fs_normalize=True, ga2m=0, **kwargs):
        """A layer of GAM ODST trees.

        Change a layer of ODST trees to make each tree only depend on at most 1 or 2 features
        to make it as a GAM or GA2M.

        Args:
            in_features: Number of features in the input tensor.
            num_trees: Number of trees in this layer.
            tree_dim: Number of response channels in the response of individual tree.
            depth: Number of splits in every tree.
            choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t.
                f(tensor, dim).sum(dim) == 1.
            bin_function: f(tensor) -> R[0, 1], computes tree leaf weights.
            initialize_response_: In-place initializer for tree output tensor.
            initialize_selection_logits_: In-place initializer for logits that select features for
                the tree. Both thresholds and scales are initialized with data-aware init
                (or .load_state_dict).
            colsample_bytree: The random proportion of features allowed in each tree. The same
                argument as in xgboost package. If less than 1, for each tree, it will only choose a
                fraction of features to train. For instance, if colsample_bytree = 0.9, each tree
                will only selects among 90% of the features.
            selectors_detach: If True, the selector will be detached before passing into the next layer.
                This will save GPU memory in the large dataset (e.g. Epsilon).
            fs_normalize: If True, we normalize the feature selectors be summed to 1. But False or
                True do not make too much difference in performance.
            ga2m: If set to 1, use GA2M, else use GAM.
            kwargs: For other old unused arguments for compatibility reasons.
        """
        if ga2m:
            # If specified as GA2M, but the tree depth is set to just 1 that can not model GA2M.
            # Change it to 2.
            if depth < 2:
                depth = 2

            # Similarly, if the colsample_by_tree is too small that each tree has only 1 feature,
            # increases it to 2.
            if (colsample_bytree < 1. and int(np.ceil(in_features * colsample_bytree)) < 2):
                colsample_bytree = 2 / in_features

        if colsample_bytree >= in_features:
            colsample_bytree = 1

        super().__init__(
            in_features=in_features,
            num_trees=num_trees,
            depth=depth,
            tree_dim=tree_dim,
            choice_function=choice_function,
            bin_function=bin_function,
            initialize_response_=initialize_response_,
            initialize_selection_logits_=initialize_selection_logits_,
            colsample_bytree=colsample_bytree,
        )
        self.selectors_detach = selectors_detach
        self.fs_normalize = fs_normalize
        self.ga2m = ga2m

        try:
            del self.feature_selection_logits
            the_depth = 1 if not self.ga2m else 2
            self.feature_selection_logits = nn.Parameter(
                torch.zeros([self.in_features, self.num_trees, the_depth]), requires_grad=True
            )
            initialize_selection_logits_(self.feature_selection_logits)
        except AttributeError:
            # No feature_selection_logits exists. Could be that it sets the col_subsample very small
            # that there is no need to optimize this. To save memory, So it's deleted in the master
            # class.
            pass

    def forward(self, input, return_feature_selectors=True, prev_feature_selectors=None):
        self.prev_feature_selectors = prev_feature_selectors

        response = super().forward(input)

        fs, self.feature_selectors = self.feature_selectors, None
        if return_feature_selectors:
            return response, fs

        return response

    def initialize(self, input, return_feature_selectors=True,
                   prev_feature_selectors=None, eps=1e-6):
        self.prev_feature_selectors = prev_feature_selectors
        response = super().initialize(input, eps=eps)
        self.feature_selectors = None

    def get_feature_selection_values(self, input, return_fss=False):
        """Get the selected features of each tree.

        Args:
            input: Input data of shape [batch_size, in_features].
            return_fss: If True, return the feature selectors.

        Returns:
            feature_values: The feature input to trees in a batch with Shape as
                [batch_size, num_trees, tree_depth].
            feature_selectors: (Optional) the feature selectors.
        """
        feature_selectors = self.get_feature_selectors()
        # ^--[in_features, num_trees, depth=1]

        # A hack to pass this value outside of this function
        self.feature_selectors = feature_selectors
        if self.selectors_detach: # To save memory
            self.feature_selectors = self.feature_selectors.detach()

        # It needs to multiply by the tree_dim
        if self.tree_dim > 1:
            shape = self.feature_selectors.shape
            self.feature_selectors = self.feature_selectors.unsqueeze(-2).expand(
                -1, -1, self.tree_dim, -1
            ).reshape(shape[0], -1, shape[-1])
            # ^--[in_features, num_trees * tree_dim, depth]

        if input.shape[1] > self.in_features:  # The rest are previous layers
            # Check incoming data
            pfs, self.prev_feature_selectors = self.prev_feature_selectors, None
            assert pfs.shape[:2] == (self.in_features, input.shape[1] - self.in_features), \
                'Previous selectors does not have the same shape as the input: %s != %s' \
                % (pfs.shape[:2], (self.in_features, input.shape[1] - self.in_features))
            fw = self.cal_prev_feat_weights(feature_selectors, pfs)

            feature_selectors = torch.cat([feature_selectors, fw], dim=0)
            # ^--[input_features, num_trees, depth=1]

        # post_process it
        feature_selectors = self.post_process(feature_selectors)

        fv = torch.einsum('bi,ind->bnd', input, feature_selectors)
        # ^--[batch_size, num_trees, depth=1,2]
        if not self.ga2m:
            fv = fv.expand(-1, -1, self.depth)
        else:
            if self.depth > 2:
                fv = fv.repeat(1, 1, int(np.ceil(self.depth / 2)))[..., :self.depth]

        if return_fss:
            return fv, feature_selectors
        return fv

    def cal_prev_feat_weights(self, myfs, pfs):
        """Calculate the feature weights of the previous trees outputs.

        To make sure it's a GAM or GA2M, the weights should be 0 if the previous tree focus on
        different (sets of) features than the current tree, and should be 1 if they are the same.

        Args:
            myfs: The current feature selector of this layer.
            pfs: The previous feature selectors.

        Returns:
            fw: The feature weights for the previous trees' outputs. Values are between 0 and 1
                with shape as [prev_trees_outputs, current_tree_outputs, depth], where depth=1 in
                GAM and depth=2 in GA2M.
        """
        # Do a row-wise inner product between prev selectors and cur ones
        if not self.ga2m:
            fw = torch.einsum('icd,ipd->pcd', myfs, pfs)
        else:
            g1 = torch.einsum("dp,dc->pc", pfs[:, :, 0], myfs[:, :, 0])
            g2 = torch.einsum("dp,dc->pc", pfs[:, :, 1], myfs[:, :, 1])
            g3 = torch.einsum("dp,dc->pc", pfs[:, :, 1], myfs[:, :, 0])
            g4 = torch.einsum("dp,dc->pc", pfs[:, :, 0], myfs[:, :, 1])

            fw = g1 * g2 + g3 * g4
            fw = fw.clamp_(max=1.).unsqueeze_(-1).repeat(1, 1, 2)
        return fw

    def post_process(self, feature_selectors):
        result = feature_selectors
        if self.fs_normalize:
            result = (feature_selectors / feature_selectors.sum(dim=0, keepdims=True))
        return result

    def get_num_trees_assigned_to_each_feature(self):
        with torch.no_grad():
            fs = self.get_feature_selectors()
            # ^-- [in_features, num_trees, 1]
            return (fs > 0).sum(dim=[1, 2])


class GAMAttODST(GAM_ODST):
    def __init__(self, in_features, num_trees, tree_dim=1, depth=6, choice_function=entmax15,
                 bin_function=entmoid15, initialize_response_=nn.init.normal_,
                 initialize_selection_logits_=nn.init.uniform_, colsample_bytree=1.,
                 selectors_detach=True, ga2m=0, prev_in_features=0, dim_att=8, **kwargs):
        """A layer of GAM ODST trees with attention mechanism.

        Change a layer of ODST trees to make each tree only depend on at most 1 or 2 features
        to make it as a GAM or GA2M. And also add an attention between layers.

        Args:
            in_features: Number of features in the input tensor.
            num_trees: Number of trees in this layer.
            tree_dim: Number of response channels in the response of individual tree.
            depth: Number of splits in every tree.
            choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t.
                f(tensor, dim).sum(dim) == 1.
            bin_function: f(tensor) -> R[0, 1], computes tree leaf weights.
            initialize_response_: In-place initializer for tree output tensor.
            initialize_selection_logits_: in-place initializer for logits that select features for
                the tree. Both thresholds and scales are initialized with data-aware init
                (or .load_state_dict).
            colsample_bytree: The random proportion of features allowed in each tree. The same
                argument as in xgboost package. If less than 1, for each tree, it will only choose a
                fraction of features to train. For instance, if colsample_bytree = 0.9, each tree
                will only selects among 90% of the features.
            selectors_detach: If True, the selector will be detached before passing into the next layer.
                This will save GPU memory in the large dataset (e.g. Epsilon).
            fs_normalize: If True, we normalize the feature selectors be summed to 1. But False or
                True do not make too much difference in performance.
            ga2m: If set to 1, use GA2M, else use GAM.
            prev_in_features: The number of previous layers' outputs.
            dim_att: The dimension of attention embedding to reduce memory consumption.
            kwargs: For other old unused arguments for compatibility reasons.
        """
        super().__init__(
            in_features=in_features,
            num_trees=num_trees,
            depth=depth,
            tree_dim=tree_dim,
            choice_function=choice_function,
            bin_function=bin_function,
            initialize_response_=initialize_response_,
            initialize_selection_logits_=initialize_selection_logits_,
            colsample_bytree=colsample_bytree,
            selectors_detach=selectors_detach,
            fs_normalize=False,
            ga2m=ga2m,
        )

        self.prev_in_features = prev_in_features
        self.dim_att = dim_att

        # Save parameter for the first layer
        if prev_in_features > 0:
            self.att_key = nn.Parameter(
                torch.zeros([prev_in_features, dim_att]), requires_grad=True
            )
            self.att_query = nn.Parameter(
                torch.zeros([dim_att, self.num_trees]), requires_grad=True
            )
            initialize_selection_logits_(self.att_key)
            initialize_selection_logits_(self.att_query)

    def cal_prev_feat_weights(self, feature_selectors, pfs):
        """Calculate the feature weights of the previous trees outputs.

        To make sure it's a GAM or GA2M, the weights should be 0 if the previous tree focus on
        different (sets of) features than the current tree, and should be 1 if they are the same.

        Args:
            feature_selectors: The current feature selector of this layer.
            pfs: The previous feature selectors.

        Returns:
            fw: The feature weights for the previous trees' outputs. Values are between 0 and 1 with
                shape as [prev_trees_outputs, current_tree_outputs, depth], where depth=1 in GAM and
                depth=2 in GA2M.
        """
        assert self.prev_in_features > 0
        fw = super().cal_prev_feat_weights(feature_selectors, pfs)
        # ^--[prev_in_feats, num_trees, depth=1,2]

        pfa = torch.einsum('pa,at->pt', self.att_key, self.att_query)
        new_fw = entmax15(fw.add(1e-6).log().add(pfa.unsqueeze_(-1)), dim=0)
        fw = fw * new_fw
        return fw

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
            kwargs: The kwargs for initializing odst trees.
        """
        layers = self.create_layers(
            in_features,
            num_trees,
            num_layers,
            tree_dim=num_classes + addi_tree_dim,
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

    def create_layers(self, in_features, num_trees, num_layers, tree_dim, **kwargs):
        """Create layers of oblivious trees.

        Args:
            in_features: The dim of input features.
            num_trees: The number of trees in a layer.
            num_layers: The number of layers.
            tree_dim: The output dimension of each tree.
            kwargs: The kwargs for initializing odst trees.
        """
        layers = []
        for i in range(num_layers):
            oddt = ODST(in_features, num_trees, tree_dim=tree_dim, **kwargs)
            in_features = in_features + num_trees * tree_dim
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
                tmp = torch.cat([l.get_feature_selectors() for l in self], dim=1)
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
            h = layer(layer_inp)
            if self.training and self.output_dropout:
                h = F.dropout(h, self.output_dropout)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
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

        num_trees = [l.get_num_trees_assigned_to_each_feature() for l in self]
        counts = torch.stack(num_trees)
        return counts


class GAMAdditiveMixin(object):
    """All Functions related to extracting GAM and GA2M graphs from the model."""

    def extract_additive_terms(
        self,
        X,
        norm_fn=lambda x: x,
        y_mu=0.0,
        y_std=1.0,
        device="cpu",
        batch_size=1024,
        tol=1e-3,
        purify=True,
    ):
        """Extract the additive terms in the GAM/GA2M model to plot the graphs.

        To extract the main and interaction terms, it runs the model on all possible input values
        and get the predicted value of each additive term. Then it returns a mapping of x and
        model's outputs y in a dataframe for each term.

        Args:
            X: Input 2d array (pandas). Note that it is the unpreprocessed data.
            norm_fn: The data preprocessing function (E.g. quantile normalization) before feeding
                into the model. Inputs: pandas X. Outputs: preprocessed outputs.
            y_mu, y_std: The outputs of the model will be multiplied by y_std and then shifted by
                y_mu. It's useful in regression problem where target y is normalized to mean 0 and
                std 1. Default: 0, 1.
            device: Use which device to run the model. Default: 'cpu'.
            batch_size: Batch size.
            tol: The tolerance error for the interaction purification that moves mass from
                interactions to mains (see the "purification" of the paper).
            purify: If True, we move all effects of the interactions to main effects.

        Returns:
            A pandas table that records all main and interaction terms. The columns include::
            feat_name: The feature name. E.g. "Hour".
            feat_idx: The feature index. E.g. 2.
            x: The unique values of the feature. E.g. [0.5, 3, 4.7].
            y: The values of the output. E.g. [-0.2, 0.3, 0.5].
            importance: The feature importance. It's calculated as the weighted average of
                the absolute value of y weighted by the counts of each unique value.
            counts: The counts of each unique value in the data. E.g. [20, 10, 3].
        """
        assert self.num_classes == 1, "Has not support > 2 classes. But should be easy."
        assert isinstance(X, pd.DataFrame)
        self.eval()

        vals, counts, terms = self._run_and_extract_vals_counts(
            X, device, batch_size, norm_fn=norm_fn, y_mu=y_mu, y_std=y_std
        )

        if purify:
            # Doing centering: do the pairwise purification
            self._purify_interactions(vals, counts, tol=tol)

        # Center the main effect
        vals[-1] += self.bias.data.item()
        for t in vals:
            # If it's an interaction term or the bias term, continue.
            if isinstance(t, tuple) or t == -1:
                continue

            weights = counts[t].values
            avg = np.average(vals[t].values, weights=weights)

            vals[-1] += avg
            vals[t] -= avg

        # Organize data frame. Initialize with the bias term.
        results = [
            {
                "feat_name": "offset",
                "feat_idx": -1,
                "x": None,
                "y": np.full(1, vals[-1]),
                "importance": -1,
                "counts": None,
            }
        ]

        for t in tqdm(vals):
            if t == -1:
                continue

            if not isinstance(t, tuple):
                x = vals[t].index.values
                y = vals[t].values
                count = counts[t].values
                tmp = np.argsort(x)
                x, y, count = x[tmp], y[tmp], count[tmp]
            else:
                # Make 2d back to 1d
                tmp = vals[t].stack()
                tmp_count = counts[t].values.reshape(-1)
                selected_entry = (tmp.values != 0) | (tmp_count > 0)
                tmp = tmp[selected_entry]
                x = tmp.index.values
                y = tmp.values
                count = tmp_count[selected_entry]

            imp = np.average(np.abs(np.array(y)), weights=np.array(count))
            results.append(
                {
                    "feat_name": (
                        X.columns[t]
                        if not isinstance(t, tuple)
                        else f"{X.columns[t[0]]}_{X.columns[t[1]]}"
                    ),
                    "feat_idx": t,
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "importance": imp,
                    "counts": count.tolist(),
                }
            )

            df = pd.DataFrame(results)
            df["tmp"] = df.feat_idx.apply(
                lambda x: x[0] * 1e10 + x[1] * 1e5 if isinstance(x, tuple) else int(x)
            )
            df = df.sort_values("tmp").drop("tmp", axis=1)
            df = df.reset_index(drop=True)
        return df

    def _run_and_extract_vals_counts(
        self, X, device, batch_size, norm_fn=lambda x: x, y_mu=0.0, y_std=1.0
    ):
        """Run the models and return the value of model's outputs and the counts.

        It runs the model on all inputs X, and returns the models's output and the counts of each
        input value for each term.

        Args:
            X: Input 2d array (pandas). Note that it is the unnormalized data.
            norm_fn: The data preprocessing function (E.g. quantile normalization) before feeding
                into the model. Inputs: pandas X. Outputs: preprocessed outputs.
            y_mu, y_std: The outputs of the model will be multiplied by y_std and then shifted by
                y_mu. It's useful in regression problem where target y is normalized to mean 0 and
                std 1. Default: 0, 1.
            device: Use which device to run the model. Default: 'cpu'.
            batch_size: Batch size.
            tol: The tolerance error for the interaction purification that moves mass from
                interactions to mains (see the "purification" of the paper).
            purify: If True, we move all effects of the interactions to main effects.

        Returns:
            vals (dict of dict): A dict that has keys as feature index and value as another dict
                that maps the unique value of input X to the output of the model. For example, if a
                model learns 2 main effects for features 1 and 2, and an interaction term between
                features 1 and 2, we could have::
                {1: {0: -0.2, 1: 0.3, 2: 1},
                 2: {1: 0.3, 2: -0.5},
                 (1, 2): {(0, 1): 1, (0, 2): 0.3, (1, 1): -1, (1, 2): -0.3, (2, 1): 0, (2, 2): 1}}.
            counts (dict of dict): Same format as `vals` but the values are the counts in the data.
                It has a dict that has keys as feature index and value as another dict that maps
                the unique value of input X to the counts of occurence in the data. For example::
                {1: {0: 10, 1: 100, 2: 90},
                 2: {1: 80, 2: 120},
                 (1, 2): {(0, 1): 10, (0, 2): 50, (1, 1): 100, (1, 2): 10, (2, 1): 20, (2, 2): 10}}.
            terms (list): all the main and interaction terms. E.g. [1, 2, (2, 3)].
        """
        with torch.no_grad():
            results = self._run_vals_with_additive_term_with_batch(
                X, device, batch_size, norm_fn=norm_fn, y_std=y_std
            )

        # Extract all additive term results
        vals, counts, terms = self._extract_vals_counts(results, X)
        vals[-1] = y_mu
        return vals, counts, terms

    def _run_vals_with_additive_term_with_batch(
        self, X, device, batch_size, norm_fn=lambda x: x, y_std=1.0
    ):
        """Run the models with additive terms using mini-batch.

        It calls self.run_with_additive_terms() with mini-batch.

        Args:
            X: Input 2d array (pandas). Note that it is the unnormalized data.
            device: Use which device to run the model. Default: 'cpu'.
            batch_size: Batch size.
            norm_fn: The data preprocessing function (E.g. quantile normalization) before feeding
                into the model. Inputs: pandas X. Outputs: preprocessed outputs.
            y_std: The outputs of the model will be multiplied by y_std. It's useful in regression
                problem where target y is normalized to std 1. Default: 1.

        Returns:
            results (numpy array): The model's output of each term. A numpy tensor of shape
                [num_data, num_unique_terms, output_dim] where 'num_unique_terms' is the total
                number of main and interaction effects, and 'output_dim' is the output_dim
                (num_classes). Usually 1.
        """

        results = process_in_chunks(
            lambda x: self.run_with_additive_terms(
                torch.tensor(norm_fn(x), device=device)
            ),
            X.values,
            batch_size=batch_size,
        )
        results = results.cpu().numpy()
        results = results * y_std
        return results

    def _extract_vals_counts(self, results, X):
        """Extracts the values and counts based on the outputs of models with additive terms.

        Args:
            results: The model's outputs of self._run_vals_with_additive_term_with_batch. It's a
                numpy tensor of shape [num_data, num_unique_terms, output_dim] that represents the
                model's output of each data on each additive term.
            X: The inputs of the data.

        Returns:
            vals (dict of dict): A dict that has keys as feature index and value as another dict
                that maps the unique value of input X to the output of the model. For example, if a
                model learns 2 main effects for features 1 and 2, and an interaction term between
                features 1 and 2, we could have::
                {1: {0: -0.2, 1: 0.3, 2: 1},
                 2: {1: 0.3, 2: -0.5},
                 (1, 2): {(0, 1): 1, (0, 2): 0.3, (1, 1): -1, (1, 2): -0.3, (2, 1): 0, (2, 2): 1}}.
            counts (dict of dict): Same format as `vals` but the values are the counts in the data.
                It has a dict that has keys as feature index and value as another dict that maps
                the unique value of input X to the counts of occurence in the data. For example::
                {1: {0: 10, 1: 100, 2: 90},
                 2: {1: 80, 2: 120},
                 (1, 2): {(0, 1): 10, (0, 2): 50, (1, 1): 100, (1, 2): 10, (2, 1): 20, (2, 2): 10}}.
            terms (list): all the main and interaction terms. E.g. [1, 2, (2, 3)].
        """
        terms = self.get_additive_terms()

        vals, counts = {}, {}
        for idx, t in enumerate(tqdm(terms)):
            if not isinstance(t, tuple):  # main effect term
                index = X.iloc[:, t]
                scores = pd.Series(results[:, idx, 0], index=index)

                tmp = scores.groupby(level=0).agg(["count", "first"])
                vals[t] = tmp["first"]
                counts[t] = tmp["count"]
            else:
                tmp = pd.Series(
                    results[:, idx, 0],
                    index=pd.MultiIndex.from_frame(X.iloc[:, list(t)]),
                )

                # One groupby to return both vals and counts
                tmp2 = tmp.groupby(level=[0, 1]).agg(["count", "first"])

                the_vals = tmp2["first"]
                the_counts = tmp2["count"]

                vals[t] = the_vals.unstack(level=-1).fillna(0.0)
                counts[t] = the_counts.unstack(level=-1).fillna(0).astype(int)

        # For each interaction tuple (i, j), initialize the main effect term i and j since they
        # will have some values during the purification.
        for t in terms:
            if not isinstance(t, tuple):
                continue

            for i in t:
                if i in vals:
                    continue
                a = X.iloc[:, i]
                the_counts = a.groupby(a).agg(["count"])
                counts[i] = the_counts["count"]
                vals[i] = the_counts["count"].copy()
                vals[i][:] = 0.0

        return vals, counts, terms

    def _purify_interactions(self, vals, counts, tol=1e-3):
        """Purify the interaction term to move the mass from interaction to the main effect.

        See the Supp. D in the paper for details. It modifies the vals in-place for arguments vals.

        Args:
            vals (dict of dict): A dict that has keys as feature index and value as another dict
                that maps the unique value of input X to the output of the model. For example, if a
                model learns 2 main effects for features 1 and 2, and an interaction term between
                features 1 and 2, we could have::
                {1: {0: -0.2, 1: 0.3, 2: 1},
                 2: {1: 0.3, 2: -0.5},
                 (1, 2): {(0, 1): 1, (0, 2): 0.3, (1, 1): -1, (1, 2): -0.3, (2, 1): 0, (2, 2): 1}}.
            counts (dict of dict): Same format as `vals` but the values are the counts in the data.
                It has a dict that has keys as feature index and value as another dict that maps
                the unique value of input X to the counts of occurence in the data. For example::
                {1: {0: 10, 1: 100, 2: 90},
                 2: {1: 80, 2: 120},
                 (1, 2): {(0, 1): 10, (0, 2): 50, (1, 1): 100, (1, 2): 10, (2, 1): 20, (2, 2): 10}}.
        """
        for t in vals:
            # If it's not an interaction term, continue.
            if not isinstance(t, tuple):
                continue

            # Continue purify the interactions until the purified average value is smaller than tol.
            biggest_epsilon = np.inf
            while biggest_epsilon > tol:
                biggest_epsilon = -np.inf

                avg = (vals[t] * counts[t]).sum(axis=1).values / counts[t].sum(
                    axis=1
                ).values
                if np.max(np.abs(avg)) > biggest_epsilon:
                    biggest_epsilon = np.max(np.abs(avg))

                vals[t] -= avg.reshape(-1, 1)
                vals[t[0]] += avg

                avg = (vals[t] * counts[t]).sum(axis=0).values / counts[t].sum(
                    axis=0
                ).values
                if np.max(np.abs(avg)) > biggest_epsilon:
                    biggest_epsilon = np.max(np.abs(avg))

                vals[t] -= avg.reshape(1, -1)
                vals[t[1]] += avg

    def get_additive_terms(self, return_inverse=False):
        """Get the additive terms in the GAM/GA2M model.

        It returns all the main and interaction effects in the NodeGAM.

        Args:
            return_inverse (bool): If True, it returns the map back from each additive term to the
                index of trees. It's useful to check which tree focuses on which feature set.

        Returns:
            tuple_terms (list): A list of integer or tuple that represents all the additive terms it
                learns. E.g. [2, 4, (2, 3), (1, 4)].
        """
        fs = torch.cat([l.get_feature_selectors() for l in self], dim=1).sum(dim=-1)
        fs[fs > 0.0] = 1.0
        # ^-- [in_features, layers*num_trees] binary features

        result = torch.unique(fs, dim=1, sorted=True, return_inverse=return_inverse)
        # ^-- ([in_features, uniq_terms], [layers*num_trees])

        terms = result
        if isinstance(result, tuple):  # return inverse=True
            terms = result[0]

        # To make additive terms human-readable, it transforms the one-hot vector into an integer,
        # and a 2-hot vector (interaction) into a tuple of integer.
        tuple_terms = self.convert_onehot_vector_to_integers(terms)

        if isinstance(result, tuple):
            return tuple_terms, result[1]
        return tuple_terms

    def convert_onehot_vector_to_integers(self, terms):
        """Make onehot or multi-hot vectors into a list of integers or tuple.

        Args:
            terms (Pytorch tensor): a one-hot matrix with each column has only one entry as 1.
                Shape: [in_features, uniq_GAM_terms].

        Returns:
            tuple_terms (list): A list of integers or tuples of all the GAM terms.
        """
        r_idx, c_idx = torch.nonzero(terms, as_tuple=True)
        tuple_terms = []
        for c in range(terms.shape[1]):
            n_interaction = (c_idx == c).sum()

            if n_interaction > 2:
                print(
                    f"WARNING: it is not a GA2M with a {n_interaction}-way term. "
                    f"Ignore this term."
                )
                continue
            if n_interaction == 1:
                tuple_terms.append(int(r_idx[c_idx == c].item()))
            elif n_interaction == 2:
                tuple_terms.append(tuple(r_idx[c_idx == c][:2].cpu().numpy()))
        return tuple_terms


class GAMBlock(GAMAdditiveMixin, ODSTBlock):
    """Node-GAM model."""

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
        l2_interactions=0.0,
        l1_interactions=0.0,
        **kwargs,
    ):
        """Initialization of Node-GAM.

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
            last_dropout: If add_last_layer is True, it adds a dropout on the weight og last
                linear year.
            l2_lambda: Add a l2 penalty on the outputs of trees.
            l2_interactions: Penalize the l2 magnitude of the output of trees that have
                pairwise interactions. Default: 0.
            l1_interactions: Penalize the l1 magnitude of the output of trees that have
                pairwise interactions. Default: 0.
            kwargs (dict): The arguments for underlying GAM ODST trees.
        """
        super().__init__(
            in_features=in_features,
            num_trees=num_trees,
            num_layers=num_layers,
            num_classes=num_classes,
            addi_tree_dim=addi_tree_dim,
            output_dropout=output_dropout,
            init_bias=init_bias,
            add_last_linear=add_last_linear,
            last_dropout=last_dropout,
            l2_lambda=l2_lambda,
            **kwargs,
        )
        self.l2_interactions = l2_interactions
        self.l1_interactions = l1_interactions

        self.inv_is_interaction = None

    def create_layers(self, in_features, num_trees, num_layers, tree_dim, **kwargs):
        """Create layers.

        Args:
            in_features: The input dimension (feature).
            num_trees: The number of trees in a layer.
            num_layers: The number of layers.
            tree_dim: The output dimension of each tree.
            kwargs (dict): The arguments for underlying GAM ODST trees.
        """
        layers = []
        for i in range(num_layers):
            # Last layer only has num_classes dim
            oddt = GAM_ODST(in_features, num_trees, tree_dim=tree_dim, **kwargs)
            layers.append(oddt)
        return layers

    def calculate_l2_penalty(self, outputs):
        """Calculate the penalty of the trees' outputs.

        It helps regularize the model.

        Args:
            outputs: The outputs of trees. A tensor of shape [batch_size, num_trees, tree_dim].
        """
        # Normal L2 weight decay on outputs
        penalty = super().calculate_l2_penalty(outputs)

        # If trees are still learning which features to take, skip the interaction penalty
        if not self[0].choice_function.is_deterministic:
            return penalty

        # Search and cache which term is interaction
        if self.inv_is_interaction is None:
            with torch.no_grad():
                terms, inv = self.get_additive_terms(return_inverse=True)
            idx_is_interactions = [
                i for i, t in enumerate(terms) if isinstance(t, tuple)
            ]
            if len(idx_is_interactions) == 0:
                return penalty

            inv_is_interaction = inv.new_zeros(*inv.shape, dtype=torch.bool)
            for idx in idx_is_interactions:
                inv_is_interaction |= inv == idx
            self.inv_is_interaction = inv_is_interaction

        outputs_interactions = outputs[:, self.inv_is_interaction, :]
        if self.l2_interactions > 0.0:
            penalty += self.l2_interactions * torch.mean(outputs_interactions**2)
        if self.l1_interactions > 0.0:
            penalty += self.l1_interactions * torch.mean(
                torch.abs(outputs_interactions)
            )

        return penalty

    def run_with_layers(self, x, return_fs=False):
        """Run the examples through the layers of trees.

        Args:
            x: The input tensor of shape [batch_size, in_features].
            return_fs: If True, it returns the feature selectors of each tree.

        Returns:
            outputs: The trees' outputs [batch_size, num_trees, tree_dim].
            prev_feature_selectors: Only returns when return_fs is True, this returns the feature
                selector of each ODST tree of shape [in_features, num_trees, tree_depth].
        """
        initial_features = x.shape[-1]
        prev_feature_selectors = None
        for layer in self:
            layer_inp = x
            h, feature_selectors = layer(
                layer_inp,
                prev_feature_selectors=prev_feature_selectors,
                return_feature_selectors=True,
            )
            if self.training and self.output_dropout:
                h = F.dropout(h, self.output_dropout)
            x = torch.cat([x, h], dim=-1)

            prev_feature_selectors = (
                feature_selectors
                if prev_feature_selectors is None
                else torch.cat([prev_feature_selectors, feature_selectors], dim=1)
            )

        outputs = x[..., initial_features:]
        if return_fs:
            return outputs, prev_feature_selectors
        return outputs


class GAMAttBlock(GAMBlock):
    """Node-GAM with attention model."""

    def create_layers(self, in_features, num_trees, num_layers, tree_dim, **kwargs):
        """Create layers of oblivious trees.

        Args:
            in_features: The dim of input features.
            num_trees: The number of trees in a layer.
            num_layers: The number of layers.
            tree_dim: The output dimension of each tree.
            kwargs: The kwargs for initializing GAMAtt ODST trees.
        """
        layers = []
        prev_in_features = 0
        for i in range(num_layers):
            # Last layer only has the dimension equal to num_classes
            oddt = GAMAttODST(
                in_features,
                num_trees,
                tree_dim=tree_dim,
                prev_in_features=prev_in_features,
                **kwargs,
            )
            layers.append(oddt)
            prev_in_features += num_trees * tree_dim
        return layers

if __name__ == "__main__":
    # Test parameters
    batch_size = 32
    in_features = 10
    num_trees = 4
    num_layers = 2
    depth = 3
    tree_dim = 1
    addi_tree_dim = 0
    output_dropout = 0.1
    last_dropout = 0.1
    colsample_bytree = 0.8
    l2_lambda = 0.01
    ga2m = 0  # Set to 1 for GA2M model
    dim_att = 8  # For GAMAtt model
    anneal_steps = 1000  # For temperature annealing

    # Create sample input data
    x = torch.randn(batch_size, in_features)
    
    # Initialize choice function with temperature annealing
    choice_fn = EM15Temp(max_temp=1.0, min_temp=0.01, steps=anneal_steps)
    
    # Simulate a distribution family with multiple parameters
    class TestFamily:
        def __init__(self, param_count):
            self.param_count = param_count
            self.name = "test"
            self.param_names = [f"param_{i}" for i in range(param_count)]
    
    # Create a test family with 2 parameters
    test_family = TestFamily(param_count=2)
    
    # Test GAMBlock with multiple models
    print("\nTesting GAMBlock with multiple models:")
    gam_models = [
        GAMBlock(
            in_features=in_features,
            num_trees=num_trees,
            num_layers=num_layers,
            num_classes=1,
            addi_tree_dim=addi_tree_dim,
            depth=depth,
            choice_function=choice_fn,
            bin_function=entmoid15,
            output_dropout=output_dropout,
            last_dropout=last_dropout,
            colsample_bytree=colsample_bytree,
            selectors_detach=True,
            add_last_linear=True,
            ga2m=ga2m,
            l2_lambda=l2_lambda
        )
        for _ in range(test_family.param_count)
    ]
    
    # Forward pass for each model
    gam_outputs = [model(x) for model in gam_models]
    
    print(f"Input shape: {x.shape}")
    print(f"Number of GAM models: {len(gam_models)}")
    for i, output in enumerate(gam_outputs):
        print(f"GAM Model {i+1} Output: {output}")
    print(f"GAM Model architecture:\n{gam_models[0]}")
    
    # Test GAMAttBlock with multiple models
    print("\nTesting GAMAttBlock with multiple models:")
    gam_att_models = [
        GAMAttBlock(
            in_features=in_features,
            num_trees=num_trees,
            num_layers=num_layers,
            num_classes=1,
            addi_tree_dim=addi_tree_dim,
            depth=depth,
            choice_function=choice_fn,
            bin_function=entmoid15,
            output_dropout=output_dropout,
            last_dropout=last_dropout,
            colsample_bytree=colsample_bytree,
            selectors_detach=True,
            add_last_linear=True,
            ga2m=ga2m,
            l2_lambda=l2_lambda,
            dim_att=dim_att
        )
        for _ in range(test_family.param_count)
    ]
    
    # Forward pass for each model
    gam_att_outputs = [model(x) for model in gam_att_models]
    
    print(f"Input shape: {x.shape}")
    print(f"Number of GAMAtt models: {len(gam_att_models)}")
    for i, output in enumerate(gam_att_outputs):
        print(f"GAMAtt Model {i+1} Output: {output}")
    print(f"GAMAtt Model architecture:\n{gam_att_models[0]}")