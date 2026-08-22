"""GAM/GA2M-constrained oblivious-tree blocks (adapted from https://github.com/zzzace2000/nodegam)."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .oblivious_trees import ODST, ODSTBlock
from .sparse_activations import entmax15, entmoid15
from .tensor_utils import process_in_chunks
from .term_extraction import (
    aggregate_term_values,
    build_terms_frame,
    center_main_effects,
    purify_interactions,
    terms_from_feature_selectors,
)


class GAM_ODST(ODST):
    def __init__(
        self,
        in_features,
        num_trees,
        tree_dim=1,
        depth=6,
        choice_function=entmax15,
        bin_function=entmoid15,
        initialize_response_=nn.init.normal_,
        initialize_selection_logits_=nn.init.uniform_,
        colsample_bytree=1.0,
        selectors_detach=True,
        fs_normalize=True,
        ga2m=0,
        **kwargs,
    ):
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
            kwargs: Additional unused keyword arguments.
        """
        if ga2m:
            # If specified as GA2M, but the tree depth is set to just 1 that can not model GA2M.
            # Change it to 2.
            if depth < 2:
                depth = 2

            # Similarly, if the colsample_by_tree is too small that each tree has only 1 feature,
            # increases it to 2.
            if (
                colsample_bytree < 1.0
                and int(np.ceil(in_features * colsample_bytree)) < 2
            ):
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
                torch.zeros([self.in_features, self.num_trees, the_depth]),
                requires_grad=True,
            )
            initialize_selection_logits_(self.feature_selection_logits)
        except AttributeError:
            # No feature_selection_logits exists. Could be that it sets the col_subsample very small
            # that there is no need to optimize this. To save memory, So it's deleted in the master
            # class.
            pass

    def forward(
        self, input, return_feature_selectors=True, prev_feature_selectors=None
    ):
        self.prev_feature_selectors = prev_feature_selectors

        response = super().forward(input)

        fs, self.feature_selectors = self.feature_selectors, None
        if return_feature_selectors:
            return response, fs

        return response

    def initialize(
        self,
        input,
        return_feature_selectors=True,
        prev_feature_selectors=None,
        eps=1e-6,
    ):
        self.prev_feature_selectors = prev_feature_selectors
        super().initialize(input, eps=eps)
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
        if self.selectors_detach:  # To save memory
            self.feature_selectors = self.feature_selectors.detach()

        # It needs to multiply by the tree_dim
        if self.tree_dim > 1:
            shape = self.feature_selectors.shape
            self.feature_selectors = (
                self.feature_selectors.unsqueeze(-2)
                .expand(-1, -1, self.tree_dim, -1)
                .reshape(shape[0], -1, shape[-1])
            )
            # ^--[in_features, num_trees * tree_dim, depth]

        if input.shape[1] > self.in_features:  # The rest are previous layers
            # Check incoming data
            pfs, self.prev_feature_selectors = self.prev_feature_selectors, None
            assert pfs.shape[:2] == (
                self.in_features,
                input.shape[1] - self.in_features,
            ), (
                "Previous selectors does not have the same shape as the input: %s != %s"
                % (pfs.shape[:2], (self.in_features, input.shape[1] - self.in_features))
            )
            fw = self.cal_prev_feat_weights(feature_selectors, pfs)

            feature_selectors = torch.cat([feature_selectors, fw], dim=0)
            # ^--[input_features, num_trees, depth=1]

        # post_process it
        feature_selectors = self.post_process(feature_selectors)

        fv = torch.einsum("bi,ind->bnd", input, feature_selectors)
        # ^--[batch_size, num_trees, depth=1,2]
        if not self.ga2m:
            fv = fv.expand(-1, -1, self.depth)
        else:
            if self.depth > 2:
                fv = fv.repeat(1, 1, int(np.ceil(self.depth / 2)))[..., : self.depth]

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
            fw = torch.einsum("icd,ipd->pcd", myfs, pfs)
        else:
            g1 = torch.einsum("dp,dc->pc", pfs[:, :, 0], myfs[:, :, 0])
            g2 = torch.einsum("dp,dc->pc", pfs[:, :, 1], myfs[:, :, 1])
            g3 = torch.einsum("dp,dc->pc", pfs[:, :, 1], myfs[:, :, 0])
            g4 = torch.einsum("dp,dc->pc", pfs[:, :, 0], myfs[:, :, 1])

            fw = g1 * g2 + g3 * g4
            fw = fw.clamp_(max=1.0).unsqueeze_(-1).repeat(1, 1, 2)
        return fw

    def post_process(self, feature_selectors):
        result = feature_selectors
        if self.fs_normalize:
            result = feature_selectors / feature_selectors.sum(dim=0, keepdims=True)
        return result

    def get_num_trees_assigned_to_each_feature(self):
        with torch.no_grad():
            fs = self.get_feature_selectors()
            # ^-- [in_features, num_trees, 1]
            return (fs > 0).sum(dim=[1, 2])


class GAMAttODST(GAM_ODST):
    def __init__(
        self,
        in_features,
        num_trees,
        tree_dim=1,
        depth=6,
        choice_function=entmax15,
        bin_function=entmoid15,
        initialize_response_=nn.init.normal_,
        initialize_selection_logits_=nn.init.uniform_,
        colsample_bytree=1.0,
        selectors_detach=True,
        ga2m=0,
        prev_in_features=0,
        dim_att=8,
        **kwargs,
    ):
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
            kwargs: Additional unused keyword arguments.
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

        pfa = torch.einsum("pa,at->pt", self.att_key, self.att_query)
        new_fw = entmax15(fw.add(1e-6).log().add(pfa.unsqueeze_(-1)), dim=0)
        fw = fw * new_fw
        return fw


class GAMAdditiveMixin:
    """Extraction of per-term GAM/GA2M graphs from a block of additive tree layers.

    Host contract: iterating the host yields tree layers exposing
    ``get_feature_selectors()`` (e.g. an ``nn.Sequential`` of ``GAM_ODST``),
    and the host provides ``run_with_additive_terms()``, ``bias``,
    ``num_classes``, and ``eval()``.
    """

    def run_with_additive_terms(self, x):
        """Return the learned output of each unique additive tree term.

        This is the direct NODE-GAM ``run_with_additive_terms`` contract.  It
        preserves the block's tree ordering and final linear weights, so term
        contributions are model outputs rather than raw input columns.
        """
        outputs = self.run_with_layers(x)
        tree_dim = self.num_classes + self.addi_tree_dim
        outputs = outputs.view(
            *outputs.shape[:-1], self.num_layers * self.num_trees, tree_dim
        )
        terms, inverse = self.get_additive_terms(return_inverse=True)

        if self.last_w is not None:
            expanded_inverse = inverse.unsqueeze(-1).expand(-1, tree_dim).reshape(-1)
            weights = expanded_inverse.new_zeros(
                expanded_inverse.shape[0], len(terms), self.num_classes,
                dtype=torch.float32,
            )
            values = self.last_w.unsqueeze(1).expand(-1, len(terms), -1)
            indices = expanded_inverse[:, None, None].expand(-1, 1, self.num_classes)
            weights.scatter_(1, indices, values)
            return torch.einsum(
                "bd,duc->buc", outputs.reshape(outputs.shape[0], -1), weights
            )

        outputs = outputs[..., : self.num_classes]
        weights = inverse.new_zeros(
            inverse.shape[0], len(terms), dtype=torch.float32
        )
        weights.scatter_(1, inverse.unsqueeze(-1), 1.0 / inverse.shape[0])
        return torch.einsum("bdc,du->buc", outputs, weights)

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
            The per-term result table produced by :func:`build_terms_frame`.
        """
        assert self.num_classes == 1, "Has not support > 2 classes. But should be easy."
        assert isinstance(X, pd.DataFrame)
        self.eval()

        with torch.no_grad():
            results = self._run_vals_with_additive_term_with_batch(
                X, device, batch_size, norm_fn=norm_fn, y_std=y_std
            )
        vals, counts = aggregate_term_values(results, X, self.get_additive_terms())
        vals[-1] = y_mu

        if purify:
            # Doing centering: do the pairwise purification
            purify_interactions(vals, counts, tol=tol)

        center_main_effects(vals, counts, bias=self.bias.data.item())

        return build_terms_frame(vals, counts, X.columns)

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
        fs = torch.cat([layer.get_feature_selectors() for layer in self], dim=1)
        # ^-- [in_features, layers*num_trees, depth]
        return terms_from_feature_selectors(fs, return_inverse=return_inverse)


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

    def create_layers(
        self, in_features, num_trees, num_layers, tree_dim, max_features=None, **kwargs
    ):
        """Create layers.

        Args:
            in_features: The input dimension (feature).
            num_trees: The number of trees in a layer.
            num_layers: The number of layers.
            tree_dim: The output dimension of each tree.
            kwargs (dict): The arguments for underlying GAM ODST trees.
        """
        del max_features
        layers = []
        for _i in range(num_layers):
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
        choice_function = self[0].choice_function
        if hasattr(choice_function, "is_deterministic") and not choice_function.is_deterministic:
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
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
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

    def create_layers(
        self, in_features, num_trees, num_layers, tree_dim, max_features=None, **kwargs
    ):
        """Create layers of oblivious trees.

        Args:
            in_features: The dim of input features.
            num_trees: The number of trees in a layer.
            num_layers: The number of layers.
            tree_dim: The output dimension of each tree.
            kwargs: The kwargs for initializing GAMAtt ODST trees.
        """
        del max_features
        layers = []
        prev_in_features = 0
        for _i in range(num_layers):
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
