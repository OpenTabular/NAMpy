import torch
import torch.nn as nn
from itertools import combinations
from ..configs.nbm_config import DefaultNBMConfig
from .basemodel import BaseModel
from ..arch_utils.nbm_utils import ConceptNNBasesNary


class NBM(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNBMConfig = DefaultNBMConfig(),
        **kwargs,
    ):
        """
        Initialize Neural Basis Model (NBM) with N-ary order interactions and basis learning.

        Parameters
        ----------
        cat_feature_info : dict
            Information about categorical features.
        num_feature_info : dict
            Information about numerical features.
        num_classes : int, optional
            Number of output classes, by default 1.
        nary : list or dict, optional
            Specifies n-ary interaction orders to model.
        num_bases : int, optional
            Number of shared basis functions.
        hidden_dims : tuple, optional
            Number of hidden units for neural MLP basis functions.
        num_subnets : int, optional
            Number of sub-networks to learn basis functions.
        dropout : float, optional
            Dropout rate for hidden layers.
        bases_dropout : float, optional
            Dropout rate for entire basis functions.
        batchnorm : bool, optional
            Whether to apply batch normalization, by default True.
        config : DefaultNAMConfig, optional
            Configuration dataclass for hyperparameters.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.num_classes = num_classes
        dropout = self.hparams.get("dropout", config.dropout)
        batchnorm = self.hparams.get("batch_norm", config.batch_norm)

        self.num_bases = self.hparams.get("num_bases", config.num_bases)
        self.num_subnets = self.hparams.get("num_subnets", config.num_subnets)
        self.batchnorm = self.hparams.get("batch_norm", config.batch_norm)
        self.bases_dropout = nn.Dropout(
            p=self.hparams.get("bases_dropout", config.bases_dropout)
        )
        # Infer total number of features
        self.input_dim = self.infer_input_dim(cat_feature_info, num_feature_info)
        self.nary = self.hparams.get("nary", config.nary) or {
            "1": list(combinations(range(self.input_dim), 1))
        }

        # Define basis networks (one per n-ary order and subnet)
        self.bases_nary_models = nn.ModuleDict()
        for order in self.nary.keys():
            for subnet in range(self.num_subnets):
                self.bases_nary_models[self.get_key(order, subnet)] = (
                    ConceptNNBasesNary(config=config)
                )

        # Featurizer (to reduce the output of the basis network)
        num_out_features = (
            sum(len(self.nary[order]) for order in self.nary.keys()) * self.num_subnets
        )
        self.featurizer = nn.Conv1d(
            in_channels=num_out_features * self.num_bases,
            out_channels=num_out_features,
            kernel_size=1,
            groups=num_out_features,
        )

        # Classifier layer
        self.classifier = nn.Linear(
            in_features=num_out_features, out_features=self.num_classes, bias=True
        )

    def infer_input_dim(self, cat_feature_info, num_feature_info):
        """
        Infers the input dimension based on categorical and numerical features.

        Parameters
        ----------
        cat_feature_info : dict
            Information about categorical features.
        num_feature_info : dict
            Information about numerical features.

        Returns
        -------
        int
            Total input dimension.
        """
        num_dim = 0
        for feature, info in num_feature_info.items():
            num_dim += info["dimension"]
        cat_dim = 0
        for feature, info in cat_feature_info.items():
            cat_dim += info["dimension"]
        return num_dim + cat_dim

    def get_key(self, order, subnet):
        """
        Helper function to generate unique keys for each subnet.

        Parameters
        ----------
        order : str
            Order of interactions (e.g., "1", "2").
        subnet : int
            Index of the subnet.

        Returns
        -------
        str
            Unique key for the basis model.
        """
        return f"ord{order}_net{subnet}"

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of NBM with n-ary order interactions and basis functions.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features.
        cat_features : dict
            Dictionary of categorical features.

        Returns
        -------
        dict
            Output predictions and feature contributions, keyed by actual feature names.
        """
        # List to hold the basis outputs
        bases = []
        feature_contributions = {}  # Dictionary to hold contributions for each feature

        # Combine all numerical features and categorical features for basis function input
        all_features = torch.cat(
            [num_features[feature] for feature in num_features]
            + [cat_features[feature] for feature in cat_features],
            dim=1,
        )

        # Track index positions to map them back to features
        num_feature_keys = list(num_features.keys())
        cat_feature_keys = list(cat_features.keys())
        all_feature_keys = num_feature_keys + cat_feature_keys

        # Compute basis outputs for each n-ary interaction order
        for order in self.nary.keys():
            for subnet in range(self.num_subnets):
                input_order = all_features[
                    :, self.nary[order]
                ]  # Use n-ary interactions
                basis_output = self.bases_dropout(
                    self.bases_nary_models[self.get_key(order, subnet)](
                        input_order.reshape(-1, input_order.shape[-1])
                    ).reshape(input_order.shape[0], input_order.shape[1], -1)
                )
                bases.append(basis_output)

        # Concatenate all bases and pass through the featurizer
        bases = torch.cat(bases, dim=-2)
        out_feats = self.featurizer(bases.reshape(input_order.shape[0], -1, 1)).squeeze(
            -1
        )
        for i, feature_idx in enumerate(self.nary[order]):
            feature_name = all_feature_keys[feature_idx[0]]
            if feature_name not in feature_contributions:
                feature_contributions[feature_name] = out_feats[:, feature_idx[0]]

        # Final classification layer
        output = self.classifier(out_feats)

        # Return output and feature contributions as a dictionary
        result = {"output": output}
        result.update(feature_contributions)

        return result
