import torch
import torch.nn as nn
import math
from scipy.stats import norm
from ..configs.nam_config import DefaultNAMConfig
from .basemodel import BaseModel


class GPNAM(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        kernel_width=0.2,
        rff_num_feat=100,
        config: DefaultNAMConfig = DefaultNAMConfig(),
        **kwargs,
    ):
        """
        Build GPNAM model with RFF-based kernel approximation.

        Parameters
        ----------
        cat_feature_info : dict
            Information about categorical features.
        num_feature_info : dict
            Information about numerical features.
        kernel_width : float, optional
            Kernel width of RFF approximation, by default 0.2.
        rff_num_feat : int, optional
            Number of random Fourier features, by default 100.
        config : DefaultNAMConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultNAMConfig().
        """
        super().__init__(**kwargs)

        # Save kernel parameters
        self.kernel_width = kernel_width
        self.rff_num_feat = rff_num_feat

        # Infer input dimension from categorical and numerical feature info
        self.input_dim = self.infer_input_dim(cat_feature_info, num_feature_info)

        # Random Fourier Feature (RFF) parameters
        self.c = (
            2
            * math.pi
            * torch.rand(rff_num_feat, self.input_dim).sort(dim=0)[0]
            / rff_num_feat
        )
        self.Z = torch.from_numpy(
            norm.ppf([each / (rff_num_feat + 1) for each in range(1, rff_num_feat + 1)])
        ).float()

        # Freeze RFF parameters
        self.c.requires_grad = False
        self.Z.requires_grad = False

        # Weight parameter for Gaussian Process (GP) component
        self.w = nn.Parameter(
            torch.zeros(self.input_dim * rff_num_feat + 1, 1), requires_grad=True
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
        num_dim = sum(num_feature_info.values())  # Sum dimensions of numerical features
        cat_dim = sum(
            cat_feature_info.values()
        )  # Sum dimensions of categorical features
        return num_dim + cat_dim

    def forward(
        self, num_features: dict, cat_features: dict, feature_of_interest=None
    ) -> dict:
        """
        Forward pass through GPNAM with RFF-based kernel approximation.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.
        feature_of_interest : str, optional
            Name of the feature to visualize the prediction for. If None, the full model prediction is returned.

        Returns
        -------
        dict
            Output predictions from the model.
        """
        # Combine numerical and categorical features
        all_features = torch.cat(
            [num_features[feature] for feature in num_features]
            + [cat_features[feature] for feature in cat_features],
            dim=1,
        )

        # Handle feature-specific contributions
        if feature_of_interest is not None:
            # Set all other features to zero or mean (can be adjusted as needed)
            for feature in num_features:
                if feature != feature_of_interest:
                    num_features[feature] = torch.zeros_like(num_features[feature])
            for feature in cat_features:
                if feature != feature_of_interest:
                    cat_features[feature] = torch.zeros_like(cat_features[feature])

            # Re-combine the modified feature set
            all_features = torch.cat(
                [num_features[feature] for feature in num_features]
                + [cat_features[feature] for feature in cat_features],
                dim=1,
            )

        # Perform Random Fourier Feature mapping
        rff_mapping = math.sqrt(2 / self.rff_num_feat) * torch.cos(
            torch.einsum("i,pq -> piq", self.Z, all_features) / self.kernel_width
            + self.c
        )
        rff_mapping = torch.transpose(rff_mapping, 1, 2).reshape(
            all_features.shape[0], -1
        )
        rff_mapping = torch.column_stack(
            (rff_mapping, torch.ones(rff_mapping.shape[0]))
        ).float()

        # Linear GP layer
        pred = rff_mapping @ self.w

        # Squeeze if necessary
        if pred.dim() == 2 and pred.shape[1] == 1:
            pred = torch.squeeze(pred, 1)

        # Return dictionary with the output
        result = {"output": pred}
        if feature_of_interest is not None:
            result["feature_contribution"] = (
                pred  # Contribution from the specific feature
            )
        return result
