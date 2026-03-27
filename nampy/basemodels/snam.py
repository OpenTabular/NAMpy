import torch

from ..configs.snam_config import DefaultSNAMConfig
from .nam import NAM


class SNAM(NAM):
    """
    Sparse Neural Additive Model (SNAM).

    This directly reuses NAM's architecture and forward structure, and adds a
    group-lasso penalty over each feature subnetwork's full parameter vector.

    Groups
    ------
    - one group per numerical feature subnet
    - one group per categorical feature subnet
    - optionally one group per interaction subnet

    The global intercept is NOT penalized.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultSNAMConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultSNAMConfig()
        super().__init__(
            cat_feature_info=cat_feature_info,
            num_feature_info=num_feature_info,
            num_classes=num_classes,
            config=config,
            **kwargs,
        )

        # Pull SNAM-specific args from kwargs first, then hparams/config.
        self.group_lasso_lambda = kwargs.get(
            "group_lasso_lambda",
            self.hparams.get("group_lasso_lambda", config.group_lasso_lambda),
        )
        self.group_lasso_include_interactions = kwargs.get(
            "group_lasso_include_interactions",
            self.hparams.get(
                "group_lasso_include_interactions",
                config.group_lasso_include_interactions,
            ),
        )

    def _iter_grouped_subnetworks(self):
        """Yield (name, module) pairs for all subnet groups included in the penalty."""
        for name, subnet in self.num_feature_networks.items():
            yield name, subnet

        for name, subnet in self.cat_feature_networks.items():
            yield name, subnet

        if self.group_lasso_include_interactions and hasattr(self, "interaction_networks"):
            for name, subnet in self.interaction_networks.items():
                yield name, subnet

    def _group_norm(self, module) -> torch.Tensor:
        """Return the L2 norm of the flattened trainable parameter vector for one subnet."""
        params = [p.reshape(-1) for p in module.parameters() if p.requires_grad]
        if not params:
            return next(self.parameters()).new_zeros(())
        theta_j = torch.cat(params)
        return torch.norm(theta_j, p=2)

    def _group_lasso_penalty(self) -> torch.Tensor:
        """Sum of L2 norms over all subnet groups."""
        penalty = next(self.parameters()).new_zeros(())
        for _, subnet in self._iter_grouped_subnetworks():
            penalty = penalty + self._group_norm(subnet)
        return penalty

    def get_group_norms(self):
        """
        Return current group norms as a plain Python dict.

        Useful for feature selection / inspection after training.
        """
        return {
            name: float(self._group_norm(subnet).detach().cpu())
            for name, subnet in self._iter_grouped_subnetworks()
        }

    def selected_groups(self, threshold: float = 1e-8):
        """Return names of groups whose parameter norm exceeds the threshold."""
        norms = self.get_group_norms()
        return [name for name, value in norms.items() if value > threshold]

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        result = super().forward(num_features=num_features, cat_features=cat_features)

        if self.group_lasso_lambda > 0.0:
            result["group_lasso_penalty"] = (
                self.group_lasso_lambda * self._group_lasso_penalty()
            )

        return result