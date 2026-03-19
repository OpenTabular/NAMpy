import torch
import torch.nn as nn


class NeuralDecisionTree(nn.Module):
    """
    Differentiable soft decision tree.

    Parameters
    ----------
    input_dim : int
        Number of input features for this tree.
    depth : int
        Tree depth. Number of leaves is 2 ** depth.
    output_dim : int, default=1
        Number of outputs per sample.
    lamda : float, default=1e-3
        Strength of the tree balance penalty.
    temperature : float, default=1.0
        Temperature for sigmoid routing. Lower = sharper splits.
    use_hard_routing_in_eval : bool, default=False
        If True, uses hard routing at evaluation time only.
    """

    def __init__(
        self,
        input_dim: int,
        depth: int,
        output_dim: int = 1,
        lamda: float = 1e-3,
        temperature: float = 1.0,
        use_hard_routing_in_eval: bool = False,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if output_dim < 1:
            raise ValueError("output_dim must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.input_dim = input_dim
        self.depth = depth
        self.output_dim = output_dim
        self.lamda = float(lamda)
        self.temperature = float(temperature)
        self.use_hard_routing_in_eval = bool(use_hard_routing_in_eval)

        self.internal_node_num_ = 2**depth - 1
        self.leaf_node_num_ = 2**depth

        # Internal-node split logits
        self.inner_nodes = nn.Linear(input_dim, self.internal_node_num_, bias=True)

        # Leaf values / responses
        self.leaf_values = nn.Parameter(
            torch.empty(self.leaf_node_num_, output_dim)
        )
        nn.init.xavier_uniform_(self.leaf_values)

        # Layer-wise penalty coefficients
        self.penalty_list = [self.lamda * (2.0 ** (-d)) for d in range(depth)]

    def _cal_penalty(self, layer_idx: int, mu_parent: torch.Tensor, gate_soft: torch.Tensor):
        """
        Tree-balance penalty for one layer.

        Parameters
        ----------
        layer_idx : int
            Layer index.
        mu_parent : Tensor of shape [B, n_nodes_layer]
            Probability of reaching each parent node in this layer.
        gate_soft : Tensor of shape [B, n_nodes_layer]
            Soft routing probability for one branch at each node.

        Returns
        -------
        Tensor
            Scalar penalty.
        """
        coeff = self.penalty_list[layer_idx]
        if coeff <= 0.0:
            return mu_parent.new_zeros(())

        eps = 1e-6
        denom = mu_parent.sum(dim=0) + eps
        alpha = (mu_parent * gate_soft).sum(dim=0) / denom
        alpha = alpha.clamp(min=eps, max=1.0 - eps)

        penalty = -0.5 * coeff * (torch.log(alpha) + torch.log(1.0 - alpha)).sum()
        return penalty

    def _forward(self, X: torch.Tensor):
        """
        Compute leaf probabilities and regularization penalty.

        Returns
        -------
        mu : Tensor of shape [B, n_leaves]
            Leaf probabilities / routing weights.
        penalty : Tensor
            Scalar tree penalty.
        """
        batch_size = X.shape[0]

        logits = self.inner_nodes(X) / self.temperature
        gate_soft = torch.sigmoid(logits)

        if self.use_hard_routing_in_eval and not self.training:
            gate_route = (logits > 0).to(gate_soft.dtype)
        else:
            gate_route = gate_soft

        mu = X.new_ones(batch_size, 1)
        penalty = X.new_zeros(())

        begin_idx = 0
        end_idx = 1

        for layer_idx in range(self.depth):
            gate_soft_layer = gate_soft[:, begin_idx:end_idx]   # [B, n_nodes_layer]
            gate_route_layer = gate_route[:, begin_idx:end_idx] # [B, n_nodes_layer]

            penalty = penalty + self._cal_penalty(layer_idx, mu, gate_soft_layer)

            # Left/right child routing weights
            # Shape: [B, n_nodes_layer, 2]
            child_prob = torch.stack(
                [gate_route_layer, 1.0 - gate_route_layer],
                dim=-1,
            )

            # Broadcast current node mass to its two children
            mu = mu.unsqueeze(-1) * child_prob
            mu = mu.reshape(batch_size, -1)

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        return mu, penalty

    def forward(self, X: torch.Tensor, return_penalty: bool = False):
        """
        Forward pass.

        Parameters
        ----------
        X : Tensor
            Input of shape [B, input_dim] or [B].
        return_penalty : bool, default=False
            Whether to also return the tree penalty.

        Returns
        -------
        Tensor or tuple(Tensor, Tensor)
            Prediction tensor of shape [B, output_dim], and optionally the penalty.
        """
        if X.ndim == 1:
            X = X.unsqueeze(-1)
        X = X.reshape(X.shape[0], -1).float()

        mu, penalty = self._forward(X)
        y_pred = mu @ self.leaf_values

        if return_penalty:
            return y_pred, penalty
        return y_pred