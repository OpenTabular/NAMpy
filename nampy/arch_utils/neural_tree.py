import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralDecisionTree(nn.Module):
    def __init__(self, input_dim, depth, output_dim=1, lamda=1e-3):
        """
        Initialize the neural decision tree with a neural network at each leaf.

        Parameters:
        -----------
        input_dim: int
            The number of input features.
        depth: int
            The depth of the tree. The number of leaves will be 2^depth.
        output_dim: int
            The number of output classes (default is 1 for regression tasks).
        lamda: float
            Regularization parameter.
        """
        super(NeuralDecisionTree, self).__init__()
        self.internal_node_num_ = 2**depth - 1
        self.leaf_node_num_ = 2**depth
        self.lamda = lamda
        self.depth = depth

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [self.lamda * (2 ** (-d)) for d in range(0, depth)]

        # Initialize internal nodes with linear layers followed by hard thresholds
        self.inner_nodes = nn.Sequential(
            nn.Linear(input_dim + 1, self.internal_node_num_, bias=False),
        )

        self.leaf_nodes = nn.Linear(self.leaf_node_num_, output_dim, bias=False)

    def forward(self, X, training=False):
        _mu = self._forward(X)
        y_pred = self.leaf_nodes(_mu)

        # Return predictions and penalty if in training mode
        # if training:
        #    return y_pred, _penalty
        # else:
        return y_pred

    def _forward(self, X):
        """Implementation of the forward pass with hard decision boundaries."""
        batch_size = X.size()[0]
        X = self._data_augment(X)

        # Get the decision boundaries for the internal nodes
        decision_boundaries = self.inner_nodes(X)

        # Apply hard thresholding to simulate binary decisions
        path_prob = (decision_boundaries > 0).float()

        # Prepare for routing at the internal nodes
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)

        # Routing samples through the tree with hard decisions
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _mu = _mu * _path_prob  # Update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """
        Compute the regularization term for internal nodes in different layers.
        """
        penalty = torch.tensor(0.0)
        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2**layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(
                _path_prob[:, node] * _mu[:, node // 2], dim=0
            ) / torch.sum(_mu[:, node // 2], dim=0)

            coeff = self.penalty_list[layer_idx]

            penalty -= 0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha))

        return penalty

    def _data_augment(self, X):
        """Add a constant input `1` onto the front of each sample."""
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1)
        X = torch.cat((bias, X), 1)

        return X
