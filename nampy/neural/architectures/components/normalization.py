import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square normalization layer.

    Attributes:
        d_model (int): The dimensionality of the input and output tensors.
        eps (float): Small value to avoid division by zero.
        weight (nn.Parameter): Learnable parameter for scaling.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )

        return output


class LayerNorm(nn.Module):
    """Layer normalization layer.

    Attributes:
        d_model (int): The dimensionality of the input and output tensors.
        eps (float): Small value to avoid division by zero.
        weight (nn.Parameter): Learnable parameter for scaling.
        bias (nn.Parameter): Learnable parameter for shifting.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return nn.functional.layer_norm(
            x, (self.weight.numel(),), self.weight, self.bias, self.eps
        )


class BatchNorm(nn.Module):
    """Batch normalization layer.

    Attributes:
        d_model (int): The dimensionality of the input and output tensors.
        eps (float): Small value to avoid division by zero.
        momentum (float): The value used for the running mean and variance computation.
    """

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, d_model: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(d_model))
        self.register_buffer("running_var", torch.ones(d_model))
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return nn.functional.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )


class InstanceNorm(nn.Module):
    """Instance normalization layer.

    Attributes:
        d_model (int): The dimensionality of the input and output tensors.
        eps (float): Small value to avoid division by zero.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Tabular rows are independent instances; spatial tensors retain their
        # conventional [batch, channels, ...] instance-normalization meaning.
        if x.ndim < 2 or x.shape[1] != self.weight.numel():
            raise ValueError(
                "InstanceNorm expects [batch, channels, ...] with the configured width."
            )
        dims = (-1,) if x.ndim == 2 else tuple(range(2, x.ndim))
        output = (x - x.mean(dim=dims, keepdim=True)) * torch.rsqrt(
            x.var(dim=dims, correction=0, keepdim=True) + self.eps
        )
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return output * self.weight.view(shape) + self.bias.view(shape)


class GroupNorm(nn.Module):
    """Group normalization layer.

    Attributes:
        num_groups (int): Number of groups to separate the channels into.
        d_model (int): The dimensionality of the input and output tensors.
        eps (float): Small value to avoid division by zero.
    """

    def __init__(self, num_groups: int, d_model: int, eps: float = 1e-5):
        super().__init__()
        if num_groups < 1 or d_model % num_groups:
            raise ValueError(
                "GroupNorm channels must be divisible by a positive group count."
            )
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        if x.ndim < 2 or x.shape[1] != self.weight.numel():
            raise ValueError(
                "GroupNorm expects [batch, channels, ...] with the configured width."
            )
        grouped = x.reshape(x.shape[0], self.num_groups, -1)
        output = (grouped - grouped.mean(dim=-1, keepdim=True)) * torch.rsqrt(
            grouped.var(dim=-1, correction=0, keepdim=True) + self.eps
        )
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return output.reshape_as(x) * self.weight.view(shape) + self.bias.view(shape)


class LearnableLayerScaling(nn.Module):
    """Learnable Layer Scaling (LLS) normalization layer.

    Attributes:
        d_model (int): The dimensionality of the input and output tensors.
    """

    def __init__(self, d_model: int):
        """Initialize LLS normalization layer."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * self.weight.unsqueeze(0)
        return output
