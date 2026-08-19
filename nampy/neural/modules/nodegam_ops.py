# nodegam_ops.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.jit import script

"""Neural Network related utils like Entmax and Modules."""


def check_numpy(x):
    """Makes sure x is a numpy array. If not, make it as one."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


def process_in_chunks(function, *args, batch_size, out=None, **kwargs):
    """Computes output by applying batch-parallel function to large data tensor in chunks.

    Args:
        function: a function(*[x[indices, ...] for x in args]) -> out[indices, ...].
        args: one or many tensors, each [num_instances, ...].
        batch_size: maximum chunk size processed in one go.
        out: memory buffer for out, defaults to torch.zeros of appropriate size and type.

    Returns:
        out: the outputs of function(data), computed in a memory-efficient (mini-batch) way.
    """
    total_size = args[0].shape[0]
    first_output = function(*[x[0:batch_size] for x in args])
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(
            *output_shape,
            dtype=first_output.dtype,
            device=first_output.device,
            layout=first_output.layout,
            **kwargs,
        )

    out[0:batch_size] = first_output
    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = function(*[x[batch_ix] for x in args])
    return out


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class Entmax15Function(Function):
    """Entropy Max (EntMax).

    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim

        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val  # same numerical stability trick as for softmax
        input = input / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (Y,) = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)

        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt**2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean**2)
        delta = (1 - ss) / rho

        # NOTE this is not exactly the same as in reference algo
        # Fortunately it seems the clamped values never wrongly
        # get selected by tau <= sorted_z. Prove this!
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


class Entmoid15(Function):
    """A highly optimized equivalent of lambda x: Entmax15([x, 0])."""

    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    @script
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input**2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    @script
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


def entmax15(input, dim=-1):
    return Entmax15Function.apply(input, dim)


entmoid15 = Entmoid15.apply


def my_one_hot(val, dim=-1):
    """Make one hot encoding along certain dimension and not just the last dimension.

    Args:
        val: A pytorch tensor.
        dim: The dimension.
    """
    max_cls = torch.argmax(val, dim=dim)
    onehot = F.one_hot(max_cls, num_classes=val.shape[dim])

    # swap back the dimension
    if dim != -1 and dim != val.ndim - 1:
        the_dim = list(range(onehot.ndim))
        the_dim.insert(dim, the_dim.pop(-1))
        onehot = onehot.permute(the_dim)

    return onehot


class _Temp(nn.Module):
    """Shared base class to do temperature annealing for EntMax/SoftMax/GumbleMax functions."""

    def __init__(self, steps, max_temp=1.0, min_temp=0.01, sample_soft=False):
        """Annealing temperature from max to min in log10 space.

        Args:
            steps: The number of steps to change from max_temp to the min_temp in log10 space.
            max_temp: The max (initial) temperature.
            min_temp: The min (final) temperature.
            sample_soft: If False, the model does a hard operation after the specified steps.
        """
        super().__init__()
        self.steps = steps
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.sample_soft = sample_soft

        # Initialize to nn Parameter to store it in the model state_dict
        self.tau = nn.Parameter(
            torch.tensor(max_temp, dtype=torch.float32), requires_grad=False
        )

    def forward(self, logits, dim=-1):
        # During training and under annealing, run a soft max operation
        if self.sample_soft or (self.training and self.tau.item() > self.min_temp):
            return self.forward_with_tau(logits, dim=dim)

        # In test time, sample a hard max
        with torch.no_grad():
            return self.discrete_op(logits, dim=dim)

    def discrete_op(self, logits, dim=-1):
        return my_one_hot(logits, dim=dim).float()

    @property
    def is_deterministic(self):
        return (not self.sample_soft) and (
            not self.training or self.tau.item() <= self.min_temp
        )

    def temp_step_callback(self, step):
        # Calculate the temp; allow fractional step!
        if step >= self.steps:
            self.tau.data = torch.tensor(self.min_temp, dtype=torch.float32)
        else:
            logmin = np.log10(self.min_temp)
            logmax = np.log10(self.max_temp)
            # Linearly interpolate it;
            logtemp = logmax + step / self.steps * (logmin - logmax)
            temp = 10**logtemp
            self.tau.data = torch.tensor(temp, dtype=torch.float32)

    def forward_with_tau(self, logits, dim):
        raise NotImplementedError()


class EM15Temp(_Temp):
    """EntMax15 with temperature annealing."""

    def forward_with_tau(self, logits, dim):
        return entmax15(logits / self.tau.item(), dim=dim)


class ModuleWithInit(nn.Module):
    """Base class for pytorch module with data-aware initializer on first batch."""

    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = nn.Parameter(
            torch.tensor(0, dtype=torch.float32), requires_grad=False
        )
        self._is_initialized_bool = None
        # Note: this module uses a separate flag self._is_initialized so as to achieve both
        # * persistence: is_initialized is saved alongside model in state_dict
        # * speed: model doesn't need to cache
        # please DO NOT use these flags in child modules
        # I change the type to torch.float32 to use apex 16 precision training

    def initialize(self, *args, **kwargs):
        """initialize module tensors using first batch of data."""
        raise NotImplementedError("Please implement ")

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)
