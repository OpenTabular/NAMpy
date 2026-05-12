import torch
import torch.nn as nn


class RunningMean(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros((1, n_dim)))
        self.register_buffer("n", torch.zeros((1,)))
        self._update_mean = True  # Toggle to control mean updating

    def forward(self, x):
        if self.training and self._update_mean:
            batch_mean = x.detach().mean(dim=0)
            self.n += 1
            self.mean = ((self.n - 1) / self.n) * self.mean + (1 / self.n) * batch_mean
        return x - self.mean


class CubicSplineLayer(nn.Module):
    def __init__(
        self,
        n_bases=10,
        min_val=0,
        max_val=1,
        learn_knots=False,
        identify=True,
        input_dim=1,
        output_dim=1,
    ):
        super().__init__()
        if n_bases < 3:
            raise ValueError("CubicSplineLayer requires at least 3 basis functions.")
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1.")
        if output_dim < 1:
            raise ValueError("output_dim must be >= 1.")

        self.learn_knots = learn_knots
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.n_bases = int(n_bases)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        if self.learn_knots:
            # The first and last knots are fixed; distances define all intervals.
            self.relative_distances = nn.Parameter(torch.rand(self.n_bases - 1))

        self.identify = identify
        feature_dim = self.input_dim * self.n_bases

        if identify:
            self.demean = RunningMean(feature_dim)
            self.linear = nn.Linear(feature_dim, self.output_dim, bias=True)
        else:
            self.linear = nn.Linear(feature_dim, self.output_dim, bias=False)

        knots = torch.linspace(self.min_val, self.max_val, self.n_bases)
        F, S = self.compute_F_and_S(knots)
        self.register_buffer("knots", knots)
        self.register_buffer("F", F)
        self.register_buffer("S", S)

    def compute_knots(self):
        """
        Compute the knots based on the fixed first and last values, and the learned relative distances.
        """
        distances = torch.nn.functional.softplus(self.relative_distances)
        total_distance = distances.sum()
        normalized_distances = (
            distances * (self.max_val - self.min_val) / total_distance
        )

        first_knot = distances.new_tensor([self.min_val])
        knots = torch.cat(
            [
                first_knot,
                self.min_val + torch.cumsum(normalized_distances, 0),
            ]
        )
        return knots

    def forward(self, x):
        x = self._prepare_input(x)
        knots, F, _ = self._current_spline_components()
        basis = [
            self.apply_spline_basis(x[:, dim], knots, F)
            for dim in range(self.input_dim)
        ]
        x = torch.cat(basis, dim=1)
        if self.identify:
            x = self.demean(x)
        return self.linear(x)

    def _prepare_input(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if x.shape[1] != self.input_dim:
            raise ValueError(
                "CubicSplineLayer expected input dimension "
                f"{self.input_dim}, got {x.shape[1]}."
            )
        return x.float()

    def _current_spline_components(self):
        if not self.learn_knots:
            return self.knots, self.F, self.S

        knots = self.compute_knots()
        F, S = self.compute_F_and_S(knots)
        return knots, F, S

    def compute_F_and_S(self, knots):
        """
        Compute the F matrix for the spline basis and S matrix for penalization.
        """
        k = len(knots)
        h = torch.diff(knots)
        if torch.any(h <= 0):
            raise ValueError("Spline knots must be strictly increasing.")

        h_shift_up = h[1:]
        device = knots.device
        dtype = knots.dtype

        D = torch.zeros((k - 2, k), device=device, dtype=dtype)
        D[:, : k - 2] += torch.diag(1 / h[: k - 2])
        D[:, 1 : k - 1] += torch.diag(-1 / h[: k - 2] - 1 / h_shift_up)
        D[:, 2:k] += torch.diag(1 / h_shift_up)

        B = torch.diag((h[: k - 2] + h_shift_up) / 3)
        if k > 3:
            off_diag = torch.diag(h_shift_up[: k - 3] / 6)
            B[:-1, 1:] += off_diag
            B[1:, :-1] += off_diag

        F_minus = torch.linalg.solve(B, D)
        F = torch.vstack(
            [
                torch.zeros(k, device=device, dtype=dtype),
                F_minus,
                torch.zeros(k, device=device, dtype=dtype),
            ]
        )
        S = D.T @ torch.linalg.solve(B, D)
        return F, S

    def apply_spline_basis(self, x, knots, F):
        """
        Apply the spline basis to the input x based on the knots and F matrix.
        """
        n = len(x)
        k = len(knots)
        base = x.new_zeros((n, k))

        for i in range(n):
            value = x[i]
            if value <= knots[0]:
                h = knots[1] - knots[0]
                xik = value - knots[0]
                c_jm = -xik * h / 3
                c_jp = -xik * h / 6
                base[i, :] += c_jm * F[0, :] + c_jp * F[1, :]
                base[i, 0] += 1 - xik / h
                base[i, 1] += xik / h
            elif value >= knots[-1]:
                j = len(knots) - 1
                h = knots[j] - knots[j - 1]
                xik = value - knots[j]
                c_jm = xik * h / 6
                c_jp = xik * h / 3
                base[i, :] += c_jm * F[j - 1, :] + c_jp * F[j, :]
                base[i, j - 1] += -xik / h
                base[i, j] += 1 + xik / h
            else:
                j = int(torch.searchsorted(knots, value, right=True).item())
                x_j = knots[j - 1]
                x_j1 = knots[j]
                h = x_j1 - x_j
                a_jm = (x_j1 - value) / h
                a_jp = (value - x_j) / h
                c_jm = ((x_j1 - value) ** 3 / h - h * (x_j1 - value)) / 6
                c_jp = ((value - x_j) ** 3 / h - h * (value - x_j)) / 6
                base[i, :] += c_jm * F[j - 1, :] + c_jp * F[j, :]
                base[i, j - 1] += a_jm
                base[i, j] += a_jp

        return base

    def get_smooth_penalty(self):
        _, _, S = self._current_spline_components()
        penalty = self.linear.weight.new_zeros(())
        for dim in range(self.input_dim):
            start = dim * self.n_bases
            stop = start + self.n_bases
            weight = self.linear.weight[:, start:stop]
            penalty = penalty + torch.einsum("ok,kl,ol->", weight, S, weight)
        return penalty

    def get_knot_distance_penalty(self):
        if self.learn_knots:
            knots = self.compute_knots()
            return (1 / torch.diff(knots).abs().clamp_min(1e-8)).sum()
        return self.linear.weight.new_zeros(())

    def get_knot_locations(self):
        knots, _, _ = self._current_spline_components()
        return knots.detach().cpu()


class TensorProductCubicSplineLayer(CubicSplineLayer):
    def __init__(
        self,
        n_bases=10,
        min_val=0,
        max_val=1,
        learn_knots=False,
        identify=True,
        input_dim=2,
        output_dim=1,
    ):
        if input_dim < 2:
            raise ValueError("TensorProductCubicSplineLayer requires input_dim >= 2.")
        super().__init__(
            n_bases=n_bases,
            min_val=min_val,
            max_val=max_val,
            learn_knots=learn_knots,
            identify=identify,
            input_dim=input_dim,
            output_dim=output_dim,
        )

        feature_dim = self.n_bases**self.input_dim
        if identify:
            self.demean = RunningMean(feature_dim)
            self.linear = nn.Linear(feature_dim, self.output_dim, bias=True)
        else:
            self.linear = nn.Linear(feature_dim, self.output_dim, bias=False)

    def forward(self, x):
        x = self._prepare_input(x)
        knots, F, _ = self._current_spline_components()
        basis = self.apply_spline_basis(x[:, 0], knots, F)
        for dim in range(1, self.input_dim):
            next_basis = self.apply_spline_basis(x[:, dim], knots, F)
            basis = (basis.unsqueeze(-1) * next_basis.unsqueeze(1)).reshape(
                x.shape[0], -1
            )

        if self.identify:
            basis = self.demean(basis)
        return self.linear(basis)

    def get_smooth_penalty(self):
        _, _, S = self._current_spline_components()
        weight = self.linear.weight.reshape(
            self.output_dim,
            *([self.n_bases] * self.input_dim),
        )
        penalty = self.linear.weight.new_zeros(())
        for dim in range(self.input_dim):
            marginal_weights = weight.movedim(dim + 1, -1).reshape(-1, self.n_bases)
            penalty = penalty + torch.einsum(
                "nk,kl,nl->", marginal_weights, S, marginal_weights
            )
        return penalty
