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
            with torch.no_grad():
                self.n.add_(1)
                self.mean.add_((x.detach().mean(dim=0) - self.mean) / self.n)
        return x - self.mean


class CubicSplineLayer(nn.Module):
    def __init__(
        self,
        n_bases=10,
        min_val=0,
        max_val=1,
        learn_knots=False,
        identify=True,
        n_outputs=1,
    ):
        super().__init__()
        if n_bases < 3:
            raise ValueError("n_bases must be at least 3 for a cubic spline layer.")
        if max_val <= min_val:
            raise ValueError("max_val must be greater than min_val.")

        self.learn_knots = learn_knots
        self.min_val = min_val
        self.max_val = max_val
        self.n_bases = n_bases

        if self.learn_knots:
            # We fix the first and last knot and only learn the distances between the intermediate knots
            self.relative_distances = nn.Parameter(
                torch.rand(n_bases - 1)
            )
        self.register_buffer("knots", torch.linspace(min_val, max_val, n_bases))

        self.identify = identify

        if identify:
            self.demean = RunningMean(n_bases)
            self.linear = nn.Linear(n_bases, n_outputs, bias=True)
        else:
            self.linear = nn.Linear(n_bases, n_outputs, bias=False)

        F, S = self.compute_F_and_S(self.knots)
        self.register_buffer("F", F)
        self.register_buffer("S", S)

    def compute_knots(self):
        """
        Compute the knots based on the fixed first and last values, and the learned relative distances.
        """
        # Ensure that the distances are positive
        distances = torch.nn.functional.softplus(self.relative_distances)

        total_distance = distances.sum()
        normalized_distances = (
            distances * (self.max_val - self.min_val) / total_distance
        )

        # Compute cumulative sum of distances, ensuring the last value is exactly max_val
        knots = torch.cat(
            [
                self.relative_distances.new_tensor([self.min_val]),
                self.min_val + torch.cumsum(normalized_distances, 0),
            ]
        )

        return knots

    def forward(self, x):
        if x.ndim == 2:
            if x.shape[1] != 1:
                raise ValueError(
                    "CubicSplineLayer expects one scalar value per sample; "
                    f"received shape {tuple(x.shape)}."
                )
            x = x[:, 0]
        elif x.ndim != 1:
            raise ValueError(
                "CubicSplineLayer expects a 1D tensor or a two-dimensional "
                f"single-column tensor; received shape {tuple(x.shape)}."
            )

        if self.learn_knots and self.training:
            # Compute the knots during training based on the relative distances
            self.knots = self.compute_knots()

            # Recompute F and S based on the updated knots
            self.F, self.S = self.compute_F_and_S(self.knots)

        x = self.apply_spline_basis(x, self.knots, self.F)
        if self.identify:
            x = self.demean(x)
        return self.linear(x)

    def compute_F_and_S(self, knots):
        """
        Compute the F matrix for the spline basis and S matrix for penalization.
        """
        k = len(knots)
        h = torch.diff(knots)
        h_shift_up = h[1:]

        D = knots.new_zeros((k - 2, k))
        eye = torch.eye(k - 2, device=knots.device, dtype=knots.dtype)
        D[:, : k - 2] += (1 / h[: k - 2]) * eye
        D[:, 1 : k - 1] += (-1 / h[: k - 2] - 1 / h_shift_up) * eye
        D[:, 2:k] += (1 / h_shift_up) * eye

        B = knots.new_zeros((k - 2, k - 2))
        B += ((h[: k - 2] + h_shift_up) / 3) * eye
        off_diagonal = torch.eye(k - 3, device=knots.device, dtype=knots.dtype)
        B[:-1, 1:] += off_diagonal * (h_shift_up[: k - 3] / 6)
        B[1:, :-1] += off_diagonal * (h_shift_up[: k - 3] / 6)

        F_minus = torch.linalg.solve(B, D)
        F = torch.vstack([knots.new_zeros(k), F_minus, knots.new_zeros(k)])
        S = D.T @ F_minus
        return F, S

    def apply_spline_basis(self, x, knots, F):
        """
        Apply the spline basis to the input x based on the knots and F matrix.
        """
        n = len(x)
        k = len(knots)
        base = x.new_zeros((n, k))

        for i in range(n):
            if x[i] < knots[0]:
                h = knots[1] - knots[0]
                xik = x[i] - knots[0]
                c_jm = -xik * h / 3
                c_jp = -xik * h / 6
                base[i, :] += c_jm * F[0, :] + c_jp * F[1, :]
                base[i, 0] += 1 - xik / h
                base[i, 1] += xik / h
            elif x[i] > knots[-1]:
                j = len(knots) - 1
                h = knots[j] - knots[j - 1]
                xik = x[i] - knots[j]
                c_jm = xik * h / 6
                c_jp = xik * h / 3
                base[i, :] += c_jm * F[j - 1, :] + c_jp * F[j, :]
                base[i, j - 1] += -xik / h
                base[i, j] += 1 + xik / h
            else:
                j = int(
                    torch.searchsorted(knots, x[i], right=True)
                    .clamp(min=1, max=k - 1)
                    .item()
                )
                x_j = knots[j - 1]
                x_j1 = knots[j]
                h = x_j1 - x_j
                a_jm = (x_j1 - x[i]) / h
                a_jp = (x[i] - x_j) / h
                c_jm = ((x_j1 - x[i]) ** 3 / h - h * (x_j1 - x[i])) / 6
                c_jp = ((x[i] - x_j) ** 3 / h - h * (x[i] - x_j)) / 6
                base[i, :] += c_jm * F[j - 1, :] + c_jp * F[j, :]
                base[i, j - 1] += a_jm
                base[i, j] += a_jp

        return base

    def get_smooth_penalty(self):
        return (self.linear.weight @ self.S * self.linear.weight).sum()

    def get_knot_distance_penalty(self):
        if self.learn_knots:
            return (1 / torch.diff(self.knots).abs()).sum()
        return self.knots.new_zeros(())
