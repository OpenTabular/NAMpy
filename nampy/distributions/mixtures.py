"""Finite atoms mixed with a scalar continuous law, independent of fitting.

References
----------
- Ospina, R., and Ferrari, S. L. P. (2010). Inflated beta distributions.
  https://doi.org/10.1007/s00362-008-0125-4
- Lambert, D. (1992). Zero-Inflated Poisson Regression, With an Application to Defects
  in Manufacturing.
  https://doi.org/10.1080/00401706.1992.10485228

The generic mixture operations also support NAMpy compositions beyond these motivating
beta and count models.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .contracts import DistributionLaw, DistributionSpec
from .laws import BetaLaw, GammaLaw
from .prediction import DistributionPrediction


def validate_mixture_weights(masses, continuous):
    """Validate rowwise normalized weights without rescaling any component."""
    if np.any((masses < 0) | (masses > 1)) or np.any(
        (continuous < 0) | (continuous > 1)
    ):
        raise ValueError(
            "Mixture probabilities must be nonnegative and no greater than one."
        )
    if not np.all(
        np.isclose(
            masses.sum(axis=1) + continuous,
            1,
            rtol=0,
            atol=8 * float(np.finfo(float).eps),
        )
    ):
        raise ValueError(
            "Mixture probabilities must lie on the simplex within floating-point roundoff; they are not renormalized."
        )


@dataclass(frozen=True)
class AtomMixtureLaw:
    """Mix a continuous law with named point masses at fixed, ordered atoms.

    Parameters consist of the base law's parameters followed by one probability
    per atom. By default the continuous weight is one minus their sum. Set
    continuous_mass_name to supply it explicitly, preserving small weights
    from coupled transforms. Explicit weights must sum to one within eight
    float64 epsilons and are never renormalized. Zero weights and a
    purely atomic law are supported, but base parameters must remain valid.
    At an atom log_prob always reports log mass, including -inf for zero mass;
    elsewhere it reports log density. Quantile endpoints use support limits of
    components with positive weight. Quantiles currently evaluate the base law
    row by row to respect its vector-of-probabilities contract.

    Operations are derived from the base capabilities. This is probability
    composition only: no derivative, fitting or inference capability is earned.
    Discrete bases, nested mixed bases and vector events remain unsupported.
    """

    base: DistributionLaw
    atoms: tuple[float, ...]
    mass_names: tuple[str, ...]
    name: str = "atom_mixture"
    continuous_mass_name: str | None = None

    def __post_init__(self):
        atoms, names = tuple(self.atoms), tuple(self.mass_names)
        if (
            not atoms
            or len(atoms) != len(names)
            or not np.all(np.isfinite(atoms))
            or any(a >= b for a, b in zip(atoms, atoms[1:], strict=False))
        ):
            raise ValueError(
                "Atoms must be finite, strictly increasing, and match mass names."
            )
        if self.base.spec.event_shape or self.base.spec.response_kind != "continuous":
            raise NotImplementedError(
                "AtomMixtureLaw requires a scalar continuous base law."
            )
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "mass_names", names)
        # Constructing metadata also rejects duplicate/empty or colliding names.
        _ = self.spec

    @property
    def spec(self):
        requirements = {
            "log_prob": {"log_prob"},
            "cdf": {"cdf"},
            "cdf_left": {"cdf"},
            "log_cdf": {"log_cdf"},
            "survival": {"survival"},
            "log_survival": {"log_survival"},
            "quantile": {"quantile", "cdf"},
            "sample": {"sample"},
            "mean": {"mean"},
            "variance": {"mean", "variance"},
        }
        operations = frozenset(
            op
            for op, needs in requirements.items()
            if needs <= self.base.spec.operations
        )
        continuous_names = (
            () if self.continuous_mass_name is None else (self.continuous_mass_name,)
        )
        parameterization = f"atoms_v1:{self.atoms!r}:{self.base.spec.name}:{self.base.spec.parameterization}"
        if continuous_names:
            parameterization = (
                f"explicit_weights_v1:{self.continuous_mass_name}:{parameterization}"
            )
        return DistributionSpec(
            self.name,
            (*self.base.spec.parameter_names, *self.mass_names, *continuous_names),
            operations,
            parameterization,
            "mixed",
            context_names=self.base.spec.context_names,
            atom_locations=self.atoms,
        )

    def validate_parameters(self, parameters, *, context):
        self.base.validate_parameters(
            {name: parameters[name] for name in self.base.spec.parameter_names},
            context=context,
        )
        masses, continuous = self._weights(parameters)
        validate_mixture_weights(masses, continuous)

    def _weights(self, parameters):
        masses = np.column_stack([parameters[name] for name in self.mass_names])
        continuous = (
            1 - masses.sum(axis=1)
            if self.continuous_mass_name is None
            else parameters[self.continuous_mass_name]
        )
        return masses, continuous

    def evaluate(
        self, operation, parameters, *, context, values=None, n_draws=None, rng=None
    ):
        if operation not in self.spec.operations:
            raise NotImplementedError(
                f"{self.name} does not support {operation!r} for this base law."
            )
        masses, continuous = self._weights(parameters)
        n = len(masses)
        base_parameters = {
            name: parameters[name] for name in self.base.spec.parameter_names
        }
        atoms = np.asarray(self.atoms)

        def base(op, *, at=None, mask=None, **kwargs):
            if mask is None:
                return self.base.evaluate(
                    op, base_parameters, context=context, values=at, **kwargs
                )
            return self.base.evaluate(
                op,
                {name: value[mask] for name, value in base_parameters.items()},
                context={name: value[mask] for name, value in context.items()},
                values=at,
                **kwargs,
            )

        if operation in {"mean", "variance"}:
            active = continuous > 0
            bm = np.zeros(n)
            if np.any(active):
                bm[active] = base("mean", mask=active)
            mean = continuous * bm + masses @ atoms
            if operation == "mean":
                return mean
            bv = np.zeros(n)
            if np.any(active):
                bv[active] = base("variance", mask=active)
            return continuous * (bv + (bm - mean) ** 2) + np.sum(
                masses * (atoms[None, :] - mean[:, None]) ** 2, axis=1
            )
        if operation == "sample":
            output = np.zeros((n_draws, n))
            active = continuous > 0
            if np.any(active):
                output[:, active] = base(
                    "sample", mask=active, n_draws=n_draws, rng=rng
                )
            u = rng.uniform(size=(n_draws, n))
            cumulative = np.cumsum(masses, axis=1)
            lower = np.zeros(n)
            for j, atom in enumerate(atoms):
                output[(u >= lower) & (u < cumulative[:, j])] = atom
                lower = cumulative[:, j]
            return output
        if operation == "quantile":
            output = np.empty((n, len(values)))
            if not len(values):
                return output
            cdf_atoms = np.column_stack(
                [base("cdf", at=np.full(n, atom)) for atom in atoms]
            )
            atom_upper = np.cumsum(masses, axis=1) + continuous[:, None] * cdf_atoms
            atom_lower = atom_upper - masses
            for i in range(n):
                if continuous[i] == 0:
                    # Ignore zero-mass atoms, including at probabilities 0/1.
                    active = masses[i] > 0
                    support = atoms[active]
                    cumulative = np.cumsum(masses[i, active])
                    output[i] = support[
                        np.minimum(
                            np.searchsorted(cumulative, values, side="left"),
                            len(support) - 1,
                        )
                    ]
                    continue
                jump = (
                    (values[:, None] > atom_lower[i])
                    & (values[:, None] <= atom_upper[i])
                    & (masses[i] > 0)
                )
                has_jump = np.any(jump, axis=1)
                adjusted = (
                    values
                    - np.sum((values[:, None] > atom_upper[i]) * masses[i], axis=1)
                ) / continuous[i]
                reduced = np.clip(adjusted, 0, 1)
                # Per-row probabilities are not a global probability grid.
                row = slice(i, i + 1)
                output[i] = base("quantile", mask=row, at=reduced)[0]
                output[i, has_jump] = atoms[np.argmax(jump[has_jump], axis=1)]
                endpoints = base("quantile", mask=row, at=np.array([0.0, 1.0]))[0]
                positive_atoms = atoms[masses[i] > 0]
                if len(positive_atoms):
                    endpoints[0] = min(endpoints[0], positive_atoms[0])
                    endpoints[1] = max(endpoints[1], positive_atoms[-1])
                output[i, values == 0] = endpoints[0]
                output[i, values == 1] = endpoints[1]
            return output
        if operation == "log_prob":
            matches = values[:, None] == atoms
            at_atom = np.any(matches, axis=1)
            output = np.full(n, -np.inf)
            active = ~at_atom & (continuous > 0)
            if np.any(active):
                output[active] = np.log(continuous[active]) + base(
                    "log_prob", mask=active, at=values[active]
                )
            with np.errstate(divide="ignore"):
                output[at_atom] = np.log(
                    np.sum(matches[at_atom] * masses[at_atom], axis=1)
                )
            return output
        if operation in {"cdf", "cdf_left", "survival"}:
            include = (
                atoms <= values[:, None]
                if operation == "cdf"
                else atoms < values[:, None]
                if operation == "cdf_left"
                else atoms > values[:, None]
            )
            result = continuous * base(
                "survival" if operation == "survival" else "cdf", at=values
            ) + np.sum(include * masses, axis=1)
            return np.clip(result, 0, 1)
        include = (
            atoms <= values[:, None]
            if operation == "log_cdf"
            else atoms > values[:, None]
        )
        with np.errstate(divide="ignore"):
            continuous_log = np.log(continuous) + base(operation, at=values)
            atom_logs = np.where(include, np.log(masses), -np.inf)
        return logsumexp(np.column_stack([continuous_log, atom_logs]), axis=1)


INFLATED_BETA_LAW = AtomMixtureLaw(
    BetaLaw(),
    (0.0, 1.0),
    ("pi0", "pi1"),
    name="inflated_beta",
    continuous_mass_name="pi_interior",
)


def inflated_beta_prediction(parameters):
    """Map interior mean/precision and all three probabilities without subtraction."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 2 or parameters.shape[1] != 5:
        raise ValueError("Expected inflated-beta natural parameters (n, 5).")
    mu, phi, pi0, pi1, pi_interior = parameters.T
    if np.any((mu <= 0) | (mu >= 1)) or np.any(phi <= 0):
        raise ValueError(
            "Inflated-beta interior mu must be inside (0,1) and phi positive."
        )
    return DistributionPrediction(
        INFLATED_BETA_LAW,
        {
            "alpha": mu * phi,
            "beta": (1 - mu) * phi,
            "pi0": pi0,
            "pi1": pi1,
            "pi_interior": pi_interior,
        },
    )


ZERO_ADJUSTED_GAMMA_LAW = AtomMixtureLaw(
    GammaLaw(), (0.0,), ("pi0",), name="zero_adjusted_gamma"
)


def zero_adjusted_gamma_prediction(parameters):
    """Map positive Gamma mean/CV and atom probability to the shared law."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 2 or parameters.shape[1] != 3:
        raise ValueError("Expected zero-adjusted Gamma natural parameters (n, 3).")
    mu, sigma, pi0 = parameters.T
    if np.any(mu <= 0) or np.any(sigma <= 0):
        raise ValueError("Zero-adjusted Gamma mu and sigma must be positive.")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        shape = 1 / sigma**2
        rate = shape / mu
    return DistributionPrediction(
        ZERO_ADJUSTED_GAMMA_LAW, {"shape": shape, "rate": rate, "pi0": pi0}
    )
