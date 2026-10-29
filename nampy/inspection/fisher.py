"""Shared Fisher diagnostic contracts and streaming sampling-error summaries.

Numerical likelihoods and parameter derivatives remain backend-owned. These
records describe mean information, not fitted-parameter confidence intervals.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

_ESTIMATORS = frozenset({"exact", "score_outer_product", "negative_hessian"})
_INPUT_SAMPLING = frozenset({"fixed", "iid", "without_replacement"})


@dataclass(frozen=True, slots=True)
class FisherDiagnostics:
    """Controls for mean likelihood-information diagnostics and sampling error."""

    estimator: str = "exact"
    every_n_epochs: int | None = None
    n_inputs: int = 64
    n_draws: int = 8
    split: str = "validation"
    include_initial: bool = True
    include_final: bool = True
    record_diagonal: str = "final"
    group_by: str = "term"
    random_state: int | None = None
    regression_scale: float | None = None
    input_sampling: str = "fixed"
    population_size: int | None = None

    def __post_init__(self):
        for name in ("n_inputs", "n_draws", "every_n_epochs", "population_size"):
            value = getattr(self, name)
            if value is None and name in {"every_n_epochs", "population_size"}:
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if self.n_draws < 2:
            raise ValueError(
                "n_draws must be at least two for sampling-error estimation."
            )
        for name, values in (
            ("estimator", _ESTIMATORS),
            ("input_sampling", _INPUT_SAMPLING),
            ("split", {"train", "validation"}),
            ("record_diagonal", {"never", "final", "all"}),
            ("group_by", {"tensor", "layer", "term"}),
        ):
            if getattr(self, name) not in values:
                raise ValueError(f"{name} must be one of {sorted(values)}.")
        for name in ("include_initial", "include_final"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean.")
        if self.random_state is not None and (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, Integral)
            or not 0 <= self.random_state < 2**32
        ):
            raise ValueError("random_state must be an integer in [0, 2**32).")
        if self.regression_scale is not None and (
            not np.isfinite(self.regression_scale) or self.regression_scale <= 0
        ):
            raise ValueError("regression_scale must be finite and positive.")
        if (
            self.population_size is not None
            and self.input_sampling != "without_replacement"
        ):
            raise ValueError(
                "population_size applies only to sampling without replacement."
            )


def normalize_fisher_diagnostics(value):
    if value is False:
        return None
    if value is True:
        return FisherDiagnostics()
    if isinstance(value, FisherDiagnostics):
        return value
    raise TypeError("fisher_diagnostics must be False, True or FisherDiagnostics.")


def _array(value, *, size=None, nonnegative=False):
    if value is None:
        return None
    array = np.array(value, dtype=np.float64, copy=True)
    if (
        array.ndim != 1
        or (size is not None and array.size != size)
        or not np.isfinite(array).all()
    ):
        raise ValueError("Fisher arrays must be finite aligned vectors.")
    if nonnegative and np.any(array < 0):
        raise ValueError(
            "Fisher variances and score/exact information must be nonnegative."
        )
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FisherGroupSummary:
    """Information and sampling uncertainty for a disjoint parameter group."""

    name: str
    n_parameters: int
    trace: float
    mean_diagonal: float
    output_sampling_variance: float | None
    estimator_variance: float | None
    conditional_standard_error: float | None
    input_estimator_variance: float | None = None
    total_estimator_variance: float | None = None
    total_standard_error: float | None = None

    def __post_init__(self):
        if not self.name or self.n_parameters < 1:
            raise ValueError("Fisher groups require a name and positive size.")
        for name in (
            "trace",
            "mean_diagonal",
            "output_sampling_variance",
            "estimator_variance",
            "conditional_standard_error",
            "input_estimator_variance",
            "total_estimator_variance",
            "total_standard_error",
        ):
            value = getattr(self, name)
            if value is not None and (
                not np.isfinite(value)
                or (
                    name not in {"trace", "mean_diagonal", "input_estimator_variance"}
                    and value < 0
                )
            ):
                raise ValueError(f"Invalid Fisher group {name}.")


@dataclass(frozen=True, slots=True)
class FisherInformationResult:
    """Immutable mean parameter information and its sampling uncertainty."""

    estimator: str
    diagonal: np.ndarray | None
    output_sampling_variance: np.ndarray | None
    estimator_variance: np.ndarray | None
    conditional_standard_error: np.ndarray | None
    parameter_names: tuple[str, ...]
    parameter_shapes: tuple[tuple[int, ...], ...]
    parameter_slices: tuple[tuple[int, int], ...]
    groups: tuple[FisherGroupSummary, ...]
    n_parameters: int
    n_inputs: int
    n_draws: int | None
    target_source: str
    conditioned_on_inputs: bool
    split: str | None
    grouping: str | None = None
    normalization: str = "mean"
    backend: str = "unknown"
    held_fixed: tuple[str, ...] = ()
    input_sampling: str = "fixed"
    input_estimator_variance: np.ndarray | None = None
    total_estimator_variance: np.ndarray | None = None
    total_standard_error: np.ndarray | None = None

    def __post_init__(self):
        if (
            self.estimator not in _ESTIMATORS
            or self.normalization != "mean"
            or self.input_sampling not in _INPUT_SAMPLING
        ):
            raise ValueError(
                "Unknown Fisher estimator, normalization or sampling scheme."
            )
        if self.n_inputs < 1 or self.n_parameters < 1:
            raise ValueError("Fisher results require inputs and parameter coordinates.")
        cursor = 0
        if not (
            len(self.parameter_names)
            == len(self.parameter_shapes)
            == len(self.parameter_slices)
        ):
            raise ValueError("Fisher parameter metadata must align.")
        for shape, (start, stop) in zip(
            self.parameter_shapes, self.parameter_slices, strict=True
        ):
            if start != cursor or stop - start != int(np.prod(shape)):
                raise ValueError(
                    "Fisher parameter slices must be contiguous and match shapes."
                )
            cursor = stop
        if (
            cursor != self.n_parameters
            or sum(g.n_parameters for g in self.groups) != cursor
        ):
            raise ValueError(
                "Fisher layout and groups must exhaust parameter coordinates."
            )
        if len({g.name for g in self.groups}) != len(self.groups):
            raise ValueError("Fisher group names must be unique.")
        for name in (
            "diagonal",
            "output_sampling_variance",
            "estimator_variance",
            "conditional_standard_error",
            "input_estimator_variance",
            "total_estimator_variance",
            "total_standard_error",
        ):
            value = _array(
                getattr(self, name),
                size=self.n_parameters,
                nonnegative=name != "input_estimator_variance"
                and (name != "diagonal" or self.estimator != "negative_hessian"),
            )
            object.__setattr__(self, name, value)
        for name in (
            "parameter_names",
            "parameter_shapes",
            "parameter_slices",
            "groups",
            "held_fixed",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class FisherSnapshot:
    """Information recorded at an initial, periodic or final fitted state."""

    stage: str
    epoch: int | None
    optimizer_step: int
    groups: tuple[FisherGroupSummary, ...]
    diagonal: np.ndarray | None = None

    def __post_init__(self):
        if self.stage not in {"initial", "epoch", "final"} or self.optimizer_step < 0:
            raise ValueError("Invalid Fisher snapshot stage or step.")
        if self.stage == "epoch" and (self.epoch is None or self.epoch < 1):
            raise ValueError("Epoch snapshots require a positive completed epoch.")
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "diagonal", _array(self.diagonal))


class _Moments:
    """Online moments of per-input means and conditional output variances."""

    def __init__(self, width):
        self.mean = np.zeros(width)
        self.m2 = np.zeros(width)
        self.within = np.zeros(width)
        self.n = 0

    def update(self, mean, within):
        self.n += 1
        delta = mean - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (mean - self.mean)
        if within is not None:
            self.within += within

    def summarize(self, *, config, population_size):
        n, r = self.n, config.n_draws
        sampled = config.estimator != "exact"
        output = self.within / n if sampled else None
        conditional = self.within / (n * n * r) if sampled else None
        if config.input_sampling == "fixed":
            return self.mean, output, conditional, None, None
        if n < 2:
            raise ValueError(
                "Input-population uncertainty requires at least two inputs."
            )
        between = self.m2 / ((n - 1) * n)
        if config.input_sampling == "without_replacement":
            if population_size is None or population_size < n:
                raise ValueError(
                    "A finite population must be at least as large as the probe."
                )
            fraction = n / population_size
            total = (1 - fraction) * between + fraction * (
                0 if conditional is None else conditional
            )
        else:
            total = between
        # An unbiased input component can be negative from finite MC noise.
        # Keep it signed rather than silently clipping and biasing the estimate.
        inputs = total - (0 if conditional is None else conditional)
        return self.mean, output, conditional, inputs, total


def aggregate_fisher(
    rows,
    *,
    config,
    parameter_names,
    parameter_shapes,
    parameter_slices,
    groups,
    backend,
    estimator,
    n_inputs,
    split=None,
    grouping=None,
    population_size=None,
    return_diagonal=True,
    held_fixed=(),
):
    """Stream per-input moments into mean Fisher and sampling-error records.

    Each row is (mean coordinate estimates, within-output coordinate variance,
    within-output variance of each group's summed estimate). Group variance
    must be calculated from joint draws, preserving coordinate covariance.
    Input variance components are unbiased signed estimates; total variances
    remain nonnegative. With fixed inputs no population error is implied.
    """
    if estimator != config.estimator:
        raise ValueError("Fisher configuration and actual estimator must agree.")
    width = parameter_slices[-1][1]
    groups = tuple((name, np.asarray(index, dtype=int)) for name, index in groups)
    indices = np.concatenate([index for _, index in groups])
    if not np.array_equal(np.sort(indices), np.arange(width)):
        raise ValueError(
            "Fisher groups must form a disjoint exhaustive parameter partition."
        )
    coordinates = _Moments(width) if return_diagonal else None
    grouped = _Moments(len(groups))
    sampled = estimator != "exact"
    for mean, within, group_within in rows:
        mean = _array(mean, size=width, nonnegative=estimator != "negative_hessian")
        within = _array(within, size=width, nonnegative=True)
        group_within = _array(group_within, size=len(groups), nonnegative=True)
        if sampled != (within is not None and group_within is not None):
            raise ValueError(
                "Sampled Fisher estimators require coordinate and direct group variances."
            )
        if coordinates is not None:
            coordinates.update(mean, within)
        grouped.update(
            np.asarray([mean[index].sum() for _, index in groups]), group_within
        )
    if grouped.n != n_inputs or n_inputs < 1:
        raise ValueError("Fisher input count differs from streamed moments.")
    population_size = (
        config.population_size if population_size is None else population_size
    )
    g = grouped.summarize(config=config, population_size=population_size)
    summaries = tuple(
        FisherGroupSummary(
            name,
            len(index),
            g[0][i],
            g[0][i] / len(index),
            None if g[1] is None else g[1][i],
            None if g[2] is None else g[2][i],
            None if g[2] is None else np.sqrt(g[2][i]),
            None if g[3] is None else g[3][i],
            None if g[4] is None else g[4][i],
            None if g[4] is None else np.sqrt(g[4][i]),
        )
        for i, (name, index) in enumerate(groups)
    )
    c = (
        (None,) * 5
        if coordinates is None
        else coordinates.summarize(config=config, population_size=population_size)
    )
    return FisherInformationResult(
        estimator,
        c[0],
        c[1],
        c[2],
        None if c[2] is None else np.sqrt(c[2]),
        tuple(parameter_names),
        tuple(parameter_shapes),
        tuple(parameter_slices),
        summaries,
        width,
        n_inputs,
        config.n_draws if sampled else None,
        "model" if sampled else "analytic",
        config.input_sampling == "fixed",
        split,
        grouping=grouping or config.group_by,
        backend=backend,
        held_fixed=tuple(held_fixed),
        input_sampling=config.input_sampling,
        input_estimator_variance=c[3],
        total_estimator_variance=c[4],
        total_standard_error=None if c[4] is None else np.sqrt(c[4]),
    )
