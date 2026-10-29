"""Distribution-based diagnostics and centiles, independent of fitting engines.

Quantile residuals follow GAMLSS's continuous/jump CDF construction. Worm
plots use detrended normal QQ points and pointwise reference envelopes, as in
gamlss/R/wp.R. These are separate from mgcv residual dispatch and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.stats import norm

from ..distributions.prediction import DistributionPrediction


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _vector(value, n, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},).")
    return result


def _check_discrete_observations(distribution, y):
    # Threshold CDFs accept noninteger arguments, whereas observed discrete
    # responses must have nonzero mass. Use the declared probability operation
    # when available; a CDF-only external law remains usable on valid responses.
    if (
        distribution.spec.response_kind == "discrete"
        and "log_prob" in distribution.spec.operations
    ):
        mass = distribution.log_prob(y)
        if np.any(np.isneginf(mass) & np.isfinite(y)):
            raise ValueError(
                "A discrete observation has zero probability (or its log mass underflowed); check response support."
            )


@dataclass(frozen=True)
class QuantileResiduals:
    """Rowwise normalized residuals, PIT values, CDF bounds and policy flags.

    All arrays have shape (n,). ``randomized`` identifies probability jumps;
    ``clipped`` records where optional clipping changed the PIT. Missing rows
    remain NaN and neither flag is set there. Clipping never changes CDF bounds.
    """

    residuals: np.ndarray
    pit: np.ndarray
    cdf_lower: np.ndarray
    cdf_upper: np.ndarray
    randomized: np.ndarray
    clipped: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _readonly(getattr(self, name)))


def quantile_residuals(
    distribution: DistributionPrediction,
    y,
    *,
    random_state=None,
    uniforms=None,
    clip=None,
) -> QuantileResiduals:
    """Compute Phi^-1(U) using F(y-) <= U <= F(y).

    Parameters
    ----------
    distribution : DistributionPrediction
        Conditional laws evaluated on the same rows as the observed responses.
    y : array-like, shape (n,)
        Observed responses, allowing NaN for missing rows. Responses must be
        within their law's observation support; no response clipping is done.
    random_state : int or numpy.random.Generator, optional
        RNG for jump rows only. Continuous rows do not consume random numbers.
    uniforms : array-like, shape (n,), optional
        Explicit finite values in [0, 1], useful for cross-language checks.
        Mutually exclusive with random_state; non-jump rows ignore these values.
    clip : float, optional
        Explicit PIT clipping to [clip, 1-clip], with 0 < clip < 0.5. Default
        preserves infinite residuals caused by zero/one CDFs or rounded tails.

    Returns
    -------
    QuantileResiduals
        Residuals plus the probabilities and flags needed to audit them. These
        are conditional diagnostics, not a claim of independent normal fitted
        residuals. Frequency expansion and weighting are not performed.
    """
    y = _vector(y, distribution.n_rows, "y")
    if np.any(np.isinf(y)):
        raise ValueError("Observed responses must be finite or NaN.")
    _check_discrete_observations(distribution, y)
    if uniforms is not None and random_state is not None:
        raise ValueError("Supply uniforms or random_state, not both.")
    epsilon = None
    if clip is not None:
        if not isinstance(clip, Real):
            raise ValueError("clip must be between zero and 0.5.")
        epsilon = float(clip)
        if not np.isfinite(epsilon) or not 0 < epsilon < 0.5 or 1 - epsilon == 1:
            raise ValueError(
                "clip must be representable away from zero and one, and between zero and 0.5."
            )
    lower, upper = distribution.cdf_left(y), distribution.cdf(y)
    lower[np.isnan(y)] = np.nan
    upper[np.isnan(y)] = np.nan
    missing = np.isnan(lower) & np.isnan(upper)
    if (
        np.any(~np.isfinite(lower[~missing]))
        or np.any(~np.isfinite(upper[~missing]))
        or np.any((lower < 0) | (upper > 1) | (lower > upper))
    ):
        raise ValueError(
            "Invalid CDF bounds: require 0 <= F(y-) <= F(y) <= 1, or two NaNs."
        )
    jumps = ~missing & (upper > lower)
    pit = upper.copy()
    if uniforms is not None:
        values = _vector(uniforms, distribution.n_rows, "uniforms")
        if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("uniforms must contain finite values in [0, 1].")
        values = values[jumps]
    else:
        values = np.random.default_rng(random_state).random(int(jumps.sum()))
    pit[jumps] = lower[jumps] + values * (upper[jumps] - lower[jumps])
    clipped = np.zeros(distribution.n_rows, dtype=bool)
    if epsilon is not None:
        adjusted = np.clip(pit, epsilon, 1 - epsilon)
        clipped = ~np.isnan(pit) & (adjusted != pit)
        pit = adjusted
    return QuantileResiduals(norm.ppf(pit), pit, lower, upper, jumps, clipped)


@dataclass(frozen=True)
class WormPanel:
    """Prepared QQ deviations for one group, with original-row identities.

    ``cubic_coefficients`` are ordered intercept, z, z**2, z**3, or None when
    fewer than four finite residuals are available (or fitting is disabled).
    Missing and infinite residuals are counted and excluded from QQ fitting.
    """

    label: str
    row_indices: np.ndarray
    theoretical: np.ndarray
    deviation: np.ndarray
    envelope_lower: np.ndarray
    envelope_upper: np.ndarray
    cubic_coefficients: np.ndarray | None
    n_total: int
    n_missing: int
    n_infinite: int


@dataclass(frozen=True)
class WormPlotData:
    """Worm panels and grouping metadata; envelopes are pointwise references.

    These envelopes assume independent standard normal observations with no
    parameter-estimation adjustment. They are neither simultaneous confidence
    bands nor fitted-model uncertainty envelopes.
    """

    panels: tuple[WormPanel, ...]
    reference_level: float
    n_ungrouped: int
    bin_edges: np.ndarray | None


def prepare_worm_plot(
    residuals,
    *,
    groups=None,
    x=None,
    n_bins=4,
    breaks=None,
    fit_cubic=True,
    reference_level=0.95,
) -> WormPlotData:
    """Prepare a detrended normal QQ plot, optionally grouped or binned.

    ``residuals`` is a vector or QuantileResiduals. Supply either scalar
    string/numeric ``groups`` or numeric ``x``. Numeric bins use quantile edges
    by default (duplicate edges collapse), or strictly increasing ``breaks``
    covering all finite x. Bins are [left,right), with the last right-closed.
    Missing group labels/nonfinite x are counted in ``n_ungrouped``. Explicit
    empty bins remain present. QQ positions follow R ppoints/qqnorm defaults.
    """
    values = np.asarray(
        residuals.residuals if isinstance(residuals, QuantileResiduals) else residuals,
        dtype=float,
    )
    if values.ndim != 1:
        raise ValueError("residuals must be a vector.")
    if not np.isfinite(reference_level) or not 0 < reference_level < 1:
        raise ValueError("reference_level must be between zero and one.")
    if groups is not None and x is not None:
        raise ValueError("Supply groups or x, not both.")
    if breaks is not None and x is None:
        raise ValueError("breaks requires x.")
    n = len(values)
    selections = []
    edges = None
    n_ungrouped = 0
    if x is not None:
        x = _vector(x, n, "x")
        valid = np.isfinite(x)
        n_ungrouped = int((~valid).sum())
        if breaks is None:
            if (
                isinstance(n_bins, bool)
                or not isinstance(n_bins, Integral)
                or n_bins < 1
            ):
                raise ValueError("n_bins must be a positive integer.")
            edges = (
                np.unique(np.quantile(x[valid], np.linspace(0, 1, int(n_bins) + 1)))
                if np.any(valid)
                else np.array([])
            )
        else:
            edges = np.asarray(breaks, dtype=float)
            if (
                edges.ndim != 1
                or len(edges) < 2
                or not np.all(np.isfinite(edges))
                or np.any(np.diff(edges) <= 0)
            ):
                raise ValueError(
                    "breaks must be finite, strictly increasing bin edges."
                )
            if np.any((x[valid] < edges[0]) | (x[valid] > edges[-1])):
                raise ValueError("breaks must cover every finite x value.")
        if len(edges) < 2:
            selections = [("all finite x", np.flatnonzero(valid))]
        else:
            membership = np.searchsorted(edges[1:-1], x, side="right")
            selections = [
                (
                    f"[{left:g}, {right:g}{']' if k == len(edges) - 2 else ')'}",
                    np.flatnonzero(valid & (membership == k)),
                )
                for k, (left, right) in enumerate(
                    zip(edges[:-1], edges[1:], strict=True)
                )
            ]
    elif groups is not None:
        labels = np.asarray(groups, dtype=object)
        if labels.shape != (n,):
            raise ValueError(f"groups must have shape ({n},).")
        buckets: dict[object, list[int]] = {}
        for i, label in enumerate(labels):
            if (
                label is None
                or isinstance(label, (float, np.floating))
                and np.isnan(label)
            ):
                n_ungrouped += 1
                continue
            if (
                not isinstance(label, (str, int, float, np.integer, np.floating))
                or isinstance(label, (float, np.floating))
                and not np.isfinite(label)
            ):
                raise ValueError(
                    "groups must contain strings, finite numeric labels, None or NaN."
                )
            buckets.setdefault(label, []).append(i)
        selections = [
            (str(label), np.asarray(indices, dtype=int))
            for label, indices in buckets.items()
        ]
    else:
        selections = [("all", np.arange(n))]
    panels = []
    for label, indices in selections:
        observed = values[indices]
        selected = indices[np.isfinite(observed)]
        order = np.argsort(values[selected], kind="stable")
        selected = selected[order]
        count = len(selected)
        a = 3 / 8 if count <= 10 else 0.5
        positions = (np.arange(1, count + 1) - a) / (count + 1 - 2 * a)
        theoretical = norm.ppf(positions)
        deviation = values[selected] - theoretical
        se = (
            np.sqrt(positions * (1 - positions) / count) / norm.pdf(theoretical)
            if count
            else np.array([])
        )
        width = norm.ppf((1 + reference_level) / 2) * se
        coefficients = (
            np.polynomial.polynomial.polyfit(theoretical, deviation, 3)
            if fit_cubic and count >= 4
            else None
        )
        panels.append(
            WormPanel(
                label,
                _readonly(selected),
                _readonly(theoretical),
                _readonly(deviation),
                _readonly(-width),
                _readonly(width),
                None if coefficients is None else _readonly(coefficients),
                len(indices),
                int(np.isnan(observed).sum()),
                int(np.isinf(observed).sum()),
            )
        )
    return WormPlotData(
        tuple(panels),
        reference_level,
        n_ungrouped,
        None if edges is None else _readonly(edges),
    )


@dataclass(frozen=True)
class CentileCurves:
    """Conditional quantiles on explicit x rows, stably sorted by x.

    ``row_indices`` records the original row order. Nonfinite x is excluded
    and counted; missing distribution rows remain NaN gaps in the curves.
    These are outcome quantiles, not confidence bands or marginal quantiles.
    """

    x: np.ndarray
    probabilities: np.ndarray
    quantiles: np.ndarray
    row_indices: np.ndarray
    n_excluded_x: int
    response_kind: str = "continuous"


def _centile_probabilities(probs):
    probabilities = np.atleast_1d(np.asarray(probs, dtype=float))
    if (
        probabilities.ndim != 1
        or not len(probabilities)
        or np.any(~np.isfinite(probabilities))
        or np.any((probabilities <= 0) | (probabilities >= 1))
        or np.any(np.diff(probabilities) <= 0)
    ):
        raise ValueError(
            "Centile probabilities must be a nonempty, strictly increasing vector inside (0, 1)."
        )
    return probabilities


def prepare_centiles(
    distribution: DistributionPrediction, x, *, probs=(0.05, 0.5, 0.95)
) -> CentileCurves:
    """Prepare conditional centiles on explicit prediction rows.

    Supply distribution predictions on a chosen covariate grid with other
    covariates fixed explicitly. No averaging, refitting, interpolation or
    covariate selection is performed here. Probabilities must be strictly
    increasing and inside (0,1), avoiding infinite endpoint curves.
    """
    x = _vector(x, distribution.n_rows, "x")
    probabilities = _centile_probabilities(probs)
    quantiles = distribution.quantile(probabilities)
    indices = np.flatnonzero(np.isfinite(x))
    indices = indices[np.argsort(x[indices], kind="stable")]
    return CentileCurves(
        _readonly(x[indices]),
        _readonly(probabilities),
        _readonly(quantiles[indices]),
        _readonly(indices),
        int((~np.isfinite(x)).sum()),
        distribution.spec.response_kind,
    )


@dataclass(frozen=True)
class CentileCalibration:
    """Observed coverage and attainable model probability for each centile.

    No independence assumptions or inferential intervals are attached. For
    discrete responses F(q_p) can exceed p; compare observed coverage to that
    attainable probability, not automatically to the nominal p.
    """

    probabilities: np.ndarray
    observed: np.ndarray
    model_probability: np.ndarray
    n_used: np.ndarray


def centile_calibration(
    distribution: DistributionPrediction, y, *, probs=(0.05, 0.5, 0.95)
) -> CentileCalibration:
    """Evaluate unweighted held-out coverage at conditional centiles.

    Rows must align with the observations. Missing/nonfinite observations or
    quantiles are excluded per probability and counts are returned. No training
    is performed, so use held-out or cross-fitted predictions for validation.
    """
    y = _vector(y, distribution.n_rows, "y")
    _check_discrete_observations(distribution, y)
    probabilities = _centile_probabilities(probs)
    quantiles = distribution.quantile(probabilities)
    observed, expected, counts = [], [], []
    for j in range(len(probabilities)):
        attainable = distribution.cdf(quantiles[:, j])
        valid = np.isfinite(y) & np.isfinite(quantiles[:, j]) & np.isfinite(attainable)
        counts.append(int(valid.sum()))
        observed.append(
            float(np.mean(y[valid] <= quantiles[valid, j])) if np.any(valid) else np.nan
        )
        expected.append(float(np.mean(attainable[valid])) if np.any(valid) else np.nan)
    return CentileCalibration(
        _readonly(probabilities),
        _readonly(observed),
        _readonly(expected),
        _readonly(counts),
    )
