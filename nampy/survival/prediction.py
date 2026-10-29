"""Shared survival predictions and exact piecewise hazard integration.

References
----------
- Royston, P., and Parmar, M. K. B. (2013). Restricted mean survival time: an
  alternative to the hazard ratio for the design and analysis of randomized trials with
  a time-to-event outcome.
  https://doi.org/10.1186/1471-2288-13-152
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from ..distributions.prediction import DistributionPrediction
from .contracts import _readonly, validate_cuts
from .data import _cuts
from .likelihood import log_entry_ratio


def survival_from_hazard(hazard, cuts, *, times=None):
    """Integrate piecewise constant hazards into survival probabilities.

    ``hazard`` is finite nonnegative float (n,k), one hazard per subject and
    interval; ``cuts`` is float (k+1,) boundaries. Optional ``times`` is a
    finite sorted float (m,) vector within the cut range, default cuts[1:].
    Returns dict with ``time`` (m,), ``cumulative_hazard`` (n,m) and
    ``survival`` (n,m), the latter exp(-cumulative_hazard). Values condition
    on survival to cuts[0]. Supports partial intervals and time zero.
    """
    grid = _cuts(cuts)
    h = np.asarray(hazard, dtype=float)
    if (
        h.ndim != 2
        or h.shape[1] != grid.size - 1
        or not np.all(np.isfinite(h))
        or np.any(h < 0)
    ):
        raise ValueError("hazard must be finite nonnegative (n,k).")
    t = grid[1:] if times is None else np.asarray(times, dtype=float)
    if (
        t.ndim != 1
        or not np.all(np.isfinite(t))
        or np.any(np.diff(t) < 0)
        or np.any((t < grid[0]) | (t > grid[-1]))
    ):
        raise ValueError("times must be sorted, finite and within cuts.")
    durations = np.clip(t[:, None] - grid[:-1], 0, np.diff(grid))
    cumulative = h @ durations.T
    return {
        "time": t.copy(),
        "cumulative_hazard": cumulative,
        "survival": np.exp(-cumulative),
    }


@dataclass(frozen=True)
class SurvivalPrediction:
    """Snapshot of interval hazards or an existing scalar event-time law.

    Curve operations return ``(n_subjects, n_times)``. Piecewise predictions
    have finite support: there is no implied tail beyond the final cut.
    These are conditional event-time distributions, not confidence bands.
    """

    hazard_rates: np.ndarray | None = None
    cuts: np.ndarray | None = None
    distribution: DistributionPrediction | None = None
    backend: str = "shared"
    cumulative_hazard_steps: np.ndarray | None = None

    def __post_init__(self):
        if self.cumulative_hazard_steps is not None:
            if self.distribution is not None or self.hazard_rates is not None:
                raise ValueError("Choose one survival representation: rates, law or cumulative-hazard steps.")
            cuts = validate_cuts(self.cuts)
            values = _readonly(self.cumulative_hazard_steps, dtype=float)
            if (values.ndim != 2 or values.shape[1] != len(cuts) or not np.isfinite(values).all()
                    or np.any(values[:, 0] != 0) or np.any(np.diff(values, axis=1) < 0)):
                raise ValueError("Step cumulative hazard must be finite (n,len(cuts)), monotone and zero at origin.")
            object.__setattr__(self, "cuts", cuts)
            object.__setattr__(self, "cumulative_hazard_steps", values)
            return
        if self.distribution is not None:
            if not isinstance(self.distribution, DistributionPrediction):
                raise TypeError("distribution must be DistributionPrediction.")
            if self.hazard_rates is not None or self.cuts is not None:
                raise ValueError(
                    "Choose interval hazards or a scalar event-time distribution."
                )
            if self.distribution.spec.name not in {"weibull", "lognormal"}:
                raise NotImplementedError(
                    "Survival views currently support Weibull/lognormal laws."
                )
            return
        cuts = validate_cuts(self.cuts)
        rates = _readonly(self.hazard_rates, dtype=float)
        if rates.ndim != 2 or rates.shape[1] != len(cuts) - 1:
            raise ValueError("hazard_rates must have shape (n_subjects, n_intervals).")
        if not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError("hazard_rates must be finite and nonnegative.")
        object.__setattr__(self, "hazard_rates", rates)
        object.__setattr__(self, "cuts", cuts)

    @property
    def n_subjects(self):
        if self.cumulative_hazard_steps is not None:
            return len(self.cumulative_hazard_steps)
        if self.distribution is not None:
            return len(next(iter(self.distribution.parameters.values())))
        assert self.hazard_rates is not None
        return len(self.hazard_rates)

    def take(self, indices):
        """Copy selected subject rows in the requested order, retaining law/support."""
        rows = np.atleast_1d(np.arange(self.n_subjects)[indices])
        if rows.ndim != 1:
            raise ValueError("Subject indices must select a one-dimensional row sequence.")
        if self.cumulative_hazard_steps is not None:
            return SurvivalPrediction(cuts=self.cuts, backend=self.backend,
                cumulative_hazard_steps=self.cumulative_hazard_steps[rows])
        if self.distribution is None:
            assert self.hazard_rates is not None
            return SurvivalPrediction(self.hazard_rates[rows], self.cuts, backend=self.backend)
        distribution = DistributionPrediction(self.distribution.law,
            {name: values[rows] for name, values in self.distribution.parameters.items()},
            context={name: values[rows] for name, values in self.distribution.context.items()})
        return SurvivalPrediction(distribution=distribution, backend=self.backend)

    @property
    def support(self):
        if self.distribution is not None:
            return (0.0, np.inf)
        assert self.cuts is not None
        return (0.0, float(self.cuts[-1]))

    def _times(self, times):
        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or not len(times)
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or np.any(times < 0)
        ):
            raise ValueError(
                "times must be nonnegative, finite and strictly increasing."
            )
        if times[-1] > self.support[1]:
            raise NotImplementedError(
                "Piecewise tail extrapolation beyond cuts is unsupported."
            )
        return times

    def _log_survival(self, times):
        if self.cumulative_hazard_steps is not None:
            return -self.cumulative_hazard_steps[:, np.searchsorted(self.cuts, times, side="right")-1]
        if self.distribution is None:
            assert self.hazard_rates is not None and self.cuts is not None
            return -survival_from_hazard(self.hazard_rates, self.cuts, times=times)[
                "cumulative_hazard"
            ]
        columns = []
        for time in times:
            columns.append(
                np.zeros(self.n_subjects)
                if time == 0
                else self.distribution.log_survival(time)
            )
        return np.column_stack(columns)

    def log_survival(self, times, *, conditional_on=None):
        times = self._times(times)
        result = self._log_survival(times)
        if conditional_on is not None:
            landmark = np.broadcast_to(
                np.asarray(conditional_on, dtype=float), (self.n_subjects,)
            )
            if not np.isfinite(landmark).all() or np.any(landmark < 0):
                raise ValueError("conditional_on must be finite and nonnegative.")
            if np.any(landmark > self.support[1]):
                raise NotImplementedError("Landmark lies beyond supported cuts.")
            if self.cumulative_hazard_steps is not None:
                denominator = -self.cumulative_hazard_steps[np.arange(self.n_subjects),
                    np.searchsorted(self.cuts, landmark, side="right")-1]
            elif self.distribution is None:
                assert self.hazard_rates is not None and self.cuts is not None
                duration = np.clip(
                    landmark[:, None] - self.cuts[:-1], 0, np.diff(self.cuts)
                )
                denominator = -np.sum(self.hazard_rates * duration, axis=1)
            else:
                denominator = np.zeros(self.n_subjects)
                positive = landmark > 0
                if positive.any():
                    values = self.distribution.log_survival(
                        np.where(positive, landmark, 1)
                    )
                    denominator[positive] = values[positive]
            result = result - denominator[:, None]
            result[times[None, :] < landmark[:, None]] = np.nan
        return result

    def survival(self, times, *, conditional_on=None):
        return np.exp(self.log_survival(times, conditional_on=conditional_on))

    def cumulative_hazard(self, times, *, conditional_on=None):
        return -self.log_survival(times, conditional_on=conditional_on)

    def hazard(self, times):
        if self.cumulative_hazard_steps is not None:
            raise NotImplementedError("Step cumulative hazards do not define a continuous hazard density.")
        times = self._times(times)
        if self.distribution is None:
            assert self.hazard_rates is not None and self.cuts is not None
            interval = np.maximum(0, np.searchsorted(self.cuts, times, side="left") - 1)
            return self.hazard_rates[:, interval].copy()
        if np.any(times == 0):
            raise NotImplementedError(
                "Parametric hazard limits at zero are not exposed."
            )
        return np.column_stack(
            [
                np.exp(
                    self.distribution.log_prob(t) - self.distribution.log_survival(t)
                )
                for t in times
            ]
        )

    def quantile(self, probabilities):
        """Event-time quantiles; NaN means not reached within piecewise support."""
        p = np.atleast_1d(np.asarray(probabilities, dtype=float))
        if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p >= 1)):
            raise ValueError("probabilities must be finite and lie in [0, 1).")
        if self.cumulative_hazard_steps is not None:
            assert self.cuts is not None
            answer = np.full((self.n_subjects, len(p)), np.nan)
            for i, values in enumerate(self.cumulative_hazard_steps):
                indices = np.asarray(np.searchsorted(values, -np.log1p(-p), side="left"))
                valid = indices < len(values)
                answer[i, valid] = self.cuts[indices[valid]]
            return answer
        if self.distribution is not None:
            return self.distribution.quantile(p)
        assert self.hazard_rates is not None and self.cuts is not None
        cumulative = np.cumsum(self.hazard_rates * np.diff(self.cuts), axis=1)
        starts = np.column_stack([np.zeros(self.n_subjects), cumulative[:, :-1]])
        answer = np.full((self.n_subjects, len(p)), np.nan)
        for j, threshold in enumerate(-np.log1p(-p)):
            for i in range(self.n_subjects):
                if threshold == 0:
                    answer[i, j] = 0
                    continue
                found = np.flatnonzero(
                    (cumulative[i] >= threshold) & (self.hazard_rates[i] > 0)
                )
                if len(found):
                    k = found[0]
                    answer[i, j] = (
                        self.cuts[k]
                        + (threshold - starts[i, k]) / self.hazard_rates[i, k]
                    )
        return answer

    def median(self):
        return self.quantile(0.5)[:, 0]

    def cumulative_hazard_at(self, time, *, conditional_on=None):
        """Rowwise H(time)-H(landmark), without constructing a time-grid matrix.

        Times and optional landmarks are scalar or one per prediction row.
        Weibull and interval increments avoid subtracting close hazards.
        """
        time = np.broadcast_to(np.asarray(time, dtype=float), (self.n_subjects,))
        entry = self._landmarks(conditional_on, time)
        if not np.isfinite(time).all() or np.any(time < 0):
            raise ValueError("time must be finite and nonnegative.")
        if np.any(time > self.support[1]):
            raise NotImplementedError("Piecewise tail extrapolation beyond cuts is unsupported.")
        if self.cumulative_hazard_steps is not None:
            values = self.cumulative_hazard_steps
            rows = np.arange(self.n_subjects)
            return (values[rows, np.searchsorted(self.cuts, time, side="right")-1]
                    - values[rows, np.searchsorted(self.cuts, entry, side="right")-1])
        if self.distribution is None:
            assert self.cuts is not None and self.hazard_rates is not None
            duration = np.maximum(
                np.minimum(time[:, None], self.cuts[1:])
                - np.maximum(entry[:, None], self.cuts[:-1]), 0,
            )
            return np.sum(self.hazard_rates * duration, axis=1)
        parameters = self.distribution.parameters
        if self.distribution.spec.name == "weibull":
            shape, scale = parameters["shape"], parameters["scale"]
            positive = time > entry
            result = np.zeros(self.n_subjects)
            fraction = np.ones(self.n_subjects)
            entered = entry > 0
            fraction[entered] = -np.expm1(
                shape[entered] * log_entry_ratio(time[entered], entry[entered])
            )
            result[positive] = np.exp(
                shape[positive] * (np.log(time[positive]) - np.log(scale[positive]))
            ) * fraction[positive]
            return result
        # Substitute a valid positive time only on the unused zero-time rows.
        value = -self.distribution.log_survival(np.where(time > 0, time, 1))
        value[time == 0] = 0
        entered = entry > 0
        if entered.any():
            denominator = self.distribution.log_survival(np.where(entered, entry, 1))
            value[entered] += denominator[entered]
        return value

    def _landmarks(self, conditional_on, horizon):
        entry = np.broadcast_to(
            np.asarray(0 if conditional_on is None else conditional_on, dtype=float),
            (self.n_subjects,),
        )
        if not np.isfinite(entry).all() or np.any(entry < 0) or np.any(entry > horizon):
            raise ValueError("Landmarks must be finite and satisfy 0 <= landmark <= horizon.")
        return entry

    def rmst(self, horizon, *, conditional_on=None):
        """Integral of S(t)/S(a) from landmark a (default zero) to horizon.

        Interval integration is exact. Parametric integration uses adaptive
        quadrature after t=a+expm1(v), with absolute tolerance 1e-10 in time
        units and relative tolerance 1e-9. The median is a quadrature break
        point when inside the interval. Failures raise; tails follow the law.
        This is remaining lifetime after a, not attained age or a confidence band.
        """
        horizon = float(horizon)
        self._times([horizon])
        entry = self._landmarks(conditional_on, horizon)
        if self.cumulative_hazard_steps is not None:
            assert self.cuts is not None
            duration = np.maximum(np.minimum(horizon, self.cuts[1:])
                - np.maximum(entry[:, None], self.cuts[:-1]), 0)
            denominator = self.cumulative_hazard_steps[np.arange(self.n_subjects),
                np.searchsorted(self.cuts, entry, side="right")-1]
            log_ratio = np.where(duration > 0, -self.cumulative_hazard_steps[:, :-1]+denominator[:, None], 0)
            return np.sum(duration*np.exp(log_ratio), axis=1)
        if self.distribution is not None:
            answer = np.zeros(self.n_subjects)
            for i in range(self.n_subjects):
                width = horizon - entry[i]
                if width == 0:
                    continue
                row = SurvivalPrediction(distribution=DistributionPrediction(
                    self.distribution.law,
                    {name: values[i:i+1] for name, values in self.distribution.parameters.items()},
                ))

                def integrand(v, row=row, landmark=entry[i]):
                    increment = row.cumulative_hazard_at(
                        landmark + np.expm1(v), conditional_on=landmark
                    )[0]
                    if np.isnan(increment) or increment < 0:
                        raise FloatingPointError("Nonfinite or negative cumulative-hazard increment.")
                    return np.exp(v-increment)

                with np.errstate(over="ignore"):
                    median = row.median()[0]
                points = ([np.log1p(median-entry[i])]
                          if entry[i] < median < horizon else None)
                integral = quad(integrand, 0, np.log1p(width), points=points,
                                epsabs=1e-10, epsrel=1e-9, limit=200, full_output=1)
                if len(integral) != 3 or not np.isfinite(integral[0]):
                    raise FloatingPointError("Parametric RMST quadrature did not converge.")
                answer[i] = integral[0]
            return answer
        assert self.hazard_rates is not None and self.cuts is not None
        if conditional_on is None:
            duration = np.clip(horizon - self.cuts[:-1], 0, np.diff(self.cuts))
        else:
            duration = np.maximum(
                np.minimum(horizon, self.cuts[1:])
                - np.maximum(entry[:, None], self.cuts[:-1]), 0,
            )
        cumulative = np.cumsum(self.hazard_rates * duration, axis=1)
        starts = np.column_stack([np.zeros(self.n_subjects), cumulative[:, :-1]])
        integral = np.broadcast_to(duration, self.hazard_rates.shape).copy()
        positive = self.hazard_rates > 0
        integral[positive] = (
            -np.expm1(
                -self.hazard_rates[positive]
                * np.broadcast_to(duration, integral.shape)[positive]
            )
            / self.hazard_rates[positive]
        )
        return np.sum(np.exp(-starts) * integral, axis=1)
