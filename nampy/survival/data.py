"""Single-event piecewise exponential data and prediction utilities."""

import numpy as np
import pandas as pd


def validate_covariates(X):
    """Reject missing/nonfinite features consistently across survival engines."""
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if frame.isna().any().any():
        raise ValueError("Missing survival covariates are not supported.")
    if not np.isfinite(frame.select_dtypes(include=np.number).to_numpy()).all():
        raise ValueError("Survival covariates must be finite.")


def piecewise_exponential_data(data, *, time, event, cuts, entry=None, id=None):
    """Split single-event right-censored follow-up into Poisson exposure rows.

    Parameters
    ----------
    data : pandas.DataFrame, shape (n,d)
        One row per subject; fixed covariates are copied to every interval.
    time : str
        Finite positive exit-time column, measured in the same units as cuts.
    event : str
        Binary column: 1 observed event, 0 right censoring.
    cuts : array-like of float, shape (k+1,)
        Strictly increasing finite interval boundaries, at least two; must
        cover every subject's entry and exit. Intervals are (left,right].
    entry : str or None, default None
        Optional delayed-entry time column; None uses zero. Each exit must
        exceed entry. Events exactly on a cut belong to the preceding interval.
    id : str or None, default None
        Unique nonmissing subject identifier column. None uses row positions.

    Returns
    -------
    pandas.DataFrame, shape (n_expanded, d+7)
        Original covariates plus ``ped_id`` (identifier), ``tstart`` (float),
        ``tend`` (float interval right boundary, also for censored rows),
        ``exposure`` (float time at risk), ``offset`` (float log exposure),
        ``ped_event`` (int 0/1), and ``interval`` (int zero-based cut index).
        Exposure sums to exit-entry and events sum to original event count.
        Partial final intervals retain the grid endpoint in tend so their
        piecewise constant hazard is evaluated consistently at prediction.
    """
    if not isinstance(data, pd.DataFrame) or not len(data):
        raise ValueError("data must be a nonempty DataFrame.")
    reserved = {
        "ped_id",
        "tstart",
        "tend",
        "exposure",
        "offset",
        "ped_event",
        "interval",
    }
    if reserved.intersection(data.columns):
        raise ValueError(
            "Input contains reserved piecewise-exponential output columns."
        )
    grid = _cuts(cuts)
    stop = np.asarray(data[time], dtype=float)
    start = (
        np.zeros(len(data)) if entry is None else np.asarray(data[entry], dtype=float)
    )
    status = np.asarray(data[event])
    if (
        not np.all(np.isfinite(stop))
        or not np.all(np.isfinite(start))
        or np.any(stop <= start)
    ):
        raise ValueError("Finite exit times must exceed entry times.")
    if not np.all(np.isin(status, [0, 1])):
        raise ValueError("event must contain only 0 and 1.")
    if start.min() < grid[0] or stop.max() > grid[-1]:
        raise ValueError("cuts must cover all entry and exit times.")
    ids = np.arange(len(data)) if id is None else data[id].to_numpy()
    if pd.isna(ids).any() or pd.Series(ids).duplicated().any():
        raise ValueError(
            "Subject IDs must be unique and nonmissing; recurrent events are unsupported."
        )
    rows = []
    for i in range(len(data)):
        for j, (left, right) in enumerate(zip(grid[:-1], grid[1:], strict=True)):
            a, b = max(left, start[i]), min(right, stop[i])
            if b <= a:
                continue
            row = data.iloc[i].to_dict()
            row.update(
                ped_id=ids[i],
                tstart=a,
                tend=right,
                exposure=b - a,
                offset=np.log(b - a),
                ped_event=int(status[i] == 1 and b == stop[i]),
                interval=j,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _cuts(cuts):
    grid = np.asarray(cuts, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("cuts must be finite, strictly increasing boundaries (k+1,).")
    return grid
