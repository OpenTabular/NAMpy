"""
Parity testing tools for comparing GAM outputs to reference implementations.

Snapshots capture a fitted model's coefficients, smoothing parameters, EDF,
covariance matrices, and predictions in a serialisable dict.  They can be
saved to disk and compared against reference snapshots produced by R's mgcv
package (or any other reference implementation) using :func:`compare_parity_snapshots`.

Typical workflow
----------------
1. Run the R reference script to produce a JSON snapshot.
2. Fit the equivalent model in Python and call :func:`build_parity_snapshot`.
3. Compare with :func:`compare_parity_snapshots` and inspect tolerances.
"""

from .compare import (
    assert_parity_snapshot_close,
    compare_parity_snapshots,
)
from .snapshots import (
    build_parity_snapshot,
    load_parity_snapshot,
    save_parity_snapshot,
)
from .trace import (
    build_optimizer_trace,
    load_optimizer_trace,
    save_optimizer_trace,
)

__all__ = [
    "build_parity_snapshot",
    "build_optimizer_trace",
    "save_parity_snapshot",
    "save_optimizer_trace",
    "load_parity_snapshot",
    "load_optimizer_trace",
    "compare_parity_snapshots",
    "assert_parity_snapshot_close",
]
