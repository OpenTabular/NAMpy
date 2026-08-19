"""Compare NAMpy vs mgcv outer-trace rows for the weighted negbin joint case.

Localizes the trace divergence seen in
negbin_est_reml_newton_joint_theta_weighted_cr before adjusting anything.
"""

import sys

sys.path.insert(0, "/home/ad32/projects/package/NAMpy")

import numpy as np

from nampy.gam import GAM
from nampy.gam.results.traces import build_optimizer_trace
from tests.mgcv_parity_utils import _make_negbin_data
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _run_mgcv_outer_trace,
)

data = _make_negbin_data()
w = np.asarray(data["w"], dtype=np.float64)

gam = GAM(
    family={"name": "negbin", "theta": 1.8, "estimate_theta": True},
    formula='y ~ s(x0, bs="cr", k=8)',
    optimize_smoothing=True,
    smoothing_method="REML",
    smoothing_optimizer="outer_newton",
)
gam.fit(data=data, sample_weight=w)
trace = build_optimizer_trace(gam)

expected = _run_mgcv_outer_trace(
    data=data,
    formula='y ~ s(x0, bs="cr", k=8)',
    family="negbin_est:1.8",
    method="REML",
    optimizer="newton",
    weights_column="w",
)

print("== NAMpy rows ==")
for row in trace["rows"]:
    print(
        f"iter={row.get('iteration')} lsp={row.get('lsp')} "
        f"score={row.get('score')} grad={row.get('grad')}"
    )
print("== mgcv rows ==")
for row in expected["rows"]:
    print(
        f"iter={row.get('iteration')} lsp={row.get('lsp')} "
        f"score={row.get('score')} grad={row.get('grad')}"
    )
