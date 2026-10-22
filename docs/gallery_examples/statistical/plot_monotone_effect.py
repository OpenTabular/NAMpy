"""
Encode a monotone dose response
===============================

Suppose an agronomy experiment measures crop yield after applying different
amounts of fertilizer. Within the studied range, domain knowledge says the
expected yield may level off but should not decrease as the dose increases.

The synthetic variables are ``fertilizer_kg_ha`` (kilograms of fertilizer per
hectare) and ``yield_t_ha`` (metric tonnes of crop per hectare). We compare an
ordinary smooth with a monotone-increasing smooth.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nampy.gam import GAM


# Generate a saturating dose response
# -----------------------------------
rng = np.random.default_rng(9)
dose = np.sort(rng.uniform(0.0, 180.0, 180))
expected_yield = 2.5 + 4.0 / (1.0 + np.exp(-(dose - 80.0) / 24.0))
data = pd.DataFrame(
    {
        "yield_t_ha": expected_yield + rng.normal(0.0, 0.45, dose.size),
        "fertilizer_kg_ha": dose,
    }
)


# Fit the two assumptions
# -----------------------
ordinary = GAM(
    formula='yield_t_ha ~ s(fertilizer_kg_ha, bs="cr", k=12)',
    family="gaussian",
    optimize_smoothing=True,
    smoothing_method="reml",
).fit(data)
increasing = GAM(
    formula='yield_t_ha ~ s(fertilizer_kg_ha, bs="mpi", k=12)',
    family="gaussian",
    optimize_smoothing=True,
    smoothing_method="gcv",
    smoothing_optimizer="bfgs",
).fit(data)


# Compare the estimated response curves
# -------------------------------------
grid = pd.DataFrame({"fertilizer_kg_ha": np.linspace(0.0, 180.0, 240)})
ordinary_fit = ordinary.predict(grid, type="response")
constrained_fit, constrained_se = increasing.predict(
    grid, type="response", return_se=True
)

fig, ax = plt.subplots(figsize=(8.4, 4.6))
ax.scatter(
    data["fertilizer_kg_ha"], data["yield_t_ha"],
    s=13, color="#9ca3af", alpha=0.48, label="Observed plots"
)
ax.plot(grid["fertilizer_kg_ha"], ordinary_fit, color="#374151", lw=2,
        label="Unconstrained smooth")
ax.fill_between(
    grid["fertilizer_kg_ha"],
    constrained_fit - 2.0 * constrained_se,
    constrained_fit + 2.0 * constrained_se,
    color="#f3b4c2", alpha=0.72, linewidth=0,
)
ax.plot(grid["fertilizer_kg_ha"], constrained_fit, color="#a61e4d", lw=2.4,
        label="Monotone increasing smooth")
ax.set(xlabel="Fertilizer dose (kg/ha)", ylabel="Expected crop yield (t/ha)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)

# The ordinary smooth can follow local noise and turn downward. The constrained
# fit cannot. This is appropriate because monotonicity was supplied by the
# scientific question—not selected after looking at which curve appeared nicer.
