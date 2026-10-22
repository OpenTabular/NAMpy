"""
Model changing travel-time variability
======================================

Longer commutes tend to take more time, but their day-to-day variability can
also increase. A mean-only model cannot describe that second pattern.

This synthetic study records ``distance_km`` (commute distance in kilometres),
``peak_hour`` (0 for off-peak trips and 1 for peak-hour trips), and
``travel_time_min`` (travel time in minutes). A distributional neural additive
model learns separate predictors for conditional location and scale.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from nampy.models import NAMLSS


# Generate trips whose spread grows with distance
# -----------------------------------------------
rng = np.random.default_rng(29)
n_trips = 800
distance_km = rng.uniform(1.0, 45.0, n_trips)
peak_hour = rng.integers(0, 2, n_trips).astype(float)
location = 8.0 + 1.25 * distance_km + 9.0 * peak_hour
standard_deviation = 2.0 + 0.18 * distance_km + 2.0 * peak_hour
travel_time_min = location + standard_deviation * rng.normal(size=n_trips)
X = pd.DataFrame({"distance_km": distance_km, "peak_hour": peak_hour})


# Fit location and scale together
# -------------------------------
model = NAMLSS(
    family="normal",
    layer_sizes=[24, 12],
    dropout=0.0,
    numerical_preprocessing="minmax",
)
with tempfile.TemporaryDirectory(prefix="nampy-travel-time-") as directory:
    model.fit(
        X,
        travel_time_min,
        max_epochs=40,
        patience=8,
        batch_size=128,
        lr=1e-2,
        random_state=29,
        checkpoint_path=Path(directory),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


# Plot the distance effect and conditional spread
# -----------------------------------------------
figures = model.plot_distributional_effects(
    X,
    reference_X=X,
    feature_names=["distance_km"],
    point_label="conditional location effect",
    envelope_label="location effect ± conditional SD",
    ylabel="Travel-time effect and spread (minutes)",
    title="Expected travel time and variability increase with distance",
    ncols=1,
    figsize=(7.4, 4.4),
)
figures[0].axes[0].set_xlabel("Commute distance (km)")

# The central curve is the change in the family-defined conditional point
# prediction relative to the empirical reference. The outer curves add and
# subtract one predicted conditional standard deviation. They describe trip
# variability, not uncertainty about the fitted mean.
