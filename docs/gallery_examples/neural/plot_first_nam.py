"""
Fit a neural additive model
===========================

This example asks the same question as the first statistical GAM example:
**how do outdoor temperature and time of day relate to electricity demand?**
Using the same study makes the modeling approach the only new concept.

The synthetic variables are outdoor ``temperature`` in degrees Celsius,
``hour`` on a 24-hour clock, and electricity ``demand`` in megawatts. A NAM
learns one neural feature function for temperature and one for hour.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from nampy.models import NAMRegressor


# Generate the same energy-demand study
# -------------------------------------
rng = np.random.default_rng(21)
n_observations = 900
temperature = rng.uniform(-5.0, 35.0, n_observations)
hour = rng.uniform(0.0, 24.0, n_observations)
demand = (
    45.0
    + 0.12 * (temperature - 18.0) ** 2
    + 5.0 * np.sin(2.0 * np.pi * (hour - 7.0) / 24.0)
    + rng.normal(0.0, 2.0, n_observations)
)
X = pd.DataFrame({"temperature": temperature, "hour": hour})
X_train, X_test, y_train, y_test = train_test_split(
    X, demand, test_size=0.2, random_state=21
)


# Train feature-wise neural networks
# ----------------------------------
# Early stopping limits unnecessary training after validation performance stops
# improving. A temporary checkpoint keeps this downloadable example from
# leaving model files in the working directory.
model = NAMRegressor(
    layer_sizes=[48, 24],
    dropout=0.0,
    numerical_preprocessing="minmax",
)
with tempfile.TemporaryDirectory(prefix="nampy-first-nam-") as directory:
    model.fit(
        X_train,
        y_train,
        max_epochs=120,
        patience=15,
        batch_size=128,
        lr=1e-2,
        random_state=21,
        checkpoint_path=Path(directory),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


# Evaluate held-out demand
# ------------------------
prediction = model.predict(X_test)
rmse = mean_squared_error(y_test, prediction) ** 0.5
r2 = r2_score(y_test, prediction)
print(f"Held-out RMSE: {rmse:.2f} MW")
print(f"Held-out R²: {r2:.3f}")


# Inspect learned feature effects
# -------------------------------
# ``reference_X=X_train`` centers contributions against the empirical training
# distribution. The reddish background shows where training observations are
# concentrated; it is a density display, not an uncertainty interval.
figures = model.plot_terms(
    X_test,
    center=True,
    reference_X=X_train,
    density=True,
    density_color="#f3b4c2",
    pages=1,
    figsize=(10, 4),
)
figures[0].axes[0].set(
    xlabel="Outdoor temperature (°C)", ylabel="Centered contribution (MW)"
)
figures[0].axes[1].set(
    xlabel="Hour of day", ylabel="Centered contribution (MW)"
)
figures[0].subplots_adjust(top=0.84)
figures[0].suptitle("Neural feature effects for electricity demand", y=0.98)

components = model.predict_components(
    X_test, center=True, reference_X=X_train
)
components.validate_additive_reconstruction(atol=1e-5)

# The learned temperature contribution is U-shaped, while the hour
# contribution follows a daily pattern. Reconstruction verifies that the named
# contributions account for the linear predictor; it does not establish that
# individual effects would remain unchanged under correlated predictors.
