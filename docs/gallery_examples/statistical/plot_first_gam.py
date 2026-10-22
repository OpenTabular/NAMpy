"""
Fit and diagnose a statistical GAM
==================================

This example asks: **how do outdoor temperature and time of day relate to
electricity demand?** The data are synthetic so that the underlying pattern
is known, but every variable has a concrete meaning and unit:

* ``temperature``: outdoor temperature in degrees Celsius;
* ``hour``: time of day on a 24-hour clock;
* ``demand``: electricity demand in megawatts (MW).

Cold and hot weather both increase demand, and use follows a daily cycle. We
fit on 80% of the observations, assess predictions on the remaining 20%, and
then inspect the estimated effects.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from nampy.models import GAMRegressor


# Generate the study data
# -----------------------
# The random seed makes the example reproducible. The equation is the complete
# data-generating process; no variables are hidden from the fitted model.
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
data = pd.DataFrame(
    {"demand": demand, "temperature": temperature, "hour": hour}
)
train, test = train_test_split(data, test_size=0.2, random_state=21)


# Fit smooth temperature and time effects
# ---------------------------------------
# A cubic regression spline can learn the U-shaped temperature effect. A
# cyclic spline is used for hour because the end of one day meets the start of
# the next. ``k`` supplies basis capacity; REML decides how much is used.
model = GAMRegressor(
    formula=(
        'demand ~ s(temperature, bs="cr", k=12) '
        '+ s(hour, bs="cc", k=12)'
    ),
    family="gaussian",
).fit(train)


# Evaluate held-out observations
# ------------------------------
prediction = model.predict(test)
rmse = mean_squared_error(test["demand"], prediction) ** 0.5
r2 = r2_score(test["demand"], prediction)
print(f"Held-out RMSE: {rmse:.2f} MW")
print(f"Held-out R²: {r2:.3f}")


# Interpret the fitted effects
# ----------------------------
# The vertical axes are centered term contributions on the additive-predictor
# scale, not demand itself. The shaded bands are pointwise uncertainty
# intervals. Together with the intercept, these terms reconstruct the fitted
# linear predictor.
rendered = model.plot(
    pages=1,
    rug=False,
    scheme=1,
    shade_col="#f3b4c2",
    scale=0,
    figsize=(10, 4),
)
figure = rendered["figures"][0]
figure.axes[0].set(xlabel="Outdoor temperature (°C)", ylabel="Contribution (MW)")
figure.axes[1].set(xlabel="Hour of day", ylabel="Contribution (MW)")
figure.subplots_adjust(top=0.84)
figure.suptitle("Estimated effects on electricity demand", y=0.98)

# The temperature curve is lowest near mild weather and rises toward both
# extremes. The hour curve is periodic and joins at the daily boundary.


# Check residual structure
# ------------------------
# A held-out score cannot tell us whether the smooths are well specified. A
# residual trend or changing spread would suggest revisiting the terms or the
# response distribution.
fitted = model.predict(train)
residual = model.gam_.residuals(type="deviance")
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.scatter(fitted, residual, s=15, color="#6b7280", alpha=0.5)
ax.set(xlabel="Fitted demand (MW)", ylabel="Deviance residual")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.16)

basis_check = model.gam_.k_check(n_rep=20, seed=21)
print(basis_check)

# Here the residual cloud has no strong fitted-value pattern and the basis
# check gives no evidence that either ``k`` is too restrictive. These checks
# support the specification; they do not prove that it is the only reasonable
# model.
