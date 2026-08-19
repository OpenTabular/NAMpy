#!/usr/bin/env python3
"""GAMResidualRegressor: frozen mgcv-parity baseline + neural correction.

The GAM stage sees only s(x0); the 2*x3 signal is invisible to it. The
neural correction is trained with the GAM link prediction as a fixed
offset, so it learns exactly what the baseline missed. This composite is
NOT an mgcv model.

Run:
    python examples/example_gam_residual.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from nampy.gam import GAM
from nampy.hybrid import GAMResidualRegressor
from nampy.models import LinRegRegressor


def main():
    rng = np.random.default_rng(0)
    n = 400
    data = pd.DataFrame(
        {"x0": rng.uniform(size=n), "x3": rng.normal(size=n)}
    )
    data["y"] = (
        np.sin(3.0 * data["x0"])
        + 2.0 * data["x3"]
        + rng.normal(scale=0.1, size=n)
    )

    hybrid = GAMResidualRegressor(
        "y ~ s(x0, k=8)",
        LinRegRegressor(numerical_preprocessing="standardization"),
        family="gaussian",
    )
    hybrid.fit(
        data,
        neural_features=["x3"],
        neural_fit_kwargs={
            "max_epochs": 60,
            "patience": 60,
            "lr": 5e-2,
            "batch_size": 64,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
    )

    gam_alone = GAM(
        formula="y ~ s(x0, k=8)",
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="reml",
    )
    gam_alone.fit(data=data)

    r2_hybrid = hybrid.score(data, data["y"])
    r2_gam = r2_score(data["y"], gam_alone.predict(data))
    print(f"GAM alone R^2 : {r2_gam:.4f}")
    print(f"Hybrid R^2    : {r2_hybrid:.4f}")

    components = hybrid.predict_components(data)
    print(f"Backend       : {components.backend}")
    print(f"Term keys     : {sorted(components.terms)}")

    frozen = np.array_equal(
        hybrid.gam_.fit_result().coef_full, gam_alone.fit_result().coef_full
    )
    print(f"GAM stage identical to standalone mgcv-parity fit: {frozen}")


if __name__ == "__main__":
    main()
