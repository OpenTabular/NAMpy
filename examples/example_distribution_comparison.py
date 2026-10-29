"""Held-out family comparison; add --neural to include a short neural fit.

Run from the repository root with the installed NAMpy environment. The short
neural budget demonstrates completion semantics, not adequate convergence.
The comparison holdout is never used for training or early stopping.
"""

import argparse
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from nampy import GAMLSS
from nampy.distributions import DistributionCandidate, compare_distributions
from nampy.neural.configuration import (
    GradientTrainingConfig,
    ObjectiveConfig,
    PreprocessingConfig,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neural", action="store_true", help="Include a short neural lognormal fit."
    )
    args = parser.parse_args()
    rng = np.random.default_rng(112)
    X = pd.DataFrame({"x": rng.uniform(-1, 1, 120)})
    V = pd.DataFrame({"x": rng.uniform(-1, 1, 36)})
    y = rng.lognormal(0.3 + 0.4 * X.x, 0.4)
    z = rng.lognormal(0.3 + 0.4 * V.x, 0.4)
    candidates = {
        "lognormal": GAMLSS(
            family="lognormal_lss",
            formula={"loc": "y ~ x", "scale": "~1"},
            optimize_smoothing=False,
        ),
        "weibull": GAMLSS(
            family="weibull_lss",
            formula={"scale": "y ~ x", "shape": "~1"},
            optimize_smoothing=False,
        ),
        # Deliberately incompatible support: the failed fit remains in the table.
        "invalid_beta": GAMLSS(
            family="beta_lss",
            formula={"mu": "y ~ x", "phi": "~1"},
            optimize_smoothing=False,
        ),
    }
    with TemporaryDirectory(prefix="nampy-family-comparison-") as checkpoints:
        if args.neural:
            from nampy.models import LinRegLSS

            candidates["neural_lognormal"] = DistributionCandidate(LinRegLSS(
                objective=ObjectiveConfig(family="lognormal"),
                preprocessing=PreprocessingConfig(numerical_preprocessing="standardization"),
                training=GradientTrainingConfig(max_epochs=3, random_state=7, patience=2,
                    checkpoint_path=checkpoints, trainer_kwargs={"logger": False,
                    "enable_progress_bar": False, "enable_model_summary": False,
                    "num_sanity_val_steps": 0})))
        result = compare_distributions(
            candidates,
            X,
            y,
            V,
            z,
            sample_weight=np.tile([1.0, 2.0, 1.0], 40),
            validation_weight=np.tile([0.0, 1.0, 2.0], 12),
            response_units="original positive response units",
        )
    rows = {row.name: row for row in result.results}
    assert rows["invalid_beta"].status == "fit_error"
    assert rows["invalid_beta"].rank is None
    assert rows["lognormal"].eligible and rows["weibull"].eligible
    if args.neural:
        assert rows["neural_lognormal"].fit_status.state == "completed"
    assert all(row.n_validation == 24 for row in result.results)
    columns = [
        "name",
        "rank",
        "status",
        "backend",
        "fit_state",
        "mean_log_prob",
        "n_scored",
    ]
    print(result.to_frame()[columns].to_string(index=False))
    print("Calibration is unweighted on positive-weight validation rows.")
    print("Use a separate test set after selection; this table is a selection result.")
    return result


if __name__ == "__main__":
    main()
