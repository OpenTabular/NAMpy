"""Mean likelihood information from exact, score and Hessian diagnostics.

Run from the repository root with ``python examples/example_fisher_diagnostics.py``.
These small fits demonstrate interfaces, not model quality or convergence.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from nampy import (
    FisherDiagnostics,
    GAMRegressor,
    GradientTrainingConfig,
    LinRegClassifier,
)


def main():
    x = np.linspace(-1, 1, 32)
    X = pd.DataFrame({"x": x})
    offset = 0.2 * x
    controls = GradientTrainingConfig(
        max_epochs=1,
        random_state=7,
        trainer_kwargs={
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
    configuration = FisherDiagnostics(n_inputs=8, n_draws=32, random_state=19)
    classifier = LinRegClassifier(
        training=controls, fisher_diagnostics=configuration
    ).fit(X, np.resize([0, 1], len(x)), offset=offset)
    neural = {
        method: classifier.fisher_information(
            X, offset=offset, config=replace(configuration, estimator=method)
        )
        for method in ("exact", "score_outer_product", "negative_hessian")
    }

    # Gaussian GAM scale is the known response variance, held fixed here.
    data = X.assign(y=0.3 + 0.7 * x + np.sin(np.arange(len(x))) * 0.1)
    statistical = GAMRegressor(
        formula="y ~ x",
        family="gaussian",
        scale=2.0,
        optimize_smoothing=False,
        fisher_diagnostics=configuration,
    ).fit(data, offset=offset)
    gam = statistical.fisher_information(data, offset=offset)
    assert gam.normalization == "mean" and gam.estimator == "exact"
    assert gam.n_draws is None and gam.estimator_variance is None
    for information in neural.values():
        assert information.normalization == "mean"
        assert np.isfinite(information.diagonal).all()
    result = {"neural": neural, "statistical": gam}
    print({name: float(info.diagonal.sum()) for name, info in neural.items()})
    print({"gam_mean_information_trace": float(gam.diagonal.sum())})
    return result


if __name__ == "__main__":
    main()
