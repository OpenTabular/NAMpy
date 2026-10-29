"""BCT and BCPE inference without fitting a statistical or neural model.

Run from the repository root:
    .venv/bin/python examples/example_boxcox_tail_probabilities.py
"""

import numpy as np
from numpy.testing import assert_allclose

from nampy.diagnostics import prepare_centiles
from nampy.distributions import BCPELaw, BCTLaw, DistributionPrediction


def main():
    x = np.linspace(0, 1, 5)
    parameters = {
        "mu": np.exp(x),
        "sigma": np.full(5, 0.4),
        "nu": np.linspace(-1, 1, 5),
    }
    for law, tau in [(BCTLaw(), 5.0), (BCPELaw(), 1.5)]:
        distribution = DistributionPrediction(
            law, {**parameters, "tau": np.full(5, tau)}
        )
        centiles = prepare_centiles(distribution, x)
        median = centiles.quantiles[:, 1]
        assert_allclose(distribution.cdf(median), 0.5, atol=2e-12)
        assert_allclose(
            distribution.quantile_residuals(median).residuals, 0, atol=2e-12
        )
        assert distribution.sample(10, random_state=7).shape == (10, 5)
        print(law.spec.name, ": x, mu, 5th centile, median, 95th centile")
        print(np.column_stack([x, parameters["mu"], centiles.quantiles]))
    print("mu is not generally the response mean or exact median.")
    print("Moments and both backends' fitting routes are unavailable for BCT and BCPE.")


if __name__ == "__main__":
    main()
