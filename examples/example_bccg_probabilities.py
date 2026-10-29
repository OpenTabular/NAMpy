"""Query BCCG probabilities without fitting either backend.

Run from the repository root with:
    .venv/bin/python examples/example_bccg_probabilities.py
"""

import numpy as np
from numpy.testing import assert_allclose

from nampy.diagnostics import prepare_centiles
from nampy.distributions import BCCGLaw, DistributionPrediction


def main():
    x = np.linspace(0, 1, 5)
    parameters = {
        "mu": np.exp(x),
        "sigma": np.full(5, 0.6),
        "nu": np.linspace(-1, 1, 5),
    }
    distribution = DistributionPrediction(BCCGLaw(), parameters)
    centiles = prepare_centiles(distribution, x)
    median = centiles.quantiles[:, 1]
    assert_allclose(distribution.cdf(median), 0.5, atol=2e-14)
    assert_allclose(distribution.quantile_residuals(median).residuals, 0, atol=2e-14)
    print("x, mu, 5th centile, median, 95th centile")
    print(np.column_stack([x, parameters["mu"], centiles.quantiles]))
    print("mu is not generally the exact median or response mean.")
    print("Conditional means:", distribution.mean())
    print("Conditional variances:", distribution.variance())
    print("Negative powers can give infinite means and undefined variances.")
    print(
        "Statistical bccg_lss supports fixed smoothing and automatic joint REML; neural fitting is unavailable."
    )


if __name__ == "__main__":
    main()
