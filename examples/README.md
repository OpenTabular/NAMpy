# Standalone examples

These are terminal-runnable verification scripts. They intentionally go beyond
the short recipes in `docs/examples/`: each script creates a synthetic
data-generating process, fits a model, checks model-specific behavior such as
additive reconstruction, and may save a plot.

For mathematical theory and guided exploration, use the root
[`docs/notebooks/`](../docs/notebooks/) collection instead. For exact public signatures,
use the [API reference](../docs/api/index.rst).

| Script | Focus |
|---|---|
| `example_gam.py` | binomial GAM classification and link-scale decomposition |
| `example_gam2.py` | Gaussian GAM smoothing with fixed, GCV, and REML fits |
| `example_scat_gam.py` | scaled-t location with fixed ancillary parameters |
| `example_zip_gam.py` | zero-inflated count prediction with fixed ancillary parameters |
| `example_cnorm_gam.py` | normal location with exact and censored responses |
| `example_clog_gam.py` | logistic location, paired censor bounds, and additive reconstruction |
| `example_cpois_gam.py` | censored Poisson counts, latent means, and prediction uncertainty |
| `example_bcg_gam.py` | censored Box-Cox Gaussian location with estimated lambda and SD |
| `example_soap_compositions.py` | known-boundary soap tensors and matrix-coordinate factor smooths |
| `example_bs_constraints.py` | derivative-penalized B-splines with explicit coefficient constraints |
| `example_gumbls_gam.py` | Gumbel location and scale with separate distribution predictors |
| `example_ziplss_gam.py` | hurdle counts with independent intensity and presence predictors |
| `example_multinom_gam.py` | two log-odds predictors and three category probabilities |
| `example_gevlss_gam.py` | generalized extreme value location, log scale, and shape |
| `example_shash_gamlss.py` | sinh-arcsinh location, scale, skewness, tail shape, and additive components |
| `example_twlss_gamlss.py` | Tweedie mean, power, and dispersion with EFS selection |
| `example_nam.py` | mixed-feature NAM regression |
| `example_gpnam.py` | GPNAM smooth additive regression |
| `example_nbm.py` | NBM main effects and a pairwise interaction |
| `example_nbm2.py` | NBM with one-hot categorical effects |
| `example_qnam.py` | non-crossing conditional quantiles |
| `example_treenam.py` | TreeNAM piecewise-additive effects |
| `example_ensemble.py` | jointly trained EnsembleTreeNAM versus TreeNAM |

Install the relevant backend and run a script from the repository root:

```bash
pip install -e .
python examples/example_gam2.py
```

The PNG files in this directory are outputs from representative scripts, not a
second source of documentation.
