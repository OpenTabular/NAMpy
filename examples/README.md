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
| `example_nam.py` | mixed-feature NAM regression |
| `example_gpnam.py` | GPNAM smooth additive regression |
| `example_nbm.py` | NBM main effects and a pairwise interaction |
| `example_nbm2.py` | NBM with one-hot categorical effects |
| `example_qnam.py` | non-crossing conditional quantiles |
| `example_treenam.py` | TreeNAM piecewise-additive effects |
| `example_ensemble.py` | jointly trained EnsembleTreeNAM versus TreeNAM |

Install the relevant backend and run a script from the repository root:

```bash
pip install -e ".[all]"
python examples/example_gam2.py
```

The PNG files in this directory are outputs from representative scripts, not a
second source of documentation.
