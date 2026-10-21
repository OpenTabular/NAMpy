# NAMpy notebooks

Four self-contained notebooks cover the shared workflow, statistical GAMs,
neural architectures, and distributional models. Each notebook defines the
data and helper functions needed to run it independently.

| Notebook | Focus |
|---|---|
| `01_getting_started.ipynb` | **Getting started with NAMpy:** fitting, prediction, interpretation, scikit-learn integration, and persistence |
| `02_statistical_gams.ipynb` | **Statistical GAMs:** formulas, smooth bases, structured effects, constraints, inference, diagnostics, and GAMLSS |
| `03_neural_models.ipynb` | **Neural models:** feature functions, interaction structures, and interpretation across architectures |
| `04_distributional_models.ipynb` | **Distributional models and ensembles:** classification, distributional parameters, quantiles, and ensemble uncertainty |

The task tables report both the capabilities declared by the model registry and
the results returned by the fitted estimators. Known reconstruction limitations
are shown directly in those tables.

From the repository root, install the complete environment and launch Jupyter:

```bash
pip install -e ".[docs]" jupyterlab
jupyter lab docs/notebooks
```
