# NAMpy notebooks

Four self-contained, theory-first notebooks cover NAMpy without splitting every
architecture into a repetitive tutorial. Each notebook defines its own small
datasets and helper functions; there is no shared notebook utility directory.
Every checked-in notebook has been executed with `FAST_MODE = True`.

| Notebook | Focus |
|---|---|
| `01_nampy_core_workflow.ipynb` | Additive theory, the shared GAM/neural estimator interface, link-scale explanations, sklearn integration, and persistence |
| `02_complete_gam_guide.ipynb` | Formula and array GAMs, bases, structured smooths, constraints, families, smoothness selection, inference, diagnostics, and GAMLSS |
| `03_neural_additive_model_zoo.ipynb` | Theory and focused examples for every architecture discovered from the installed registry |
| `04_tasks_distributional_models_and_ensembles.ipynb` | Runtime regression/classification/LSS sweeps, all registered distribution families, ensembles, and the final capability report |

The runtime tables distinguish registered support from successful demonstration.
In this checkout, the NAMformer task variants fit and predict but their extracted
marginal components do not reconstruct the link output; the notebooks report
that result explicitly instead of hiding it behind a static checkmark.

From the repository root, install the complete environment and launch Jupyter:

```bash
pip install -e ".[all,docs]" jupyterlab
jupyter lab docs/notebooks
```

Set `FAST_MODE = False` inside a notebook for larger datasets and meaningful
training budgets. The fast-mode comparisons are interface demonstrations, not
competitive benchmarks.
