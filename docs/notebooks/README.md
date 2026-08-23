# Model notebooks

Compact, visual tutorials for every supported NAMpy model. Each notebook pairs
the essential theory with a runnable fit, predictive checks, additive
explanations, term importance, and model-specific plots. Start with
`00_overview.ipynb`; every checked-in notebook includes executed outputs.

`01_gam.ipynb` is a longer energy-system case study covering penalized
likelihood, smooth construction, smoothing criteria and optimizers, response
families, multi-linear-predictor models, shape constraints, inference, and
diagnostics.

| Notebook | Model |
|---|---|
| `01_gam.ipynb` | mgcv-aligned GAM and sklearn adapters |
| `02_linreg.ipynb` | LinReg |
| `03_nam.ipynb` | NAM |
| `04_snam.ipynb` | SNAM |
| `05_sian.ipynb` | SIAN |
| `06_gpnam.ipynb` | GPNAM |
| `07_igann.ipynb` | IGANN |
| `08_nbm.ipynb` | NBM |
| `09_spam.ipynb` | SPAM |
| `10_nbm_spam.ipynb` | NBM-SPAM |
| `11_treenam.ipynb` | TreeNAM |
| `13_nodegam.ipynb` | NodeGAM |
| `14_natt.ipynb` | NATT |
| `15_namformer.ipynb` | NAMformer |
| `16_qnam.ipynb` | QNAM |
| `17_spline_nam.ipynb` | SplineNAM |
| `18_neural_ensemble.ipynb` | independent NeuralEnsemble |

From the repository root, install the complete environment and launch Jupyter:

```bash
pip install -e ".[all,docs]" jupyterlab
jupyter lab docs/notebooks
```
