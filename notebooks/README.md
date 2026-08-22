# Model notebooks

Concise conceptual and API tutorials for every supported NAMpy model. Start
with `00_overview.ipynb`; training cells are disabled by default.

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
| `12_ensemble_treenam.ipynb` | jointly trained EnsembleTreeNAM |
| `13_nodegam.ipynb` | NodeGAM |
| `14_natt.ipynb` | NATT |
| `15_namformer.ipynb` | NAMformer |
| `16_qnam.ipynb` | QNAM |
| `17_spline_nam.ipynb` | SplineNAM |
| `18_neural_ensemble.ipynb` | independent NeuralEnsemble |

From the repository root, install the complete environment and launch Jupyter:

```bash
pip install -e ".[all,docs]" jupyterlab
jupyter lab notebooks
```
