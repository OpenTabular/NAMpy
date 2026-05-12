# NAMpy: Interpretable Additive Tabular Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

NAMpy provides interpretable additive neural models for tabular regression,
classification, and distributional regression. Its high-level estimators follow
the scikit-learn API and use PyTorch/Lightning internally.

## Installation

```bash
pip install nampy
```

Install from source:

```bash
git clone https://github.com/Ananyapam7/NAMpy.git
cd NAMpy
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from nampy.models import NAMRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = NAMRegressor(numerical_preprocessing="standardization")
model.fit(X_train, y_train, max_epochs=100, lr=1e-3)

score = model.score(X_test, y_test)
predictions = model.predict(X_test)
print(f"R2 score: {score:.4f}")
```

Classification uses the same estimator pattern:

```python
from nampy.models import NAMClassifier

model = NAMClassifier(numerical_preprocessing="ple", n_bins=50)
model.fit(X, y, max_epochs=150, lr=1e-4)
probabilities = model.predict_proba(X)
```

## Distributional Regression

`NAMLSS` models full response distributions instead of only the conditional
mean. It supports families such as normal, poisson, gamma, beta, student-t,
negative binomial, categorical, quantile, and robust normal distributions.

```python
from nampy.models import NAMLSS

model = NAMLSS()
model.fit(X, y, max_epochs=150, lr=1e-4, patience=10, family="normal")
distribution_parameters = model.predict(X_test)
```

## Available Model Wrappers

- Regression: `NAMRegressor`, `GPNAMRegressor`, `NBMRegressor`,
  `NATTRegressor`, `NAMformerRegressor`, `NodeGAMRegressor`,
  `SplineNAMRegressor`, `TreeNAMRegressor`, `SparseNAMRegressor`,
  `LinRegRegressor`
- Classification: `NAMClassifier`, `GPNAMClassifier`, `NBMClassifier`,
  `NATTClassifier`, `NAMformerClassifier`, `NodeGAMClassifier`,
  `SplineNAMClassifier`,
  `LinRegClassifier`
- Distributional regression: `NAMLSS`, `GPNAMLSS`, `NBMLSS`, `NATTLSS`,
  `NAMformerLSS`, `NodeGAMLSS`, `SplineNAMLSS`, `LinRegLSS`, `QNAM`

## Interpretability and Diagnostics

Fitted sklearn-style wrappers expose generic additive-model helpers:

```python
terms = model.predict_terms(X_test)
raw = model.predict_feature_vals(X_test)
prediction = raw["prediction"]
regularization = raw["regularization"]
importance = model.feature_importance(X_test)
model.plot_terms(X_test)
info = model.summary()
```

## Documentation

Build the documentation locally with:

```bash
make docs
```

The generated HTML is written to `docs/_build/html/index.html`.

## Citation

If you use NAMpy in research, please cite:

```bibtex
@software{nampy2024,
  title={NAMpy: Interpretable Tabular Deep Learning},
  author={De, Ananyapam and Thielmann, Anton},
  year={2024},
  url={https://github.com/Ananyapam7/NAMpy}
}
```

## Links

- Documentation: see `docs/` or run `make docs`
- Source code: https://github.com/Ananyapam7/NAMpy
- Issue tracker: https://github.com/Ananyapam7/NAMpy/issues
- PyPI: https://pypi.org/project/nampy/

## License

NAMpy is released under the MIT License. See [LICENSE](LICENSE).
