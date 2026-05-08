# NAMpy: Interpretable (Additive) Tabular Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

NAMpy provides interpretable additive neural models for tabular data, with support for **regression**, **classification**, and **distributional regression** tasks. Models implement scikit-learn's `BaseEstimator` interface, so they integrate with standard scikit-learn workflows for fitting, prediction, and evaluation.

## Key Features

- **Scikit-learn Compatible**: Consistent API with sklearn estimators
- **10+ Model Architectures**: NAM, GPNAM, NBM, NATT, NAMformer, and more
- **Three Task Types**: Regression, classification, and distributional regression (LSS)
- **Interpretable**: Additive structure supports feature-level interpretation
- **PyTorch Backend**: Built on modern deep learning tooling
- **Extensible**: Interfaces for custom model implementations

Most models are available for `regression`, `classification` and distributional regression, denoted by `LSS`.
Some models are specialized: `QNAM` is distributional-only, while `TreeNAM` and `SNAM` are currently regression-only.

## Integrated Models:

1. NAM
2. GPNAM
3. NBM
4. NATT
5. NAMformer
6. QNAM
7. Linear Regression (Neural)
8. NodeGAM
9. TreeNAM (Regressor)
10. SNAM (Regressor)

## Installation

### From PyPI (recommended)

```bash
pip install nampy
```

### From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/OpenTabular/NAMpy.git
cd NAMpy
pip install -e .
```

### From GitHub

Install directly from a specific branch or tag:

```bash
pip install git+https://github.com/OpenTabular/NAMpy.git@main
```

### Requirements

- Python >= 3.10
- PyTorch
- Lightning
- scikit-learn
- pandas
- numpy

## Quick Start

### Fit a Model

All NAMpy models implement sklearn `BaseEstimator` methods, including `.fit`. This enables standard tooling such as scikit-learn model selection and evaluation utilities.

```python
from nampy.models import NAMClassifier

# Initialize and fit your model
model = NAMClassifier(
    numerical_preprocessing="ple",
    n_bins=50
)

# X can be a DataFrame or any array-like that can be converted to a DataFrame.
model.fit(X, y, max_epochs=150, lr=1e-04)
```

### Make Predictions

Use the standard prediction methods:

```python
# Simple predictions
preds = model.predict(X)

# Predict probabilities (for classification)
preds = model.predict_proba(X)
```

### Regression Example

```python
from nampy.models import NAMRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = NAMRegressor(numerical_preprocessing="standardization")
model.fit(X_train, y_train, max_epochs=100, lr=1e-3)

# Evaluate
score = model.score(X_test, y_test)
print(f"R² Score: {score:.4f}")

```

## Distributional Regression with NAMLSS

NAMpy provides distributional regression through the `NAMLSS` module, which models the full response distribution rather than only the mean. This is useful when variability, skewness, or kurtosis are as important as the central tendency. Most models in NAMpy are also available as distributional models.

### Key Features of NAMLSS:

- **Full Distribution Modeling**: Unlike traditional regression models that predict a single value (e.g., the mean), `NAMLSS` models the entire distribution of the response variable. This supports predictions of quantiles, variance, and higher moments.
- **Customizable Distribution Types**: `NAMLSS` supports a variety of distribution families (e.g., Normal, Poisson, Gamma, Beta), making it suitable for response variables ranging from continuous to count data.
- **Location, Scale, Shape Parameters**: The model predicts parameters corresponding to the location, scale, and shape of the distribution, providing direct access to distributional characteristics.
- **Enhanced Predictive Uncertainty**: By modeling the full distribution, `NAMLSS` provides explicit predictive uncertainty estimates for downstream decisions.



### Available Distribution Classes:

`NAMLSS` includes a range of distribution classes for statistical modeling needs. The available distribution classes include:

- `normal`: Normal Distribution for modeling continuous data with a symmetric distribution around the mean.
- `poisson`: Poisson Distribution for modeling count data that for instance represent the number of events occurring within a fixed interval.
- `gamma`: Gamma Distribution for modeling continuous data that is skewed and bounded at zero, often used for waiting times.
- `beta`: Beta Distribution for modeling data that is bounded between 0 and 1, useful for proportions and percentages.
- `dirichlet`: Dirichlet Distribution for modeling multivariate data where individual components are correlated, and the sum is constrained to 1.
- `studentt`: Student's T-Distribution for modeling data with heavier tails than the normal distribution, useful when the sample size is small.
- `negativebinom`: Negative Binomial Distribution for modeling count data with over-dispersion relative to the Poisson distribution.
- `inversegamma`: Inverse Gamma Distribution, often used as a prior distribution in Bayesian inference for scale parameters.
- `categorical`: Categorical Distribution for modeling categorical data with more than two categories.
- `quantile`: Quantile regression for estimating conditional quantiles.
- `robustnormal`: Robust Normal Distribution for heavy-tailed targets.

These distribution classes allow `NAMLSS` to model a wide variety of data types and distributions.


### Getting Started with NAMLSS:

To integrate distributional regression into your workflow with `NAMLSS`, initialize the model with the desired configuration, similar to other NAMpy models:

```python
from nampy.models import NAMLSS

# Initialize the NAMLSS model
model = NAMLSS()

# Fit the model to your data
model.fit(
    X, 
    y, 
    max_epochs=150, 
    lr=1e-04, 
    patience=10,     
    family="normal"  # define your distribution
)

# Predict distribution parameters
dist_params = model.predict(X_test)
```


## Implement Your Own Model

NAMpy supports integration of custom models into the existing logic. Implement a PyTorch model and define its forward pass, but inherit from NAMpy's `BaseModel` rather than `nn.Module`. Each NAMpy model takes three main arguments: the number of classes (e.g., 1 for regression or 2 for binary classification), `cat_feature_info`, and `num_feature_info` for categorical and numerical feature information, respectively. These are passed as dictionaries, with variable names as the keys. Additionally, you can provide a config argument, which can either be a custom configuration or one of the provided default configs.

A key aspect of NAMpy is that the inputs to the forward passes are dictionaries of tensors. This supports models that treat different data types differently and directly maps feature/variable predictions to input features in additive models. 

Example workflow for a custom model:


1. First, define your config:
Use a dataclass to specify hyperparameters and other settings for your model.

```python
from dataclasses import dataclass

@dataclass
class MyConfig:
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
```

2. Second, define your model:
Define your custom model as you would for an `nn.Module`. The main difference is that you will inherit from `BaseModel` and use the provided feature information to construct your layers. To integrate your model into the existing API, define the architecture and the forward pass. Note that the forward pass must return a dictionary with the key "output" for the final model prediction. This can be multi-dimensional, for example for classification or distributional regression. Beyond that, the dictionary can contain anything but often includes single feature/variable predictions for further processing or plotting.

```python
from nampy.basemodels import BaseModel
import torch
import torch.nn as nn

class MyCustomModel(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        total_input_size = sum([input_shape for input_shape in num_feature_info.values()]) + len(cat_feature_info)
        
        # Define a simple MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(total_input_size, 128),  # Adjust the hidden layer size as needed
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of the NAM model.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.

        Returns
        -------
        dict
            Dictionary containing the output tensor and the original feature values.
        """
        # Concatenate all numerical features into a single tensor
        num_features_tensor = torch.cat([num_features[key] for key in num_features.keys()], dim=1)

        # Concatenate all categorical features into a single tensor
        cat_features_tensor = torch.cat([cat_features[key] for key in cat_features.keys()], dim=1)

        # Concatenate all features into a single input tensor
        input_tensor = torch.cat([num_features_tensor, cat_features_tensor], dim=1)

        # Pass the concatenated tensor through the MLP
        output = self.mlp(input_tensor)

        # return a dictionary, with the key "output" for the final predictions
        # This is used, for when the model (e.g. for plotting) also returns feature predictions
        return {"output": output}


```

3. Leverage the NAMpy API:
You can build a regression, classification or distributional regression model that can leverage all of NAMpy's built-in methods, by using the following:

```python
from nampy.models import SklearnBaseRegressor

class MyRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=MyCustomModel, config=MyConfig, **kwargs)
```

4. Train and evaluate your model:
You can now fit, evaluate, and predict with your custom model using the same APIs as other NAMpy models. For classification or distributional regression, inherit from `SklearnBaseClassifier` or `SklearnBaseLSS` respectively.

```python
regressor = MyRegressor(numerical_preprocessing="ple")
regressor.fit(X_train, y_train, max_epochs=50)
predictions = regressor.predict(X_test)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## Citation

If you use NAMpy in your research, please cite:

```bibtex
@software{nampy2024,
  title={NAMpy: Interpretable Tabular Deep Learning},
  author={Thielmann, Anton},
  year={2024},
  url={https://github.com/OpenTabular/NAMpy}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

Comprehensive documentation is available:

- **Build Locally**: `make docs` (then open `docs/_build/html/index.html`)
- **Read the Docs**: https://nampy.readthedocs.io (coming soon)
- **GitHub Pages**: https://opentabular.github.io/NAMpy (coming soon)

Documentation includes:
- Installation guide
- Quick start tutorial
- Comprehensive user guide
- API reference (auto-generated)
- Model comparison guide
- Examples and tutorials
- FAQ

## Links

- **Documentation**: See `docs/` directory or build with `make docs`
- **Source Code**: https://github.com/OpenTabular/NAMpy
- **Issue Tracker**: https://github.com/OpenTabular/NAMpy/issues
- **PyPI**: https://pypi.org/project/nampy/

## Acknowledgments

NAMpy builds upon research in neural additive models and interpretable machine learning. Special thanks to the open-source community and contributors.

---

Made with ❤️ by the OpenTabular team
