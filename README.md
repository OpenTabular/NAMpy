# NAMGCV: Interpretable (Additive) Tabular Deep Learning

:exclamation:
📚 The paper_list.md includes interesting papers on Additive Models and their 1-2 sentence summaries. If you know any further interesting papers on the topic, please include them in this list. 📝 
:exclamation:

NAMGCV is a Python package that brings the power of advanced deep learning architectures to tabular data, offering a suite of models for regression, classification, and distributional regression tasks. Designed with ease of use in mind, NAMGCV models adhere to scikit-learn's `BaseEstimator` interface, making them highly compatible with the familiar scikit-learn ecosystem. This means you can fit, predict, and evaluate using NAMGCV models just as you would with any traditional scikit-learn model, but with the added performance and flexibility of deep learning.


All models are available for `regression`, `classification` and distributional regression, denoted by `LSS`.
Hence, they are available as e.g. `NAMRegressor`, `NAMClassifier` or `NAMLSS`.

## Integrated Models:

1. NAM
2. GPNAM
3. NBM
4. NATT
5. NAMformer
6. QNAM
7. Linear Regression (Neural)
8. GAM (Base architecture)


## Fit a Model
Fitting a model in NAMGCV is as simple as it gets. All models in NAMGCV are sklearn BaseEstimators. Thus the `.fit` method is implemented for all of them. Additionally, this allows for using all other sklearn inherent methods such as their built in hyperparameter optimization tools.

```python
from NAMGCV.models import NAMClassifier
# Initialize and fit your model
model = NAMClassifier(
    numerical_preprocessing="ple",
    n_bins=50
)

# X can be a dataframe or something that can be easily transformed into a pd.DataFrame as a np.array
model.fit(X, y, max_epochs=150, lr=1e-04)
```

Predictions are also easily obtained:
```python
# simple predictions
preds = model.predict(X)

# Predict probabilities
preds = model.predict_proba(X)
```


## Distributional Regression with NAMLSS

NAMGCV introduces an approach to distributional regression through its `NAMLSS` module, allowing users to model the full distribution of a response variable, not just its mean. This method is particularly valuable in scenarios where understanding the variability, skewness, or kurtosis of the response distribution is as crucial as predicting its central tendency. All available moedls in NAMGCV are also available as distributional models.

### Key Features of NAMLSS:

- **Full Distribution Modeling**: Unlike traditional regression models that predict a single value (e.g., the mean), `NAMLSS` models the entire distribution of the response variable. This allows for more informative predictions, including quantiles, variance, and higher moments.
- **Customizable Distribution Types**: `NAMLSS` supports a variety of distribution families (e.g., Gaussian, Poisson, Binomial), making it adaptable to different types of response variables, from continuous to count data.
- **Location, Scale, Shape Parameters**: The model predicts parameters corresponding to the location, scale, and shape of the distribution, offering a nuanced understanding of the data's underlying distributional characteristics.
- **Enhanced Predictive Uncertainty**: By modeling the full distribution, `NAMLSS` provides richer information on predictive uncertainty, enabling more robust decision-making processes in uncertain environments.



### Available Distribution Classes:

`NAMLSS` offers a wide range of distribution classes to cater to various statistical modeling needs. The available distribution classes include:

- `normal`: Normal Distribution for modeling continuous data with a symmetric distribution around the mean.
- `poisson`: Poisson Distribution for modeling count data that for instance represent the number of events occurring within a fixed interval.
- `gamma`: Gamma Distribution for modeling continuous data that is skewed and bounded at zero, often used for waiting times.
- `beta`: Beta Distribution for modeling data that is bounded between 0 and 1, useful for proportions and percentages.
- `dirichlet`: Dirichlet Distribution for modeling multivariate data where individual components are correlated, and the sum is constrained to 1.
- `studentt`: Student's T-Distribution for modeling data with heavier tails than the normal distribution, useful when the sample size is small.
- `negativebinom`: Negative Binomial Distribution for modeling count data with over-dispersion relative to the Poisson distribution.
- `inversegamma`: Inverse Gamma Distribution, often used as a prior distribution in Bayesian inference for scale parameters.
- `categorical`: Categorical Distribution for modeling categorical data with more than two categories.

These distribution classes allow `NAMLSS` to flexibly model a wide variety of data types and distributions, providing users with the tools needed to capture the full complexity of their data.


### Getting Started with NAMGCVLSS:

To integrate distributional regression into your workflow with `NAMLSS`, start by initializing the model with your desired configuration, similar to other NAMGCV models:

```python
from NAMGCV.models import NAMLSS

# Initialize the NAMGCVLSS model
model = NAMLSS()

# Fit the model to your data
model.fit(
    X, 
    y, 
    max_epochs=150, 
    lr=1e-04, 
    patience=10,     
    family="normal" # define your distribution
    )

```


### Implement Your Own Model

NAMGCV allows users to easily integrate their custom models into the existing logic. This process is designed to be straightforward, making it simple to create a PyTorch model and define its forward pass. Instead of inheriting from `nn.Module`, you inherit from NAMGCV's `BaseModel`. Each NAMGCV model takes three main arguments: the number of classes (e.g., 1 for regression or 2 for binary classification), `cat_feature_info`, and `num_feature_info` for categorical and numerical feature information, respectively. These are passed as dictionaries, with variable names as the keys. Additionally, you can provide a config argument, which can either be a custom configuration or one of the provided default configs.

One of the key advantages of using NAMGCV is that the inputs to the forward passes are dictionaries of tensors. While this might be unconventional, it is highly beneficial for models that treat different data types differently and directly maps feature/variable predictions to input features in additive models. 

Here's how you can implement a custom model with NAMGCV:


1. First, define your config:
The configuration class allows you to specify hyperparameters and other settings for your model. This can be done using a simple dataclass.

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
Define your custom model just as you would for an `nn.Module`. The main difference is that you will inherit from `BaseModel` and use the provided feature information to construct your layers. To integrate your model into the existing API, you only need to define the architecture and the forward pass. Note, that the forward pass must return a dictionary with the key "outpu" for the final model prediction. This can be multi-dimensinoal, e.g. for classification or distributional regression. Beyond that, the dictionary can contain anything but commonly includes single feature/variable predictions for e.g. further processing/plotting.

```python
from NAMGCV.base_models import BaseModel
import torch
import torch.nn

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

3. Leverage the NAMGCV API:
You can build a regression, classification or distributional regression model that can leverage all of NAMGCVs built-in methods, by using the following:

```python
from NAMGCV.models import SklearnBaseRegressor

class MyRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=MyCustomModel, config=MyConfig, **kwargs)
```

4. Train and evaluate your model:
You can now fit, evaluate, and predict with your custom model just like with any other NAMGCV model. For classification or distributional regression, inherit from `SklearnBaseClassifier` or `SklearnBaseLSS` respectively.

```python
regressor = MyRegressor(numerical_preprocessing="ple")
regressor.fit(X_train, y_train, max_epochs=50)
```

